# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
HTTP API with SSE streaming for Delia CLI.

Provides endpoints for the TypeScript CLI frontend:
- POST /api/agent/run - Run agent with SSE streaming
- GET /api/health - Health check
- GET /api/sessions - List sessions
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator

import structlog
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .backend_manager import backend_manager
from .llm import call_llm
from .routing import select_model, detect_chat_task_type
from .mcp_server import _select_optimal_backend_v2
from .tools.agent import AgentConfig, AgentResult, run_agent_loop
from .tools.builtins import get_default_tools
from .tools.parser import ParsedToolCall
from .tools.executor import ToolResult
from .types import Workspace

log = structlog.get_logger()


async def sse_event(event: str, data: dict[str, Any]) -> str:
    """Format an SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def agent_run_stream(
    task: str,
    model: str | None = None,
    workspace: str | None = None,
    max_iterations: int = 10,
    tools: list[str] | None = None,
    backend_type: str | None = None,
    allow_write: bool = False,
    allow_exec: bool = False,
    yolo: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Run agent and yield SSE events.

    Args:
        task: Task description
        model: Model override
        workspace: Workspace directory to confine operations
        max_iterations: Max tool call iterations
        tools: Optional tool filter
        backend_type: Backend type preference
        allow_write: Enable file write tool (--allow-write)
        allow_exec: Enable shell exec tool (--allow-exec)
        yolo: Skip confirmation prompts (--yolo)

    Events:
        - status: Routing/model selection status
        - thinking: Agent is processing
        - tool_call: Tool is being called
        - tool_result: Tool returned result
        - token: Streaming token (when available)
        - error: Error occurred
        - done: Agent completed
    """
    start_time = time.time()

    # Yield initial status event
    yield await sse_event("status", {
        "phase": "routing",
        "message": "Selecting backend...",
    })

    # Select backend
    try:
        backend_provider, backend_obj = await _select_optimal_backend_v2(
            task, None, "analyze", backend_type
        )
    except Exception as e:
        yield await sse_event("error", {"message": f"Backend selection failed: {e}"})
        return

    backend_name = backend_obj.name if backend_obj else "unknown"
    backend_id = backend_obj.id if backend_obj else "unknown"
    yield await sse_event("status", {
        "phase": "routing",
        "message": f"Backend: {backend_name}",
        "details": {"backend": backend_id}
    })

    # Select model
    try:
        selected_model = await select_model(
            task_type="analyze",
            content_size=len(task),
            model_override=model,
            content=task,
        )
    except Exception as e:
        yield await sse_event("error", {"message": f"Model selection failed: {e}"})
        return

    yield await sse_event("status", {
        "phase": "model",
        "message": f"Model: {selected_model}",
        "details": {
            "tier": "analyze",
            "model": selected_model,
            "backend": backend_id,
            "task_type": "agent",
        }
    })

    # Set up workspace
    workspace_obj = Workspace(root=workspace) if workspace else None

    # Set up tool registry with permission flags
    registry = get_default_tools(
        workspace=workspace_obj,
        allow_write=allow_write,
        allow_exec=allow_exec,
    )

    # Log permission status
    if allow_write or allow_exec:
        log.info(
            "agent_permissions_enabled",
            allow_write=allow_write,
            allow_exec=allow_exec,
            yolo=yolo,
        )
        yield await sse_event("status", {
            "phase": "model",
            "message": f"Permissions: write={'yes' if allow_write else 'no'} exec={'yes' if allow_exec else 'no'}",
            "details": {
                "allow_write": allow_write,
                "allow_exec": allow_exec,
                "yolo": yolo,
            }
        })

    # Filter tools if specified
    if tools:
        registry = registry.filter(tools)

    # Track events for streaming
    tool_calls_made: list[str] = []

    # Create event queue for streaming from agent loop
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    # Create LLM callable that yields events
    async def streaming_llm_call(
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> str:
        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "tool")
                prompt_parts.append(f"[Tool Result - {tool_name}]\n{content}")

        combined_prompt = "\n\n".join(prompt_parts)

        # Signal LLM call starting
        await event_queue.put({"type": "llm_start"})

        result = await call_llm(
            model=selected_model,
            prompt=combined_prompt,
            system=system,
            task_type="agent",
            original_task="agent",
            language="unknown",
            backend_obj=backend_obj,
        )

        if result.get("success"):
            response = result.get("response", "")
            # Signal LLM call done
            await event_queue.put({"type": "llm_done", "response": response})
            return response
        else:
            error = result.get("error", "LLM call failed")
            await event_queue.put({"type": "llm_error", "error": error})
            raise RuntimeError(error)

    # Wrap tool execution to emit events
    original_registry = registry

    class StreamingToolRegistry:
        """Wrapper that emits events on tool calls."""

        def __init__(self, inner: Any, queue: asyncio.Queue):
            self._inner = inner
            self._queue = queue

        def get(self, name: str):
            tool = self._inner.get(name)
            if not tool:
                return None

            # Wrap handler to emit events
            original_handler = tool.handler

            async def wrapped_handler(**kwargs):
                # Emit tool_call event
                await self._queue.put({
                    "type": "tool_call",
                    "name": name,
                    "args": kwargs,
                })

                start = time.time()
                try:
                    result = await original_handler(**kwargs)
                    elapsed_ms = int((time.time() - start) * 1000)

                    # Emit tool_result event
                    await self._queue.put({
                        "type": "tool_result",
                        "name": name,
                        "success": True,
                        "output": result[:500] if len(result) > 500 else result,
                        "elapsed_ms": elapsed_ms,
                    })

                    return result
                except Exception as e:
                    elapsed_ms = int((time.time() - start) * 1000)
                    await self._queue.put({
                        "type": "tool_result",
                        "name": name,
                        "success": False,
                        "output": str(e),
                        "elapsed_ms": elapsed_ms,
                    })
                    raise

            # Return modified tool
            from .tools.registry import ToolDefinition
            return ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                handler=wrapped_handler,
            )

        def get_tool_prompt(self):
            return self._inner.get_tool_prompt()

        def get_openai_schemas(self):
            return self._inner.get_openai_schemas()

        def list_tools(self):
            return self._inner.list_tools()

        def __len__(self):
            return len(self._inner)

        def __contains__(self, name: str):
            return name in self._inner

    streaming_registry = StreamingToolRegistry(registry, event_queue)

    # Agent config
    agent_config = AgentConfig(
        max_iterations=max_iterations,
        timeout_per_tool=30.0,
        total_timeout=300.0,
        parallel_tools=True,
        native_tool_calling=False,
    )

    # Run agent in background task
    agent_task = asyncio.create_task(
        run_agent_loop(
            call_llm=streaming_llm_call,
            prompt=task,
            system_prompt=None,
            registry=streaming_registry,
            model=selected_model,
            config=agent_config,
        )
    )

    # Stream events from queue while agent runs
    try:
        while not agent_task.done():
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)

                if event["type"] == "tool_call":
                    tool_calls_made.append(event["name"])
                    yield await sse_event("tool_call", {
                        "name": event["name"],
                        "args": event["args"],
                    })

                elif event["type"] == "tool_result":
                    yield await sse_event("tool_result", {
                        "name": event["name"],
                        "success": event["success"],
                        "output": event["output"],
                        "elapsed_ms": event["elapsed_ms"],
                    })

                elif event["type"] == "llm_start":
                    yield await sse_event("thinking", {"status": "Generating response..."})

                elif event["type"] == "llm_done":
                    # Response will come from agent result
                    pass

                elif event["type"] == "llm_error":
                    yield await sse_event("error", {"message": event["error"]})

            except asyncio.TimeoutError:
                # No event ready, continue loop
                continue

        # Get agent result
        result: AgentResult = await agent_task

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Validate response quality (same as chat/delegation tools)
        quality_score = 0.0
        if result.success and result.response:
            from .quality import validate_response
            quality_result = validate_response(result.response, "analyze")

            # Emit quality status
            yield await sse_event("status", {
                "phase": "quality",
                "message": f"Quality: {quality_result.overall:.0%}",
                "details": {
                    "quality_score": quality_result.overall,
                    "tier": "analyze",
                    "model": selected_model,
                    "backend": backend_id,
                }
            })

            quality_score = quality_result.overall

            # Log quality for monitoring
            log.info(
                "agent_response_quality",
                quality=quality_result.overall,
                valid=quality_result.is_valid,
                model=selected_model,
                iterations=result.iterations,
            )

            # If quality is very low, log warning
            if quality_result.overall < 0.3:
                log.warning(
                    "low_quality_agent_response",
                    quality=quality_result.overall,
                    reason=quality_result.reason,
                    model=selected_model,
                )

        # Yield final response
        yield await sse_event("response", {"content": result.response})

        # Yield done event with summary (includes quality score)
        yield await sse_event("done", {
            "success": result.success,
            "iterations": result.iterations,
            "tool_calls": tool_calls_made,
            "elapsed_ms": elapsed_ms,
            "model": selected_model,
            "backend": backend_name,
            "stopped_reason": result.stopped_reason,
            "quality": quality_score,
        })

    except Exception as e:
        log.error("agent_stream_error", error=str(e))
        yield await sse_event("error", {"message": str(e)})
        yield await sse_event("done", {
            "success": False,
            "iterations": 0,
            "tool_calls": tool_calls_made,
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "model": selected_model,
            "backend": backend_name,
            "stopped_reason": "error",
        })


async def agent_run_handler(request: Request) -> StreamingResponse:
    """Handle POST /api/agent/run - SSE streaming agent execution."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    task = body.get("task")
    if not task:
        return JSONResponse({"error": "Missing 'task' field"}, status_code=400)

    model = body.get("model")
    workspace = body.get("workspace")
    max_iterations = body.get("max_iterations", 10)
    tools = body.get("tools")  # Optional list of tool names
    backend_type = body.get("backend_type")

    # Permission flags (all disabled by default for security)
    allow_write = body.get("allow_write", False)
    allow_exec = body.get("allow_exec", False)
    yolo = body.get("yolo", False)

    return StreamingResponse(
        agent_run_stream(
            task=task,
            model=model,
            workspace=workspace,
            max_iterations=max_iterations,
            tools=tools,
            backend_type=backend_type,
            allow_write=allow_write,
            allow_exec=allow_exec,
            yolo=yolo,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def health_handler(request: Request) -> JSONResponse:
    """Handle GET /api/health - Health check."""
    backends = []
    for backend in backend_manager.get_enabled_backends():
        backends.append({
            "id": backend.id,
            "name": backend.name,
            "provider": backend.provider,
            "type": backend.type,
        })

    return JSONResponse({
        "status": "ok",
        "backends": backends,
    })


async def status_handler(request: Request) -> JSONResponse:
    """Handle GET /api/status - Full system status with MDAP metrics.

    Returns comprehensive status including:
    - Backend health and scores
    - Voting statistics (consensus rate, rejection reasons)
    - Quality metrics per tier
    - Routing configuration
    - Usage statistics
    """
    from .routing import BackendScorer
    from .config import get_backend_metrics, config
    from .voting_stats import get_voting_stats_tracker

    # Get health status from BackendManager
    health_status = await backend_manager.get_health_status()

    # Add performance scores to each backend
    weights = backend_manager.get_scoring_weights()
    scorer = BackendScorer(weights=weights)
    backend_lookup = {b.id: b for b in backend_manager.backends.values()}

    for backend_info in health_status["backends"]:
        backend_obj = backend_lookup.get(backend_info["id"])
        if backend_obj and backend_info.get("enabled"):
            score = scorer.score(backend_obj)
            backend_info["score"] = round(score, 3)

            # Add metrics summary if available
            metrics = get_backend_metrics(backend_info["id"])
            if metrics.total_requests > 0:
                backend_info["metrics"] = {
                    "success_rate": round(metrics.success_rate * 100, 1),
                    "latency_p50_ms": round(metrics.latency_p50, 1),
                    "throughput_tps": round(metrics.tokens_per_second, 1),
                    "total_requests": metrics.total_requests,
                }

    # Get voting stats
    voting_tracker = get_voting_stats_tracker()
    voting_stats = voting_tracker.get_stats()

    # Get quality config
    from .quality import get_quality_validator
    validator = get_quality_validator()
    quality_config = {
        "min_response_length": validator.config.min_response_length,
        "max_response_length_tokens": validator.config.max_response_length_tokens,
        "ngram_uniqueness_threshold": validator.config.ngram_uniqueness_threshold,
        "min_vocabulary_diversity": validator.config.min_vocabulary_diversity,
    }

    return JSONResponse({
        "status": health_status["status"],
        "active_backend": health_status.get("active_backend"),
        "backends": health_status["backends"],
        "routing": health_status.get("routing", {}),
        "voting": voting_stats,
        "quality_config": quality_config,
    })


async def models_handler(request: Request) -> JSONResponse:
    """Handle GET /api/models - List available models per tier.

    Returns models configured for each tier with quality metrics.
    """
    from .voting_stats import get_voting_stats_tracker

    # Get tier stats for quality info
    voting_tracker = get_voting_stats_tracker()
    stats = voting_tracker.get_stats()
    tier_stats = stats.get("tiers", {})

    # Get models from active backend
    active_backend = None
    for backend in backend_manager.get_enabled_backends():
        active_backend = backend
        break

    if not active_backend:
        return JSONResponse({
            "error": "No active backend",
            "models": {},
        }, status_code=503)

    models = {}
    for tier in ["quick", "coder", "moe", "thinking"]:
        model_name = active_backend.models.get(tier)
        tier_info = tier_stats.get(tier, {})

        models[tier] = {
            "model": model_name,
            "backend": active_backend.id,
            "quality_ema": tier_info.get("quality_ema", 0.5),
            "avg_quality": tier_info.get("avg_quality", 0.0),
            "calls": tier_info.get("calls", 0),
            "rejection_rate": tier_info.get("rejection_rate", 0.0),
            "consensus_rate": tier_info.get("consensus_rate", 0.0),
        }

    return JSONResponse({
        "active_backend": active_backend.id,
        "models": models,
    })


async def backends_handler(request: Request) -> JSONResponse:
    """Handle GET /api/backends - List backends with scores and metrics.

    Returns all backends with:
    - Health status
    - Performance scores (from BackendScorer)
    - Live metrics (success rate, latency, throughput)
    - Affinity scores
    """
    from .routing import BackendScorer
    from .config import get_backend_metrics, get_affinity_tracker

    weights = backend_manager.get_scoring_weights()
    scorer = BackendScorer(weights=weights)
    affinity_tracker = get_affinity_tracker()

    backends = []
    for backend in backend_manager.backends.values():
        # Base info
        info = {
            "id": backend.id,
            "name": backend.name,
            "provider": backend.provider,
            "type": backend.type,
            "enabled": backend.enabled,
            "url": backend.url,
            "models": backend.models,
        }

        if backend.enabled:
            # Add score
            info["score"] = round(scorer.score(backend), 3)

            # Add live metrics
            metrics = get_backend_metrics(backend.id)
            if metrics.total_requests > 0:
                info["metrics"] = {
                    "success_rate": round(metrics.success_rate * 100, 1),
                    "latency_p50_ms": round(metrics.latency_p50, 1),
                    "latency_p95_ms": round(metrics.latency_p95, 1),
                    "throughput_tps": round(metrics.tokens_per_second, 1),
                    "total_requests": metrics.total_requests,
                }

            # Add affinity scores per task type
            affinities = {}
            for task_type in ["quick", "coder", "moe", "thinking"]:
                affinity = affinity_tracker.get_affinity(backend.id, task_type)
                if affinity != 0.5:  # Only include non-default
                    affinities[task_type] = round(affinity, 3)
            if affinities:
                info["affinities"] = affinities

        backends.append(info)

    # Sort by score (enabled first, then by score)
    backends.sort(key=lambda b: (not b.get("enabled", False), -b.get("score", 0)))

    return JSONResponse({
        "backends": backends,
        "scoring_weights": weights,
    })


async def sessions_list_handler(request: Request) -> JSONResponse:
    """Handle GET /api/sessions - List sessions."""
    from .session_manager import SessionManager

    manager = SessionManager()
    sessions = manager.list_sessions()

    return JSONResponse({
        "sessions": sessions,
    })


async def session_get_handler(request: Request) -> JSONResponse:
    """Handle GET /api/sessions/{id} - Get session."""
    from .session_manager import SessionManager

    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)

    manager = SessionManager()
    session = manager.get_session(session_id)

    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    return JSONResponse(session.to_dict())


async def chat_stream(
    session_id: str | None,
    message: str,
    model: str | None = None,
    backend_type: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream chat response.

    Events:
        - session: Session created/loaded
        - thinking: Processing
        - token: Streaming token
        - response: Full response
        - error: Error occurred
        - done: Complete
    """
    from .session_manager import SessionManager

    start_time = time.time()
    manager = SessionManager()

    # Get or create session
    if session_id:
        session = manager.get_session(session_id)
        if not session:
            yield await sse_event("error", {"message": f"Session {session_id} not found"})
            return
    else:
        session = manager.create_session(metadata={"source": "chat"})
        session_id = session.session_id
        yield await sse_event("session", {"id": session_id, "created": True})

    # Check for explicit model/tier commands BEFORE routing
    import re
    model_command = None
    message_lower = message.lower().strip()

    # Detect explicit model switch commands
    model_switch_patterns = [
        (r"\b(use|load|switch to|activate)\b.*(quick|coder|coding|moe|thinking)\b.*(model|tier)?", "tier_switch"),
        (r"\b(quick|coder|coding|moe|thinking)\b\s*(model|tier|mode)", "tier_mention"),
        (r"^(quick|coder|coding|moe|thinking)$", "tier_only"),
    ]

    # Map aliases to canonical tier names
    tier_aliases = {"coding": "coder", "quick": "quick", "coder": "coder", "moe": "moe", "thinking": "thinking"}

    for pattern, cmd_type in model_switch_patterns:
        match = re.search(pattern, message_lower)
        if match:
            # Extract the tier from the match
            matched_text = match.group(0)
            for alias, tier in tier_aliases.items():
                if alias in matched_text:
                    model_command = tier
                    break
            break

    # If explicit model command, override routing
    if model_command:
        detected_task = model_command
        yield await sse_event("status", {
            "phase": "model",
            "message": f"Switching to {model_command} tier",
            "details": {"tier": model_command}
        })
    else:
        yield await sse_event("status", {
            "phase": "routing",
            "message": "Selecting backend...",
        })

    # Select backend
    try:
        backend_provider, backend_obj = await _select_optimal_backend_v2(
            message, None, "analyze", backend_type
        )
    except Exception as e:
        yield await sse_event("error", {"message": f"Backend selection failed: {e}"})
        return

    backend_name = backend_obj.name if backend_obj else "unknown"
    backend_id = backend_obj.id if backend_obj else "unknown"
    if not model_command:
        yield await sse_event("status", {
            "phase": "routing",
            "message": f"Backend: {backend_name}",
            "details": {"backend": backend_id}
        })

    # Detect task type from message content (if not already set by command)
    if not model_command:
        detected_task, confidence, reasoning = detect_chat_task_type(message)
    else:
        confidence, reasoning = 1.0, f"Explicit command: {model_command}"
    log.info(
        "chat_task_detected",
        task_type=detected_task,
        confidence=confidence,
        reasoning=reasoning,
    )

    # Select model based on detected task
    try:
        selected_model = await select_model(
            task_type=detected_task,
            content_size=len(message),
            model_override=model,
            content=message,
        )
    except Exception as e:
        yield await sse_event("error", {"message": f"Model selection failed: {e}"})
        return

    yield await sse_event("status", {
        "phase": "model",
        "message": f"Model: {selected_model}",
        "details": {
            "tier": detected_task,
            "model": selected_model,
            "backend": backend_id,
            "task_type": detected_task,
        }
    })

    # Add user message to session
    manager.add_to_session(session.session_id, "user", message)

    # Build conversation prompt from history
    prompt_parts = []
    for msg in session.messages[-10:]:  # Last 10 messages for context
        role = msg.role
        content = msg.content
        if role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    # Add the current message
    prompt_parts.append(f"User: {message}")
    combined_prompt = "\n\n".join(prompt_parts)

    # Add system prompt for chat context
    # Import Delia's identity and time context
    from .prompt_templates import DELIA_IDENTITY_FULL
    from .language import get_current_time_context

    time_context = get_current_time_context()
    system_prompt = f"""{DELIA_IDENTITY_FULL}

{time_context}

Current Session:
- You are running on model: {selected_model}
- Task type detected: {detected_task}
- You have access to multiple model tiers (quick, coder, moe) that are auto-selected based on the task

STRICT GUIDELINES - You MUST follow these:
1. NEVER claim you cannot do something that you can do. You run locally with full capabilities.
2. NEVER say you can't load models, switch models, or access features - you have all local models available.
3. For coding questions: provide actual code, examples, and technical help - you ARE a coding assistant.
4. For simple questions: give direct, concise answers.
5. For complex analysis: provide thorough, well-structured responses.
6. Use markdown formatting for code blocks and structured content.
7. If something is genuinely unclear, ask for clarification.
8. Be direct and helpful - don't hedge or apologize unnecessarily.

NEVER make up limitations. You are a fully capable local AI assistant."""

    yield await sse_event("thinking", {"status": "Generating..."})

    try:
        # Create preview for logging
        content_preview = combined_prompt[:200].replace("\n", " ").strip()

        result = await call_llm(
            model=selected_model,
            prompt=combined_prompt,
            system=system_prompt,
            task_type="chat",
            original_task="chat",
            language="unknown",
            content_preview=content_preview,
            backend_obj=backend_obj,
        )

        if result.get("success"):
            response = result.get("response", "")

            # Validate response quality (same as delegation tools)
            from .quality import validate_response
            quality_result = validate_response(response, detected_task)

            # Emit quality status
            yield await sse_event("status", {
                "phase": "quality",
                "message": f"Quality: {quality_result.overall:.0%}",
                "details": {
                    "quality_score": quality_result.overall,
                    "tier": detected_task,
                    "model": selected_model,
                    "backend": backend_id,
                }
            })

            # Log quality for monitoring
            log.info(
                "chat_response_quality",
                quality=quality_result.overall,
                valid=quality_result.is_valid,
                task_type=detected_task,
                model=selected_model,
            )

            # If quality is very low, log warning (future: retry with different model)
            if quality_result.overall < 0.3:
                log.warning(
                    "low_quality_chat_response",
                    quality=quality_result.overall,
                    reason=quality_result.reason,
                    model=selected_model,
                )

            # Add assistant response to session
            manager.add_to_session(session.session_id, "assistant", response, model=selected_model)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Yield response
            yield await sse_event("response", {"content": response})

            # Yield done with quality info
            yield await sse_event("done", {
                "success": True,
                "session_id": session.session_id,
                "model": selected_model,
                "backend": backend_name,
                "elapsed_ms": elapsed_ms,
                "quality": quality_result.overall,
            })
        else:
            error = result.get("error", "LLM call failed")
            yield await sse_event("error", {"message": error})
            yield await sse_event("done", {
                "success": False,
                "session_id": session.session_id,
                "model": selected_model,
                "backend": backend_name,
                "elapsed_ms": int((time.time() - start_time) * 1000),
            })

    except Exception as e:
        log.error("chat_stream_error", error=str(e))
        yield await sse_event("error", {"message": str(e)})
        yield await sse_event("done", {
            "success": False,
            "session_id": session.session_id,
            "model": selected_model,
            "backend": backend_name,
            "elapsed_ms": int((time.time() - start_time) * 1000),
        })


async def chat_handler(request: Request) -> StreamingResponse:
    """Handle POST /api/chat - SSE streaming chat."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    message = body.get("message")
    if not message:
        return JSONResponse({"error": "Missing 'message' field"}, status_code=400)

    session_id = body.get("session_id")
    model = body.get("model")
    backend_type = body.get("backend_type")

    return StreamingResponse(
        chat_stream(
            session_id=session_id,
            message=message,
            model=model,
            backend_type=backend_type,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def session_delete_handler(request: Request) -> JSONResponse:
    """Handle DELETE /api/sessions/{id} - Delete session."""
    from .session_manager import SessionManager

    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)

    manager = SessionManager()
    success = manager.delete_session(session_id)

    if not success:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    return JSONResponse({"deleted": True, "session_id": session_id})


# Create Starlette app
routes = [
    Route("/api/agent/run", agent_run_handler, methods=["POST"]),
    Route("/api/chat", chat_handler, methods=["POST"]),
    Route("/api/health", health_handler, methods=["GET"]),
    Route("/api/status", status_handler, methods=["GET"]),
    Route("/api/models", models_handler, methods=["GET"]),
    Route("/api/backends", backends_handler, methods=["GET"]),
    Route("/api/sessions", sessions_list_handler, methods=["GET"]),
    Route("/api/sessions/{session_id}", session_get_handler, methods=["GET"]),
    Route("/api/sessions/{session_id}", session_delete_handler, methods=["DELETE"]),
]

app = Starlette(
    routes=routes,
    on_startup=[],
    on_shutdown=[],
)

# Add CORS for CLI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_api(host: str = "0.0.0.0", port: int = 8201) -> None:
    """Run the API server."""
    import uvicorn

    log.info("api_server_starting", host=host, port=port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_api()
