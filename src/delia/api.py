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
- POST /api/agent/confirm - Confirm or deny dangerous tool execution
- GET /api/health - Health check
- GET /api/sessions - List sessions
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import structlog
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .backend_manager import backend_manager, shutdown_backends
from .llm import call_llm, init_llm_module
from .routing import select_model, detect_chat_task_type, get_router
from .orchestration import detect_intent, get_orchestration_executor
from .stats import StatsService
from .queue import ModelQueue
from .tools.agent import AgentConfig, AgentResult, run_agent_loop
from .tools.builtins import get_default_tools
from .tools.parser import ParsedToolCall
from .tools.executor import ToolResult
from .types import Workspace

log = structlog.get_logger()

# =============================================================================
# Global Services
# =============================================================================

# Initialize shared services
stats_service = StatsService()
model_queue = ModelQueue()

# Pending approval requests (approval_id -> asyncio.Event, result)
_pending_approvals: dict[str, tuple[asyncio.Event, dict]] = {}


def _update_stats_sync(
    model_tier: str,
    task_type: str,
    original_task: str,
    tokens: int,
    elapsed_ms: int,
    content_preview: str,
    enable_thinking: bool,
    backend: str,
) -> None:
    """Update stats synchronously."""
    stats_service.record_call(
        model_tier=model_tier,
        task_type=task_type,
        original_task=original_task,
        tokens=tokens,
        elapsed_ms=elapsed_ms,
        content_preview=content_preview,
        enable_thinking=enable_thinking,
        backend=backend,
    )


async def _save_all_stats_async() -> None:
    """Save all stats to disk."""
    await stats_service.save_all()


def _save_stats_background() -> None:
    """Schedule stats saving as a background task (non-blocking)."""
    asyncio.create_task(_save_all_stats_async())


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: Starlette):
    """
    Lifespan context manager for proper startup/shutdown.
    
    Ensures:
    - Initialization of OrchestrationService (loads affinity, prewarm)
    - Initialization of LLM module (model queue, stats)
    - Background state persistence task
    - Clean shutdown of backend HTTP clients
    - Proper cleanup of pending confirmations
    - Graceful handling of in-flight requests
    - Distributed discovery (mDNS)
    """
    from .orchestration import get_orchestration_service
    from .discovery import get_discovery_engine
    
    log.info("api_server_startup")
    
    # Initialize the unified orchestration service
    # This loads affinity and prewarm data from disk
    service = get_orchestration_service()
    log.info("orchestration_service_ready", stats=service.get_stats())
    
    # Initialize LLM module (sets up model queue, etc.)
    init_llm_module(
        stats_callback=_update_stats_sync,
        save_stats_callback=_save_stats_background,
        model_queue=model_queue,
    )
    log.info("llm_module_initialized")
    
    # Start background state persistence task
    persistence_task = asyncio.create_task(_state_persistence_loop(service))
    
    # Start distributed discovery in background (can be slow)
    discovery = get_discovery_engine()
    api_port = app.state.port if hasattr(app.state, "port") else 34589
    discovery_task = asyncio.create_task(discovery.start(port=api_port))
    
    yield
    
    # Shutdown: cleanup resources
    log.info("api_server_shutdown_starting")
    
    # Stop discovery
    discovery_task.cancel()
    await discovery.stop()
    
    # Cancel persistence task
    persistence_task.cancel()
    try:
        await persistence_task
    except asyncio.CancelledError:
        pass
    
    # Save final state
    try:
        await service.save_state()
        log.info("orchestration_state_saved_on_shutdown")
    except Exception as e:
        log.warning("orchestration_state_save_failed", error=str(e))
    
    # Clear any pending confirmations (they'll timeout anyway)
    _pending_confirmations.clear()
    
    # Shutdown backend HTTP clients
    try:
        await shutdown_backends()
    except Exception as e:
        log.warning("shutdown_backends_error", error=str(e))
    
    log.info("api_server_shutdown_complete")


async def _state_persistence_loop(service) -> None:
    """
    Background task that periodically saves orchestration state.
    
    Runs every 5 minutes to persist affinity, prewarm, and melon data.
    """
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            await service.save_state()
            log.debug("orchestration_state_persisted_periodic")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("state_persistence_error", error=str(e))


# =============================================================================
# Confirmation State Management
# =============================================================================

@dataclass
class PendingConfirmation:
    """A pending confirmation request for a dangerous tool call."""
    confirm_id: str
    tool_name: str
    args: dict[str, Any]
    event: asyncio.Event = field(default_factory=asyncio.Event)
    confirmed: bool | None = None  # None = pending, True = confirmed, False = denied
    allow_all: bool = False  # If True, skip future confirmations for this session


# Global store for pending confirmations (keyed by confirm_id)
_pending_confirmations: dict[str, PendingConfirmation] = {}

# Confirmation timeout in seconds
CONFIRMATION_TIMEOUT = 60


async def wait_for_confirmation(confirm_id: str) -> PendingConfirmation:
    """Wait for a confirmation response or timeout."""
    confirmation = _pending_confirmations.get(confirm_id)
    if not confirmation:
        raise ValueError(f"No pending confirmation with ID: {confirm_id}")

    try:
        await asyncio.wait_for(confirmation.event.wait(), timeout=CONFIRMATION_TIMEOUT)
    except asyncio.TimeoutError:
        # Timeout = denied
        confirmation.confirmed = False
        log.warning("confirmation_timeout", confirm_id=confirm_id, tool=confirmation.tool_name)

    return confirmation


def create_confirmation(tool_name: str, args: dict[str, Any]) -> PendingConfirmation:
    """Create a new pending confirmation."""
    confirm_id = str(uuid.uuid4())[:8]
    confirmation = PendingConfirmation(
        confirm_id=confirm_id,
        tool_name=tool_name,
        args=args,
    )
    _pending_confirmations[confirm_id] = confirmation
    log.info("confirmation_created", confirm_id=confirm_id, tool=tool_name)
    return confirmation


def resolve_confirmation(confirm_id: str, confirmed: bool, allow_all: bool = False) -> bool:
    """Resolve a pending confirmation. Returns True if found and resolved."""
    confirmation = _pending_confirmations.get(confirm_id)
    if not confirmation:
        log.warning("confirmation_not_found", confirm_id=confirm_id)
        return False

    confirmation.confirmed = confirmed
    confirmation.allow_all = allow_all
    confirmation.event.set()
    log.info("confirmation_resolved", confirm_id=confirm_id, confirmed=confirmed, allow_all=allow_all)
    return True


def cleanup_confirmation(confirm_id: str) -> None:
    """Remove a confirmation from the pending store."""
    _pending_confirmations.pop(confirm_id, None)


# =============================================================================
# SSE Helpers
# =============================================================================

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
        _, backend_obj = await get_router().select_optimal_backend(
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

    # Configure security manager based on API flags
    from .security import get_security_manager, ApprovalMode
    security_mgr = get_security_manager()
    if yolo:
        security_mgr.policy.approval_mode = ApprovalMode.YOLO
    elif allow_write and allow_exec:
        security_mgr.policy.approval_mode = ApprovalMode.AUTO
    else:
        # Deny operations that aren't explicitly allowed
        security_mgr.policy.approval_mode = ApprovalMode.DENY

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
            from .text_utils import strip_thinking_tags
            response = strip_thinking_tags(result.get("response", ""))
            # Signal LLM call done
            await event_queue.put({"type": "llm_done", "response": response})
            return response
        else:
            error = result.get("error", "LLM call failed")
            await event_queue.put({"type": "llm_error", "error": error})
            raise RuntimeError(error)

    # Wrap tool execution to emit events
    original_registry = registry

    # Track if user chose "allow all" for confirmations in this session
    session_allow_all = {"value": False}

    class StreamingToolRegistry:
        """Wrapper that emits events on tool calls and handles confirmations."""

        def __init__(self, inner: Any, queue: asyncio.Queue, require_confirm: bool):
            self._inner = inner
            self._queue = queue
            self._require_confirm = require_confirm

        def get(self, name: str):
            tool = self._inner.get(name)
            if not tool:
                return None

            # Wrap handler to emit events
            original_handler = tool.handler
            is_dangerous = tool.dangerous
            require_confirm = self._require_confirm

            async def wrapped_handler(**kwargs):
                # Check if this dangerous tool needs confirmation
                if is_dangerous and require_confirm and not session_allow_all["value"]:
                    # Create confirmation request
                    confirmation = create_confirmation(name, kwargs)

                    # Emit confirm event for CLI to display
                    await self._queue.put({
                        "type": "confirm",
                        "confirm_id": confirmation.confirm_id,
                        "tool": name,
                        "args": kwargs,
                        "message": f"Execute {name}?",
                    })

                    # Wait for user response (or timeout)
                    await wait_for_confirmation(confirmation.confirm_id)

                    # Check result
                    if not confirmation.confirmed:
                        cleanup_confirmation(confirmation.confirm_id)
                        # Emit tool_result showing it was denied
                        await self._queue.put({
                            "type": "tool_result",
                            "name": name,
                            "success": False,
                            "output": "Operation cancelled by user",
                            "elapsed_ms": 0,
                        })
                        return "Error: Operation cancelled by user"

                    # Check if user chose "allow all"
                    if confirmation.allow_all:
                        session_allow_all["value"] = True

                    cleanup_confirmation(confirmation.confirm_id)

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

            # Return modified tool with same dangerous flag
            from .tools.registry import ToolDefinition
            return ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                handler=wrapped_handler,
                permission_level=tool.permission_level,
                dangerous=tool.dangerous,
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

    # require_confirmation = True unless --yolo was passed
    require_confirmation = not yolo
    streaming_registry = StreamingToolRegistry(registry, event_queue, require_confirmation)

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
        model_val = active_backend.models.get(tier)
        
        # Handle list of models
        primary_model = None
        all_models = []
        if isinstance(model_val, list):
            all_models = model_val
            if model_val:
                primary_model = model_val[0]
        else:
            primary_model = model_val
            if model_val:
                all_models = [model_val]

        tier_info = tier_stats.get(tier, {})

        models[tier] = {
            "model": primary_model,
            "all_models": all_models,
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
    # Select backend
    try:
        _, backend_obj = await get_router().select_optimal_backend(
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
    from .prompts import build_system_prompt, ModelRole
    from .language import get_current_time_context

    time_context = get_current_time_context()
    base_prompt = build_system_prompt(role=ModelRole.ASSISTANT)
    system_prompt = f"""{base_prompt}

{time_context}

Current Session:
- Running on: {selected_model}
- Task detected: {detected_task}
- Available tiers: quick, coder, moe (auto-selected)

IMPORTANT - Stay in character as Delia:
1. You have FULL capabilities - never claim otherwise
2. Be DIRECT and SASSY - no generic AI hedging
3. For coding: provide real code and examples
4. For opinions/preferences: EXPRESS them! You're Delia, not a generic bot
5. When uncertain, be playful about it (not apologetic)
6. Use markdown for code blocks
7. Keep your personality balanced - you're a high-end AI orchestrator, not a one-trick bot.

FORBIDDEN phrases:
- "As an AI, I don't have..."
- "I'm just a language model..."
- "I cannot have preferences..."

You are Delia. You have personality. Use it. """

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
            from .text_utils import strip_thinking_tags
            response = strip_thinking_tags(result.get("response", ""))

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

            # Extract token count from result
            tokens = result.get("tokens", 0)
            
            # Update affinity tracker - successful chat response
            from .config import get_affinity_tracker
            affinity_tracker = get_affinity_tracker()
            # Use quality score from validation for more nuanced affinity learning
            affinity_tracker.update(backend_id, detected_task, quality=quality_result.overall)
            
            # Award melons based on quality! 🍈
            from .melons import award_melons_for_quality
            melons_awarded = award_melons_for_quality(
                model_id=selected_model,
                task_type=detected_task,
                quality_score=quality_result.overall,
            )
            
            log.debug(
                "chat_affinity_updated",
                backend=backend_id,
                task_type=detected_task,
                quality=quality_result.overall,
                melons=melons_awarded,
                success=True,
            )

            # Yield done with quality info and tokens
            yield await sse_event("done", {
                "success": True,
                "session_id": session.session_id,
                "model": selected_model,
                "backend": backend_name,
                "elapsed_ms": elapsed_ms,
                "quality": quality_result.overall,
                "tokens": tokens,
            })
        else:
            error = result.get("error", "LLM call failed")
            
            # Update affinity tracker - failed response
            from .config import get_affinity_tracker
            affinity_tracker = get_affinity_tracker()
            affinity_tracker.update(backend_id, detected_task, quality=0.0)
            
            log.debug(
                "chat_affinity_updated",
                backend=backend_id,
                task_type=detected_task,
                quality=0.0,
                success=False,
            )
            
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
        
        # Update affinity tracker - exception
        try:
            from .config import get_affinity_tracker
            affinity_tracker = get_affinity_tracker()
            affinity_tracker.update(backend_id, detected_task, quality=0.0)
        except Exception:
            pass  # Don't fail on affinity update error
        
        yield await sse_event("error", {"message": str(e)})
        yield await sse_event("done", {
            "success": False,
            "session_id": session.session_id,
            "model": selected_model,
            "backend": backend_name,
            "elapsed_ms": int((time.time() - start_time) * 1000),
        })


async def chat_handler(request: Request) -> StreamingResponse:
    """Handle POST /api/chat - SSE streaming chat.
    
    Modes:
    - Default (nlp_orchestrated=True): NLP-based orchestration. Models are tools.
      Delia detects intent and orchestrates voting/comparison as needed.
      Models receive role-specific prompts, NO tools.
    
    - simple=True: Basic single-model chat with no orchestration.
    
    - orchestrated=True (legacy): Tool-based orchestration.
      Models see tools and decide when to use them.
      Kept for backward compatibility but NLP mode is better.
    """
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
    
    # Mode selection (in priority order):
    # 1. simple=True → Basic chat (no orchestration)
    # 2. orchestrated=True → Legacy tool-based orchestration
    # 3. allow_write/allow_exec → Force tool-based mode for file/shell ops
    # 4. Default → NEW NLP-based orchestration (models are tools!)

    simple_mode = body.get("simple", False)
    legacy_orchestrated = body.get("orchestrated", False)
    include_file_tools = body.get("include_file_tools", True)  # Web search enabled by default
    workspace = body.get("workspace")

    # Permission flags - when set, force tool-based mode
    allow_write = body.get("allow_write", False)
    allow_exec = body.get("allow_exec", False)
    yolo = body.get("yolo", False)

    # If tool permissions are requested, use tool-based orchestration
    if allow_write or allow_exec:
        legacy_orchestrated = True
    
    # Simple mode - basic single model chat
    if simple_mode:
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
    
    # Legacy tool-based orchestration (for backward compat)
    # Also used when allow_write/allow_exec is requested
    if legacy_orchestrated:
        return StreamingResponse(
            chat_agent_stream(
                session_id=session_id,
                message=message,
                model=model,
                backend_type=backend_type,
                include_file_tools=include_file_tools,
                workspace=workspace,
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

    # DEFAULT: NEW NLP-based orchestration
    # Models are tools - Delia detects intent and orchestrates
    return StreamingResponse(
        chat_nlp_orchestrated_stream(
            session_id=session_id,
            message=message,
            model=model,
            backend_type=backend_type,
            include_file_tools=include_file_tools,
            workspace=workspace,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def chat_agent_stream(
    session_id: str | None,
    message: str,
    model: str | None = None,
    backend_type: str | None = None,
    include_file_tools: bool = True,  # Web search and file tools enabled by default
    workspace: str | None = None,
    allow_write: bool = False,
    allow_exec: bool = False,
    yolo: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Stream orchestrated chat response with tool calling.

    This version uses run_agent_loop with orchestration tools,
    allowing the LLM to delegate to other models, run batch comparisons,
    or use deep thinking.

    When allow_write/allow_exec is True, file/shell operations are enabled.
    Dangerous operations will prompt for confirmation unless yolo=True.

    Events:
        - session: Session created/loaded
        - status: Phase status updates (routing, model, tools)
        - thinking: Processing indicator
        - tool_call: Tool being called (name, args)
        - tool_result: Tool execution result
        - confirm: Confirmation needed for dangerous operation
        - token: Streaming token (if supported)
        - response: Full response
        - error: Error occurred
        - done: Complete with metadata
    """
    from .session_manager import SessionManager
    from .tools.orchestration import get_orchestration_tools
    from .tools.builtins import get_default_tools
    from .prompts import build_system_prompt, ModelRole

    start_time = time.time()
    manager = SessionManager()
    require_confirm = not yolo  # Require confirmation unless yolo mode
    session_allow_all = {"value": False}  # Mutable container for allow-all state

    # Get or create session
    if session_id:
        session = manager.get_session(session_id)
        if not session:
            yield await sse_event("error", {"message": f"Session {session_id} not found"})
            return
    else:
        session = manager.create_session(metadata={"source": "orchestrated_chat"})
        session_id = session.session_id
        yield await sse_event("session", {"id": session_id, "created": True})

    # Emit permission status
    if allow_write or allow_exec:
        yield await sse_event("status", {
            "phase": "permissions",
            "message": f"Permissions: write={allow_write} exec={allow_exec} yolo={yolo}",
            "details": {"allow_write": allow_write, "allow_exec": allow_exec, "yolo": yolo}
        })

    # Build tool registry - start with orchestration tools
    registry = get_orchestration_tools()

    # Optionally add file/web tools with permission flags
    if include_file_tools:
        workspace_obj = Workspace(root=workspace) if workspace else None
        file_registry = get_default_tools(
            workspace=workspace_obj,
            allow_write=allow_write,
            allow_exec=allow_exec,
        )
        for tool_name in file_registry.list_tools():
            if tool := file_registry.get(tool_name):
                try:
                    registry.register(tool)
                except ValueError:
                    pass  # Skip duplicates
    
    yield await sse_event("status", {
        "phase": "routing",
        "message": "Selecting backend...",
    })
    
    # Select backend
    try:
        _, backend_obj = await get_router().select_optimal_backend(
            message, None, "analyze", backend_type
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
    
    # Detect task type from message content
    detected_task, confidence, reasoning = detect_chat_task_type(message)
    log.info(
        "chat_task_detected",
        task_type=detected_task,
        confidence=confidence,
        reasoning=reasoning,
        orchestrated=True,
    )
    
    # Select model based on detected task (not hardcoded "analyze")
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
            "model": selected_model,
            "backend": backend_id,
            "orchestrated": True,
            "tools": registry.list_tools(),
        }
    })
    
    # Detect native tool calling support
    use_native = backend_obj.supports_native_tool_calling if backend_obj else False
    tools_schemas = registry.get_openai_schemas() if use_native else None
    
    # Add user message to session
    manager.add_to_session(session.session_id, "user", message)
    
    # Build conversation from history
    prompt_parts = []
    for msg in session.messages[-10:]:  # Last 10 messages for context
        if msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt_parts.append(f"User: {message}")
    combined_prompt = "\n\n".join(prompt_parts)
    
    # Create LLM callable for the agent loop
    async def chat_llm_call(messages: list[dict], system: str | None) -> str:
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
        
        combined = "\n\n".join(prompt_parts)
        content_preview = combined[:200].replace("\n", " ").strip()
        
        result = await call_llm(
            model=selected_model,
            prompt=combined,
            system=system,
            task_type="chat",
            original_task="orchestrated_chat",
            language="unknown",
            content_preview=content_preview,
            backend_obj=backend_obj,
            tools=tools_schemas,
            tool_choice="auto" if tools_schemas else None,
        )
        
        if result.get("success"):
            from .text_utils import strip_thinking_tags
            return strip_thinking_tags(result.get("response", ""))
        raise RuntimeError(result.get("error", "LLM call failed"))
    
    yield await sse_event("thinking", {"status": "Processing with orchestration tools..."})
    
    # Create agent config - keep iterations LOW to prevent runaway tool calling
    # Most chat needs 0-2 iterations; more usually means model is confused
    config = AgentConfig(
        max_iterations=3,  # Reduced from 5
        timeout_per_tool=20.0,
        total_timeout=60.0,  # 1 minute - chat should be fast
        parallel_tools=True,
        native_tool_calling=use_native,
    )
    
    # Track tool calls for real-time display
    from .tools.agent import build_system_prompt as agent_build_system_prompt
    from .tools.executor import execute_tools
    from .tools.parser import parse_tool_calls, has_tool_calls

    # Simple helper to build messages list
    def build_messages(prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": prompt}]
    
    all_tool_calls = []
    all_tool_results = []
    
    # Build system prompt with tools
    legacy_system = build_system_prompt(role=ModelRole.ASSISTANT)
    full_system = agent_build_system_prompt(legacy_system, registry, use_native)
    messages = build_messages(combined_prompt)
    
    # Run agent loop manually to emit events in real-time
    try:
        for iteration in range(config.max_iterations):
            # Check timeout
            if time.time() - start_time > config.total_timeout:
                yield await sse_event("error", {"message": "Agent timed out"})
                break
            
            yield await sse_event("status", {
                "phase": "iteration",
                "message": f"Iteration {iteration + 1}/{config.max_iterations}",
                "details": {"iteration": iteration + 1}
            })
            
            # Call LLM
            try:
                response_text = await chat_llm_call(messages, full_system)
            except Exception as e:
                yield await sse_event("error", {"message": f"LLM call failed: {e}"})
                response_text = f"Error: {e}"
                break
            
            # Check for tool calls
            if not has_tool_calls(response_text):
                # No tool calls - agent is done
                break
            
            # Parse tool calls
            tool_calls = parse_tool_calls(response_text, native_mode=use_native)
            
            if not tool_calls:
                # Couldn't parse - treat as done
                break
            
            # Emit tool calls IN REAL-TIME
            yield await sse_event("tool_call", {
                "iteration": iteration + 1,
                "calls": [
                    {"name": tc.name, "args": tc.arguments}
                    for tc in tool_calls
                ],
            })
            
            # Check for dangerous tools that need confirmation
            approved_calls = []
            denied_calls = []

            for tc in tool_calls:
                tool_def = registry.get(tc.name)
                is_dangerous = tool_def and tool_def.dangerous

                if is_dangerous and require_confirm and not session_allow_all["value"]:
                    # Emit confirm event and wait for response
                    confirmation = create_confirmation(tc.name, tc.arguments)

                    yield await sse_event("confirm", {
                        "confirm_id": confirmation.confirm_id,
                        "tool": tc.name,
                        "args": tc.arguments,
                        "message": f"Allow {tc.name}?",
                    })

                    # Wait for user response (or timeout)
                    await wait_for_confirmation(confirmation.confirm_id)

                    if confirmation.confirmed:
                        if confirmation.allow_all:
                            session_allow_all["value"] = True
                        approved_calls.append(tc)
                    else:
                        denied_calls.append(tc)
                        yield await sse_event("tool_result", {
                            "name": tc.name,
                            "success": False,
                            "output_preview": "Operation denied by user",
                        })

                    cleanup_confirmation(confirmation.confirm_id)
                else:
                    approved_calls.append(tc)

            if not approved_calls:
                # All tools were denied, continue to next iteration
                continue

            # Execute approved tools
            yield await sse_event("thinking", {"status": f"Executing {len(approved_calls)} tool(s)..."})

            results = await execute_tools(
                approved_calls,
                registry,
                timeout=config.timeout_per_tool,
                parallel=config.parallel_tools,
            )

            # Update tool_calls to only approved ones for downstream processing
            tool_calls = approved_calls
            
            all_tool_calls.extend(tool_calls)
            all_tool_results.extend(results)
            
            # Emit tool results
            for tc, result in zip(tool_calls, results):
                yield await sse_event("tool_result", {
                    "name": tc.name,
                    "success": result.success,
                    "output_preview": result.output[:200] + "..." if len(result.output) > 200 else result.output,
                })
            
            # Add to conversation
            if use_native:
                messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in tool_calls
                    ]
                })
            else:
                messages.append({"role": "assistant", "content": response_text})
            
            # Add tool results
            from .tools.agent import format_tool_result
            for tc, result in zip(tool_calls, results):
                messages.append(format_tool_result(tc.id, tc.name, result.output))
        
        # Final response is the last LLM output without tool calls
        response = response_text
        
        # Add assistant response to session
        manager.add_to_session(
            session.session_id, 
            "assistant", 
            response, 
            model=selected_model
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Emit summary of all tool calls
        if all_tool_calls:
            yield await sse_event("tools", {
                "calls": [
                    {"name": tc.name, "args": tc.arguments}
                    for tc in all_tool_calls
                ],
                "count": len(all_tool_calls),
            })
        
        # Emit response
        yield await sse_event("response", {"content": response})
        
        # Update affinity tracker - successful chat response
        # This helps Delia learn which backends work well for which task types
        from .config import get_affinity_tracker
        affinity_tracker = get_affinity_tracker()
        affinity_tracker.update(backend_id, detected_task, quality=1.0)
        
        log.debug(
            "chat_affinity_updated",
            backend=backend_id,
            task_type=detected_task,
            quality=1.0,
            success=True,
        )
        
        # Emit done with full metadata
        yield await sse_event("done", {
            "success": True,
            "session_id": session_id,
            "model": selected_model,
            "backend": backend_name,
            "iterations": iteration + 1,
            "tools_used": [tc.name for tc in all_tool_calls] if all_tool_calls else [],
            "elapsed_ms": elapsed_ms,
            "orchestrated": True,
        })
        
    except Exception as e:
        log.error("chat_agent_stream_error", error=str(e))
        
        # Update affinity tracker - failed chat response
        try:
            from .config import get_affinity_tracker
            affinity_tracker = get_affinity_tracker()
            affinity_tracker.update(backend_id, detected_task, quality=0.0)
            
            log.debug(
                "chat_affinity_updated",
                backend=backend_id,
                task_type=detected_task,
                quality=0.0,
                success=False,
            )
        except Exception:
            pass  # Don't fail on affinity/melon update error
        
        yield await sse_event("error", {"message": str(e)})
        yield await sse_event("done", {
            "success": False,
            "session_id": session_id,
            "model": selected_model,
            "backend": backend_name,
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "orchestrated": True,
        })


async def chat_nlp_orchestrated_stream(
    session_id: str | None,
    message: str,
    model: str | None = None,
    backend_type: str | None = None,
    include_file_tools: bool = True,
    workspace: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    NLP-Orchestrated Chat Stream - The New Paradigm.
    
    Uses the unified OrchestrationService for all orchestration logic.
    This ensures feature parity between Chat/API and MCP modes.
    
    The OrchestrationService handles:
    - Frustration detection and penalties
    - Intent detection (3-tier NLP)
    - Prewarm tracking (EMA learning)
    - Orchestration execution (voting, comparison, etc.)
    - Quality validation
    - Affinity and melon rewards
    
    Events:
        - session: Session created/loaded
        - intent: Detected intent (task_type, orchestration_mode, role)
        - status: Phase status updates
        - thinking: Processing indicator
        - frustration: Repeat detected, penalty applied
        - orchestration: Voting/comparison details
        - response: Full response
        - error: Error occurred
        - done: Complete with metadata
    """
    from .session_manager import SessionManager
    from .orchestration import get_orchestration_service, detect_intent
    from .orchestration.result import OrchestrationMode
    
    start_time = time.time()
    manager = SessionManager()
    service = get_orchestration_service()
    
    # Get or create session
    if session_id:
        session = manager.get_session(session_id)
        if not session:
            yield await sse_event("error", {"message": f"Session {session_id} not found"})
            return
    else:
        session = manager.create_session(metadata={"source": "nlp_orchestrated_chat"})
        session_id = session.session_id
        yield await sse_event("session", {"id": session_id, "created": True})
    
    # Pre-process: Check for frustration (for SSE events only)
    # The actual penalty is handled by the service, but we emit events here
    is_repeat, repeat_count = service._check_repeat(session_id, message) if session_id else (False, 0)

    if is_repeat and repeat_count >= 2:
        yield await sse_event("frustration", {
            "detected": True,
            "repeat_count": repeat_count,
            "message": f"Repeat detected ({repeat_count}x)",
        })
    
    # Pre-process: Detect intent (for SSE events)
    # The service also does this, but we need it early for UI feedback
    intent = detect_intent(message)
    
    # Auto-upgrade to VOTING if repeated question
    if is_repeat and repeat_count >= 2:
        if intent.orchestration_mode == OrchestrationMode.NONE:
            intent.orchestration_mode = OrchestrationMode.VOTING
            intent.k_votes = min(3 + repeat_count, 5)
            intent.reasoning = f"auto-voting due to {repeat_count}x repeat; {intent.reasoning}"
    
    yield await sse_event("intent", {
        "task_type": intent.task_type,
        "orchestration_mode": intent.orchestration_mode.value,
        "model_role": intent.model_role.value,
        "confidence": round(intent.confidence, 2),
        "reasoning": intent.reasoning,
        "needs_orchestration": intent.orchestration_mode != OrchestrationMode.NONE,
    })
    
    # Select backend (for SSE events)
    yield await sse_event("status", {
        "phase": "routing",
        "message": "Selecting optimal backend...",
    })
    
    try:
        _, backend_obj = await get_router().select_optimal_backend(
            message, None, intent.task_type, backend_type
        )
    except Exception as e:
        yield await sse_event("error", {"message": f"Backend selection failed: {e}"})
        return
    
    backend_name = backend_obj.name if backend_obj else "unknown"
    backend_id = backend_obj.id if backend_obj else "unknown"
    
    yield await sse_event("status", {
        "phase": "routing",
        "message": f"Backend: {backend_name}",
        "details": {
            "backend": backend_id,
            "task_type": intent.task_type,
            "orchestration": intent.orchestration_mode.value,
        }
    })
    
    # Add user message to session
    manager.add_to_session(session.session_id, "user", message)
    
    # Emit thinking status based on mode
    if intent.orchestration_mode == OrchestrationMode.VOTING:
        yield await sse_event("thinking", {
            "status": f"K-voting with k={intent.k_votes} for reliable answer...",
        })
    elif intent.orchestration_mode == OrchestrationMode.COMPARISON:
        yield await sse_event("thinking", {
            "status": "Running multi-model comparison...",
        })
    elif intent.orchestration_mode == OrchestrationMode.DEEP_THINKING:
        yield await sse_event("thinking", {
            "status": "Deep analysis with extended reasoning...",
        })
    else:
        yield await sse_event("thinking", {"status": "Generating response..."})
    
    # Execute via unified OrchestrationService
    # This handles: frustration penalty, prewarm update, quality validation, rewards
    try:
        async for event in service.process_stream(
            message=message,
            session_id=session_id,
            backend_type=backend_type,
            model_override=model,
            include_file_tools=include_file_tools,
        ):
            # Prepare data payload
            data = {
                "message": event.message,
                **event.details
            }
            
            # Map fields for client compatibility
            if event.event_type in ("token", "response"):
                data["content"] = event.message
            elif event.event_type == "thinking":
                data["status"] = event.message

            # Pass events to the client
            yield await sse_event(event.event_type, data)
            
            # Record final response in session if it's a 'response' event
            if event.event_type == "response" and event.message:
                model_detail = event.details.get("model", "unknown")
                if isinstance(model_detail, list):
                    model_detail = model_detail[0] if model_detail else "unknown"

                manager.add_to_session(
                    session.session_id,
                    "assistant",
                    event.message,
                    model=str(model_detail)
                )

        # Success: yield done event
        yield await sse_event("done", {
            "success": True,
            "session_id": session_id,
            "backend": backend_name,
            "elapsed_ms": int((time.time() - start_time) * 1000),
        })

    except Exception as e:
        log.error("nlp_orchestration_error", error=str(e))
        yield await sse_event("error", {"message": str(e)})
        yield await sse_event("done", {
            "success": False,
            "session_id": session_id,
            "backend": backend_name,
            "elapsed_ms": int((time.time() - start_time) * 1000),
        })


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


async def session_compact_handler(request: Request) -> JSONResponse:
    """Handle POST /api/sessions/{id}/compact - Compact session history.

    Compacts the conversation history using LLM-based summarization.
    Reduces token count while preserving key information.

    Query params:
        - force: If true, force compaction even if below threshold
    """
    from .session_manager import SessionManager
    from .compaction import ConversationCompactor

    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)

    # Check for force param
    force = request.query_params.get("force", "").lower() in ("true", "1", "yes")

    manager = SessionManager()
    session = manager.get_session(session_id)

    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    # Perform compaction
    compactor = ConversationCompactor()
    result = await compactor.compact(session, force=force)

    if not result.success:
        return JSONResponse({
            "success": False,
            "error": result.error or "Compaction failed",
            "session_id": session_id,
        }, status_code=400)

    return JSONResponse({
        "success": True,
        "session_id": session_id,
        "messages_compacted": result.messages_compacted,
        "tokens_saved": result.tokens_saved,
        "compression_ratio": result.compression_ratio,
        "summary_preview": result.summary[:200] if result.summary else None,
    })


async def session_stats_handler(request: Request) -> JSONResponse:
    """Handle GET /api/sessions/{id}/stats - Get session compaction stats.

    Returns current token usage and whether compaction is recommended.
    """
    from .session_manager import SessionManager
    from .compaction import ConversationCompactor

    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)

    manager = SessionManager()
    session = manager.get_session(session_id)

    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    compactor = ConversationCompactor()
    stats = compactor.get_compaction_stats(session)

    return JSONResponse({
        "session_id": session_id,
        "total_messages": stats.get("total_messages", 0),
        "total_tokens": stats.get("total_tokens", 0),
        "needs_compaction": stats.get("needs_compaction", False),
        "threshold_tokens": stats.get("threshold_tokens", 18000),
        "compactable_messages": stats.get("compactable_messages", 0),
    })


async def agent_confirm_handler(request: Request) -> JSONResponse:
    """Handle POST /api/agent/confirm - Confirm or deny dangerous tool execution.

    Request body:
        - confirm_id: The confirmation ID from the confirm event
        - confirmed: True to allow, False to deny
        - allow_all: If True, skip all future confirmations in this session (optional)
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    confirm_id = body.get("confirm_id")
    if not confirm_id:
        return JSONResponse({"error": "Missing 'confirm_id' field"}, status_code=400)

    confirmed = body.get("confirmed", False)
    allow_all = body.get("allow_all", False)

    success = resolve_confirmation(confirm_id, confirmed, allow_all)

    if not success:
        return JSONResponse(
            {"error": f"No pending confirmation with ID: {confirm_id}"},
            status_code=404
        )

    return JSONResponse({
        "confirm_id": confirm_id,
        "confirmed": confirmed,
        "allow_all": allow_all,
    })


# Create Starlette app with lifespan for proper shutdown
routes = [
    Route("/api/agent/run", agent_run_handler, methods=["POST"]),
    Route("/api/agent/confirm", agent_confirm_handler, methods=["POST"]),
    Route("/api/chat", chat_handler, methods=["POST"]),
    Route("/api/health", health_handler, methods=["GET"]),
    Route("/api/status", status_handler, methods=["GET"]),
    Route("/api/models", models_handler, methods=["GET"]),
    Route("/api/backends", backends_handler, methods=["GET"]),
    Route("/api/sessions", sessions_list_handler, methods=["GET"]),
    Route("/api/sessions/{session_id}", session_get_handler, methods=["GET"]),
    Route("/api/sessions/{session_id}", session_delete_handler, methods=["DELETE"]),
    Route("/api/sessions/{session_id}/compact", session_compact_handler, methods=["POST"]),
    Route("/api/sessions/{session_id}/stats", session_stats_handler, methods=["GET"]),
]

app = Starlette(
    routes=routes,
    lifespan=lifespan,
)

# Add CORS for CLI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_api(host: str = "0.0.0.0", port: int = 34589) -> None:
    """Run the API server with proper shutdown handling.
    
    Uses SO_REUSEADDR to allow immediate port reuse after shutdown,
    and configures a fast graceful shutdown timeout.
    """
    import signal
    import socket
    import uvicorn

    log.info("api_server_starting", host=host, port=port)
    
    # Create socket with SO_REUSEADDR for immediate port reuse
    # This allows reopening the server immediately after closing
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        log.error("port_bind_failed", host=host, port=port, error=str(e))
        sock.close()
        raise SystemExit(f"Cannot bind to {host}:{port} - {e}") from e
    
    # Store port for discovery engine
    app.state.port = port
    
    # Configure uvicorn with the pre-bound socket
    # - fd: Use our pre-bound socket with SO_REUSEADDR
    # - timeout_graceful_shutdown: Max 3 seconds to close connections
    # - timeout_keep_alive: Don't hold connections open too long
    config = uvicorn.Config(
        app,
        fd=sock.fileno(),
        log_level="info",
        ws="wsproto",  # Use wsproto to avoid websockets deprecation warnings
        timeout_graceful_shutdown=3,  # Fast shutdown - max 3 seconds
        timeout_keep_alive=5,  # Don't hold keep-alive connections too long
    )
    
    server = uvicorn.Server(config)
    
    # Install signal handlers for graceful shutdown
    def handle_signal(signum, frame):
        log.info("signal_received", signal=signum)
        server.should_exit = True
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    try:
        server.run()
    finally:
        # Ensure socket is closed
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Delia API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=34589, help="Port to bind")
    args = parser.parse_args()
    run_api(host=args.host, port=args.port)
