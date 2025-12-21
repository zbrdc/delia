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
Orchestration tools for multi-model chat.

These tools wrap the MCP orchestration functions (delegate, batch, think, etc.)
as ToolDefinitions so they can be used within run_agent_loop for chat mode.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from .registry import ToolRegistry, ToolDefinition

log = structlog.get_logger()


# =============================================================================
# Handler Implementations
# =============================================================================


async def _delegate_handler(
    task: str,
    content: str,
    model: str | None = None,
    backend_type: str | None = None,
    language: str | None = None,
    final: bool = False,
) -> str | dict[str, Any]:
    """
    Internal handler that calls the delegate implementation.

    Wraps the MCP delegate tool for use within the agent loop.

    Args:
        final: If True, signals that this delegation's result should be
               returned directly without reloading the orchestrating model.
               Use when the delegated model's response IS the final answer.
    """
    from ..mcp_server import _delegate_impl
    from ..routing import get_router
    
    log.info(
        "orchestration_delegate",
        task=task,
        model=model,
        backend_type=backend_type,
        language=language,
        content_len=len(content),
        final=final,
    )

    try:
        _, backend_obj = await get_router().select_optimal_backend(
            content, None, task, backend_type
        )

        result = await _delegate_impl(
            task=task,
            content=content,
            file=None,
            model=model,
            language=language,
            context=None,
            symbols=None,
            include_references=False,
            backend=backend_type,
            backend_obj=backend_obj,
            files=None,
            include_metadata=True,
            max_tokens=None,
            session_id=None,
        )

        # If final=True, return dict that executor will recognize
        # This signals the agent loop to return this result directly
        # without reloading the orchestrating model
        if final:
            return {"__result__": result, "__is_final__": True}
        return result
    except Exception as e:
        log.error("orchestration_delegate_error", error=str(e))
        return f"Delegate error: {e}"


async def _batch_handler(tasks: str) -> str:
    """
    Internal handler for batch parallel execution.
    
    Args:
        tasks: JSON array of task objects, e.g.:
            [{"task": "review", "content": "...", "model": "coder"}]
    """
    from ..mcp_server import batch as mcp_batch
    
    log.info("orchestration_batch", tasks_json_len=len(tasks))
    
    try:
        # Validate JSON before calling
        parsed = json.loads(tasks)
        if not isinstance(parsed, list):
            return "Error: tasks must be a JSON array"
        
        result = await mcp_batch(tasks)
        return result
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
        log.error("orchestration_batch_error", error=str(e))
        return f"Batch error: {e}"


async def _batch_vote_handler(
    prompt: str,
    task: str = "analyze",
    k: int = 3,
    backends: str | None = None,
) -> str:
    """
    Run the same prompt across multiple backends and vote on best result.
    
    Uses MDAP k-voting for consensus across different models/backends.
    """
    from ..mcp_server import batch_vote as mcp_batch_vote
    
    log.info(
        "orchestration_batch_vote",
        prompt_len=len(prompt),
        k=k,
        backends=backends,
    )
    
    try:
        result = await mcp_batch_vote(
            prompt=prompt,
            task=task,
            k=k,
            backends=backends,
        )
        return result
    except Exception as e:
        log.error("orchestration_batch_vote_error", error=str(e))
        return f"Batch vote error: {e}"


async def _think_handler(
    problem: str,
    context: str = "",
    depth: str = "normal",
    final: bool = False,
) -> str | dict[str, Any]:
    """
    Internal handler for deep thinking.

    Uses extended thinking models for complex reasoning.

    Args:
        final: If True, return result directly without reloading orchestrator.
    """
    from ..mcp_server import think as mcp_think

    log.info(
        "orchestration_think",
        depth=depth,
        problem_len=len(problem),
        context_len=len(context),
        final=final,
    )

    try:
        result = await mcp_think(problem, context, depth)
        if final:
            return {"__result__": result, "__is_final__": True}
        return result
    except Exception as e:
        log.error("orchestration_think_error", error=str(e))
        return f"Think error: {e}"


async def _health_handler() -> str:
    """
    Internal handler for backend health status.
    """
    from ..mcp_server import health as mcp_health
    
    log.info("orchestration_health")
    
    try:
        result = await mcp_health()
        return result
    except Exception as e:
        log.error("orchestration_health_error", error=str(e))
        return f"Health check error: {e}"


async def _models_handler() -> str:
    """
    Internal handler for listing available models.
    """
    from ..mcp_server import models as mcp_models
    
    log.info("orchestration_models")
    
    try:
        result = await mcp_models()
        return result
    except Exception as e:
        log.error("orchestration_models_error", error=str(e))
        return f"Models listing error: {e}"


async def _switch_backend_handler(backend_id: str) -> str:
    """
    Internal handler for switching the active backend.
    """
    from ..mcp_server import switch_backend as mcp_switch_backend
    
    log.info("orchestration_switch_backend", backend_id=backend_id)
    
    try:
        result = await mcp_switch_backend(backend_id)
        return result
    except Exception as e:
        log.error("orchestration_switch_backend_error", error=str(e))
        return f"Switch backend error: {e}"


async def _compare_handler(
    prompt: str,
    models: str,  # Comma-separated model names
    include_synthesis: bool = True,
) -> str:
    """
    Compare responses from multiple models.
    
    Handles both multi-backend (parallel) and single-backend (sequential) cases.
    On single backend, models are loaded sequentially.
    """
    from ..backend_manager import backend_manager
    from ..llm import call_llm
    from ..routing import select_model
    import time
    
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    
    if len(model_list) < 2:
        return "Error: Need at least 2 models to compare. Provide comma-separated model names."
    
    log.info("orchestration_compare", models=model_list, prompt_len=len(prompt))
    
    # Get available backends
    enabled_backends = backend_manager.get_enabled_backends()
    
    results = []
    
    try:
        for i, model_name in enumerate(model_list):
            start = time.time()
            
            # Try to find a backend that has this model
            backend_obj = None
            for backend in enabled_backends:
                # Check if model is in any tier or matches a configured model
                all_models = list(backend.models.values()) if backend.models else []
                if model_name in all_models or any(model_name in m for m in all_models):
                    backend_obj = backend
                    break
            
            if not backend_obj:
                # Use active backend and let it try to load the model
                backend_obj = backend_manager.get_active_backend()
            
            if not backend_obj:
                results.append({
                    "model": model_name,
                    "response": f"Error: No backend available for {model_name}",
                    "elapsed_ms": 0,
                    "success": False,
                })
                continue
            
            # Call LLM with specific model
            result = await call_llm(
                model=model_name,
                prompt=prompt,
                system="You are a helpful AI assistant. Provide a clear, well-structured response.",
                task_type="analyze",
                original_task="compare",
                language="unknown",
                content_preview=prompt[:100],
                backend_obj=backend_obj,
            )
            
            elapsed_ms = int((time.time() - start) * 1000)
            
            if result.get("success"):
                results.append({
                    "model": model_name,
                    "response": result.get("response", ""),
                    "elapsed_ms": elapsed_ms,
                    "success": True,
                })
            else:
                results.append({
                    "model": model_name,
                    "response": f"Error: {result.get('error', 'Unknown error')}",
                    "elapsed_ms": elapsed_ms,
                    "success": False,
                })
        
        # Format output
        output_parts = ["# Model Comparison\n"]
        
        for r in results:
            status = "✓" if r["success"] else "✗"
            output_parts.append(f"## {status} {r['model']} ({r['elapsed_ms']}ms)\n")
            output_parts.append(f"{r['response']}\n")
        
        # Add synthesis if requested and we have multiple successful responses
        successful = [r for r in results if r["success"]]
        if include_synthesis and len(successful) >= 2:
            output_parts.append("\n## 📊 Comparison Summary\n")
            output_parts.append(f"Models compared: {', '.join(r['model'] for r in successful)}\n")
            output_parts.append(f"Total time: {sum(r['elapsed_ms'] for r in results)}ms\n")
        
        return "\n".join(output_parts)
        
    except Exception as e:
        log.error("orchestration_compare_error", error=str(e))
        return f"Compare error: {e}"


async def _vote_handler(
    prompt: str,
    k: int | None = None,
    model: str | None = None,
    target_accuracy: float = 0.9999,
) -> str:
    """
    K-voting for self-consistency using MDAP framework.
    
    Uses VotingConsensus with:
    - First-to-ahead-by-k voting
    - Red-flagging for quality issues
    - Semantic similarity matching
    - Auto kmin calculation
    """
    from ..backend_manager import backend_manager
    from ..llm import call_llm
    from ..routing import select_model
    from ..voting import VotingConsensus
    from ..quality import ResponseQualityValidator
    from ..voting_stats import get_voting_stats_tracker
    import time
    
    start_time = time.time()
    
    # Get backend and model
    backend_obj = backend_manager.get_active_backend()
    if not backend_obj:
        return "Error: No backend available"
    
    selected_model = model or await select_model(
        task_type="analyze",
        content_size=len(prompt),
        content=prompt,
    )
    
    # Calculate optimal k if not provided (MDAP formula)
    # Estimate task complexity for kmin calculation
    from ..voting import estimate_task_complexity
    task_steps = estimate_task_complexity(prompt)
    
    if k is None:
        k = VotingConsensus.calculate_kmin(
            total_steps=task_steps,
            target_accuracy=target_accuracy,
            base_accuracy=0.95,  # Conservative estimate
        )
    
    k = max(2, min(k, 7))  # Clamp to reasonable range
    
    log.info(
        "orchestration_vote",
        k=k,
        model=selected_model,
        prompt_len=len(prompt),
        estimated_steps=task_steps,
    )
    
    # Initialize voting consensus with quality validator
    validator = ResponseQualityValidator()
    consensus = VotingConsensus(
        k=k,
        quality_validator=validator,
        similarity_threshold=0.85,
        max_response_length=700,
    )
    
    # Get voting stats tracker for metrics
    voting_tracker = get_voting_stats_tracker()
    
    # Determine tier for stats
    tier = "coder"  # Default for voting tasks
    if backend_obj.models:
        for t, m in backend_obj.models.items():
            if m == selected_model:
                tier = t
                break
    
    try:
        max_attempts = k * 3  # Allow some red-flagged responses
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            # Get a response
            # Higher temperature for diverse samples in k-voting
            result = await call_llm(
                model=selected_model,
                prompt=prompt,
                system="You are a helpful AI assistant. Provide a clear, accurate, well-structured response.",
                task_type="analyze",
                original_task="vote",
                language="unknown",
                content_preview=prompt[:100],
                backend_obj=backend_obj,
                temperature=0.7,  # Higher temp for sample diversity
            )
            
            if not result.get("success"):
                continue
            
            response = result.get("response", "")
            
            # Add vote and check for consensus
            vote_result = consensus.add_vote(response)
            
            if vote_result.red_flagged:
                # Record red-flag in stats
                voting_tracker.record_rejection(
                    reason=vote_result.red_flag_reason or "unknown",
                    backend_id=backend_obj.id,
                    tier=tier,
                    response_preview=response[:100],
                )
                continue
            
            if vote_result.consensus_reached:
                # Record successful consensus
                voting_tracker.record_consensus(
                    votes_cast=vote_result.total_votes,
                    k=k,
                    tier=tier,
                    backend_id=backend_obj.id,
                    success=True,
                )
                
                elapsed_ms = int((time.time() - start_time) * 1000)
                prob = VotingConsensus.voting_probability(k, 0.95)
                
                return f"""# ✓ K-Voting Consensus Reached

{vote_result.winning_response}

---
**Voting Stats**
- Consensus: {vote_result.votes_for_winner}/{k} votes
- Total attempts: {vote_result.total_votes}
- Confidence: {prob:.2%}
- Model: {selected_model}
- Time: {elapsed_ms}ms"""
        
        # No consensus reached - get best response
        best_response, metadata = consensus.get_best_response()
        
        # Record failed consensus
        voting_tracker.record_consensus(
            votes_cast=metadata.total_votes,
            k=k,
            tier=tier,
            backend_id=backend_obj.id,
            success=False,
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        if best_response:
            return f"""# ⚠ K-Voting (No Full Consensus)

{best_response}

---
**Voting Stats**
- Best response: {metadata.winning_votes}/{k} votes needed
- Unique responses: {metadata.unique_responses}
- Red-flagged: {metadata.red_flagged_count}
- Total attempts: {metadata.total_votes}
- Model: {selected_model}
- Time: {elapsed_ms}ms"""
        else:
            return f"Error: All {max_attempts} voting attempts failed (red-flagged)"
        
    except Exception as e:
        log.error("orchestration_vote_error", error=str(e))
        return f"Vote error: {e}"


async def _compute_handler(
    operation: str,
    input_value: str,
) -> str:
    """
    Programmatic computation for operations LLMs struggle with.
    
    Handles:
    - reverse: Reverse a string
    - uppercase: Convert to uppercase
    - lowercase: Convert to lowercase
    - length: Get string length
    - sort: Sort characters alphabetically
    """
    log.info("orchestration_compute", operation=operation, input_len=len(input_value))
    
    try:
        op = operation.lower().strip()
        
        if op == "reverse":
            result = input_value[::-1]
            return f'"{input_value}" reversed is "{result}"'
        
        elif op == "uppercase":
            result = input_value.upper()
            return f'"{input_value}" in uppercase is "{result}"'
        
        elif op == "lowercase":
            result = input_value.lower()
            return f'"{input_value}" in lowercase is "{result}"'
        
        elif op == "length":
            result = len(input_value)
            return f'"{input_value}" has {result} characters'
        
        elif op == "sort":
            result = ''.join(sorted(input_value))
            return f'"{input_value}" sorted alphabetically is "{result}"'
        
        elif op == "count":
            # Count occurrences of each character
            from collections import Counter
            counts = Counter(input_value)
            formatted = ', '.join(f'{char}: {count}' for char, count in counts.most_common())
            return f'Character counts in "{input_value}": {formatted}'
        
        else:
            return f"Unknown operation: {operation}. Available: reverse, uppercase, lowercase, length, sort, count"
    
    except Exception as e:
        log.error("orchestration_compute_error", error=str(e))
        return f"Compute error: {e}"



async def _session_compact_handler(
    session_id: str,
    force: bool = False,
) -> str:
    """Internal handler for session compaction."""
    from ..session_manager import get_session_manager
    log.info("orchestration_session_compact", session_id=session_id, force=force)
    try:
        sm = get_session_manager()
        result = await sm.compact_session(session_id, force=force)
        return json.dumps(result, indent=2)
    except Exception as e:
        log.error("orchestration_session_compact_error", error=str(e))
        return f"Compaction error: {e}"


async def _session_stats_handler(
    session_id: str,
) -> str:
    """Internal handler for session stats."""
    from ..session_manager import get_session_manager
    log.info("orchestration_session_stats", session_id=session_id)
    try:
        sm = get_session_manager()
        stats = sm.get_compaction_stats(session_id)
        if stats is None:
            return f"Error: Session not found: {session_id}"
        return json.dumps(stats, indent=2)
    except Exception as e:
        log.error("orchestration_session_stats_error", error=str(e))
        return f"Stats error: {e}"


async def _project_memories_handler(
    reload: bool = False,
) -> str:
    """Internal handler for project memories."""
    from ..project_memory import list_project_memories, reload_project_memories
    log.info("orchestration_project_memories", reload=reload)
    try:
        if reload:
            reload_project_memories()
        memories = list_project_memories()
        return json.dumps(memories, indent=2)
    except Exception as e:
        log.error("orchestration_project_memories_error", error=str(e))
        return f"Memories error: {e}"


# =============================================================================
# Registry Builder
# =============================================================================


def get_orchestration_tools() -> ToolRegistry:
    """
    Get registry with multi-model orchestration tools.
    
    These tools allow the chat agent to:
    - Delegate tasks to specific model tiers
    - Run parallel batch comparisons
    - Use deep thinking for complex problems
    - Check backend health and available models
    
    Returns:
        ToolRegistry with orchestration tools registered
    """
    registry = ToolRegistry()
    
    # Delegate tool - send tasks to specific backends/models
    registry.register(ToolDefinition(
        name="delegate",
        description="""Send a task to a specific backend or model tier.

Use when:
- User wants a specific model tier (quick/coder/moe/thinking)
- User wants to use local vs remote GPU
- Task needs a specialized model (e.g., code review → coder tier)

Model Tiers:
- quick: Fast responses, simple tasks (7B-14B models)
- coder: Code-specialized, technical tasks (14B code models)
- moe: Complex reasoning, deep analysis (30B+ MoE models)
- thinking: Extended reasoning with thinking steps

Task Types:
- quick, summarize: Simple/fast tasks
- review, analyze: Code analysis tasks
- generate: Code generation
- plan, critique: Deep reasoning tasks

IMPORTANT: Use final=true when the delegated model's response IS the final answer.
This skips reloading the orchestrating model, saving ~5-10 seconds of GPU swap time.""",
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "enum": ["quick", "summarize", "review", "analyze", "generate", "plan", "critique"],
                    "description": "Task type - determines default model tier"
                },
                "content": {
                    "type": "string",
                    "description": "The prompt or content to process"
                },
                "model": {
                    "type": "string",
                    "description": "Force model tier: quick, coder, moe, or thinking"
                },
                "backend_type": {
                    "type": "string",
                    "enum": ["local", "remote"],
                    "description": "Force local or remote GPU backend"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language hint: python, typescript, rust, go, etc."
                },
                "final": {
                    "type": "boolean",
                    "description": "If true, return delegated result directly without reloading orchestrator. Use when delegation IS the final answer.",
                    "default": False
                }
            },
            "required": ["task", "content"]
        },
        handler=_delegate_handler,
    ))
    
    # Batch tool - parallel execution across backends
    registry.register(ToolDefinition(
        name="batch",
        description="""Process multiple tasks in PARALLEL across all available GPUs.

Use when:
- User wants to compare responses from different models
- User has multiple independent tasks to run
- User wants parallel processing for speed

The tasks are distributed round-robin across all enabled backends.

Example tasks JSON:
[
  {"task": "review", "content": "code here", "model": "coder"},
  {"task": "analyze", "content": "same code", "model": "moe"}
]""",
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "string",
                    "description": "JSON array of task objects. Each object has: task (string), content (string), and optionally model (string)"
                }
            },
            "required": ["tasks"]
        },
        handler=_batch_handler,
    ))
    
    # Batch Vote tool - voting across multiple backends
    registry.register(ToolDefinition(
        name="batch_vote",
        description="""Run the SAME prompt across multiple backends and VOTE on the best result.

Uses MDAP k-voting consensus for mathematically guaranteed accuracy.
Each backend responds independently, then responses are voted on.

Use when:
- High-stakes question requiring verification from multiple models
- User wants consensus from different LLMs
- "What do all the models think about this?"
- "Get a second/third opinion on this"

This is MORE RELIABLE than single-model voting because different models
have different failure modes - if they agree, you can be very confident!

Example:
- batch_vote(prompt="Is this architecture secure?", k=3)
- batch_vote(prompt="Review this code", backends="ollama-local,ollama-remote")""",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to send to all backends"
                },
                "task": {
                    "type": "string",
                    "description": "Task type (analyze, review, generate, etc.)",
                    "default": "analyze"
                },
                "k": {
                    "type": "integer",
                    "description": "Votes needed for consensus (default: 3)",
                    "default": 3
                },
                "backends": {
                    "type": "string",
                    "description": "Comma-separated backend IDs (default: all enabled)"
                }
            },
            "required": ["prompt"]
        },
        handler=_batch_vote_handler,
    ))
    
    # Think tool - deep reasoning
    registry.register(ToolDefinition(
        name="think",
        description="""Deep reasoning for complex problems using extended thinking.

Use when:
- User asks for thorough or careful analysis
- Problem requires multi-step reasoning
- User explicitly wants "deep thinking" or "careful consideration"
- Complex architecture or design decisions

Depth Levels:
- quick: Fast answer, minimal reasoning (14B model)
- normal: Balanced reasoning with thinking (14B coder)
- deep: Thorough multi-step analysis (30B+ MoE model)

IMPORTANT: Use final=true when the thinking result IS the final answer.
This skips reloading the orchestrating model, saving ~5-10 seconds of GPU swap time.""",
        parameters={
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The problem or question to think through deeply"
                },
                "context": {
                    "type": "string",
                    "description": "Supporting information - code, docs, constraints",
                    "default": ""
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "normal", "deep"],
                    "description": "Reasoning depth level",
                    "default": "normal"
                },
                "final": {
                    "type": "boolean",
                    "description": "If true, return result directly without reloading orchestrator. Use when thinking IS the final answer.",
                    "default": False
                }
            },
            "required": ["problem"]
        },
        handler=_think_handler,
    ))
    
    # Health tool - check backend status
    registry.register(ToolDefinition(
        name="health",
        description="""Check health status of all GPU backends.

Use when user asks about:
- Backend availability ("what's available?")
- Which models are currently loaded
- System status and capabilities
- Connection issues""",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=_health_handler,
    ))
    
    # Models tool - list available models
    registry.register(ToolDefinition(
        name="models",
        description="""List all available models across all backends.

Use when user asks:
- "What models are available?"
- "Which backends have which models?"
- "What can I use?"

Returns model tiers (quick/coder/moe/thinking) and specific model names.""",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=_models_handler,
    ))
    
    # Switch backend tool
    registry.register(ToolDefinition(
        name="switch_backend",
        description="""Switch to a specific backend by ID.

Use when user wants to:
- Use a specific backend explicitly
- Switch from local to remote or vice versa
- Use a named backend configuration""",
        parameters={
            "type": "object",
            "properties": {
                "backend_id": {
                    "type": "string",
                    "description": "The backend ID to switch to (e.g., 'ollama-local', 'ollama-remote')"
                }
            },
            "required": ["backend_id"]
        },
        handler=_switch_backend_handler,
    ))
    
    # Compare tool - compare responses from multiple models
    registry.register(ToolDefinition(
        name="compare",
        description="""Compare responses from multiple models.

IMPORTANT: Works on single GPU! Models are loaded sequentially.

Use when user asks to:
- Compare models: "what do qwen and deepseek think about this"
- Get multiple perspectives: "compare different models on this code"
- A/B test responses: "how does model X vs model Y handle this"

The tool handles single-backend setups by swapping models sequentially.
On multi-backend setups, runs in parallel.""",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or task to compare across models"
                },
                "models": {
                    "type": "string",
                    "description": "Comma-separated model names (e.g., 'qwen2.5:14b,deepseek-coder:6.7b')"
                },
                "include_synthesis": {
                    "type": "boolean",
                    "description": "Include a summary comparing the responses",
                    "default": True
                }
            },
            "required": ["prompt", "models"]
        },
        handler=_compare_handler,
    ))
    
    # Vote tool - MDAP k-voting for self-consistency
    registry.register(ToolDefinition(
        name="vote",
        description="""MDAP K-voting for mathematically guaranteed response quality.

Based on "Massively Decomposed Agentic Processes" paper:
- First-to-ahead-by-k voting
- Red-flagging for quality issues
- Semantic similarity matching
- Auto kmin calculation: Θ(ln s) scaling

Use when:
- User wants a RELIABLE answer: "make sure", "verify", "double-check"
- High-stakes questions requiring confidence
- Complex analysis where consistency matters
- User explicitly asks for voting or consensus

Features:
- Auto-calculates optimal k based on task complexity
- Red-flags low-quality responses (too long, repetitive, incoherent)
- Tracks voting statistics for dashboard tuning
- Returns confidence probability with result""",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question to vote on"
                },
                "k": {
                    "type": "integer",
                    "description": "Votes needed for consensus (auto-calculated if not provided)"
                },
                "model": {
                    "type": "string",
                    "description": "Specific model (optional, auto-selected based on task)"
                },
                "target_accuracy": {
                    "type": "number",
                    "description": "Target accuracy (0.0-1.0, default: 0.9999)",
                    "default": 0.9999
                }
            },
            "required": ["prompt"]
        },
        handler=_vote_handler,
    ))
    
    # Compute tool - programmatic operations for things LLMs struggle with
    registry.register(ToolDefinition(
        name="compute",
        description="""Programmatic string operations for tasks LLMs struggle with.

USE THIS FOR:
- String reversal: "What is X spelled backwards?" → compute(operation="reverse", input_value="X")
- Case changes: uppercase, lowercase
- String analysis: length, sort, character counts

EXAMPLES:
- "elbow backwards" → compute("reverse", "elbow") → "woble"
- "hello uppercase" → compute("uppercase", "hello") → "HELLO"

This gives EXACT CORRECT answers, not LLM guesses!""",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["reverse", "uppercase", "lowercase", "length", "sort", "count"],
                    "description": "The operation to perform"
                },
                "input_value": {
                    "type": "string",
                    "description": "The string to operate on"
                }
            },
            "required": ["operation", "input_value"]
        },
        handler=_compute_handler,
    ))
    

    # Session Compaction
    registry.register(ToolDefinition(
        name="session_compact",
        description="Compact a session's conversation history using LLM summarization. Reduces token count while preserving key info.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to compact"},
                "force": {"type": "boolean", "description": "Force compaction even if below threshold", "default": False},
            },
            "required": ["session_id"]
        },
        handler=_session_compact_handler,
    ))

    # Session Stats
    registry.register(ToolDefinition(
        name="session_stats",
        description="Get compaction statistics for a session, including current token usage and recommendations.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to check"},
            },
            "required": ["session_id"]
        },
        handler=_session_stats_handler,
    ))

    # Project Memories
    registry.register(ToolDefinition(
        name="project_memories",
        description="List project memories (DELIA.md files) loaded into context. Shows instruction hierarchy and size.",
        parameters={
            "type": "object",
            "properties": {
                "reload": {"type": "boolean", "description": "Force reload of all project memories", "default": False},
            }
        },
        handler=_project_memories_handler,
    ))

    return registry
