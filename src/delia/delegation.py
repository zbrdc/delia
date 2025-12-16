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
Delegation helper functions for the delegate tool.

This module contains helper functions for validating, preparing,
and executing delegate requests. Functions that need runtime
dependencies (like call_llm, tracker) receive them via a context object.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from .config import config, get_affinity_tracker, get_prewarm_tracker
from .backend_manager import backend_manager
from .routing import BackendScorer
from .file_helpers import read_files, read_serena_memory
from .language import detect_language, get_system_prompt
from .prompt_templates import create_structured_prompt
from .text_utils import strip_thinking_tags
from .tokens import count_tokens
from .validation import (
    VALID_MODELS,
    validate_content,
    validate_file_path,
    validate_model_hint,
    validate_task,
)
from .quality import validate_response

if TYPE_CHECKING:
    from .backend_manager import BackendConfig

log = structlog.get_logger()


@dataclass
class DelegateContext:
    """Runtime context for delegate operations.

    Holds references to functions and objects that are defined in mcp_server.py
    and need to be passed to delegation helpers.
    """

    # Async function to select model: (task_type, content_len, model_override, content) -> model_name
    select_model: Callable[[str, int, str | None, str], Coroutine[Any, Any, str]]

    # Function to get active backend: () -> BackendConfig | None
    get_active_backend: Callable[[], Any]

    # Async function to call LLM: (model, prompt, system, thinking, **kwargs) -> dict
    call_llm: Callable[..., Coroutine[Any, Any, dict]]

    # Function to get current client ID: () -> str | None
    get_client_id: Callable[[], str | None]

    # Tracker object with update_last_request method
    tracker: Any


async def validate_delegate_request(
    task: str,
    content: str,
    file: str | None = None,
    model: str | None = None,
) -> tuple[bool, str]:
    """Validate all inputs for delegate request.

    Args:
        task: Task type (review, analyze, generate, etc.)
        content: Content to process
        file: Optional file path
        model: Optional model override

    Returns:
        Tuple of (is_valid, error_message)
    """
    valid, error = validate_task(task)
    if not valid:
        return False, f"Error: {error}"

    valid, error = validate_content(content)
    if not valid:
        return False, f"Error: {error}"

    valid, error = validate_file_path(file)
    if not valid:
        return False, f"Error: {error}"

    valid, error = validate_model_hint(model)
    if not valid:
        return False, f"Error: {error}"

    return True, ""


async def prepare_delegate_content(
    content: str,
    context: str | None = None,
    symbols: str | None = None,
    include_references: bool = False,
    files: str | None = None,
    session_context: str | None = None,
) -> str:
    """Prepare content with context, files, and symbol focus.

    Args:
        content: The main task content/prompt
        context: Comma-separated Serena memory names to include
        symbols: Comma-separated symbol names to focus on
        include_references: Whether references to symbols are included
        files: Comma-separated file paths to read and include (Delia reads directly)
        session_context: Optional conversation history from session manager

    Returns:
        Prepared content string with all context assembled
    """
    parts = []

    # Add session history at the start for conversation continuity
    if session_context:
        parts.append("### Previous Conversation\n" + session_context + "\n")

    # Load files directly from disk (efficient - no Claude serialization)
    if files:
        file_contents = read_files(files)
        if file_contents:
            for path, file_content in file_contents:
                # Detect language from extension for syntax highlighting hint
                ext = Path(path).suffix.lstrip(".")
                lang_hint = ext if ext else ""
                parts.append(f"### File: `{path}`\n```{lang_hint}\n{file_content}\n```")
            log.info("files_loaded", count=len(file_contents), paths=[p for p, _ in file_contents])

    # Load Serena memory context if specified
    if context:
        memory_names = [m.strip() for m in context.split(",")]
        for mem_name in memory_names:
            mem_content = read_serena_memory(mem_name)
            if mem_content:
                parts.append(f"### Context from '{mem_name}':\n{mem_content}")
                log.info("context_memory_loaded", memory=mem_name)

    # Add symbol focus hint if symbols specified
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        symbol_hint = f"### Focus Symbols: {', '.join(symbol_list)}"
        if include_references:
            symbol_hint += "\n_References to these symbols are included below._"
        parts.append(symbol_hint)
        log.info("context_symbol_focus", symbols=symbol_list, include_references=include_references)

    # Add task content
    if parts:
        parts.append(f"---\n\n### Task:\n{content}")
        return "\n\n".join(parts)
    else:
        return content


def determine_task_type(task: str) -> str:
    """Map user task to internal task type.

    Args:
        task: User-provided task name

    Returns:
        Internal task type string
    """
    task_map = {
        "review": "review",
        "analyze": "analyze",
        "generate": "generate",
        "summarize": "summarize",
        "critique": "critique",
        "quick": "quick",
        "plan": "plan",
        "think": "analyze",  # Treat direct think tasks as analyze-tier by default
    }
    return task_map.get(task, "analyze")


async def select_delegate_model(
    ctx: DelegateContext,
    task_type: str,
    content: str,
    model_override: str | None = None,
    backend: str | None = None,
    backend_obj: Any | None = None,
) -> tuple[str, str, Any]:
    """Select appropriate model and backend for the task.

    Args:
        ctx: Delegate context with runtime dependencies
        task_type: Internal task type
        content: Content to process (used for length-based selection)
        model_override: Optional model tier override
        backend: Optional backend override
        backend_obj: Optional backend config object

    Returns:
        Tuple of (selected_model, tier, target_backend)
    """
    # Determine model tier from task type
    if task_type in config.moe_tasks:
        tier = "moe"
    elif task_type in config.coder_tasks:
        tier = "coder"
    elif task_type == "quick" or task_type == "summarize":
        tier = "quick"
    else:
        tier = "quick"  # Default

    # Override tier with model hint if provided
    if model_override and model_override in VALID_MODELS:
        tier = model_override

    # Get the actual model name from backend manager or fall back to config.py
    selected_model = None

    # First priority: use backend_obj if provided (passed from delegate())
    if backend_obj:
        selected_model = backend_obj.models.get(tier)
        if selected_model:
            log.info("model_from_backend_obj", backend=backend_obj.id, tier=tier, model=selected_model)

    # Try simplified backend selection (replaces complex backend_manager)
    if not selected_model:
        selected_model = await ctx.select_model(task_type, len(content), model_override, content)
        log.info("model_from_simplified_selection", tier=tier, model=selected_model)

    # Use provided backend or fall back to active backend
    target_backend = backend or ctx.get_active_backend()

    return selected_model, tier, target_backend


async def execute_delegate_call(
    ctx: DelegateContext,
    selected_model: str,
    content: str,
    system: str,
    task_type: str,
    original_task: str,
    detected_language: str,
    target_backend: Any,
    backend_obj: Any | None = None,
    max_tokens: int | None = None,
) -> tuple[str, int]:
    """Execute the LLM call and return response with metadata.

    Args:
        ctx: Delegate context with runtime dependencies
        selected_model: Model name to use
        content: Prepared content/prompt
        system: System prompt
        task_type: Internal task type
        original_task: Original user task name
        detected_language: Detected programming language
        target_backend: Backend to use
        backend_obj: Optional backend config object
        max_tokens: Optional token limit

    Returns:
        Tuple of (response_text, tokens_used)

    Raises:
        Exception: If LLM call fails
    """
    enable_thinking = task_type in config.thinking_tasks

    # Create a preview for the recent calls log
    content_preview = content[:200].replace("\n", " ").strip()

    result = await ctx.call_llm(
        selected_model,
        content,
        system,
        enable_thinking,
        task_type=task_type,
        original_task=original_task,
        language=detected_language,
        content_preview=content_preview,
        backend=target_backend,
        backend_obj=backend_obj,
        max_tokens=max_tokens,
    )

    # Record task-backend affinity with quality score
    success = result.get("success", False)
    if backend_obj is not None:
        backend_id = getattr(backend_obj, "id", None)
        if backend_id:
            tracker = get_affinity_tracker()
            if success:
                # Validate response quality for nuanced affinity learning
                response_text = result.get("response", "")
                quality_result = validate_response(response_text, task_type)
                tracker.update(backend_id, task_type, quality=quality_result.overall)
            else:
                # API failure = quality 0.0
                tracker.update(backend_id, task_type, quality=0.0)

    if not success:
        error_msg = result.get("error", "Unknown error")
        raise Exception(f"LLM call failed: {error_msg}")

    response_text = result.get("response", "")
    tokens = result.get("tokens", 0)

    # Strip thinking tags from models like Qwen3, DeepSeek-R1
    response_text = strip_thinking_tags(response_text)

    return response_text, tokens


async def execute_hedged_call(
    ctx: DelegateContext,
    backends: list["BackendConfig"],
    selected_model: str,
    content: str,
    system: str,
    task_type: str,
    original_task: str,
    detected_language: str,
    delay_ms: int = 50,
    max_tokens: int | None = None,
) -> tuple[str, int, "BackendConfig"]:
    """Execute hedged LLM calls to multiple backends with staggered starts.

    Sends requests to multiple backends with a delay between starts.
    Returns the first successful response and cancels remaining tasks.
    Only the winning backend is recorded for affinity tracking.

    Args:
        ctx: Delegate context with runtime dependencies
        backends: List of backends to try (sorted by preference)
        selected_model: Model name to use
        content: Prepared content/prompt
        system: System prompt
        task_type: Internal task type
        original_task: Original user task name
        detected_language: Detected programming language
        delay_ms: Milliseconds between starting each backend request
        max_tokens: Optional token limit

    Returns:
        Tuple of (response_text, tokens_used, winning_backend)

    Raises:
        Exception: If all backends fail
    """
    if not backends:
        raise ValueError("No backends provided for hedged call")

    # Single backend - fall back to regular execution
    if len(backends) == 1:
        response_text, tokens = await execute_delegate_call(
            ctx=ctx,
            selected_model=selected_model,
            content=content,
            system=system,
            task_type=task_type,
            original_task=original_task,
            detected_language=detected_language,
            target_backend=backends[0].id,
            backend_obj=backends[0],
            max_tokens=max_tokens,
        )
        return response_text, tokens, backends[0]

    enable_thinking = task_type in config.thinking_tasks
    content_preview = content[:200].replace("\n", " ").strip()

    # Create tasks with staggered starts
    async def call_with_delay(
        backend: "BackendConfig", delay: float
    ) -> tuple[dict[str, Any], "BackendConfig"]:
        """Execute a backend call after an optional delay."""
        if delay > 0:
            await asyncio.sleep(delay)
        result = await ctx.call_llm(
            selected_model,
            content,
            system,
            enable_thinking,
            task_type=task_type,
            original_task=original_task,
            language=detected_language,
            content_preview=content_preview,
            backend=backend.id,
            backend_obj=backend,
            max_tokens=max_tokens,
        )
        return result, backend

    # Start tasks with staggered delays
    delay_seconds = delay_ms / 1000.0
    tasks = {
        asyncio.create_task(
            call_with_delay(backend, i * delay_seconds),
            name=f"hedge_{backend.id}",
        ): backend
        for i, backend in enumerate(backends)
    }

    pending = set(tasks.keys())
    winner: tuple[str, int, "BackendConfig"] | None = None
    errors: list[str] = []

    try:
        while pending and winner is None:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                backend = tasks[task]
                try:
                    result, winning_backend = task.result()
                    if result.get("success", False):
                        response_text = result.get("response", "")
                        tokens = result.get("tokens", 0)
                        # Strip thinking tags
                        response_text = strip_thinking_tags(response_text)
                        winner = (response_text, tokens, winning_backend)

                        # Record affinity with quality score for winning backend
                        quality_result = validate_response(response_text, task_type)
                        tracker = get_affinity_tracker()
                        tracker.update(
                            winning_backend.id, task_type, quality=quality_result.overall
                        )

                        log.info(
                            "hedged_call_winner",
                            backend=winning_backend.id,
                            tokens=tokens,
                            remaining_cancelled=len(pending),
                        )
                        break
                    else:
                        error = result.get("error", "Unknown error")
                        errors.append(f"{backend.id}: {error}")
                        # Record failure for this backend (quality 0.0)
                        tracker = get_affinity_tracker()
                        tracker.update(backend.id, task_type, quality=0.0)
                except asyncio.CancelledError:
                    # Task was cancelled by us, ignore
                    pass
                except Exception as e:
                    errors.append(f"{backend.id}: {e!s}")
                    tracker = get_affinity_tracker()
                    tracker.update(backend.id, task_type, quality=0.0)
    finally:
        # Cancel any remaining pending tasks
        for task in pending:
            task.cancel()
        # Wait for cancellations to complete
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    if winner:
        return winner

    # All backends failed
    error_summary = "; ".join(errors) if errors else "All backends failed"
    raise Exception(f"Hedged call failed: {error_summary}")


async def execute_voting_call(
    ctx: DelegateContext,
    backends: list["BackendConfig"],
    selected_model: str,
    content: str,
    system: str,
    task_type: str,
    original_task: str,
    detected_language: str,
    voting_k: int = 2,
    delay_ms: int = 50,
    max_tokens: int | None = None,
) -> tuple[str, int, "BackendConfig", dict[str, Any]]:
    """Execute k-voting consensus call to multiple backends.

    Implements the MDAP "first-to-ahead-by-k" voting mechanism for
    mathematically-guaranteed reliability.

    With k=3 and base accuracy p=0.99:
        P(correct) = 1/(1+((1-p)/p)^k) = 0.999999

    Formula verified with Wolfram Alpha.

    Args:
        ctx: Delegate context with runtime dependencies
        backends: List of backends to try (sorted by preference)
        selected_model: Model name to use
        content: Prepared content/prompt
        system: System prompt
        task_type: Internal task type
        original_task: Original user task name
        detected_language: Detected programming language
        voting_k: Number of matching votes needed (default 2)
        delay_ms: Milliseconds between starting each backend request
        max_tokens: Optional token limit

    Returns:
        Tuple of (response_text, tokens_used, winning_backend, voting_metadata)

    Raises:
        Exception: If consensus cannot be reached
    """
    from .voting import VotingConsensus, estimate_task_complexity
    from .quality import get_quality_validator
    from .voting_stats import get_voting_stats_tracker

    if not backends:
        raise ValueError("No backends provided for voting call")

    # Single backend - fall back to regular execution
    if len(backends) == 1:
        response_text, tokens = await execute_delegate_call(
            ctx=ctx,
            selected_model=selected_model,
            content=content,
            system=system,
            task_type=task_type,
            original_task=original_task,
            detected_language=detected_language,
            target_backend=backends[0].id,
            backend_obj=backends[0],
            max_tokens=max_tokens,
        )
        return response_text, tokens, backends[0], {"votes": 1, "k": 1, "mode": "single"}

    # Initialize voting consensus
    quality_validator = get_quality_validator()
    consensus = VotingConsensus(
        k=voting_k,
        quality_validator=quality_validator,
        max_response_length=700,  # MDAP paper threshold
    )

    enable_thinking = task_type in config.thinking_tasks
    content_preview = content[:200].replace("\n", " ").strip()

    # Track backend responses for affinity
    backend_responses: dict[str, tuple[str, int]] = {}

    async def call_with_delay(
        backend: "BackendConfig", delay: float
    ) -> tuple[dict[str, Any], "BackendConfig"]:
        """Execute a backend call after an optional delay."""
        if delay > 0:
            await asyncio.sleep(delay)
        result = await ctx.call_llm(
            selected_model,
            content,
            system,
            enable_thinking,
            task_type=task_type,
            original_task=original_task,
            language=detected_language,
            content_preview=content_preview,
            backend=backend.id,
            backend_obj=backend,
            max_tokens=max_tokens,
        )
        return result, backend

    # Start tasks with staggered delays
    delay_seconds = delay_ms / 1000.0
    tasks = {
        asyncio.create_task(
            call_with_delay(backend, i * delay_seconds),
            name=f"vote_{backend.id}",
        ): backend
        for i, backend in enumerate(backends)
    }

    pending = set(tasks.keys())
    winner: tuple[str, int, "BackendConfig"] | None = None
    errors: list[str] = []
    voting_metadata: dict[str, Any] = {
        "k": voting_k,
        "votes_cast": 0,
        "red_flagged": 0,
        "unique_responses": 0,
        "mode": "voting",
    }

    try:
        while pending and winner is None:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                backend = tasks[task]
                try:
                    result, responding_backend = task.result()
                    if result.get("success", False):
                        response_text = result.get("response", "")
                        tokens = result.get("tokens", 0)
                        response_text = strip_thinking_tags(response_text)

                        # Store for affinity tracking
                        backend_responses[responding_backend.id] = (response_text, tokens)

                        # Add vote to consensus
                        vote_result = consensus.add_vote(response_text)
                        voting_metadata["votes_cast"] += 1

                        if vote_result.red_flagged:
                            voting_metadata["red_flagged"] += 1
                            log.info(
                                "voting_red_flag",
                                backend=responding_backend.id,
                                reason=vote_result.red_flag_reason,
                            )
                            # Record low quality for this backend
                            tracker = get_affinity_tracker()
                            tracker.update(responding_backend.id, task_type, quality=0.2)
                            # Track rejection in voting stats
                            voting_tracker = get_voting_stats_tracker()
                            voting_tracker.record_rejection(
                                reason=vote_result.red_flag_reason or "unknown",
                                backend_id=responding_backend.id,
                                tier=task_type,
                                response_preview=response_text[:100],
                            )
                            continue

                        if vote_result.consensus_reached:
                            winner = (
                                vote_result.winning_response or response_text,
                                tokens,
                                responding_backend,
                            )
                            voting_metadata["winning_votes"] = vote_result.votes_for_winner

                            # Record high quality for winning backend
                            quality_result = validate_response(response_text, task_type)
                            tracker = get_affinity_tracker()
                            tracker.update_with_outcome(
                                responding_backend.id,
                                task_type,
                                succeeded=True,
                                efficiency=min(1.0, 500 / max(tokens, 1)),
                            )

                            log.info(
                                "voting_consensus_reached",
                                backend=responding_backend.id,
                                k=voting_k,
                                votes=vote_result.votes_for_winner,
                                total_votes=vote_result.total_votes,
                            )
                            # Track consensus success in voting stats
                            voting_tracker = get_voting_stats_tracker()
                            voting_tracker.record_consensus(
                                votes_cast=vote_result.total_votes,
                                k=voting_k,
                                tier=task_type,
                                backend_id=responding_backend.id,
                                success=True,
                            )
                            voting_tracker.record_quality(task_type, quality_result.overall)
                            break
                    else:
                        error = result.get("error", "Unknown error")
                        errors.append(f"{backend.id}: {error}")
                        tracker = get_affinity_tracker()
                        tracker.update(backend.id, task_type, quality=0.0)

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    errors.append(f"{backend.id}: {e!s}")
                    tracker = get_affinity_tracker()
                    tracker.update(backend.id, task_type, quality=0.0)

    finally:
        # Cancel remaining pending tasks
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    if winner:
        return winner[0], winner[1], winner[2], voting_metadata

    # No consensus - get best response
    best_response, consensus_meta = consensus.get_best_response()
    voting_metadata["unique_responses"] = consensus_meta.unique_responses

    # Track failed consensus in voting stats
    voting_tracker = get_voting_stats_tracker()
    voting_tracker.record_consensus(
        votes_cast=voting_metadata["votes_cast"],
        k=voting_k,
        tier=task_type,
        backend_id="none",
        success=False,
    )

    if best_response:
        # Find which backend gave this response
        for backend_id, (resp, tokens) in backend_responses.items():
            if resp == best_response:
                backend_obj = next(
                    (b for b in backends if b.id == backend_id), backends[0]
                )
                log.warning(
                    "voting_no_consensus_using_best",
                    backend=backend_id,
                    votes=consensus_meta.winning_votes,
                    k=voting_k,
                )
                # Track quality of fallback response
                quality_result = validate_response(best_response, task_type)
                voting_tracker.record_quality(task_type, quality_result.overall)
                return best_response, tokens, backend_obj, voting_metadata

    # All backends failed
    error_summary = "; ".join(errors) if errors else "All backends failed"
    raise Exception(f"Voting call failed: {error_summary}")


def finalize_delegate_response(
    ctx: DelegateContext,
    response_text: str,
    selected_model: str,
    tokens: int,
    elapsed_ms: int,
    detected_language: str,
    target_backend: Any,
    tier: str,
    include_metadata: bool = True,
    voting_metadata: dict[str, Any] | None = None,
) -> str:
    """Add metadata footer and update tracking.

    Args:
        ctx: Delegate context with runtime dependencies
        response_text: Raw response from LLM
        selected_model: Model that was used
        tokens: Tokens used
        elapsed_ms: Time elapsed in milliseconds
        detected_language: Detected programming language
        target_backend: Backend that was used
        tier: Model tier that was used
        include_metadata: If False, return response without footer (saves ~30 tokens)
        voting_metadata: Optional voting consensus info (k, votes, red-flagged)

    Returns:
        Final response string, optionally with metadata footer
    """
    # Update tracker with actual token count and model tier
    client_id = ctx.get_client_id()
    if client_id and ctx.tracker:
        ctx.tracker.update_last_request(client_id, tokens=tokens, model_tier=tier)

    # Return without metadata if requested (saves Claude tokens)
    if not include_metadata:
        return response_text

    # Extract backend name (handle both string and BackendConfig)
    backend_name = target_backend.name if hasattr(target_backend, "name") else str(target_backend)

    # Build metadata footer
    base_meta = f"Model: {selected_model} | Tokens: {tokens} | Time: {elapsed_ms}ms | Backend: {backend_name}"

    # Add voting info if k-voting was used
    if voting_metadata and voting_metadata.get("mode") == "voting":
        k = voting_metadata.get("k", 2)
        votes = voting_metadata.get("votes_cast", 0)
        red_flagged = voting_metadata.get("red_flagged", 0)
        # Show voting mode with MDAP guarantee indicator
        voting_info = f" | Voting: k={k}, votes={votes}"
        if red_flagged > 0:
            voting_info += f", flagged={red_flagged}"
        base_meta += voting_info

    return f"""{response_text}

---
_{base_meta}_"""


async def delegate_impl(
    ctx: DelegateContext,
    task: str,
    content: str,
    file: str | None = None,
    model: str | None = None,
    language: str | None = None,
    context: str | None = None,
    symbols: str | None = None,
    include_references: bool = False,
    backend: str | None = None,
    backend_obj: Any | None = None,
    files: str | None = None,
    include_metadata: bool = True,
    max_tokens: int | None = None,
    session_id: str | None = None,
) -> str:
    """Core implementation for delegate - can be called directly by batch().

    Args:
        ctx: Delegate context with runtime dependencies
        task: Task type (review, analyze, generate, etc.)
        content: Content to process
        file: Optional file path for context
        model: Optional model tier override
        language: Optional language hint
        context: Comma-separated Serena memory names
        symbols: Comma-separated symbol names to focus on
        include_references: If True, indicates that references to symbols are included
        backend: Override backend ID
        backend_obj: Backend config object
        files: Comma-separated file paths (Delia reads directly)
        include_metadata: If False, skip the metadata footer
        max_tokens: Limit response tokens
        session_id: Optional session ID for conversation history tracking

    Returns:
        Response string from LLM, optionally with metadata footer
    """
    start_time = time.time()

    # Validate request
    valid, error = await validate_delegate_request(task, content, file, model)
    if not valid:
        return error

    # Session handling - load context and record user message
    session_context = None
    session_manager = None
    if session_id:
        from .session_manager import get_session_manager
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)
        if session:
            # Get conversation history (limit to 6000 tokens to leave room for response)
            session_context = session.get_context_window(max_tokens=6000)
            # Record user's message
            session_manager.add_to_session(
                session_id, 
                "user", 
                content,  # The user's input
                tokens=0,  # We don't count input tokens yet
                model="",
                task_type=task
            )

    # Prepare content with context, files, symbols, and session history
    prepared_content = await prepare_delegate_content(
        content, 
        context, 
        symbols, 
        include_references, 
        files,
        session_context=session_context,
    )

    # Map task to internal type
    task_type = determine_task_type(task)

    # Create enhanced, structured prompt with templates
    prepared_content = create_structured_prompt(
        task_type=task_type,
        content=prepared_content,
        file_path=file,
        language=language,
        symbols=symbols.split(",") if symbols else None,
        context_files=context.split(",") if context else None,
    )

    # Detect language and get system prompt
    detected_language = language or detect_language(prepared_content, file or "")
    system = get_system_prompt(detected_language, task_type)

    # Select model and backend
    selected_model, tier, target_backend = await select_delegate_model(
        ctx, task_type, prepared_content, model, backend, backend_obj
    )

    # Check execution mode: voting > hedging > single
    # Voting provides MDAP mathematical guarantees via k-voting consensus
    # Hedging provides faster response via first-success racing
    # Only used when no specific backend was explicitly requested
    voting_config = backend_manager.routing_config.get("voting", {})
    hedging_config = backend_manager.routing_config.get("hedging", {})

    no_explicit_backend = backend is None and backend_obj is None

    use_voting = voting_config.get("enabled", False) and no_explicit_backend
    use_hedging = (
        hedging_config.get("enabled", False)
        and no_explicit_backend
        and not use_voting  # Voting takes priority
    )

    # Track voting metadata for response
    voting_metadata: dict[str, Any] | None = None

    # Execute the LLM call
    try:
        if use_voting:
            # MDAP k-voting consensus execution
            # Provides mathematical reliability: P(correct|k=3) = 0.999999
            enabled_backends = backend_manager.get_enabled_backends()
            weights = backend_manager.get_scoring_weights()
            scorer = BackendScorer(weights=weights)

            vote_max = voting_config.get("max_backends", 3)
            vote_delay = voting_config.get("delay_ms", 50)

            # Calculate k: auto-adjust based on task complexity or use fixed value
            if voting_config.get("auto_kmin", True):
                from .voting import estimate_task_complexity, VotingConsensus

                estimated_steps = estimate_task_complexity(prepared_content)
                vote_k = VotingConsensus.calculate_kmin(
                    total_steps=estimated_steps,
                    target_accuracy=0.9999,
                )
                log.debug(
                    "auto_kmin_calculated",
                    estimated_steps=estimated_steps,
                    vote_k=vote_k,
                )
            else:
                vote_k = voting_config.get("k", 2)

            # Get top N backends by score for voting
            vote_backends = scorer.select_top_n(
                enabled_backends,
                n=vote_max,
                task_type=task_type,
            )

            if len(vote_backends) > 1:
                log.info(
                    "using_voting_execution",
                    backends=[b.id for b in vote_backends],
                    k=vote_k,
                    mode="mdap_consensus",
                )
                response_text, tokens, winning_backend, voting_metadata = (
                    await execute_voting_call(
                        ctx,
                        vote_backends,
                        selected_model,
                        prepared_content,
                        system,
                        task_type,
                        task,
                        detected_language,
                        voting_k=vote_k,
                        delay_ms=vote_delay,
                        max_tokens=max_tokens,
                    )
                )
                target_backend = winning_backend.id
            else:
                # Only one backend, fall back to normal execution
                response_text, tokens = await execute_delegate_call(
                    ctx,
                    selected_model,
                    prepared_content,
                    system,
                    task_type,
                    task,
                    detected_language,
                    target_backend,
                    backend_obj,
                    max_tokens=max_tokens,
                )

        elif use_hedging:
            # First-success racing (faster but no consensus)
            enabled_backends = backend_manager.get_enabled_backends()
            weights = backend_manager.get_scoring_weights()
            scorer = BackendScorer(weights=weights)

            hedge_max = hedging_config.get("max_backends", 2)
            hedge_delay = hedging_config.get("delay_ms", 50)

            hedge_backends = scorer.select_top_n(
                enabled_backends,
                n=hedge_max,
                task_type=task_type,
            )

            if len(hedge_backends) > 1:
                log.info(
                    "using_hedged_execution",
                    backends=[b.id for b in hedge_backends],
                    delay_ms=hedge_delay,
                )
                response_text, tokens, winning_backend = await execute_hedged_call(
                    ctx,
                    hedge_backends,
                    selected_model,
                    prepared_content,
                    system,
                    task_type,
                    task,
                    detected_language,
                    delay_ms=hedge_delay,
                    max_tokens=max_tokens,
                )
                target_backend = winning_backend.id
            else:
                response_text, tokens = await execute_delegate_call(
                    ctx,
                    selected_model,
                    prepared_content,
                    system,
                    task_type,
                    task,
                    detected_language,
                    target_backend,
                    backend_obj,
                    max_tokens=max_tokens,
                )
        else:
            # Single backend execution
            response_text, tokens = await execute_delegate_call(
                ctx,
                selected_model,
                prepared_content,
                system,
                task_type,
                task,
                detected_language,
                target_backend,
                backend_obj,
                max_tokens=max_tokens,
            )
    except Exception as e:
        return f"Error: {e!s}"

    # Record assistant response to session
    if session_id and session_manager:
        session_manager.add_to_session(
            session_id,
            "assistant",
            response_text,
            tokens=tokens,
            model=selected_model,
            task_type=task
        )

    # Update prewarm tracker with tier usage
    prewarm_tracker = get_prewarm_tracker()
    prewarm_tracker.update(tier)

    # Calculate timing
    elapsed_ms = int((time.time() - start_time) * 1000)

    # Finalize response with metadata
    return finalize_delegate_response(
        ctx,
        response_text,
        selected_model,
        tokens,
        elapsed_ms,
        detected_language,
        target_backend,
        tier,
        include_metadata=include_metadata,
        voting_metadata=voting_metadata,
    )


async def get_delegate_signals(
    ctx: DelegateContext,
    task: str,
    content: str,
    file: str | None = None,
    model: str | None = None,
    language: str | None = None,
    context: str | None = None,
    symbols: str | None = None,
    include_references: bool = False,
    files: str | None = None,
) -> dict[str, Any]:
    """Get estimation signals for a delegate request without executing.

    Returns token counts, recommended model/tier, backend availability,
    and context fit information. Useful for agents to make cost/quality
    decisions before committing to an LLM call.

    Args:
        ctx: Delegate context with runtime dependencies
        task: Task type (review, analyze, generate, etc.)
        content: Content to process
        file: Optional file path for context
        model: Optional model tier override
        language: Optional language hint
        context: Comma-separated Serena memory names
        symbols: Comma-separated symbol names to focus on
        include_references: If True, indicates references are included
        files: Comma-separated file paths

    Returns:
        Dict with estimation signals:
        - estimated_tokens: Token count for prepared content
        - recommended_tier: Model tier (quick/coder/moe/thinking)
        - recommended_model: Specific model name
        - backend_id: Backend that would be used
        - backend_available: Whether backend is healthy
        - context_limit_tokens: Max tokens for recommended tier
        - content_fits: Whether content fits in context window
        - detected_language: Detected programming language
        - task_type: Internal task type mapping
    """
    # Validate request
    valid, error = await validate_delegate_request(task, content, file, model)
    if not valid:
        return {"error": error, "valid": False}

    # Prepare content (same as delegate_impl)
    prepared_content = await prepare_delegate_content(
        content, context, symbols, include_references, files
    )

    # Map task to internal type
    task_type = determine_task_type(task)

    # Create structured prompt (same as delegate_impl)
    prepared_content = create_structured_prompt(
        task_type=task_type,
        content=prepared_content,
        file_path=file,
        language=language,
        symbols=symbols.split(",") if symbols else None,
        context_files=context.split(",") if context else None,
    )

    # Detect language
    detected_language = language or detect_language(prepared_content, file or "")

    # Select model and backend (without executing)
    selected_model, tier, target_backend = await select_delegate_model(
        ctx, task_type, prepared_content, model, None, None
    )

    # Count tokens
    estimated_tokens = count_tokens(prepared_content)

    # Get context limit for the tier
    tier_config = {
        "quick": config.model_quick,
        "coder": config.model_coder,
        "moe": config.model_moe,
        "thinking": config.model_thinking,
    }.get(tier, config.model_quick)

    context_limit = tier_config.context_tokens
    max_input_kb = tier_config.max_input_kb

    # Check if content fits (leave room for output)
    # Use max_input_kb as practical limit, not full context window
    content_bytes = len(prepared_content.encode("utf-8"))
    content_fits = content_bytes <= (max_input_kb * 1024)

    # Get backend info
    backend_id = None
    backend_available = False
    if hasattr(target_backend, "id"):
        backend_id = target_backend.id
        backend_available = getattr(target_backend, "enabled", True)
    elif target_backend:
        backend_id = str(target_backend)
        backend_available = True

    return {
        "valid": True,
        "estimated_tokens": estimated_tokens,
        "recommended_tier": tier,
        "recommended_model": selected_model,
        "backend_id": backend_id,
        "backend_available": backend_available,
        "context_limit_tokens": context_limit,
        "max_input_kb": max_input_kb,
        "content_fits": content_fits,
        "content_size_kb": round(content_bytes / 1024, 1),
        "detected_language": detected_language,
        "task_type": task_type,
    }
