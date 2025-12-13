# Copyright (C) 2023 the project owner
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

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from .config import config
from .file_helpers import read_files, read_serena_memory
from .language import detect_language, get_system_prompt
from .prompt_templates import create_structured_prompt
from .validation import (
    VALID_MODELS,
    validate_content,
    validate_file_path,
    validate_model_hint,
    validate_task,
)

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
) -> str:
    """Prepare content with context, files, and symbol focus.

    Args:
        content: The main task content/prompt
        context: Comma-separated Serena memory names to include
        symbols: Comma-separated symbol names to focus on
        include_references: Whether references to symbols are included
        files: Comma-separated file paths to read and include (Delia reads directly)

    Returns:
        Prepared content string with all context assembled
    """
    parts = []

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
    # Call LLM
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

    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        raise Exception(f"LLM call failed: {error_msg}")

    response_text = result.get("response", "")
    tokens = result.get("tokens", 0)

    # Strip thinking tags
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1].strip()

    return response_text, tokens


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

    # Add concise metadata footer
    return f"""{response_text}

---
_Model: {selected_model} | Tokens: {tokens} | Time: {elapsed_ms}ms | Backend: {backend_name}_"""


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

    Returns:
        Response string from LLM, optionally with metadata footer
    """
    start_time = time.time()

    # Validate request
    valid, error = await validate_delegate_request(task, content, file, model)
    if not valid:
        return error

    # Prepare content with context, files, and symbols
    prepared_content = await prepare_delegate_content(content, context, symbols, include_references, files)

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

    # Execute the LLM call
    try:
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
    )
