# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Orchestration tool implementations for Delia MCP tools.

Provides implementation functions for:
- think: Deep reasoning with thinking-capable models
- batch: Parallel task execution across backends
- delegate: Single task delegation to LLM backends
- chain: Sequential task execution with piping
- workflow: DAG workflow with conditional branching
- agent: Autonomous agent with tool use
- session management (compact, stats, list, delete)
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ..container import get_container
from ..orchestration.service import get_orchestration_service
from ..text_utils import strip_thinking_tags
from .. import llm
from ..routing import get_router
from ..language import detect_language
from ..delegation import (
    start_prewarm_task,
    _delegate_with_voting,
    _delegate_with_tot,
    _get_delegate_context,
    get_delegate_signals,
    prepare_delegate_content,
    determine_task_type,
    _select_delegate_model_impl,
    _delegate_impl,
)

from .handlers_ace import (
    get_ace_tracker,
    check_ace_gate,
    inject_ace_reminder,
    auto_trigger_reflection,
)

if TYPE_CHECKING:
    from typing import Any

log = structlog.get_logger()


# =============================================================================
# Orchestration Tool Implementations
# =============================================================================


async def think_impl(
    problem: str,
    context: str = "",
    depth: str = "normal",
    session_id: str | None = None,
) -> str:
    """Implementation of the think tool."""
    # ACE Framework Enforcement
    project_path = str(Path.cwd())
    ace_tracker = get_ace_tracker()
    ace_tracker.record_task_start(project_path, "think")

    container = get_container()
    if depth == "quick":
        model_hint = "quick"
    elif depth == "deep":
        model_hint = "thinking"
    else:
        model_hint = "thinking"

    thinking_prompt = f"Think through this problem step by step:\n\n## Problem\n{problem}\n"
    if context:
        thinking_prompt += f"\n## Context\n{context}\n"
    thinking_prompt += (
        "\n## Instructions\n1. Break down the problem into components\n"
        "2. Consider different approaches\n3. Reason through each step\n"
        "4. Conclusion\n\nThink deeply before answering."
    )

    service = get_orchestration_service()
    result = await service.process(
        message=thinking_prompt, session_id=session_id, model_override=model_hint
    )

    container.stats_service.increment_task("think")
    from ..mcp_server import save_all_stats_async

    await save_all_stats_async()

    # Auto-trigger reflection for learning
    response_text = result.result.response
    success = not response_text.startswith("Error:")
    outcome = response_text[:200] if len(response_text) > 200 else response_text
    await auto_trigger_reflection(
        task_type="think",
        task_description=problem[:100],
        outcome=outcome,
        success=success,
        project_path=project_path,
    )

    return inject_ace_reminder(response_text, project_path)


async def batch_impl(
    tasks: str,
    include_metadata: bool = True,
    max_tokens: int | None = None,
    session_id: str | None = None,
) -> str:
    """Implementation of the batch tool."""
    # ACE Framework Enforcement
    project_path = str(Path.cwd())
    ace_tracker = get_ace_tracker()
    ace_tracker.record_task_start(project_path, "batch")

    container = get_container()
    start_time = time.time()
    try:
        task_list = json.loads(tasks)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"
    if not isinstance(task_list, list):
        return "Error: tasks must be a JSON array"
    if len(task_list) == 0:
        return "Error: tasks array is empty"

    available = await container.backend_manager.check_all_health()
    backend_assignments = container.model_router.assign_backends_to_tasks(
        task_list, available, container.backend_manager
    )

    from ..mcp_server import current_client_id, current_username

    captured_client_id = current_client_id.get()
    captured_username = current_username.get()

    async def run_task(
        i: int, t: dict, backend_id: str, client_id: str | None, username: str | None
    ) -> str:
        current_client_id.set(client_id)
        current_username.set(username)
        task_type = t.get("task", "analyze")
        result = await _delegate_impl(
            task=task_type,
            content=t.get("content", ""),
            file=t.get("file"),
            model=t.get("model"),
            language=t.get("language"),
            context=t.get("context"),
            symbols=t.get("symbols"),
            include_references=t.get("include_references", False),
            backend=backend_id,
            files=t.get("files"),
            include_metadata=t.get("include_metadata", include_metadata),
            max_tokens=t.get("max_tokens", max_tokens),
            session_id=session_id,
        )
        return f"### Task {i + 1}: {task_type}\n\n{result}"

    results = await asyncio.gather(
        *[
            run_task(i, t, backend_assignments[i], captured_client_id, captured_username)
            for i, t in enumerate(task_list)
        ]
    )
    elapsed_ms = int((time.time() - start_time) * 1000)
    result = (
        f"# Batch Results\n\n{chr(10).join(results)}\n\n---\n"
        f"_Total tasks: {len(task_list)} | Total time: {elapsed_ms}ms_"
    )

    # Auto-trigger reflection for batch learning
    await auto_trigger_reflection(
        task_type="batch",
        task_description=f"Batch of {len(task_list)} tasks",
        outcome=f"Completed {len(task_list)} tasks in {elapsed_ms}ms",
        success=True,
        project_path=project_path,
    )

    return inject_ace_reminder(result, project_path)


async def delegate_tool_impl(
    task: str,
    content: str,
    file: str | None = None,
    model: str | None = None,
    language: str | None = None,
    context: str | None = None,
    symbols: str | None = None,
    include_references: bool = False,
    backend_type: str | None = None,
    files: str | None = None,
    include_metadata: bool = True,
    max_tokens: int | None = None,
    dry_run: bool = False,
    session_id: str | None = None,
    stream: bool = False,
    reliable: bool = False,
    voting_k: int = 3,
    tot: bool = False,
    auto_context: bool = False,
) -> str:
    """Implementation of the delegate tool."""
    from ..orchestration.result import ModelRole
    from ..prompts import build_system_prompt

    # ACE Gating: Require auto_context before delegation
    project_path = str(Path(file).parent) if file else str(Path.cwd())
    gate_error = check_ace_gate("delegate", project_path)
    if gate_error:
        return gate_error

    start_prewarm_task()

    # ACE Framework Enforcement: Record task start
    ace_tracker = get_ace_tracker()
    ace_tracker.record_task_start(project_path, task)

    if reliable:
        return await _delegate_with_voting(
            task=task,
            content=content,
            file=file,
            model=model,
            language=language,
            context=context,
            symbols=symbols,
            include_references=include_references,
            backend_type=backend_type,
            files=files,
            include_metadata=include_metadata,
            max_tokens=max_tokens,
            session_id=session_id,
            voting_k=voting_k,
        )

    if tot:
        return await _delegate_with_tot(
            task=task,
            content=content,
            file=file,
            model=model,
            language=language,
            context=context,
            symbols=symbols,
            include_references=include_references,
            backend_type=backend_type,
            files=files,
            include_metadata=include_metadata,
            max_tokens=max_tokens,
            session_id=session_id,
        )

    if dry_run:
        ctx = _get_delegate_context()
        signals = await get_delegate_signals(
            ctx,
            task,
            content,
            file,
            model,
            language,
            context,
            symbols,
            include_references,
            files,
        )
        return json.dumps(signals, indent=2)

    _, backend_obj = await get_router().select_optimal_backend(
        content, file, task, backend_type
    )

    if stream and backend_obj:
        ctx = _get_delegate_context()
        prepared_content = await prepare_delegate_content(
            content,
            context,
            symbols,
            include_references,
            files,
            auto_context=auto_context,
        )
        task_type = determine_task_type(task)
        context_header = f"Task: {task_type}\n"
        if file:
            context_header += f"File: {file}\n"
        if language:
            context_header += f"Language: {language}\n"
        if symbols:
            context_header += f"Symbols: {symbols}\n"
        if context:
            context_header += f"Context Files: {context}\n"
        prepared_content = f"{context_header}\n{prepared_content}"

        detected_language = language or detect_language(prepared_content, file or "")
        role_map = {
            "review": ModelRole.CODE_REVIEWER,
            "generate": ModelRole.CODE_GENERATOR,
            "analyze": ModelRole.ANALYST,
            "summarize": ModelRole.SUMMARIZER,
            "plan": ModelRole.ARCHITECT,
            "critique": ModelRole.ANALYST,
        }
        role = role_map.get(task_type, ModelRole.ASSISTANT)
        system = build_system_prompt(role=role)
        if detected_language:
            system += f"\n\nPrimary language: {detected_language}"

        selected_model, tier, _source = await _select_delegate_model_impl(
            ctx, task_type, prepared_content, model, None, backend_obj
        )

        full_response = ""
        total_tokens = 0
        elapsed_ms = 0
        start_time = time.time()

        async for chunk in llm.call_llm_stream(
            model=selected_model,
            prompt=prepared_content,
            system=system,
            task_type=task_type,
            original_task=task,
            language=detected_language,
            backend_obj=backend_obj,
            max_tokens=max_tokens,
        ):
            if chunk.text:
                full_response += chunk.text
            if chunk.done:
                total_tokens = chunk.tokens
                if chunk.metadata:
                    elapsed_ms = chunk.metadata.get("elapsed_ms", 0)
                if chunk.error:
                    return f"Error: {chunk.error}"

        if elapsed_ms == 0:
            elapsed_ms = int((time.time() - start_time) * 1000)
        full_response = strip_thinking_tags(full_response)

        if include_metadata:
            result = (
                f"{full_response}\n\n---\n"
                f"_[OK] {task} (streamed) | {tier} tier | {elapsed_ms}ms | {selected_model}_"
            )
            return inject_ace_reminder(result, project_path)
        return inject_ace_reminder(full_response, project_path)

    result = await _delegate_impl(
        task,
        content,
        file,
        model,
        language,
        context,
        symbols,
        include_references,
        backend=backend_type,
        backend_obj=backend_obj,
        files=files,
        include_metadata=include_metadata,
        max_tokens=max_tokens,
        session_id=session_id,
        auto_context=auto_context,
    )

    # Auto-trigger reflection for learning
    success = not result.startswith("Error:")
    task_desc = f"{task}: {content[:100]}..." if len(content) > 100 else f"{task}: {content}"
    outcome = result[:200] if len(result) > 200 else result
    await auto_trigger_reflection(
        task_type=task,
        task_description=task_desc,
        outcome=outcome,
        success=success,
        project_path=project_path,
    )

    return inject_ace_reminder(result, project_path)


# =============================================================================
# Session Management Implementations
# =============================================================================


async def session_compact_impl(session_id: str, force: bool = False) -> str:
    """Implementation of the session_compact tool."""
    from ..session_manager import get_session_manager

    sm = get_session_manager()
    result = await sm.compact_session(session_id, force=force)
    return json.dumps(result, indent=2)


async def session_stats_impl(session_id: str) -> str:
    """Implementation of the session_stats tool."""
    from ..session_manager import get_session_manager

    sm = get_session_manager()
    stats = sm.get_compaction_stats(session_id)
    if stats is None:
        return f"Error: Session {session_id} not found."
    return json.dumps(stats, indent=2)


async def session_list_impl(client_id: str | None = None) -> str:
    """Implementation of the session_list tool."""
    from ..session_manager import get_session_manager

    sm = get_session_manager()
    sessions = sm.list_sessions(client_id=client_id)
    return json.dumps(sessions, indent=2)


async def session_delete_impl(session_id: str) -> str:
    """Implementation of the session_delete tool."""
    from ..session_manager import get_session_manager

    sm = get_session_manager()
    success = sm.delete_session(session_id)
    return "Session deleted." if success else f"Error: Session {session_id} not found."


# =============================================================================
# Chain/Workflow/Agent Implementations
# =============================================================================


async def chain_impl(
    steps: str, session_id: str | None = None, continue_on_error: bool = False
) -> str:
    """Implementation of the chain tool."""
    # ACE Framework Enforcement
    project_path = str(Path.cwd())
    ace_tracker = get_ace_tracker()
    ace_tracker.record_task_start(project_path, "chain")

    from ..task_chain import parse_chain_steps, execute_chain
    from ..delegation import _get_delegate_context

    steps_list = parse_chain_steps(steps)
    ctx = _get_delegate_context()
    result = await execute_chain(steps_list, ctx, session_id, continue_on_error)
    return inject_ace_reminder(json.dumps(result.to_dict(), indent=2), project_path)


async def workflow_impl(
    definition: str, session_id: str | None = None, max_retries: int = 1
) -> str:
    """Implementation of the workflow tool."""
    # ACE Framework Enforcement
    project_path = str(Path.cwd())
    ace_tracker = get_ace_tracker()
    ace_tracker.record_task_start(project_path, "workflow")

    from ..task_workflow import parse_workflow_definition, execute_workflow
    from ..delegation import _get_delegate_context

    wf = parse_workflow_definition(definition)
    ctx = _get_delegate_context()
    result = await execute_workflow(wf, ctx, session_id, max_retries)
    return inject_ace_reminder(json.dumps(result.to_dict(), indent=2), project_path)


async def agent_impl(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    max_iterations: int = 10,
    tools: str | None = None,
    backend_type: str | None = None,
    workspace: str | None = None,
) -> str:
    """Implementation of the agent tool."""
    # ACE Framework Enforcement
    project_path = str(Path.cwd())
    ace_tracker = get_ace_tracker()
    ace_tracker.record_task_start(project_path, "agent")

    from .agent import run_agent_loop, AgentConfig
    from .builtins import get_default_tools
    from ..llm import call_llm

    config = AgentConfig(max_iterations=max_iterations)
    registry = get_default_tools()

    # Custom model selection if not provided
    if not model:
        from ..routing import select_model

        model = await select_model("agentic", len(prompt))

    result = await run_agent_loop(
        call_llm=call_llm,
        prompt=prompt,
        system_prompt=system_prompt,
        registry=registry,
        model=model,
        config=config,
    )

    return inject_ace_reminder(result.response, project_path)
