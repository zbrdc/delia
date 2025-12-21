# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
High-level MCP tool handlers for Delia.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

import structlog
from fastmcp import FastMCP

from ..container import get_container
from ..orchestration.result import OrchestrationMode, ModelRole, DetectedIntent
from ..orchestration.executor import get_orchestration_executor
from ..orchestration.service import get_orchestration_service
from ..text_utils import strip_thinking_tags
from .. import llm
from ..routing import get_router, select_model
from ..language import detect_language
from ..validation import validate_content, validate_file_path, validate_task
from ..config import get_affinity_tracker, get_backend_health, get_prewarm_tracker
from ..messages import StatusEvent, get_display_event, get_status_message
from ..voting import VotingConsensus
from ..voting_stats import get_voting_stats_tracker
from ..quality import ResponseQualityValidator
from ..types import Workspace
from ..delegation import (
    start_prewarm_task,
    _delegate_with_voting,
    _delegate_with_tot,
    _get_delegate_context,
    get_delegate_signals,
    prepare_delegate_content,
    determine_task_type,
    _select_delegate_model_impl,
    _delegate_impl
)

log = structlog.get_logger()


async def think_impl(
    problem: str,
    context: str = "",
    depth: str = "normal",
    session_id: str | None = None,
) -> str:
    """Implementation of the think tool."""
    container = get_container()
    if depth == "quick":
        model_hint = "quick"
    elif depth == "deep":
        model_hint = "thinking"
    else:
        model_hint = "thinking"

    thinking_prompt = f"Think through this problem step by step:\n\n## Problem\n{problem}\n"
    if context: thinking_prompt += f"\n## Context\n{context}\n"
    thinking_prompt += "\n## Instructions\n1. Break down the problem into components\n2. Consider different approaches\n3. Reason through each step\n4. Conclusion\n\nThink deeply before answering."

    service = get_orchestration_service()
    result = await service.process(message=thinking_prompt, session_id=session_id, model_override=model_hint)

    container.stats_service.increment_task("think")
    from ..mcp_server import save_all_stats_async
    await save_all_stats_async()

    return result.result.response


async def batch_impl(
    tasks: str,
    include_metadata: bool = True,
    max_tokens: int | None = None,
    session_id: str | None = None,
) -> str:
    """Implementation of the batch tool."""
    container = get_container()
    start_time = time.time()
    try:
        task_list = json.loads(tasks)
    except json.JSONDecodeError as e: return f"Error: Invalid JSON - {e}"
    if not isinstance(task_list, list): return "Error: tasks must be a JSON array"
    if len(task_list) == 0: return "Error: tasks array is empty"

    available = await container.backend_manager.check_all_health()
    backend_assignments = container.model_router.assign_backends_to_tasks(task_list, available, container.backend_manager)

    from ..mcp_server import current_client_id, current_username
    from ..delegation import _delegate_impl

    captured_client_id = current_client_id.get()
    captured_username = current_username.get()

    async def run_task(i: int, t: dict, backend_id: str, client_id: str | None, username: str | None) -> str:
        current_client_id.set(client_id)
        current_username.set(username)
        task_type = t.get("task", "analyze")
        result = await _delegate_impl(
            task=task_type, content=t.get("content", ""), file=t.get("file"),
            model=t.get("model"), language=t.get("language"), context=t.get("context"),
            symbols=t.get("symbols"), include_references=t.get("include_references", False),
            backend=backend_id, files=t.get("files"),
            include_metadata=t.get("include_metadata", include_metadata),
            max_tokens=t.get("max_tokens", max_tokens), session_id=session_id,
        )
        return f"### Task {i + 1}: {task_type}\n\n{result}"

    results = await asyncio.gather(*[run_task(i, t, backend_assignments[i], captured_client_id, captured_username) for i, t in enumerate(task_list)])
    elapsed_ms = int((time.time() - start_time) * 1000)
    return f"# Batch Results\n\n{chr(10).join(results)}\n\n---\n_Total tasks: {len(task_list)} | Total time: {elapsed_ms}ms_"


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
    start_prewarm_task()

    if reliable:
        return await _delegate_with_voting(
            task=task, content=content, file=file, model=model, language=language,
            context=context, symbols=symbols, include_references=include_references,
            backend_type=backend_type, files=files, include_metadata=include_metadata,
            max_tokens=max_tokens, session_id=session_id, voting_k=voting_k,
        )

    if tot:
        return await _delegate_with_tot(
            task=task, content=content, file=file, model=model, language=language,
            context=context, symbols=symbols, include_references=include_references,
            backend_type=backend_type, files=files, include_metadata=include_metadata,
            max_tokens=max_tokens, session_id=session_id,
        )

    if dry_run:
        ctx = _get_delegate_context()
        signals = await get_delegate_signals(
            ctx, task, content, file, model, language, context, symbols, include_references, files,
        )
        return json.dumps(signals, indent=2)

    _, backend_obj = await get_router().select_optimal_backend(content, file, task, backend_type)

    if stream and backend_obj:
        ctx = _get_delegate_context()
        prepared_content = await prepare_delegate_content(
            content, context, symbols, include_references, files, auto_context=auto_context,
        )
        task_type = determine_task_type(task)
        context_header = f"Task: {task_type}\n"
        if file: context_header += f"File: {file}\n"
        if language: context_header += f"Language: {language}\n"
        if symbols: context_header += f"Symbols: {symbols}\n"
        if context: context_header += f"Context Files: {context}\n"
        prepared_content = f"{context_header}\n{prepared_content}"

        detected_language = language or detect_language(prepared_content, file or "")
        from ..prompts import build_system_prompt
        role_map = {"review": ModelRole.CODE_REVIEWER, "generate": ModelRole.CODE_GENERATOR, "analyze": ModelRole.ANALYST, "summarize": ModelRole.SUMMARIZER, "plan": ModelRole.ARCHITECT, "critique": ModelRole.ANALYST}
        role = role_map.get(task_type, ModelRole.ASSISTANT)
        system = build_system_prompt(role=role)
        if detected_language: system += f"\n\nPrimary language: {detected_language}"

        selected_model, tier, _source = await _select_delegate_model_impl(
            ctx, task_type, prepared_content, model, None, backend_obj
        )

        full_response = ""
        total_tokens = 0
        elapsed_ms = 0
        start_time = time.time()

        async for chunk in llm.call_llm_stream(
            model=selected_model, prompt=prepared_content, system=system, task_type=task_type,
            original_task=task, language=detected_language, backend_obj=backend_obj, max_tokens=max_tokens,
        ):
            if chunk.text: full_response += chunk.text
            if chunk.done:
                total_tokens = chunk.tokens
                if chunk.metadata: elapsed_ms = chunk.metadata.get("elapsed_ms", 0)
                if chunk.error: return f"Error: {chunk.error}"

        if elapsed_ms == 0: elapsed_ms = int((time.time() - start_time) * 1000)
        full_response = strip_thinking_tags(full_response)

        if include_metadata:
            return f"{full_response}\n\n---\n_[OK] {task} (streamed) | {tier} tier | {elapsed_ms}ms | {selected_model}_"
        return full_response

    return await _delegate_impl(
        task, content, file, model, language, context, symbols, include_references,
        backend=backend_type, backend_obj=backend_obj, files=files,
        include_metadata=include_metadata, max_tokens=max_tokens, session_id=session_id, auto_context=auto_context,
    )


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


async def chain_impl(steps: str, session_id: str | None = None, continue_on_error: bool = False) -> str:
    """Implementation of the chain tool."""
    from ..task_chain import parse_chain_steps, execute_chain
    from ..delegation import get_delegate_context
    steps_list = parse_chain_steps(steps)
    ctx = get_delegate_context()
    result = await execute_chain(steps_list, ctx, session_id, continue_on_error)
    return json.dumps(result.to_dict(), indent=2)


async def workflow_impl(definition: str, session_id: str | None = None, max_retries: int = 1) -> str:
    """Implementation of the workflow tool."""
    from ..task_workflow import parse_workflow_definition, execute_workflow
    from ..delegation import get_delegate_context
    wf = parse_workflow_definition(definition)
    ctx = get_delegate_context()
    result = await execute_workflow(wf, ctx, session_id, max_retries)
    return json.dumps(result.to_dict(), indent=2)


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
    from .agent import run_agent_loop, AgentConfig
    from .registry import get_default_registry
    from ..llm import call_llm
    
    config = AgentConfig(max_iterations=max_iterations)
    registry = get_default_registry()
    
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
    
    return result.response



# =============================================================================
# Playbook Tools (ACE Framework)
# =============================================================================

async def get_playbook_impl(task_type: str = "general", limit: int = 5) -> str:
    """Get strategic playbook bullets for a task type."""
    from ..playbook import playbook_manager
    
    bullets = playbook_manager.get_top_bullets(task_type, limit)
    if not bullets:
        # Try project-level playbook if task-specific is empty
        bullets = playbook_manager.get_top_bullets("project", limit)
    
    if not bullets:
        return json.dumps({
            "status": "empty",
            "message": f"No playbook bullets found for '{task_type}'. Run 'delia index --summarize' to generate.",
            "bullets": []
        })
    
    result = {
        "task_type": task_type,
        "bullet_count": len(bullets),
        "bullets": [
            {
                "id": b.id,
                "content": b.content,
                "section": b.section,
                "utility_score": round(b.utility_score, 3),
                "helpful_count": b.helpful_count,
                "harmful_count": b.harmful_count,
            }
            for b in bullets
        ]
    }
    return json.dumps(result, indent=2)


async def report_feedback_impl(bullet_id: str, task_type: str, helpful: bool) -> str:
    """Report whether a playbook bullet was helpful for a task."""
    from ..playbook import playbook_manager
    
    success = playbook_manager.record_feedback(bullet_id, task_type, helpful)
    
    if success:
        log.info("playbook_feedback_recorded", bullet_id=bullet_id, task_type=task_type, helpful=helpful)
        return json.dumps({
            "status": "recorded",
            "bullet_id": bullet_id,
            "helpful": helpful,
            "message": f"Feedback recorded. Bullet {'helped' if helpful else 'did not help'} with task."
        })
    else:
        return json.dumps({
            "status": "not_found",
            "bullet_id": bullet_id,
            "message": f"Bullet '{bullet_id}' not found in playbook '{task_type}'."
        })


async def get_project_context_impl() -> str:
    """Get high-level project understanding from playbook and summaries."""
    from ..playbook import playbook_manager
    from ..orchestration.summarizer import get_summarizer
    
    # Get project playbook bullets
    bullets = playbook_manager.get_top_bullets("project", limit=10)
    
    # Get project overview from summarizer
    summarizer = get_summarizer()
    overview = summarizer.get_project_overview() if hasattr(summarizer, 'get_project_overview') else None
    
    result = {
        "playbook_bullets": [
            {"id": b.id, "content": b.content, "section": b.section}
            for b in bullets
        ],
        "project_overview": overview,
        "playbook_stats": playbook_manager.get_stats(),
    }
    return json.dumps(result, indent=2)


async def playbook_stats_impl(task_type: str | None = None) -> str:
    """Get playbook statistics and utility scores."""
    from ..playbook import playbook_manager
    
    if task_type:
        bullets = playbook_manager.load_playbook(task_type)
        return json.dumps({
            "task_type": task_type,
            "bullet_count": len(bullets),
            "bullets": [
                {
                    "id": b.id,
                    "content": b.content[:100] + "..." if len(b.content) > 100 else b.content,
                    "utility_score": round(b.utility_score, 3),
                    "helpful": b.helpful_count,
                    "harmful": b.harmful_count,
                }
                for b in bullets
            ]
        }, indent=2)
    else:
        return json.dumps(playbook_manager.get_stats(), indent=2)


def register_tool_handlers(mcp: FastMCP):
    """Register all high-level tool handlers with FastMCP."""
    
    container = get_container()

    @mcp.tool()
    async def delegate(
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
        """
        Execute a task with intelligent 3-tier model selection.
        """
        return await delegate_tool_impl(
            task, content, file, model, language, context, symbols, include_references,
            backend_type, files, include_metadata, max_tokens, dry_run, session_id,
            stream, reliable, voting_k, tot, auto_context
        )

    @mcp.tool()
    async def think(
        problem: str,
        context: str = "",
        depth: str = "normal",
        session_id: str | None = None,
    ) -> str:
        """
        Deep reasoning for complex problems with extended thinking.
        """
        return await think_impl(problem, context, depth, session_id)

    @mcp.tool()
    async def batch(
        tasks: str,
        include_metadata: bool = True,
        max_tokens: int | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Execute multiple tasks in PARALLEL across all available backends.
        """
        return await batch_impl(tasks, include_metadata, max_tokens, session_id)

    @mcp.tool()
    async def chain(
        steps: str,
        session_id: str | None = None,
        continue_on_error: bool = False,
    ) -> str:
        """
        Execute a chain of tasks sequentially with output piping.
        """
        return await chain_impl(steps, session_id, continue_on_error)

    @mcp.tool()
    async def workflow(
        definition: str,
        session_id: str | None = None,
        max_retries: int = 1,
    ) -> str:
        """
        Execute a DAG workflow with conditional branching and retry logic.
        """
        return await workflow_impl(definition, session_id, max_retries)

    @mcp.tool()
    async def agent(
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_iterations: int = 10,
        tools: str | None = None,
        backend_type: str | None = None,
        workspace: str | None = None,
    ) -> str:
        """
        Run an autonomous agent that can use tools to complete tasks.
        """
        return await agent_impl(prompt, system_prompt, model, max_iterations, tools, backend_type, workspace)

    @mcp.tool()
    async def session_compact(session_id: str, force: bool = False) -> str:
        """
        Compact a session's conversation history using LLM summarization.
        """
        return await session_compact_impl(session_id, force)

    @mcp.tool()
    async def session_stats(session_id: str) -> str:
        """
        Get compaction statistics for a session.
        """
        return await session_stats_impl(session_id)

    @mcp.tool()
    async def session_list(client_id: str | None = None) -> str:
        """
        List active conversation sessions.
        """
        return await session_list_impl(client_id)

    @mcp.tool()
    async def session_delete(session_id: str) -> str:
        """
        Delete a conversation session and its history.
        """
        return await session_delete_impl(session_id)

    # =========================================================================
    # Playbook Tools (ACE Framework)
    # =========================================================================

    @mcp.tool()
    async def get_playbook(
        task_type: str = "general",
        limit: int = 5,
    ) -> str:
        """
        Get strategic playbook bullets for a task type.
        
        Returns learned lessons, strategies, and project-specific guidance
        that have proven helpful for similar tasks.
        
        Args:
            task_type: Type of task (general, coding, testing, architecture, project)
            limit: Maximum number of bullets to return (default 5)
        
        Returns:
            JSON with bullet IDs, content, utility scores, and usage counts
        """
        return await get_playbook_impl(task_type, limit)

    @mcp.tool()
    async def report_feedback(
        bullet_id: str,
        task_type: str,
        helpful: bool,
    ) -> str:
        """
        Report whether a playbook bullet was helpful for a task.
        
        This feedback updates the bullet's utility score, improving future
        recommendations. Call this after completing a task to close the
        learning loop.
        
        Args:
            bullet_id: The bullet ID (e.g., "strat-a1b2c3d4")
            task_type: The playbook containing this bullet
            helpful: True if the bullet helped, False if it was harmful/irrelevant
        
        Returns:
            Confirmation of feedback recording
        """
        return await report_feedback_impl(bullet_id, task_type, helpful)

    @mcp.tool()
    async def get_project_context() -> str:
        """
        Get high-level project understanding from playbooks and summaries.
        
        Returns project-specific bullets about tech stack, patterns,
        conventions, and key directories. Use this at the start of a
        session to understand the codebase you're working with.
        
        Returns:
            JSON with project bullets, overview, and playbook statistics
        """
        return await get_project_context_impl()

    @mcp.tool()
    async def playbook_stats(task_type: str | None = None) -> str:
        """
        Get playbook statistics and utility scores.
        
        Shows which bullets are most/least effective. Use to identify
        low-utility bullets that may need pruning.
        
        Args:
            task_type: Specific playbook to analyze (None for global stats)
        
        Returns:
            JSON with bullet counts, utility distributions, and recommendations
        """
        return await playbook_stats_impl(task_type)
