# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
High-level MCP tool handlers for Delia.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
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


# =============================================================================
# ACE Framework Dynamic Enforcement
# =============================================================================

class ACEEnforcementTracker:
    """Track ACE compliance dynamically across tool calls."""
    
    def __init__(self):
        self._playbook_queries: dict[str, float] = {}  # path -> last query timestamp
        self._pending_tasks: dict[str, dict] = {}  # path -> {task, start_time}
        self._compliance_warnings: int = 0
    
    def record_playbook_query(self, path: str):
        """Record that playbooks were queried for a project."""
        self._playbook_queries[path] = time.time()
        log.debug("ace_playbook_queried", path=path)
    
    def record_task_start(self, path: str, task_type: str):
        """Record that a coding task started."""
        self._pending_tasks[path] = {
            "task_type": task_type,
            "start_time": time.time(),
            "playbook_queried": path in self._playbook_queries and 
                               (time.time() - self._playbook_queries.get(path, 0)) < 300  # 5 min window
        }
    
    def check_compliance(self, path: str) -> dict | None:
        """Check if ACE workflow was followed. Returns warning if not."""
        pending = self._pending_tasks.get(path)
        if not pending:
            return None
        
        if not pending.get("playbook_queried"):
            self._compliance_warnings += 1
            return {
                "warning": "ACE_PLAYBOOK_NOT_QUERIED",
                "message": f"Started {pending['task_type']} task without querying playbooks first.",
                "action_required": f"Call get_playbook(task_type='{pending['task_type']}', path='{path}') before coding.",
                "total_warnings": self._compliance_warnings,
            }
        return None
    
    def get_dynamic_reminder(self, path: str, response: str) -> str | None:
        """Generate dynamic reminder to inject into response."""
        warning = self.check_compliance(path)
        if warning:
            return (
                f"\n\n⚠️ **ACE Framework Reminder**: {warning['message']}\n"
                f"Action: {warning['action_required']}\n"
                f"Then call `confirm_ace_compliance()` after completing the task."
            )
        
        # Check if task completed without confirmation
        pending = self._pending_tasks.get(path)
        if pending and (time.time() - pending.get("start_time", 0)) > 60:  # Task > 1 min
            return (
                "\n\n📋 **ACE Reminder**: Don't forget to close the learning loop!\n"
                "Call `confirm_ace_compliance(task_description='...', bullets_applied='...', ...)` "
                "then `report_feedback()` for helpful bullets."
            )
        return None
    
    def clear_task(self, path: str):
        """Clear pending task after confirmation."""
        self._pending_tasks.pop(path, None)
    
    def was_playbook_queried(self, path: str) -> bool:
        """Check if playbook was queried recently for this project."""
        if path not in self._playbook_queries:
            return False
        # 5 minute window
        return (time.time() - self._playbook_queries[path]) < 300


# Global tracker instance
_ace_tracker = ACEEnforcementTracker()

def get_ace_tracker() -> ACEEnforcementTracker:
    return _ace_tracker


def inject_ace_reminder(response: str, project_path: str) -> str:
    """Inject ACE Framework reminder into tool response if needed.
    
    This ensures agents are consistently reminded to follow the ACE workflow.
    """
    tracker = get_ace_tracker()
    reminder = tracker.get_dynamic_reminder(project_path, response)
    if reminder:
        return response + reminder
    return response


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
    if context: thinking_prompt += f"\n## Context\n{context}\n"
    thinking_prompt += "\n## Instructions\n1. Break down the problem into components\n2. Consider different approaches\n3. Reason through each step\n4. Conclusion\n\nThink deeply before answering."

    service = get_orchestration_service()
    result = await service.process(message=thinking_prompt, session_id=session_id, model_override=model_hint)

    container.stats_service.increment_task("think")
    from ..mcp_server import save_all_stats_async
    await save_all_stats_async()

    return inject_ace_reminder(result.result.response, project_path)


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
    result = f"# Batch Results\n\n{chr(10).join(results)}\n\n---\n_Total tasks: {len(task_list)} | Total time: {elapsed_ms}ms_"
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
    start_prewarm_task()
    
    # ACE Framework Enforcement: Record task start
    project_path = str(Path(file).parent) if file else str(Path.cwd())
    ace_tracker = get_ace_tracker()
    ace_tracker.record_task_start(project_path, task)

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
            result = f"{full_response}\n\n---\n_[OK] {task} (streamed) | {tier} tier | {elapsed_ms}ms | {selected_model}_"
            return inject_ace_reminder(result, project_path)
        return inject_ace_reminder(full_response, project_path)

    result = await _delegate_impl(
        task, content, file, model, language, context, symbols, include_references,
        backend=backend_type, backend_obj=backend_obj, files=files,
        include_metadata=include_metadata, max_tokens=max_tokens, session_id=session_id, auto_context=auto_context,
    )
    return inject_ace_reminder(result, project_path)


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
    # ACE Framework Enforcement
    project_path = str(Path.cwd())
    ace_tracker = get_ace_tracker()
    ace_tracker.record_task_start(project_path, "chain")

    from ..task_chain import parse_chain_steps, execute_chain
    from ..delegation import get_delegate_context
    steps_list = parse_chain_steps(steps)
    ctx = get_delegate_context()
    result = await execute_chain(steps_list, ctx, session_id, continue_on_error)
    return inject_ace_reminder(json.dumps(result.to_dict(), indent=2), project_path)


async def workflow_impl(definition: str, session_id: str | None = None, max_retries: int = 1) -> str:
    """Implementation of the workflow tool."""
    # ACE Framework Enforcement
    project_path = str(Path.cwd())
    ace_tracker = get_ace_tracker()
    ace_tracker.record_task_start(project_path, "workflow")

    from ..task_workflow import parse_workflow_definition, execute_workflow
    from ..delegation import get_delegate_context
    wf = parse_workflow_definition(definition)
    ctx = get_delegate_context()
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

    return inject_ace_reminder(result.response, project_path)



# =============================================================================
# Playbook Tools (ACE Framework)
# =============================================================================

async def get_playbook_impl(task_type: str = "general", limit: int = 15, path: str | None = None) -> str:
    """Get strategic playbook bullets for a task type."""
    from pathlib import Path
    from ..playbook import playbook_manager
    
    # Set project path if provided (ensures project-specific playbooks)
    project_path = path or str(Path.cwd())
    if path:
        playbook_manager.set_project(Path(path))
    
    # Record playbook query for ACE enforcement
    get_ace_tracker().record_playbook_query(project_path)
    
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


async def report_feedback_impl(bullet_id: str, task_type: str, helpful: bool, path: str | None = None) -> str:
    """Report whether a playbook bullet was helpful for a task."""
    from pathlib import Path
    from ..playbook import playbook_manager
    
    # Set project path if provided
    if path:
        playbook_manager.set_project(Path(path))
    
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


async def get_project_context_impl(path: str | None = None) -> str:
    """Get high-level project understanding from playbook and summaries."""
    from pathlib import Path
    from ..playbook import playbook_manager
    from ..orchestration.summarizer import get_summarizer
    from ..mcp_server import set_project_context
    
    # Set project path for both playbooks and MCP context
    if path:
        playbook_manager.set_project(Path(path))
        set_project_context(path)
    
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
        
        Offloads work to configured LLM backends. Only use when user explicitly
        requests delegation ("delegate", "offload", "use local model").
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


    # =========================================================================
    # Playbook Tools (ACE Framework)
    # =========================================================================

    @mcp.tool()
    async def get_playbook(
        task_type: str = "general",
        limit: int = 15,
        path: str | None = None,
    ) -> str:
        """
        Get strategic playbook bullets for a task type.

        Returns learned lessons, strategies, and project-specific guidance
        that have proven helpful for similar tasks.

        Args:
            task_type: Type of task. Options:
                - coding: Code patterns, style, implementation
                - testing: Test frameworks, coverage patterns
                - architecture: Design decisions, ADRs
                - debugging: Bug investigation patterns
                - project: Tech stack, conventions
                - git: Branching, commits, PR guidelines
                - security: Auth, validation, secrets
                - deployment: CI/CD, environments
                - api: REST/GraphQL patterns
                - performance: Optimization, caching
            limit: Maximum number of bullets to return (default 5)
            path: Project path to load playbooks from (defaults to current directory)

        Returns:
            JSON with bullet IDs, content, utility scores, and usage counts
        """
        return await get_playbook_impl(task_type, limit, path)

    @mcp.tool()
    async def report_feedback(
        bullet_id: str,
        task_type: str,
        helpful: bool,
        path: str | None = None,
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
            path: Project path for the playbook (defaults to current directory)
        
        Returns:
            Confirmation of feedback recording
        """
        return await report_feedback_impl(bullet_id, task_type, helpful, path)

    @mcp.tool()
    async def get_project_context(path: str | None = None) -> str:
        """
        Get high-level project understanding from playbooks and summaries.
        
        Returns project-specific bullets about tech stack, patterns,
        conventions, and key directories. Use this at the start of a
        session to understand the codebase you're working with.
        
        Args:
            path: Project path to load context from (defaults to current directory)
        
        Returns:
            JSON with project bullets, overview, and playbook statistics
        """
        return await get_project_context_impl(path)

    @mcp.tool()
    async def set_project(path: str) -> str:
        """
        Set the active project context for dynamic instruction generation.
        
        Call this when switching between projects to ensure playbooks and
        ACE guidance are loaded from the correct project directory.
        
        Args:
            path: Absolute or relative path to the project directory
        
        Returns:
            Confirmation with detected AI agents and loaded playbooks
        """
        from pathlib import Path
        from ..playbook import playbook_manager
        from ..mcp_server import set_project_context
        from ..agent_sync import detect_ai_agents
        
        project_path = Path(path).resolve()
        if not project_path.exists():
            return json.dumps({"error": f"Project path does not exist: {path}"})
        
        # Update both playbook manager and MCP context
        playbook_manager.set_project(project_path)
        set_project_context(str(project_path))
        
        # Get project info
        agents = detect_ai_agents(project_path)
        detected = [info["description"] for aid, info in agents.items() if info.get("exists")]
        
        stats = playbook_manager.get_stats()
        
        return json.dumps({
            "status": "project_context_set",
            "path": str(project_path),
            "detected_agents": detected,
            "playbook_stats": stats,
        }, indent=2)

    @mcp.tool()
    async def auto_context(
        message: str,
        path: str | None = None,
    ) -> str:
        """
        Automatically detect context and load relevant playbooks from user message.

        Call this ONCE at the start of processing a user request. It analyzes the
        message, detects the task type (coding/testing/debugging/etc), and returns
        all relevant playbook bullets and profile recommendations.

        This replaces the need to manually call get_playbook with a task_type.

        Args:
            message: The user's message or request to analyze
            path: Optional project path. If not provided, uses current directory.

        Returns:
            JSON with detected context, relevant bullets, and profile recommendations

        Example:
            auto_context(message="Fix the bug in the login handler")
            -> Returns debugging + coding bullets automatically
        """
        from pathlib import Path as PyPath
        from ..context_detector import (
            get_context_manager,
            get_relevant_profiles,
            detect_with_learning,
        )
        from ..playbook import get_playbook_manager

        # Set project path first
        project_path = PyPath(path).resolve() if path else PyPath.cwd()

        # Detect context using learning-enhanced detection
        context = detect_with_learning(message, project_path)

        # Also update the context manager for backwards compatibility
        ctx_mgr = get_context_manager()
        ctx_mgr._current_context = context
        ctx_mgr._last_message = message

        pm = get_playbook_manager()
        pm.set_project(project_path)

        # Collect bullets for all detected tasks
        all_bullets = []
        for task_type in context.all_tasks():
            bullets = pm.get_top_bullets(task_type, limit=5 if task_type == context.primary_task else 3)
            for b in bullets:
                all_bullets.append({
                    "id": b.id,
                    "task_type": task_type,
                    "content": b.content,
                    "utility_score": b.utility_score,
                    "is_primary": task_type == context.primary_task,
                })

        # Get relevant profiles
        profiles = get_relevant_profiles(context)

        # Build response
        result = {
            "detected_context": {
                "primary_task": context.primary_task,
                "secondary_tasks": context.secondary_tasks,
                "confidence": context.confidence,
                "matched_keywords": context.matched_keywords[:5],
            },
            "bullets": all_bullets,
            "profiles_to_load": profiles,
            "instructions": (
                f"Detected task type: {context.primary_task}. "
                f"Apply the {len(all_bullets)} bullets above to your work. "
                f"After completing, call report_feedback() for each bullet_id."
            ),
        }

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def record_detection_feedback(
        message: str,
        detected_task: str,
        correct_task: str,
        path: str | None = None,
    ) -> str:
        """
        Record feedback on context detection to improve future accuracy.

        Call this when the auto_context detection was incorrect. This teaches
        the system to recognize similar patterns in the future.

        Args:
            message: The original message that was analyzed
            detected_task: What auto_context detected (e.g., "coding")
            correct_task: What should have been detected (e.g., "debugging")
            path: Optional project path

        Returns:
            JSON with learning results

        Example:
            record_detection_feedback(
                message="the server is down",
                detected_task="project",
                correct_task="debugging"
            )
            -> Learns to associate "server is down" with debugging
        """
        from pathlib import Path as PyPath
        from ..context_detector import get_pattern_learner, TaskType

        project_path = PyPath(path).resolve() if path else PyPath.cwd()
        learner = get_pattern_learner(project_path)

        was_correct = detected_task == correct_task
        result = learner.record_feedback(
            message=message,
            detected_task=detected_task,  # type: ignore
            correct_task=correct_task,  # type: ignore
            was_correct=was_correct,
        )

        return json.dumps({
            "success": True,
            "was_correct": was_correct,
            "patterns_updated": result["patterns_updated"],
            "patterns_added": result["patterns_added"],
            "message": (
                "Detection was correct, patterns reinforced."
                if was_correct else
                f"Learning from feedback: {result['patterns_added']} new patterns added."
            ),
        }, indent=2)

    @mcp.tool()
    async def get_learning_stats(
        path: str | None = None,
    ) -> str:
        """
        Get statistics about learned detection patterns.

        Returns information about patterns learned from feedback,
        their effectiveness, and suggestions for pruning.

        Args:
            path: Optional project path

        Returns:
            JSON with pattern learning statistics
        """
        from pathlib import Path as PyPath
        from ..context_detector import get_pattern_learner

        project_path = PyPath(path).resolve() if path else PyPath.cwd()
        learner = get_pattern_learner(project_path)

        stats = learner.get_stats()

        return json.dumps({
            "project": str(project_path),
            "learned_patterns": stats,
            "suggestions": {
                "prune_command": "Use prune_learned_patterns() to remove ineffective patterns",
                "effectiveness_threshold": 0.4,
            },
        }, indent=2)

    @mcp.tool()
    async def prune_learned_patterns(
        min_effectiveness: float = 0.3,
        min_uses: int = 5,
        path: str | None = None,
    ) -> str:
        """
        Remove learned patterns that are consistently wrong.

        Prunes patterns with low effectiveness after sufficient usage.

        Args:
            min_effectiveness: Minimum effectiveness to keep (0-1, default 0.3)
            min_uses: Minimum uses before pruning (default 5)
            path: Optional project path

        Returns:
            JSON with pruning results
        """
        from pathlib import Path as PyPath
        from ..context_detector import get_pattern_learner

        project_path = PyPath(path).resolve() if path else PyPath.cwd()
        learner = get_pattern_learner(project_path)

        removed = learner.prune_ineffective(min_effectiveness, min_uses)

        return json.dumps({
            "success": True,
            "patterns_removed": removed,
            "message": f"Pruned {removed} ineffective patterns." if removed > 0 else "No patterns needed pruning.",
        }, indent=2)


