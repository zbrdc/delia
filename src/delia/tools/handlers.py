# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
High-level MCP tool handlers for Delia.

This module registers all MCP tools with FastMCP. Implementation functions
are in separate modules:
- handlers_ace.py: ACE enforcement classes and helpers
- handlers_orchestration.py: delegate, think, batch, chain, workflow, agent
- handlers_playbook.py: playbook management tools
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from fastmcp import FastMCP

from ..container import get_container
from ..language import detect_language

# Import from refactored modules
from .handlers_ace import (
    ACE_EXEMPT_TOOLS,
    CHECKPOINT_REQUIRED_TOOLS,
    ACEEnforcementTracker,
    ACEEnforcementManager,
    get_ace_tracker,
    get_ace_manager,
    check_ace_gate,
    check_checkpoint_gate,
    record_checkpoint,
    record_search,
    get_phase_injection,
    inject_ace_reminder,
    auto_trigger_reflection,
)
from .handlers_orchestration import (
    think_impl,
    batch_impl,
    delegate_tool_impl,
    session_compact_impl,
    session_stats_impl,
    session_list_impl,
    session_delete_impl,
    chain_impl,
    workflow_impl,
    agent_impl,
)
from .handlers_playbook import (
    get_playbook_impl,
    report_feedback_impl,
    get_project_context_impl,
    playbook_stats_impl,
)

log = structlog.get_logger()

# Re-export for backwards compatibility
_ace_manager = get_ace_manager()


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
        prior_context: str | None = None,
        working_files: str | None = None,
        code_snippet: str | None = None,
    ) -> str:
        """
        Automatically detect context and load relevant playbooks from message + conversation.

        Call this ONCE at the start of processing a user request. It analyzes the
        message, detects the task type (coding/testing/debugging/etc), and returns
        all relevant playbook bullets and profile recommendations.

        IMPORTANT: Pass prior_context when the conversation has shifted topics.
        For example, if the assistant offered "Would you like me to commit this fix?"
        and the user says "yes", pass that assistant message as prior_context so
        git patterns are detected.

        This replaces the need to manually call get_playbook with a task_type.

        Args:
            message: The user's message or request to analyze
            path: Optional project path. If not provided, uses current directory.
            prior_context: Optional recent assistant response(s) to include in detection.
                          Use this when the user's message is short/ambiguous but the
                          conversation context makes the task type clear.
            working_files: Optional JSON array of file paths being edited (e.g., '["test_foo.py", "src/auth.py"]').
                          File patterns are used to boost detection (test files -> testing, Dockerfile -> deployment).
            code_snippet: Optional code content being edited. Code patterns are analyzed
                         (e.g., @pytest.fixture -> testing, FastAPI decorators -> api).

        Returns:
            JSON with detected context, relevant bullets, and profile recommendations

        Example:
            auto_context(message="yes, do it")
            -> Detects "project" (ambiguous)

            auto_context(message="yes, do it",
                        prior_context="Would you like me to commit this fix to dev?")
            -> Detects "git" (context-aware)

            auto_context(message="fix this", working_files='["test_auth.py"]')
            -> Detects "testing" (file pattern)

            auto_context(message="update this", code_snippet="@pytest.fixture\\ndef client():")
            -> Detects "testing" (code pattern)
        """
        from pathlib import Path as PyPath
        from ..context_detector import (
            get_context_manager,
            get_relevant_profiles,
            detect_with_learning,
            detect_task_type_enhanced,
        )
        from ..playbook import get_playbook_manager

        # Set project path first
        project_path = PyPath(path).resolve() if path else PyPath.cwd()

        # Parse working_files from JSON string
        files_list: list[str] | None = None
        if working_files:
            try:
                files_list = json.loads(working_files)
                if not isinstance(files_list, list):
                    files_list = [str(files_list)]
            except json.JSONDecodeError:
                # Treat as single file path or comma-separated
                files_list = [f.strip() for f in working_files.split(",") if f.strip()]

        # Combine message with prior context for detection
        # Prior context (e.g., assistant's last message) helps when user response is short
        detection_text = message
        if prior_context:
            # Weight prior context less than current message by putting it second
            # The detection will pick up keywords from both
            detection_text = f"{message}\n\n[Prior context]: {prior_context}"

        # Use enhanced detection if files or code provided, otherwise use learning-based
        file_context = None
        detected_language = None
        if files_list or code_snippet:
            enhanced_context = detect_task_type_enhanced(
                detection_text,
                working_files=files_list,
                code_snippet=code_snippet,
            )
            context = enhanced_context
            file_context = enhanced_context.file_context
            detected_language = enhanced_context.detected_language
        else:
            context = detect_with_learning(detection_text, project_path)

        # Also update the context manager for backwards compatibility
        ctx_mgr = get_context_manager()
        ctx_mgr._current_context = context
        ctx_mgr._last_message = message

        pm = get_playbook_manager()
        pm.set_project(project_path)

        # Collect bullets with semantic retrieval if embeddings available
        # Falls back to utility × recency if no embeddings
        all_bullets = []
        try:
            from ..ace.retrieval import get_retriever
            retriever = get_retriever()

            for task_type in context.all_tasks():
                limit = 5 if task_type == context.primary_task else 3
                bullets = pm.load_playbook(task_type)

                # Try semantic retrieval first
                try:
                    scored = await retriever.retrieve(
                        bullets=bullets,
                        query=detection_text,
                        project_path=project_path,
                        limit=limit,
                    )
                except Exception:
                    # Fallback to utility × recency
                    scored = retriever.retrieve_by_utility(bullets, limit=limit)

                for s in scored:
                    all_bullets.append({
                        "id": s.bullet.id,
                        "task_type": task_type,
                        "content": s.bullet.content,
                        "relevance_score": s.relevance_score,
                        "utility_score": s.utility_score,
                        "recency_score": s.recency_score,
                        "final_score": s.final_score,
                        "is_primary": task_type == context.primary_task,
                    })
        except Exception as e:
            # Fallback to simple retrieval
            log.warning("retrieval_fallback", error=str(e))
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

        # Get relevant profiles based on context
        profiles_dir = project_path / ".delia" / "profiles"
        templates_dir = PyPath(__file__).parent.parent / "templates" / "profiles"

        # Determine which profiles are relevant for this context
        relevant_names = set(get_relevant_profiles(context))

        # Add file-detected profiles (language/framework specific)
        if file_context and file_context.profile_hints:
            relevant_names.update(file_context.profile_hints)

        # Discover all available profiles
        all_available = set()
        if profiles_dir.exists():
            all_available.update(p.name for p in profiles_dir.glob("*.md"))

        # Load relevant profiles with full content
        loaded_profiles = []
        for profile_name in sorted(relevant_names):
            content = None

            # Try project-specific profile first
            profile_path = profiles_dir / profile_name
            if profile_path.exists():
                try:
                    content = profile_path.read_text()
                except Exception as e:
                    log.debug("profile_read_failed", profile=profile_name, error=str(e))

            # Fallback to template
            if content is None:
                template_path = templates_dir / profile_name
                if template_path.exists():
                    try:
                        content = template_path.read_text()
                    except Exception as e:
                        log.debug("template_read_failed", profile=profile_name, error=str(e))

            if content:
                loaded_profiles.append({
                    "name": profile_name,
                    "content": content,
                })

        # List other available profiles (not loaded, but agent can request)
        loaded_names = {p["name"] for p in loaded_profiles}
        other_available = sorted(all_available - loaded_names)

        # Collect bullet IDs for easy reference
        bullet_ids = [b["id"] for b in all_bullets]

        # Task-specific tool recommendations
        # Memory tools are useful for ALL tasks - they persist knowledge across sessions
        MEMORY_TOOLS = [
            {"tool": "list_memories", "use": "Check existing project knowledge"},
            {"tool": "read_memory", "use": "Load relevant documented insights"},
            {"tool": "write_memory", "use": "Persist learnings for future sessions"},
        ]

        TASK_TOOLS: dict[str, list[dict[str, str]]] = {
            "coding": [
                {"tool": "lsp_get_symbols", "use": "Understand file structure before editing"},
                {"tool": "lsp_find_references", "use": "Find all usages before refactoring"},
                {"tool": "lsp_goto_definition", "use": "Navigate to function/class definitions"},
            ],
            "debugging": [
                {"tool": "lsp_find_references", "use": "Trace where problematic code is called"},
                {"tool": "lsp_goto_definition", "use": "Jump to source of errors"},
            ],
            "testing": [
                {"tool": "lsp_get_symbols", "use": "See what functions need test coverage"},
                {"tool": "lsp_find_references", "use": "Find existing test patterns"},
            ],
            "architecture": [
                {"tool": "lsp_get_symbols", "use": "Survey module structure"},
            ],
            "git": [],
            "project": [
                {"tool": "lsp_find_symbol", "use": "Find classes/functions by name"},
            ],
            "security": [
                {"tool": "lsp_find_references", "use": "Trace sensitive data flow"},
            ],
            "api": [
                {"tool": "lsp_get_symbols", "use": "See endpoint structure"},
                {"tool": "lsp_find_references", "use": "Find endpoint usages"},
            ],
            "deployment": [],
            "performance": [
                {"tool": "lsp_find_references", "use": "Find hot path callers"},
            ],
        }

        # Get recommended tools for detected task
        # Start with task-specific tools
        recommended_tools = list(TASK_TOOLS.get(context.primary_task, []))
        # Add tools from secondary tasks
        for secondary in context.secondary_tasks[:2]:
            for tool in TASK_TOOLS.get(secondary, []):
                if tool not in recommended_tools:
                    recommended_tools.append(tool)
        # Always include memory tools - they're useful for ALL tasks
        recommended_tools.extend(MEMORY_TOOLS)

        # Register with ACE enforcement tracker
        tracker = get_ace_tracker()
        tracker.record_ace_started(str(project_path))

        # Build response
        result = {
            "detected_context": {
                "primary_task": context.primary_task,
                "secondary_tasks": context.secondary_tasks,
                "confidence": context.confidence,
                "matched_keywords": context.matched_keywords[:10],
                "detected_language": detected_language,
                "detected_frameworks": file_context.frameworks if file_context else None,
            },
            "bullets": all_bullets,
            "bullet_ids": bullet_ids,  # Easy reference for complete_task()
            "profiles": loaded_profiles,
            "other_available_profiles": other_available,
            "recommended_tools": recommended_tools,
            "ace_workflow": {
                "current_step": "APPLY",
                "next_step": "complete_task()",
                "instruction": (
                    f"1. APPLY the {len(loaded_profiles)} profiles and {len(all_bullets)} bullets below to your work. "
                    f"2. Track which bullet IDs you actually use. "
                    f"3. When done, call complete_task(success=True/False, bullets_applied='{json.dumps(bullet_ids[:3])}...')"
                ),
            },
            "instructions": (
                f"Task: {context.primary_task}. "
                + (f"Language: {detected_language}. " if detected_language else "")
                + (f"Other profiles via get_profile(): {', '.join(other_available[:5])}. " if other_available else "")
                + f"WORKFLOW: Apply bullets → complete_task(success, bullets_applied)"
            ),
        }

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def get_profile(
        name: str,
        path: str | None = None,
    ) -> str:
        """
        Load a specific profile by name.

        Use this when auto_context indicates additional profiles are available
        and you need their guidance for your current task.

        Args:
            name: Profile filename (e.g., "deployment.md", "security.md")
            path: Optional project path. If not provided, uses current directory.

        Returns:
            Profile content or error message

        Example:
            get_profile(name="deployment.md")
            get_profile(name="api.md", path="/path/to/project")
        """
        from pathlib import Path as PyPath
        from ..playbook import get_playbook_manager

        # Use current project context if no path provided
        if path:
            project_path = PyPath(path).resolve()
        else:
            pm = get_playbook_manager()
            project_path = pm.playbook_dir.parent.parent

        profiles_dir = project_path / ".delia" / "profiles"
        templates_dir = PyPath(__file__).parent.parent / "templates" / "profiles"

        # Normalize name
        if not name.endswith(".md"):
            name = f"{name}.md"

        content = None

        # Try project-specific profile first
        profile_path = profiles_dir / name
        if profile_path.exists():
            try:
                content = profile_path.read_text()
                source = "project"
            except Exception as e:
                return json.dumps({"error": f"Failed to read profile: {e}"})

        # Fallback to template
        if content is None:
            template_path = templates_dir / name
            if template_path.exists():
                try:
                    content = template_path.read_text()
                    source = "template"
                except Exception as e:
                    return json.dumps({"error": f"Failed to read template: {e}"})

        if content is None:
            # List available profiles
            available = []
            if profiles_dir.exists():
                available.extend(p.name for p in profiles_dir.glob("*.md"))
            return json.dumps({
                "error": f"Profile '{name}' not found",
                "available_profiles": sorted(available),
            }, indent=2)

        return json.dumps({
            "name": name,
            "source": source,
            "content": content,
        }, indent=2)

    @mcp.tool()
    async def complete_task(
        success: bool,
        bullets_applied: str,
        task_summary: str | None = None,
        failure_reason: str | None = None,
        new_insight: str | None = None,
        path: str | None = None,
    ) -> str:
        """
        Complete an ACE task and update bullet feedback in one call.

        **CRITICAL**: Call this when you finish a task to close the ACE learning loop.
        This replaces calling report_feedback() multiple times.

        The ACE workflow:
        1. auto_context() → get bullets + profiles
        2. Apply bullets to your work
        3. complete_task() → report success/failure + which bullets you used

        Args:
            success: Whether the task completed successfully
            bullets_applied: JSON array of bullet IDs that were applied (e.g., '["strat-abc123", "strat-def456"]')
            task_summary: Brief description of what was accomplished
            failure_reason: If success=False, why did it fail?
            new_insight: If you learned something new, describe it for future playbook addition
            path: Optional project path

        Returns:
            JSON with feedback recorded and any new bullets added

        Example:
            complete_task(
                success=True,
                bullets_applied='["strat-a1b2c3", "strat-d4e5f6"]',
                task_summary="Implemented JWT authentication with refresh tokens"
            )

            complete_task(
                success=False,
                bullets_applied='["strat-a1b2c3"]',
                failure_reason="Library version conflict with existing deps",
                new_insight="Check dependency compatibility before recommending libraries"
            )
        """
        from pathlib import Path as PyPath
        from ..playbook import get_playbook_manager

        pm = get_playbook_manager()
        # Only set project if path explicitly provided - otherwise use current context
        if path:
            project_path = PyPath(path).resolve()
            pm.set_project(project_path)
        else:
            # Use the project already set by auto_context/set_project
            project_path = pm.playbook_dir.parent.parent  # .delia/playbooks -> .delia -> project

        # Parse bullets_applied
        try:
            bullet_ids = json.loads(bullets_applied) if bullets_applied else []
            if not isinstance(bullet_ids, list):
                bullet_ids = [bullet_ids]
        except json.JSONDecodeError:
            # Try comma-separated
            bullet_ids = [b.strip() for b in bullets_applied.split(",") if b.strip()]

        # Record feedback for each bullet
        feedback_recorded = []
        for bullet_id in bullet_ids:
            # Find which task_type this bullet belongs to
            for task_type in ["coding", "testing", "debugging", "architecture", "git",
                             "security", "api", "performance", "deployment", "project"]:
                bullets = pm.load_playbook(task_type)
                if any(b.id == bullet_id for b in bullets):
                    recorded = pm.record_feedback(bullet_id, task_type, helpful=success)
                    if recorded:
                        feedback_recorded.append({
                            "bullet_id": bullet_id,
                            "task_type": task_type,
                            "marked": "helpful" if success else "harmful",
                        })
                    break

        result = {
            "status": "completed",
            "success": success,
            "feedback_recorded": feedback_recorded,
            "bullets_updated": len(feedback_recorded),
        }

        # Determine task type from the bullets or default to coding
        task_type = "coding"
        if feedback_recorded:
            task_type = feedback_recorded[0]["task_type"]

        # =====================================================================
        # ACE LEARNING LOOP: Reflector → Curator pipeline
        # =====================================================================
        # Learn from BOTH success AND failure to maximize ACE improvements.
        # Success: Learn what worked well (patterns to reinforce)
        # Failure: Learn what went wrong (anti-patterns to avoid)
        # =====================================================================
        if success or (not success and failure_reason):
            try:
                from ..ace.reflector import get_reflector
                from ..ace.curator import get_curator

                reflector = get_reflector()
                curator = get_curator(str(project_path))

                # Determine outcome message
                if success:
                    outcome_msg = task_summary or "Task completed successfully"
                else:
                    outcome_msg = failure_reason

                # Reflect on the task execution to extract insights
                reflection = await reflector.reflect(
                    task_description=task_summary or "Task execution",
                    task_type=task_type,
                    task_succeeded=success,
                    outcome=outcome_msg,
                    tool_calls=None,  # Could be passed in future
                    applied_bullets=bullet_ids,
                    error_trace=failure_reason if not success else None,
                    user_feedback=new_insight,
                )

                # Curate: Apply delta updates based on reflection
                curation = await curator.curate(reflection, auto_prune=False)

                result["ace_reflection"] = {
                    "insights_extracted": len(reflection.insights),
                    "bullets_tagged_helpful": len(reflection.bullets_to_tag_helpful),
                    "bullets_tagged_harmful": len(reflection.bullets_to_tag_harmful),
                    "root_causes": reflection.root_causes[:2] if reflection.root_causes else [],
                }
                result["ace_curation"] = {
                    "bullets_added": curation.bullets_added,
                    "bullets_removed": curation.bullets_removed,
                    "dedup_prevented": curation.dedup_prevented,
                    "feedback_recorded": curation.feedback_recorded,
                }

                log.info(
                    "ace_reflection_complete",
                    task_succeeded=success,
                    insights=len(reflection.insights),
                    curation_added=curation.bullets_added,
                )
            except Exception as e:
                log.warning("ace_reflection_failed", error=str(e))
                result["ace_reflection_error"] = str(e)

        # If user provided explicit insight (regardless of reflection), add it
        if new_insight:
            try:
                from ..ace.curator import get_curator
                curator = get_curator(str(project_path))

                added, existing_id = await curator.add_bullet(
                    task_type=task_type,
                    content=new_insight,
                    section="learned_strategies" if success else "failure_modes",
                    source="reflector",
                )

                if added:
                    result["new_bullet_added"] = True
                    result["insight_recorded"] = new_insight
                else:
                    result["new_bullet_added"] = False
                    result["similar_bullet_exists"] = existing_id
            except Exception as e:
                result["insight_error"] = str(e)

        if task_summary:
            result["task_summary"] = task_summary
        if failure_reason:
            result["failure_reason"] = failure_reason

        log.info(
            "ace_task_completed",
            success=success,
            bullets_updated=len(feedback_recorded),
            new_insight=bool(new_insight),
        )

        return json.dumps(result, indent=2)

    @mcp.tool()
    async def reflect(
        task_description: str,
        task_type: str = "coding",
        success: bool = True,
        outcome: str | None = None,
        error_trace: str | None = None,
        applied_bullets: str | None = None,
        path: str | None = None,
    ) -> str:
        """
        Manually trigger ACE Reflector to analyze a task execution.

        Use this tool when you want to analyze what happened during a task
        and extract insights for the playbook. The Reflector will:
        1. Analyze the task outcome (success/failure)
        2. Identify what worked and what didn't
        3. Extract strategic insights
        4. Optionally curate the playbook with new bullets

        Args:
            task_description: What was the task trying to accomplish
            task_type: coding, testing, debugging, architecture, git, etc.
            success: Whether the task succeeded
            outcome: Description of what happened
            error_trace: If failed, the error message or stack trace
            applied_bullets: JSON array of bullet IDs that were applied
            path: Optional project path

        Returns:
            JSON with reflection insights and curation results

        Example:
            reflect(
                task_description="Implement user authentication",
                task_type="coding",
                success=False,
                outcome="Tests failed due to missing mock",
                error_trace="AssertionError: mock not configured"
            )
        """
        from pathlib import Path as PyPath
        from ..playbook import get_playbook_manager

        pm = get_playbook_manager()
        # Only set project if path explicitly provided
        if path:
            project_path = PyPath(path).resolve()
            pm.set_project(project_path)
        else:
            project_path = pm.playbook_dir.parent.parent

        # Parse applied_bullets
        bullet_ids = []
        if applied_bullets:
            try:
                bullet_ids = json.loads(applied_bullets)
                if not isinstance(bullet_ids, list):
                    bullet_ids = [bullet_ids]
            except json.JSONDecodeError:
                bullet_ids = [b.strip() for b in applied_bullets.split(",") if b.strip()]

        try:
            from ..ace.reflector import get_reflector
            from ..ace.curator import get_curator

            reflector = get_reflector()
            curator = get_curator(str(project_path))

            # Run reflection
            reflection = await reflector.reflect(
                task_description=task_description,
                task_type=task_type,
                task_succeeded=success,
                outcome=outcome or ("Task completed successfully" if success else "Task failed"),
                tool_calls=None,
                applied_bullets=bullet_ids,
                error_trace=error_trace,
                user_feedback=None,
            )

            # Curate the playbook based on reflection
            curation = await curator.curate(reflection, auto_prune=False)

            result = {
                "reflection": {
                    "task_succeeded": reflection.task_succeeded,
                    "task_type": reflection.task_type,
                    "insights_count": len(reflection.insights),
                    "insights": [
                        {"content": i.content, "type": i.insight_type.value, "confidence": i.confidence}
                        for i in reflection.insights[:5]  # Limit to 5
                    ],
                    "root_causes": reflection.root_causes[:3] if reflection.root_causes else [],
                    "correct_approaches": reflection.correct_approaches[:3] if reflection.correct_approaches else [],
                    "bullets_tagged_helpful": reflection.bullets_to_tag_helpful,
                    "bullets_tagged_harmful": reflection.bullets_to_tag_harmful,
                },
                "curation": {
                    "bullets_added": curation.bullets_added,
                    "bullets_removed": curation.bullets_removed,
                    "bullets_merged": curation.bullets_merged,
                    "dedup_prevented": curation.dedup_prevented,
                    "feedback_recorded": curation.feedback_recorded,
                },
            }

            log.info(
                "manual_reflection_complete",
                task_type=task_type,
                success=success,
                insights=len(reflection.insights),
                curation_added=curation.bullets_added,
            )

            return json.dumps(result, indent=2)

        except Exception as e:
            log.error("reflection_failed", error=str(e))
            return json.dumps({
                "error": str(e),
                "message": "Reflection failed. Check if LLM backend is available.",
            }, indent=2)

    # =========================================================================
    # ACE Workflow Checkpoint Tools
    # =========================================================================

    @mcp.tool()
    async def check_ace_status(path: str | None = None) -> str:
        """
        Check whether ACE workflow was followed and what actions are needed.

        **IMPORTANT**: Call this tool at the start of EVERY conversation to see
        if onboarding/playbooks are available and what context to load.

        This is a gate check - it directs you to the proper workflow based on
        what's already been set up for the project.

        Args:
            path: Optional project path

        Returns:
            JSON with ACE status, available playbooks, and recommended actions
        """
        from pathlib import Path as PyPath
        from ..playbook import get_playbook_manager
        from ..context_detector import get_context_manager

        pm = get_playbook_manager()
        # Only set project if path explicitly provided - otherwise use current context
        if path:
            project_path = PyPath(path).resolve()
            pm.set_project(project_path)
        else:
            project_path = pm.playbook_dir.parent.parent

        stats = pm.get_stats()
        tracker = get_ace_tracker()

        # Check if playbooks exist
        has_playbooks = stats.get("total_bullets", 0) > 0
        playbook_queried = tracker.was_playbook_queried(str(project_path))

        if not has_playbooks:
            return json.dumps({
                "status": "no_playbooks",
                "message": "ACE Framework not initialized for this project.",
                "action_required": f"Run `init_project(path='{project_path}')` to initialize ACE Framework.",
                "alternative": "Or manually call `scan_codebase()` followed by `analyze_and_index()`",
            }, indent=2)

        if not playbook_queried:
            return json.dumps({
                "status": "playbook_not_loaded",
                "message": "Playbooks available but not loaded for this session.",
                "action_required": "Call `auto_context(message='<your task>')` to load relevant playbooks.",
                "available_playbooks": stats.get("playbooks", {}),
                "bullet_count": stats.get("total_bullets", 0),
            }, indent=2)

        return json.dumps({
            "status": "ready",
            "message": "ACE Framework active. Playbooks loaded.",
            "playbook_stats": stats,
            "reminder": "Apply playbook bullets to your work. Call report_feedback() when done.",
        }, indent=2)

    @mcp.tool()
    async def think_about_task_adherence() -> str:
        """
        Reflect on whether you are still on track with the current task.

        **IMPORTANT**: This tool should ALWAYS be called BEFORE you insert,
        replace, or delete code. It prompts you to verify alignment with
        project patterns and user intent.

        Returns:
            Reflection prompts to ensure task adherence
        """
        # Record checkpoint - unlocks file modification tools
        record_checkpoint()

        return json.dumps({
            "reflection_prompts": [
                "Are you deviating from the task at hand?",
                "Have you loaded all relevant playbook bullets for this task type?",
                "Is your implementation aligned with the project's code style and conventions?",
                "Do you need any additional information before modifying code?",
                "Would it be better to ask the user for clarification first?",
            ],
            "checklist": {
                "playbook_loaded": "Did you call auto_context() for this task?",
                "patterns_applied": "Are you applying the bullets from the playbook?",
                "style_consistent": "Does your code match existing project patterns?",
                "scope_appropriate": "Are changes scoped to what was requested?",
            },
            "guidance": (
                "If the conversation has drifted from the original task, "
                "acknowledge this and suggest how to proceed. "
                "It's better to pause and clarify than to make large changes "
                "that might not align with user intent."
            ),
        }, indent=2)

    @mcp.tool()
    async def think_about_collected_info() -> str:
        """
        Reflect on whether you have collected enough information.

        **IMPORTANT**: This tool should ALWAYS be called after a non-trivial
        sequence of searching/reading operations (find_symbol, grep, read_file, etc).

        Returns:
            Reflection prompts to assess information completeness
        """
        return json.dumps({
            "reflection_prompts": [
                "Have you collected all the information needed for this task?",
                "Is there missing context that could be acquired with available tools?",
                "Should you use LSP tools (lsp_find_references, lsp_goto_definition) for deeper understanding?",
                "Are there memory files that might contain relevant project knowledge?",
                "Do you need to ask the user for clarification on any points?",
            ],
            "tool_suggestions": {
                "semantic_navigation": ["lsp_find_references", "lsp_goto_definition", "lsp_get_symbols"],
                "pattern_search": ["search_for_pattern", "find_symbol"],
                "knowledge_retrieval": ["read_memory", "list_memories"],
            },
            "guidance": (
                "Think step by step about what information is missing. "
                "If you can acquire it with available tools, do so. "
                "If not, ask the user for clarification before proceeding."
            ),
        }, indent=2)

    @mcp.tool()
    async def think_about_completion() -> str:
        """
        Reflect on whether the task is truly complete.

        **IMPORTANT**: Call this tool when you believe the task is done.
        It prompts verification steps before declaring completion.

        Returns:
            Completion verification checklist
        """
        return json.dumps({
            "reflection_prompts": [
                "Have you performed ALL steps required by the task?",
                "Is it appropriate to run tests? If so, have you done that?",
                "Should linting/formatting be run? If so, have you done that?",
                "Are there non-code files (docs, config) that should be updated?",
                "Should new tests be written to cover the changes?",
                "METHODOLOGY CAPTURE: What did you do differently that worked? Document HOW, not just WHAT.",
            ],
            "checklist": {
                "code_changes": "All required code changes complete?",
                "tests": "Tests passing? New tests needed?",
                "linting": "Code formatted and linted?",
                "documentation": "Docs updated if needed?",
                "methodology": "Captured reusable methodology as playbook bullets?",
                "ace_complete": "Called complete_task(success, bullets_applied)?",
            },
            "ace_workflow": {
                "final_step": "complete_task(success=True/False, bullets_applied='[\"strat-xxx\", ...]')",
                "purpose": "Records feedback for all bullets in one call, closing the ACE learning loop",
            },
            "methodology_capture": {
                "question": "What did I do differently that found issues others might miss?",
                "action": "Add methodology as playbook bullet: playbook(action='add', task_type='debugging', content='...')",
                "example": "If you used 'grep -c' to verify counts, add that technique as a bullet",
            },
            "guidance": (
                "Read relevant memory files to see what should be done when a task completes. "
                "For exploration-only tasks, tests and linting may not be needed. "
                "For code changes, verify the full validation workflow. "
                "IMPORTANT: Capture reusable methodology as playbook bullets so other models can learn. "
                "ALWAYS call complete_task() to close the ACE learning loop."
            ),
        }, indent=2)

    @mcp.tool()
    async def read_initial_instructions() -> str:
        """
        Get the Delia ACE Framework instructions manual.

        **CRITICAL**: If you haven't already read the ACE Framework instructions,
        call this tool IMMEDIATELY at the start of the conversation. Some MCP
        clients do not automatically display system prompts.

        Returns:
            Complete ACE Framework instructions for this project
        """
        from pathlib import Path as PyPath
        from ..playbook import get_playbook_manager
        from ..mcp_server import _build_dynamic_instructions

        pm = get_playbook_manager()
        # Use current project context if set, otherwise default to CWD
        if pm.playbook_dir.exists():
            project_path = pm.playbook_dir.parent.parent
        else:
            project_path = PyPath.cwd()
            pm.set_project(project_path)

        # Get full dynamic instructions
        instructions = _build_dynamic_instructions()

        stats = pm.get_stats()

        return json.dumps({
            "message": "Delia ACE Framework Instructions Manual",
            "instructions": instructions,
            "playbook_summary": stats,
            "quick_start": [
                "1. Call auto_context(message='<task>') to load relevant playbooks",
                "2. Apply returned bullets to your work",
                "3. Call think_about_task_adherence() before modifying code",
                "4. Call think_about_completion() when you think you're done",
                "5. Call report_feedback(bullet_id, task_type, helpful) to close the loop",
            ],
            "note": "You have hereby read the ACE Framework manual and do not need to read it again.",
        }, indent=2)

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

    # =========================================================================
    # Git History Tools
    # =========================================================================

    @mcp.tool()
    async def git_log(
        path: str = ".",
        file: str | None = None,
        n: int = 10,
        since: str | None = None,
        author: str | None = None,
        oneline: bool = False,
    ) -> str:
        """
        Show git commit history.

        Args:
            path: Repository path (default current directory)
            file: Filter to specific file
            n: Number of commits to show (default 10)
            since: Date filter (e.g., "2024-01-01", "1 week ago")
            author: Author filter (partial match)
            oneline: Compact one-line format

        Returns:
            Formatted commit history
        """
        from .coding import git_log as git_log_impl
        result = await git_log_impl(path, file, n, since, author, oneline)
        return json.dumps({"result": result})

    @mcp.tool()
    async def git_blame(
        file: str,
        path: str = ".",
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """
        Show line-by-line authorship for a file.

        Args:
            file: File to blame
            path: Repository path
            start_line: Start line (1-indexed)
            end_line: End line (1-indexed)

        Returns:
            Blame output with commit, author, date, and line content
        """
        from .coding import git_blame as git_blame_impl
        result = await git_blame_impl(file, path, start_line, end_line)
        return json.dumps({"result": result})

    @mcp.tool()
    async def git_show(
        commit: str,
        file: str | None = None,
        path: str = ".",
        stat: bool = False,
    ) -> str:
        """
        Show commit details and diff.

        Args:
            commit: Commit hash or reference (e.g., "HEAD", "abc123", "HEAD~3")
            file: Specific file to show changes for
            path: Repository path
            stat: Show diffstat instead of full diff

        Returns:
            Commit details with diff
        """
        from .coding import git_show as git_show_impl
        result = await git_show_impl(commit, file, path, stat)
        return json.dumps({"result": result})

    # =========================================================================
    # Bulk File Operations
    # =========================================================================

    @mcp.tool()
    async def read_files(
        paths: str,
    ) -> str:
        """
        Read multiple files in one call.

        More efficient than calling read_file N times.

        Args:
            paths: JSON array of file paths (e.g., '["src/main.py", "src/utils.py"]')

        Returns:
            Dict mapping path to content (or error message)
        """
        import json as json_mod
        from .files import read_files as read_files_impl

        try:
            path_list = json_mod.loads(paths)
            if not isinstance(path_list, list):
                return json.dumps({"error": "paths must be a JSON array"})
        except json_mod.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        result = await read_files_impl(path_list)
        return json.dumps({"result": result})

    @mcp.tool()
    async def edit_files(
        edits: str,
    ) -> str:
        """
        Apply multiple edits across files atomically.

        All edits are validated before any are applied.

        Args:
            edits: JSON array of edit objects, each with:
                - path: File path
                - old_text: Text to find
                - new_text: Text to replace with
                Example: '[{"path": "a.py", "old_text": "foo", "new_text": "bar"}]'

        Returns:
            Dict mapping path to result message
        """
        import json as json_mod
        from .files import edit_files as edit_files_impl

        try:
            edit_list = json_mod.loads(edits)
            if not isinstance(edit_list, list):
                return json.dumps({"error": "edits must be a JSON array"})
        except json_mod.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        result = await edit_files_impl(edit_list)
        return json.dumps({"result": result})

    # =========================================================================
    # Tool Discovery
    # =========================================================================

    @mcp.tool()
    async def list_tools(
        category: str | None = None,
    ) -> str:
        """
        List available tools, optionally filtered by category.

        Use this to discover what tools are available and find the right
        tool for a task.

        Args:
            category: Filter by category (file_ops, lsp, git, testing, ace,
                     orchestration, admin, search, general). If None, lists all.

        Returns:
            Tools grouped by category with descriptions
        """
        from .registry import TOOL_CATEGORIES

        # Get all MCP tools from the server
        tools_info = mcp.list_tools()

        # Group by category (for now, we'll categorize based on name patterns)
        categorized: dict[str, list[dict]] = {cat: [] for cat in TOOL_CATEGORIES}

        # Categorization rules based on tool name patterns
        category_patterns = {
            "file_ops": ["read_file", "write_file", "edit_file", "list_dir", "find_file",
                        "search_for_pattern", "delete_file", "create_directory", "read_files", "edit_files"],
            "lsp": ["lsp_"],
            "git": ["git_"],
            "testing": ["run_tests"],
            "ace": ["auto_context", "complete_task", "get_playbook", "report_feedback",
                   "get_project_context", "playbook", "check_ace_status", "think_about_", "reflect"],
            "orchestration": ["delegate", "think", "batch", "chain", "workflow", "agent"],
            "admin": ["health", "models", "switch_", "set_project", "init_project",
                     "mcp_servers", "session", "admin"],
            "search": ["semantic_search", "codebase_graph", "get_related_files", "explain_dependency"],
        }

        for tool in tools_info:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            tool_desc = tool.description if hasattr(tool, 'description') else ""

            assigned = False
            for cat, patterns in category_patterns.items():
                for pattern in patterns:
                    if tool_name.startswith(pattern) or tool_name == pattern:
                        categorized[cat].append({"name": tool_name, "description": tool_desc[:100]})
                        assigned = True
                        break
                if assigned:
                    break

            if not assigned:
                categorized["general"].append({"name": tool_name, "description": tool_desc[:100]})

        # Filter if category specified
        if category:
            if category not in TOOL_CATEGORIES:
                return json.dumps({
                    "error": f"Unknown category: {category}",
                    "valid_categories": list(TOOL_CATEGORIES.keys()),
                })
            return json.dumps({
                "category": category,
                "description": TOOL_CATEGORIES[category],
                "tools": categorized.get(category, []),
                "tool_count": len(categorized.get(category, [])),
            }, indent=2)

        # Return all categories with counts
        summary = {
            "total_tools": sum(len(tools) for tools in categorized.values()),
            "categories": {
                cat: {
                    "description": TOOL_CATEGORIES[cat],
                    "count": len(tools),
                    "tools": [t["name"] for t in tools],
                }
                for cat, tools in categorized.items()
                if tools  # Only include non-empty categories
            },
        }
        return json.dumps(summary, indent=2)

    @mcp.tool()
    async def describe_tool(
        name: str,
    ) -> str:
        """
        Get detailed information about a specific tool.

        Args:
            name: Name of the tool to describe

        Returns:
            Full tool description with parameters and examples
        """
        tools_info = mcp.list_tools()

        for tool in tools_info:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            if tool_name == name:
                return json.dumps({
                    "name": tool_name,
                    "description": tool.description if hasattr(tool, 'description') else "",
                    "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                }, indent=2)

        return json.dumps({"error": f"Tool not found: {name}"})

    # =========================================================================
    # ACE Manager Admin
    # =========================================================================

    @mcp.tool()
    async def ace_manager_stats() -> str:
        """
        Get statistics about per-project ACE enforcement.

        Shows which projects have active ACE trackers and allows cleanup
        of stale trackers.

        Returns:
            Active projects and their ACE state
        """
        stats = _ace_manager.get_stats()

        # Add per-project details
        project_details = {}
        for project in stats["projects"]:
            tracker = _ace_manager.get_tracker(project)
            project_details[project] = {
                "ace_started": tracker.is_ace_started(project),
                "playbook_queried": tracker.was_playbook_queried(project),
                "last_activity": tracker.get_last_activity(),
            }

        return json.dumps({
            "result": {
                "active_projects": stats["active_projects"],
                "projects": project_details,
            }
        }, indent=2)

    @mcp.tool()
    async def ace_manager_cleanup(
        max_age_hours: float = 1.0,
    ) -> str:
        """
        Clean up stale ACE trackers for inactive projects.

        Args:
            max_age_hours: Remove trackers inactive for longer than this (default 1 hour)

        Returns:
            Number of trackers cleaned up
        """
        max_age_seconds = int(max_age_hours * 3600)
        removed = _ace_manager.cleanup_stale(max_age_seconds)

        return json.dumps({
            "result": {
                "trackers_removed": removed,
                "remaining_projects": len(_ace_manager.list_projects()),
            }
        }, indent=2)
