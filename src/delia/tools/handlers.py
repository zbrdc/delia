# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
High-level MCP tool handlers for Delia.

This module registers all MCP tools with FastMCP. Implementation functions
are in separate modules:
- handlers_enforcement.py: Framework enforcement classes and helpers
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
from .handlers_enforcement import (
    EXEMPT_TOOLS,
    CHECKPOINT_REQUIRED_TOOLS,
    EnforcementTracker,
    EnforcementManager,
    get_tracker,
    get_manager,
    check_context_gate,
    check_checkpoint_gate,
    record_checkpoint,
    record_search,
    get_phase_injection,
    inject_reminder,
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
_enforcement_manager = get_manager()


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
    # Context Detection (Delia Framework)
    # Legacy tools removed - use consolidated: playbook(), project()
    # =========================================================================

    @mcp.tool()
    async def auto_context(
        message: str,
        path: str | None = None,
        prior_context: str | None = None,
        working_files: str | None = None,
        code_snippet: str | None = None,
    ) -> str:
        """Detect task type from message and load relevant playbook bullets and profiles."""
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
            from ..learning.retrieval import get_retriever
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

        # Register with enforcement tracker (store message for feedback learning)
        tracker = get_tracker()
        tracker.record_context_started(
            str(project_path),
            task_type=context.primary_task,
            message=message,
        )

        # Build response with system time for agent awareness
        from ..language import get_current_time_context
        system_time = get_current_time_context()

        result = {
            "system_time": system_time,
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
            "delia_workflow": {
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
        """Load a specific profile by name (e.g., "deployment.md", "security.md")."""
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
        """Record task outcome and update playbook bullet feedback. Closes the Delia learning loop."""
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

        # Get system time for agent awareness
        from ..language import get_current_time_context
        system_time = get_current_time_context()

        result = {
            "system_time": system_time,
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
        # DETECTION FEEDBACK LOOP: Auto-learn from misdetections
        # =====================================================================
        # Compare what was detected by auto_context vs what was actually worked on.
        # If they differ, record feedback to improve future detection.
        # =====================================================================
        try:
            from .handlers_enforcement import get_tracker
            from ..context_detector import get_pattern_learner

            tracker = get_tracker()
            detected_task, original_message = tracker.get_detection_context(str(project_path))

            if detected_task and original_message and detected_task != task_type:
                # Misdetection occurred - record feedback to learn
                learner = get_pattern_learner(project_path)
                feedback_result = learner.record_feedback(
                    message=original_message,
                    detected_task=detected_task,
                    correct_task=task_type,
                    was_correct=False,
                )
                result["detection_feedback"] = {
                    "detected": detected_task,
                    "actual": task_type,
                    "patterns_learned": feedback_result.get("patterns_added", 0),
                    "message": f"Learned: '{original_message[:50]}...' → {task_type} (was {detected_task})",
                }
                log.info(
                    "detection_feedback_auto_recorded",
                    detected=detected_task,
                    actual=task_type,
                    patterns_added=feedback_result.get("patterns_added", 0),
                )
            elif detected_task and original_message and detected_task == task_type:
                # Detection was correct - reinforce patterns
                learner = get_pattern_learner(project_path)
                learner.record_feedback(
                    message=original_message,
                    detected_task=detected_task,
                    correct_task=task_type,
                    was_correct=True,
                )
        except Exception as e:
            log.debug("detection_feedback_skipped", error=str(e))

        # =====================================================================
        # DELIA LEARNING LOOP: Reflector → Curator pipeline
        # =====================================================================
        # Learn from BOTH success AND failure to maximize learning improvements.
        # Success: Learn what worked well (patterns to reinforce)
        # Failure: Learn what went wrong (anti-patterns to avoid)
        # =====================================================================
        if success or (not success and failure_reason):
            try:
                from ..learning.reflector import get_reflector
                from ..learning.curator import get_curator

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

                result["reflection"] = {
                    "insights_extracted": len(reflection.insights),
                    "bullets_tagged_helpful": len(reflection.bullets_to_tag_helpful),
                    "bullets_tagged_harmful": len(reflection.bullets_to_tag_harmful),
                    "root_causes": reflection.root_causes[:2] if reflection.root_causes else [],
                }
                result["curation"] = {
                    "bullets_added": curation.bullets_added,
                    "bullets_removed": curation.bullets_removed,
                    "dedup_prevented": curation.dedup_prevented,
                    "feedback_recorded": curation.feedback_recorded,
                }

                log.info(
                    "reflection_complete",
                    task_succeeded=success,
                    insights=len(reflection.insights),
                    curation_added=curation.bullets_added,
                )
            except Exception as e:
                log.warning("reflection_failed", error=str(e))
                result["reflection_error"] = str(e)

        # If user provided explicit insight (regardless of reflection), add it
        if new_insight:
            try:
                from ..learning.curator import get_curator
                curator = get_curator(str(project_path))

                add_result = await curator.add_bullet(
                    task_type=task_type,
                    content=new_insight,
                    section="learned_strategies" if success else "failure_modes",
                    source="reflector",
                )

                if add_result.get("added"):
                    result["new_bullet_added"] = True
                    result["insight_recorded"] = new_insight
                    result["bullet_id"] = add_result.get("bullet_id")
                elif add_result.get("quality_rejected"):
                    result["new_bullet_added"] = False
                    result["quality_rejected"] = True
                    result["rejection_reason"] = add_result.get("reason")
                else:
                    result["new_bullet_added"] = False
                    result["similar_bullet_exists"] = add_result.get("bullet_id")
            except Exception as e:
                result["insight_error"] = str(e)

        if task_summary:
            result["task_summary"] = task_summary
        if failure_reason:
            result["failure_reason"] = failure_reason

        log.info(
            "task_completed",
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
        """Analyze task execution and extract insights for the playbook."""
        from pathlib import Path as PyPath
        from ..playbook import get_playbook_manager
        from ..language import get_current_time_context
        
        system_time = get_current_time_context()

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
            from ..learning.reflector import get_reflector
            from ..learning.curator import get_curator

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
                "system_time": system_time,
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
                "system_time": system_time,
                "error": str(e),
                "message": "Reflection failed. Check if LLM backend is available.",
            }, indent=2)

    # =========================================================================
    # Workflow Checkpoint Tools
    # =========================================================================

    @mcp.tool()
    async def check_status(path: str | None = None) -> str:
        """Check Delia Framework status and get recommended actions for the current project."""
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
        tracker = get_tracker()

        # Get system time for agent awareness
        from ..language import get_current_time_context
        system_time = get_current_time_context()

        # Check if playbooks exist
        has_playbooks = stats.get("total_bullets", 0) > 0
        playbook_queried = tracker.was_playbook_queried(str(project_path))

        if not has_playbooks:
            return json.dumps({
                "system_time": system_time,
                "status": "no_playbooks",
                "message": "Delia Framework not initialized for this project.",
                "action_required": f"Run `init_project(path='{project_path}')` to initialize the framework.",
                "alternative": "Or manually call `scan_codebase()` followed by `analyze_and_index()`",
            }, indent=2)

        if not playbook_queried:
            return json.dumps({
                "system_time": system_time,
                "status": "playbook_not_loaded",
                "message": "Playbooks available but not loaded for this session.",
                "action_required": "Call `auto_context(message='<your task>')` to load relevant playbooks.",
                "available_playbooks": stats.get("playbooks", {}),
                "bullet_count": stats.get("total_bullets", 0),
            }, indent=2)

        return json.dumps({
            "system_time": system_time,
            "status": "ready",
            "message": "Delia Framework active. Playbooks loaded.",
            "playbook_stats": stats,
            "reminder": "Apply playbook bullets to your work. Call report_feedback() when done.",
        }, indent=2)

    @mcp.tool()
    async def think_about_task_adherence() -> str:
        """Reflection checkpoint before code modifications. Verifies alignment with task and patterns."""
        from ..language import get_current_time_context
        
        # Record checkpoint - unlocks file modification tools
        record_checkpoint()

        return json.dumps({
            "system_time": get_current_time_context(),
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
        """Reflection checkpoint after search/reading. Verifies information completeness."""
        from ..language import get_current_time_context
        
        return json.dumps({
            "system_time": get_current_time_context(),
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
        """Reflection checkpoint before completion. Verifies all steps are done."""
        from ..language import get_current_time_context
        
        return json.dumps({
            "system_time": get_current_time_context(),
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
                "workflow_complete": "Called complete_task(success, bullets_applied)?",
            },
            "workflow": {
                "final_step": "complete_task(success=True/False, bullets_applied='[\"strat-xxx\", ...]')",
                "purpose": "Records feedback for all bullets in one call, closing the Delia learning loop",
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
                "ALWAYS call complete_task() to close the Delia learning loop."
            ),
        }, indent=2)

    @mcp.tool()
    async def snapshot_context(
        task_summary: str,
        pending_items: str,
        key_decisions: str | None = None,
        files_modified: str | None = None,
        next_steps: str | None = None,
        path: str | None = None,
    ) -> str:
        """Save task state to memory for continuation in a new conversation."""
        from datetime import datetime
        from ..playbook import get_playbook_manager

        pm = get_playbook_manager()
        if path:
            project_path = Path(path).resolve()
        elif pm.playbook_dir.exists():
            project_path = pm.playbook_dir.parent.parent
        else:
            project_path = Path.cwd()

        memories_dir = project_path / ".delia" / "memories"
        memories_dir.mkdir(parents=True, exist_ok=True)

        # Parse JSON inputs
        try:
            pending_list = json.loads(pending_items) if pending_items else []
        except json.JSONDecodeError:
            pending_list = [pending_items]  # Treat as single item if not valid JSON

        try:
            decisions_dict = json.loads(key_decisions) if key_decisions else {}
        except json.JSONDecodeError:
            decisions_dict = {"note": key_decisions} if key_decisions else {}

        try:
            files_list = json.loads(files_modified) if files_modified else []
        except json.JSONDecodeError:
            files_list = [files_modified] if files_modified else []

        # Build snapshot content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Task Snapshot",
            "",
            f"*Captured: {timestamp}*",
            "",
            "## Summary",
            task_summary,
            "",
            "## Pending Items",
        ]

        for item in pending_list:
            lines.append(f"- [ ] {item}")
        lines.append("")

        if decisions_dict:
            lines.append("## Key Decisions")
            for key, value in decisions_dict.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if files_list:
            lines.append("## Files Modified")
            for f in files_list:
                lines.append(f"- `{f}`")
            lines.append("")

        if next_steps:
            lines.append("## Next Steps")
            lines.append(next_steps)
            lines.append("")

        lines.extend([
            "---",
            "*To continue: Read this file at the start of your next session.*",
        ])

        content = "\n".join(lines)
        snapshot_path = memories_dir / "task_snapshot.md"
        snapshot_path.write_text(content)

        return json.dumps({
            "status": "snapshot_saved",
            "path": str(snapshot_path.relative_to(project_path)),
            "message": (
                "Task state captured. Suggest user start a new conversation. "
                "The next agent should call read_memory(name='task_snapshot') to resume."
            ),
            "pending_count": len(pending_list),
        }, indent=2)

    @mcp.tool()
    async def read_initial_instructions() -> str:
        """Get the Delia Framework instructions manual and playbook summary."""
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
            "message": "Delia Framework Instructions Manual",
            "instructions": instructions,
            "playbook_summary": stats,
            "quick_start": [
                "1. Call auto_context(message='<task>') to load relevant playbooks",
                "2. Apply returned bullets to your work",
                "3. Call think_about_task_adherence() before modifying code",
                "4. Call think_about_completion() when you think you're done",
                "5. Call report_feedback(bullet_id, task_type, helpful) to close the loop",
            ],
            "note": "You have hereby read the Delia Framework manual and do not need to read it again.",
        }, indent=2)

    @mcp.tool()
    async def record_detection_feedback(
        message: str,
        detected_task: str,
        correct_task: str,
        path: str | None = None,
    ) -> str:
        """Record feedback when auto_context detection was incorrect to improve future accuracy."""
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
        """Get statistics about learned detection patterns and their effectiveness."""
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
        """Remove learned patterns with low effectiveness after sufficient usage."""
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
        """Show git commit history with optional file, date, and author filters."""
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
        """Show line-by-line authorship for a file with optional line range."""
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
        """Show commit details and diff for a specific commit."""
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
        """Read multiple files in one call. Paths: JSON array of file paths."""
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
        """Apply multiple edits atomically. Edits: JSON array of {path, old_text, new_text}."""
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
        """List available tools by category (file_ops, lsp, git, framework, orchestration, admin, search)."""
        from .registry import TOOL_CATEGORIES

        # Get all MCP tools from the server (async, returns dict of name -> Tool)
        tools_dict = await mcp.get_tools()
        tools_info = list(tools_dict.values())

        # Group by category (for now, we'll categorize based on name patterns)
        categorized: dict[str, list[dict]] = {cat: [] for cat in TOOL_CATEGORIES}

        # Categorization rules based on tool name patterns
        category_patterns = {
            "file_ops": ["read_file", "write_file", "edit_file", "list_dir", "find_file",
                        "search_for_pattern", "delete_file", "create_directory", "read_files", "edit_files"],
            "lsp": ["lsp_"],
            "git": ["git_"],
            "testing": ["run_tests"],
            "framework": ["auto_context", "complete_task", "get_playbook", "report_feedback",
                   "get_project_context", "playbook", "check_status", "think_about_", "reflect"],
            "orchestration": ["delegate", "think", "batch", "chain", "workflow", "agent"],
            "admin": ["health", "models", "switch_", "set_project", "init_project",
                     "mcp_servers", "session", "admin"],
            "search": ["semantic_search", "codebase_graph", "get_related_files", "explain_dependency"],
        }

        for tool in tools_info:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            tool_desc = getattr(tool, 'description', None) or ""

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
        """Get detailed information about a specific tool."""
        tools_dict = await mcp.get_tools()

        if name in tools_dict:
            tool = tools_dict[name]
            return json.dumps({
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters if hasattr(tool, 'parameters') else {},
            }, indent=2)

        return json.dumps({"error": f"Tool not found: {name}"})

    # =========================================================================
    # Framework Manager Admin
    # =========================================================================

    @mcp.tool()
    async def framework_stats() -> str:
        """Get per-project Delia Framework enforcement statistics."""
        stats = _enforcement_manager.get_stats()

        # Add per-project details
        project_details = {}
        for project in stats["projects"]:
            tracker = _enforcement_manager.get_tracker(project)
            project_details[project] = {
                "context_started": tracker.is_context_started(project),
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
    async def framework_cleanup(
        max_age_hours: float = 1.0,
    ) -> str:
        """Clean up stale framework trackers for inactive projects."""
        max_age_seconds = int(max_age_hours * 3600)
        removed = _enforcement_manager.cleanup_stale(max_age_seconds)

        return json.dumps({
            "result": {
                "trackers_removed": removed,
                "remaining_projects": len(_enforcement_manager.list_projects()),
            }
        }, indent=2)
