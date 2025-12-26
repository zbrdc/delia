# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Delia Framework core tools - ALWAYS registered regardless of profile.

These tools implement the Delia learning loop:
1. auto_context() - Load relevant context at task start
2. think_about_*() - Reflection checkpoints during work
3. complete_task() - Record outcomes and close learning loop

Framework tools are the #1 most important tools for Delia.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from fastmcp import FastMCP

from ..container import get_container
from ..context import get_project_path

# Import enforcement helpers
from .handlers_enforcement import (
    get_tracker,
    record_checkpoint,
)

log = structlog.get_logger()


def register_framework_tools(mcp: FastMCP):
    """Register Delia Framework tools - ALWAYS registered."""

    @mcp.tool()
    async def auto_context(
        message: str,
        path: str | None = None,
        prior_context: str | None = None,
        working_files: str | None = None,
        code_snippet: str | None = None,
    ) -> str:
        """Detect task type from message and load relevant playbook bullets and profiles.

        WHEN TO USE: Call this IMMEDIATELY after receiving a task. It is CRITICAL for success.
        Retrieves relevant project patterns and framework-specific profiles.
        Provides recommended tools optimized for the detected task.
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
        project_path = PyPath(path).resolve() if path else get_project_path()

        # Parse working_files from JSON string
        files_list: list[str] | None = None
        if working_files:
            try:
                files_list = json.loads(working_files)
                if not isinstance(files_list, list):
                    files_list = [str(files_list)]
            except json.JSONDecodeError:
                files_list = [f.strip() for f in working_files.split(",") if f.strip()]

        # Combine message with prior context for detection
        detection_text = message
        if prior_context:
            detection_text = f"{message}\n\n[Prior context]: {prior_context}"

        # Use enhanced detection if files or code provided
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

        # Update context manager for backwards compatibility
        ctx_mgr = get_context_manager()
        ctx_mgr._current_context = context
        ctx_mgr._last_message = message

        pm = get_playbook_manager()
        pm.set_project(project_path)

        # Collect bullets with semantic retrieval
        all_bullets = []
        try:
            from ..learning.retrieval import get_retriever
            retriever = get_retriever()

            for task_type in context.all_tasks():
                limit = 5 if task_type == context.primary_task else 3
                bullets = pm.load_playbook(task_type)

                try:
                    scored = await retriever.retrieve(
                        bullets=bullets,
                        query=detection_text,
                        project_path=project_path,
                        limit=limit,
                    )
                except Exception:
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

        # Get relevant profiles
        profiles_dir = project_path / ".delia" / "profiles"
        templates_dir = PyPath(__file__).parent.parent / "templates" / "profiles"

        relevant_names = set(get_relevant_profiles(context))
        if file_context and file_context.profile_hints:
            relevant_names.update(file_context.profile_hints)

        all_available = set()
        if profiles_dir.exists():
            all_available.update(p.name for p in profiles_dir.glob("*.md"))

        loaded_profiles = []
        for profile_name in sorted(relevant_names):
            content = None
            profile_path = profiles_dir / profile_name
            if profile_path.exists():
                try:
                    content = profile_path.read_text()
                except Exception:
                    pass
            if content is None:
                template_path = templates_dir / profile_name
                if template_path.exists():
                    try:
                        content = template_path.read_text()
                    except Exception:
                        pass
            if content:
                loaded_profiles.append({"name": profile_name, "content": content})

        loaded_names = {p["name"] for p in loaded_profiles}
        other_available = sorted(all_available - loaded_names)
        bullet_ids = [b["id"] for b in all_bullets]

        # Tool recommendations
        from ._tool_recommendations import get_recommended_tools
        recommended_tools = get_recommended_tools(context)

        # Register with enforcement tracker
        tracker = get_tracker()
        tracker.record_context_started(
            str(project_path),
            task_type=context.primary_task,
            message=message,
        )

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
            "bullet_ids": bullet_ids,
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

        if path:
            project_path = PyPath(path).resolve()
        else:
            pm = get_playbook_manager()
            project_path = pm.playbook_dir.parent.parent

        profiles_dir = project_path / ".delia" / "profiles"
        templates_dir = PyPath(__file__).parent.parent / "templates" / "profiles"

        if not name.endswith(".md"):
            name = f"{name}.md"

        content = None
        source = None

        profile_path = profiles_dir / name
        if profile_path.exists():
            try:
                content = profile_path.read_text()
                source = "project"
            except Exception as e:
                return json.dumps({"error": f"Failed to read profile: {e}"})

        if content is None:
            template_path = templates_dir / name
            if template_path.exists():
                try:
                    content = template_path.read_text()
                    source = "template"
                except Exception as e:
                    return json.dumps({"error": f"Failed to read template: {e}"})

        if content is None:
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
        if path:
            project_path = PyPath(path).resolve()
            pm.set_project(project_path)
        else:
            project_path = pm.playbook_dir.parent.parent

        # Parse bullets_applied
        try:
            bullet_ids = json.loads(bullets_applied) if bullets_applied else []
            if not isinstance(bullet_ids, list):
                bullet_ids = [bullet_ids]
        except json.JSONDecodeError:
            bullet_ids = [b.strip() for b in bullets_applied.split(",") if b.strip()]

        # Record feedback for each bullet
        feedback_recorded = []
        for bullet_id in bullet_ids:
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

        from ..language import get_current_time_context
        system_time = get_current_time_context()

        result = {
            "system_time": system_time,
            "status": "completed",
            "success": success,
            "feedback_recorded": feedback_recorded,
            "bullets_updated": len(feedback_recorded),
        }

        task_type = "coding"
        if feedback_recorded:
            task_type = feedback_recorded[0]["task_type"]

        # Detection feedback loop
        try:
            from ..context_detector import get_pattern_learner

            tracker = get_tracker()
            detected_task, original_message = tracker.get_detection_context(str(project_path))

            if detected_task and original_message and detected_task != task_type:
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
                }
            elif detected_task and original_message and detected_task == task_type:
                learner = get_pattern_learner(project_path)
                learner.record_feedback(
                    message=original_message,
                    detected_task=detected_task,
                    correct_task=task_type,
                    was_correct=True,
                )
        except Exception as e:
            log.debug("detection_feedback_skipped", error=str(e))

        # Delia learning loop: Reflector → Curator
        if success or (not success and failure_reason):
            try:
                from ..learning.reflector import get_reflector
                from ..learning.curator import get_curator

                reflector = get_reflector()
                curator = get_curator(str(project_path))

                outcome_msg = task_summary or "Task completed successfully" if success else failure_reason

                reflection = await reflector.reflect(
                    task_description=task_summary or "Task execution",
                    task_type=task_type,
                    task_succeeded=success,
                    outcome=outcome_msg,
                    tool_calls=None,
                    applied_bullets=bullet_ids,
                    error_trace=failure_reason if not success else None,
                    user_feedback=new_insight,
                )

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
            except Exception as e:
                log.warning("reflection_failed", error=str(e))
                result["reflection_error"] = str(e)

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
        if path:
            project_path = PyPath(path).resolve()
            pm.set_project(project_path)
        else:
            project_path = pm.playbook_dir.parent.parent

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

            curation = await curator.curate(reflection, auto_prune=False)

            result = {
                "system_time": system_time,
                "reflection": {
                    "task_succeeded": reflection.task_succeeded,
                    "task_type": reflection.task_type,
                    "insights_count": len(reflection.insights),
                    "insights": [
                        {"content": i.content, "type": i.insight_type.value, "confidence": i.confidence}
                        for i in reflection.insights[:5]
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
        from ..language import get_current_time_context

        pm = get_playbook_manager()
        if path:
            project_path = PyPath(path).resolve()
            pm.set_project(project_path)
        else:
            project_path = pm.playbook_dir.parent.parent

        stats = pm.get_stats()
        tracker = get_tracker()
        system_time = get_current_time_context()

        has_playbooks = stats.get("total_bullets", 0) > 0
        playbook_queried = tracker.was_playbook_queried(str(project_path))

        if not has_playbooks:
            return json.dumps({
                "system_time": system_time,
                "status": "no_playbooks",
                "message": "Delia Framework not initialized for this project.",
                "action_required": f"Run `init_project(path='{project_path}')` to initialize the framework.",
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
            "reminder": "Apply playbook bullets to your work. Call complete_task() when done.",
        }, indent=2)

    @mcp.tool()
    async def think_about_task_adherence() -> str:
        """Reflection checkpoint before code modifications. Verifies alignment with task and patterns."""
        from ..language import get_current_time_context

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
                "acknowledge this and suggest how to proceed."
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
                "Should you use LSP tools for deeper understanding?",
                "Are there memory files that might contain relevant project knowledge?",
                "Do you need to ask the user for clarification on any points?",
            ],
            "tool_suggestions": {
                "semantic_navigation": ["lsp_find_references", "lsp_goto_definition", "lsp_get_symbols"],
                "pattern_search": ["search_for_pattern", "find_file"],
                "knowledge_retrieval": ["memory(action='read')", "memory(action='list')"],
            },
            "guidance": (
                "Think step by step about what information is missing. "
                "If you can acquire it with available tools, do so."
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
                "METHODOLOGY CAPTURE: What did you do differently that worked?",
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
                "final_step": "complete_task(success=True/False, bullets_applied='[...]')",
                "purpose": "Records feedback for all bullets, closing the Delia learning loop",
            },
            "guidance": (
                "IMPORTANT: Capture reusable methodology as playbook bullets. "
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
            project_path = get_project_path()

        memories_dir = project_path / ".delia" / "memories"
        memories_dir.mkdir(parents=True, exist_ok=True)

        try:
            pending_list = json.loads(pending_items) if pending_items else []
        except json.JSONDecodeError:
            pending_list = [pending_items]

        try:
            decisions_dict = json.loads(key_decisions) if key_decisions else {}
        except json.JSONDecodeError:
            decisions_dict = {"note": key_decisions} if key_decisions else {}

        try:
            files_list = json.loads(files_modified) if files_modified else []
        except json.JSONDecodeError:
            files_list = [files_modified] if files_modified else []

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
                "The next agent should call memory(action='read', name='task_snapshot') to resume."
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
        if pm.playbook_dir.exists():
            project_path = pm.playbook_dir.parent.parent
        else:
            project_path = get_project_path()
            pm.set_project(project_path)

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
                "5. Call complete_task(success, bullets_applied) to close the loop",
            ],
        }, indent=2)

    # =========================================================================
    # Detection Feedback Tools
    # =========================================================================

    @mcp.tool()
    async def record_detection_feedback(
        message: str,
        detected_task: str,
        correct_task: str,
        path: str | None = None,
    ) -> str:
        """Record feedback when auto_context detection was incorrect to improve future accuracy."""
        from pathlib import Path as PyPath
        from ..context_detector import get_pattern_learner

        project_path = PyPath(path).resolve() if path else get_project_path()
        learner = get_pattern_learner(project_path)

        was_correct = detected_task == correct_task
        result = learner.record_feedback(
            message=message,
            detected_task=detected_task,
            correct_task=correct_task,
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

        project_path = PyPath(path).resolve() if path else get_project_path()
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

        project_path = PyPath(path).resolve() if path else get_project_path()
        learner = get_pattern_learner(project_path)

        removed = learner.prune_ineffective(min_effectiveness, min_uses)

        return json.dumps({
            "success": True,
            "patterns_removed": removed,
            "message": f"Pruned {removed} ineffective patterns." if removed > 0 else "No patterns needed pruning.",
        }, indent=2)
