# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
ACE Framework enforcement and tracking for Delia MCP tools.

Provides:
- ACEEnforcementTracker: Per-project ACE workflow tracking
- ACEEnforcementManager: Thread-safe manager for multiple projects
- Helper functions for ACE gating and reminders
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from typing import Any

log = structlog.get_logger()


# =============================================================================
# ACE Framework Dynamic Enforcement
# =============================================================================

# Tools that can be used without calling auto_context first
ACE_EXEMPT_TOOLS = {
    "auto_context",
    "check_ace_status",
    "read_initial_instructions",
    "health",
    "models",
    "set_project",
    "get_project_context",
    "get_playbook",
    "dashboard_url",
    "queue_status",
    "mcp_servers",
    "init_project",
    "scan_codebase",
}


class ACEEnforcementTracker:
    """Track ACE compliance dynamically across tool calls.

    Provides two enforcement mechanisms:
    1. Soft enforcement: record_playbook_query() and check_compliance()
    2. Hard enforcement: record_ace_started() and require_ace_started()
    """

    def __init__(self) -> None:
        self._playbook_queries: dict[str, float] = {}  # path -> last query timestamp
        self._ace_started: dict[str, float] = {}  # path -> timestamp when auto_context called
        self._pending_tasks: dict[str, dict] = {}  # path -> {task, start_time}
        self._compliance_warnings: int = 0
        self._gating_enabled: bool = True  # Can be disabled for testing

    def record_ace_started(self, path: str) -> None:
        """Record that auto_context was called for a project."""
        self._ace_started[path] = time.time()
        self._playbook_queries[path] = time.time()  # Also counts as playbook query
        log.info("ace_workflow_started", path=path)

    def is_ace_started(self, path: str) -> bool:
        """Check if auto_context was called recently (within 30 minutes)."""
        if path not in self._ace_started:
            return False
        # 30 minute window for a single task session
        return (time.time() - self._ace_started[path]) < 1800

    def require_ace_started(self, path: str, tool_name: str) -> dict | None:
        """Check if ACE was started. Returns error dict if not, None if OK.

        Use this for tool gating - returns an error response if ACE wasn't started.
        """
        if not self._gating_enabled:
            return None

        if tool_name in ACE_EXEMPT_TOOLS:
            return None

        if self.is_ace_started(path):
            return None

        # ACE not started - return error for tool gating
        return {
            "error": "ACE_WORKFLOW_REQUIRED",
            "message": (
                "You must call auto_context() before using this tool. "
                "ACE Framework ensures you get project-specific guidance before making changes."
            ),
            "action": f'auto_context(message="your task description", path="{path}")',
            "tool_blocked": tool_name,
        }

    def record_playbook_query(self, path: str) -> None:
        """Record that playbooks were queried for a project."""
        self._playbook_queries[path] = time.time()
        log.debug("ace_playbook_queried", path=path)

    def record_task_start(self, path: str, task_type: str) -> None:
        """Record that a coding task started."""
        self._pending_tasks[path] = {
            "task_type": task_type,
            "start_time": time.time(),
            "playbook_queried": path in self._playbook_queries
            and (time.time() - self._playbook_queries.get(path, 0)) < 300,  # 5 min window
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

    def clear_task(self, path: str) -> None:
        """Clear pending task after confirmation."""
        self._pending_tasks.pop(path, None)

    def was_playbook_queried(self, path: str) -> bool:
        """Check if playbook was queried recently for this project."""
        if path not in self._playbook_queries:
            return False
        # 5 minute window
        return (time.time() - self._playbook_queries[path]) < 300

    def get_last_activity(self) -> float:
        """Get timestamp of last activity on this tracker."""
        timestamps = []
        if self._playbook_queries:
            timestamps.append(max(self._playbook_queries.values()))
        if self._ace_started:
            timestamps.append(max(self._ace_started.values()))
        if self._pending_tasks:
            timestamps.extend(t.get("start_time", 0) for t in self._pending_tasks.values())
        return max(timestamps) if timestamps else 0


class ACEEnforcementManager:
    """Manages per-project ACE enforcement trackers.

    Ensures that each project has its own isolated ACE state,
    preventing cross-contamination when agents work on multiple
    projects concurrently.
    """

    def __init__(self) -> None:
        self._trackers: dict[str, ACEEnforcementTracker] = {}
        self._lock = threading.Lock()

    def get_tracker(self, project_path: str) -> ACEEnforcementTracker:
        """Get or create tracker for a project.

        Args:
            project_path: Absolute path to the project

        Returns:
            ACEEnforcementTracker for that project
        """
        # Normalize path
        normalized = str(Path(project_path).resolve())

        with self._lock:
            if normalized not in self._trackers:
                self._trackers[normalized] = ACEEnforcementTracker()
                log.debug("ace_tracker_created", project=normalized)
            return self._trackers[normalized]

    def cleanup_stale(self, max_age_seconds: int = 3600) -> int:
        """Remove trackers for projects inactive > max_age.

        Args:
            max_age_seconds: Maximum age in seconds (default 1 hour)

        Returns:
            Number of trackers removed
        """
        now = time.time()
        with self._lock:
            stale = [
                p
                for p, t in self._trackers.items()
                if (now - t.get_last_activity()) > max_age_seconds
            ]
            for p in stale:
                del self._trackers[p]
                log.debug("ace_tracker_cleaned", project=p)
            return len(stale)

    def list_projects(self) -> list[str]:
        """List all projects with active trackers."""
        with self._lock:
            return list(self._trackers.keys())

    def get_stats(self) -> dict:
        """Get statistics about tracked projects."""
        with self._lock:
            return {
                "active_projects": len(self._trackers),
                "projects": list(self._trackers.keys()),
            }


# Global manager instance (replaces single tracker)
_ace_manager = ACEEnforcementManager()


def get_ace_tracker(project_path: str | None = None) -> ACEEnforcementTracker:
    """Get ACE enforcement tracker for a project.

    Args:
        project_path: Project path. If None, uses current working directory.

    Returns:
        ACEEnforcementTracker for that project
    """
    path = project_path or str(Path.cwd())
    return _ace_manager.get_tracker(path)


def get_ace_manager() -> ACEEnforcementManager:
    """Get the global ACE enforcement manager."""
    return _ace_manager


def check_ace_gate(tool_name: str, path: str | None = None) -> str | None:
    """Check if ACE workflow was followed. Returns error JSON if not, None if OK.

    Call this at the start of tools that modify code or delegate work.
    Returns None if OK to proceed, or a JSON error string to return immediately.

    Args:
        tool_name: Name of the tool being called
        path: Project path (uses cwd if not provided)
    """
    project_path = str(Path(path).resolve()) if path else str(Path.cwd())
    tracker = get_ace_tracker(project_path)
    error = tracker.require_ace_started(project_path, tool_name)

    if error:
        return json.dumps({"result": error}, indent=2)
    return None


def inject_ace_reminder(response: str, project_path: str) -> str:
    """Inject ACE Framework reminder into tool response if needed.

    This ensures agents are consistently reminded to follow the ACE workflow.
    """
    tracker = get_ace_tracker(project_path)
    reminder = tracker.get_dynamic_reminder(project_path, response)
    if reminder:
        return response + reminder
    return response


async def auto_trigger_reflection(
    task_type: str,
    task_description: str,
    outcome: str,
    success: bool = True,
    project_path: str | None = None,
    applied_bullets: list[str] | None = None,
) -> dict | None:
    """
    Auto-trigger reflection for tool completions.

    This enables automatic learning from delegate/think/batch executions
    without requiring manual complete_task() calls.

    Args:
        task_type: Type of task (quick, analyze, etc.)
        task_description: What was requested
        outcome: Result summary
        success: Whether task succeeded
        project_path: Project directory
        applied_bullets: Bullet IDs used (if any)

    Returns:
        Reflection result dict or None if failed
    """
    try:
        from ..ace.reflector import get_reflector
        from ..ace.curator import get_curator

        proj_path = Path(project_path) if project_path else Path.cwd()
        reflector = get_reflector()
        curator = get_curator(str(proj_path))

        # Run reflection
        reflection = await reflector.reflect(
            task_description=task_description,
            task_type=task_type,
            task_succeeded=success,
            outcome=outcome,
            tool_calls=None,
            applied_bullets=applied_bullets or [],
            error_trace=None if success else outcome,
            user_feedback=None,
        )

        # Curate playbook with insights
        curation = await curator.curate(reflection, auto_prune=False)

        log.info(
            "auto_reflection_complete",
            task_type=task_type,
            success=success,
            insights=len(reflection.insights),
            bullets_added=curation.bullets_added,
        )

        return {
            "insights_extracted": len(reflection.insights),
            "bullets_added": curation.bullets_added,
            "dedup_prevented": curation.dedup_prevented,
        }
    except Exception as e:
        log.warning("auto_reflection_failed", task_type=task_type, error=str(e))
        return None
