# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Delia Framework enforcement and tracking for MCP tools.

Provides:
- EnforcementTracker: Per-project workflow tracking
- EnforcementManager: Thread-safe manager for multiple projects
- Helper functions for context gating and reminders
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
# Delia Framework Dynamic Enforcement
# =============================================================================

# Tools that can be used without calling auto_context first
EXEMPT_TOOLS = {
    "auto_context",
    "check_status",
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

# Tools that REQUIRE think_about_task_adherence() before use
# These modify files and need the checkpoint to prevent drift
CHECKPOINT_REQUIRED_TOOLS = {
    # File modification tools
    "write_file",
    "edit_file",
    "delete_file",
    # LSP modification tools
    "lsp_replace_symbol_body",
    "lsp_insert_before_symbol",
    "lsp_insert_after_symbol",
    "lsp_rename_symbol",
}

# Project root markers (in priority order)
PROJECT_MARKERS = [".delia", ".git", "pyproject.toml", "package.json", "Cargo.toml"]


def resolve_project_path(path: str | None) -> str:
    """Resolve a file or directory path to its project root.

    Args:
        path: File or directory path. If None, uses cwd.

    Returns:
        Project root directory path (string)
    """
    if path is None:
        return str(Path.cwd())

    p = Path(path).resolve()

    # If path is a file, start from its parent directory
    if p.is_file():
        p = p.parent

    # Walk up looking for project markers
    for parent in [p, *p.parents]:
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                return str(parent)

    # No marker found - return the resolved directory
    return str(p)


from ..config import config

class EnforcementTracker:
    """Track Delia Framework compliance dynamically across tool calls.

    Provides two enforcement mechanisms:
    1. Soft enforcement: record_playbook_query() and check_compliance()
    2. Hard enforcement: record_context_started() and require_context_started()
    """

    def __init__(self) -> None:
        self._playbook_queries: dict[str, float] = {}  # path -> last query timestamp
        self._context_started: dict[str, float] = {}  # path -> timestamp when auto_context called
        self._pending_tasks: dict[str, dict] = {}  # path -> {task, start_time}
        self._compliance_warnings: int = 0
        self._gating_enabled: bool = config.strict_mode  # Controlled by config
        # Checkpoint tracking for hard gating
        self._checkpoint_called: dict[str, float] = {}  # path -> timestamp when think_about_task_adherence called
        # Phase tracking: search → checkpoint → modify
        self._current_phase: dict[str, str] = {}  # path -> current phase (search/checkpoint/modify)
        self._search_count: dict[str, int] = {}  # path -> number of search/read operations
        # Context-shift detection
        self._last_task_type: dict[str, str] = {}  # path -> last detected task type
        # Detection feedback tracking
        self._last_message: dict[str, str] = {}  # path -> original message for feedback

    def record_context_started(
        self, path: str, task_type: str | None = None, message: str | None = None
    ) -> None:
        """Record that auto_context was called for a project."""
        self._context_started[path] = time.time()
        self._playbook_queries[path] = time.time()  # Also counts as playbook query
        if task_type:
            self._last_task_type[path] = task_type
        if message:
            self._last_message[path] = message
        log.info("workflow_started", path=path, task_type=task_type)

    def get_detection_context(self, path: str) -> tuple[str | None, str | None]:
        """Get the last detected task type and message for feedback purposes."""
        return self._last_task_type.get(path), self._last_message.get(path)

    def is_context_started(self, path: str) -> bool:
        """Check if auto_context was called recently (within 30 minutes)."""
        if path not in self._context_started:
            return False
        # 30 minute window for a single task session
        return (time.time() - self._context_started[path]) < 1800

    def require_context_started(self, path: str, tool_name: str) -> dict | None:
        """Check if context was started. Returns error dict if not, None if OK.

        Use this for tool gating - returns an error response if context wasn't started.
        """
        if not self._gating_enabled:
            return None

        if tool_name in EXEMPT_TOOLS:
            return None

        if self.is_context_started(path):
            return None

        # Context not started - return error for tool gating
        return {
            "error": "WORKFLOW_REQUIRED",
            "message": (
                "⛔ [FRAMEWORK REJECTION] Workflow Violation Detected.\n"
                "You attempted to use a tool without establishing project context.\n"
                "MANDATORY: You MUST call `auto_context()` first to load project patterns.\n"
                "IF YOU DO NOT FOLLOW THESE RULES IN PLACE IT WILL MAKE ME EXTREMELY SAD WITH YOU."
            ),
            "action": f'auto_context(message="your task description", path="{path}")',
            "tool_blocked": tool_name,
        }

    def record_playbook_query(self, path: str) -> None:
        """Record that playbooks were queried for a project."""
        self._playbook_queries[path] = time.time()
        log.debug("playbook_queried", path=path)

    def record_task_start(self, path: str, task_type: str) -> None:
        """Record that a coding task started."""
        self._pending_tasks[path] = {
            "task_type": task_type,
            "start_time": time.time(),
            "playbook_queried": path in self._playbook_queries
            and (time.time() - self._playbook_queries.get(path, 0)) < 300,  # 5 min window
        }

    def check_compliance(self, path: str) -> dict | None:
        """Check if Delia workflow was followed. Returns warning if not."""
        pending = self._pending_tasks.get(path)
        if not pending:
            return None

        if not pending.get("playbook_queried"):
            self._compliance_warnings += 1
            return {
                "warning": "PLAYBOOK_NOT_QUERIED",
                "message": f"Started {pending['task_type']} task without querying playbooks first.",
                "action_required": f"Call get_playbook(task_type='{pending['task_type']}', path='{path}') before coding.",
                "total_warnings": self._compliance_warnings,
            }
        return None

    def get_dynamic_reminder(self, path: str, response: str, tool_name: str | None = None) -> str | None:
        """Generate professional guidance to inject into response."""
        warning = self.check_compliance(path)
        if warning:
            return (
                f"\n\n[GUIDANCE] Delia context: {warning['message']}\n"
                f"Action: {warning['action_required']}\n"
                f"Call complete_task() once the implementation is verified."
            )

        # Check for context shift if tool name provided
        if tool_name:
            shift_reminder = self.check_context_shift(path, tool_name)
            if shift_reminder:
                return shift_reminder

        # Check if task completed without confirmation
        pending = self._pending_tasks.get(path)
        if pending and (time.time() - pending.get("start_time", 0)) > 60:  # Task > 1 min
            return (
                "\n\n[STRATEGY] Task appears substantially complete. "
                "Close the loop with complete_task(success=True, ...) to record learned patterns."
            )
        return None

    def clear_task(self, path: str) -> None:
        """Clear pending task after confirmation."""
        self._pending_tasks.pop(path, None)
        # Also clear checkpoint when task completes
        self._checkpoint_called.pop(path, None)
        self._current_phase.pop(path, None)
        self._search_count.pop(path, None)

    # =========================================================================
    # Context-Shift Detection
    # =========================================================================

    # Tool patterns that imply specific task types
    TOOL_TASK_HINTS: dict[str, str] = {
        # Git tools
        "git_commit": "git", "git_push": "git", "git_status": "git",
        # Testing tools
        "pytest": "testing", "run_tests": "testing",
        # Debug tools
        "debug": "debugging", "breakpoint": "debugging",
        # Deployment
        "docker": "deployment", "deploy": "deployment",
    }

    def check_context_shift(self, path: str, tool_name: str) -> str | None:
        """Check if current tool implies a different task type than last auto_context."""
        last_type = self._last_task_type.get(path)
        if not last_type:
            return None

        implied_type = None
        tool_lower = tool_name.lower()

        for pattern, task_type in self.TOOL_TASK_HINTS.items():
            if pattern in tool_lower:
                implied_type = task_type
                break

        if tool_name in CHECKPOINT_REQUIRED_TOOLS and last_type in ("project", "debugging"):
            implied_type = "coding"

        if implied_type and implied_type != last_type:
            return (
                f"\n\n[CONTEXT] Shift detected: Current work ({implied_type}) differs from active context ({last_type}).\n"
                f"Refresh context: auto_context(message='...') to load relevant patterns.\n"
            )
        return None

    # =========================================================================
    # Checkpoint Tracking (Hard Gating)
    # =========================================================================

    def record_checkpoint_called(self, path: str) -> None:
        """Record that think_about_task_adherence() was called."""
        self._checkpoint_called[path] = time.time()
        self._current_phase[path] = "checkpoint"
        log.info("checkpoint_called", path=path)

    def is_checkpoint_valid(self, path: str) -> bool:
        """Check if checkpoint was called recently (within 10 minutes)."""
        if path not in self._checkpoint_called:
            return False
        return (time.time() - self._checkpoint_called[path]) < 600

    def require_checkpoint(self, path: str, tool_name: str) -> dict | None:
        """Check if checkpoint was called. Returns blocking error if not."""
        if not self._gating_enabled:
            return None

        if tool_name not in CHECKPOINT_REQUIRED_TOOLS:
            return None

        if self.is_checkpoint_valid(path):
            self._current_phase[path] = "modify"
            return None

        return {
            "error": "CHECKPOINT_REQUIRED",
            "message": (
                f"⛔ [SAFETY LOCK] Write access denied for {tool_name}.\n"
                "MANDATORY: You MUST call `think_about_task_adherence()` before modifying files.\n"
                "This ensures your changes align with the project's architectural standards.\n"
                "IF YOU DO NOT FOLLOW THESE RULES IN PLACE IT WILL MAKE ME EXTREMELY SAD WITH YOU."
            ),
            "action": "think_about_task_adherence()",
            "tool_blocked": tool_name,
            "reason": "Hard gating prevents modifications without validation checkpoint.",
        }

    # =========================================================================
    # Phase Tracking (search → checkpoint → modify)
    # =========================================================================

    def record_search_operation(self, path: str) -> None:
        """Record a search/read operation."""
        self._current_phase[path] = "search"
        self._search_count[path] = self._search_count.get(path, 0) + 1

    def get_phase_warning(self, path: str, tool_name: str) -> str | None:
        """Get warning if phase transition is suspicious."""
        phase = self._current_phase.get(path, "unknown")
        search_count = self._search_count.get(path, 0)

        if tool_name in CHECKPOINT_REQUIRED_TOOLS:
            if phase == "search" and search_count >= 3:
                return (
                    "[GUIDANCE] High search volume detected without validation. "
                    "Call think_about_collected_info() or think_about_task_adherence() to verify state.\n\n"
                )
        return None

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
        if self._context_started:
            timestamps.append(max(self._context_started.values()))
        if self._pending_tasks:
            timestamps.extend(t.get("start_time", 0) for t in self._pending_tasks.values())
        return max(timestamps) if timestamps else 0


class EnforcementManager:
    """Manages per-project Delia Framework enforcement trackers.

    Ensures that each project has its own isolated enforcement state,
    preventing cross-contamination when agents work on multiple
    projects concurrently.
    """

    def __init__(self) -> None:
        self._trackers: dict[str, EnforcementTracker] = {}
        self._lock = threading.Lock()

    def get_tracker(self, project_path: str) -> EnforcementTracker:
        """Get or create tracker for a project.

        Args:
            project_path: Absolute path to the project

        Returns:
            EnforcementTracker for that project
        """
        # Normalize path
        normalized = str(Path(project_path).resolve())

        with self._lock:
            if normalized not in self._trackers:
                self._trackers[normalized] = EnforcementTracker()
                log.debug("tracker_created", project=normalized)
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
                log.debug("tracker_cleaned", project=p)
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
_enforcement_manager = EnforcementManager()


def get_tracker(project_path: str | None = None) -> EnforcementTracker:
    """Get enforcement tracker for a project.

    Args:
        project_path: Project path. If None, uses current working directory.

    Returns:
        EnforcementTracker for that project
    """
    path = project_path or str(Path.cwd())
    return _enforcement_manager.get_tracker(path)


def get_manager() -> EnforcementManager:
    """Get the global enforcement manager."""
    return _enforcement_manager


def check_context_gate(tool_name: str, path: str | None = None) -> str | None:
    """Check if Delia workflow was followed. Returns error JSON if not, None if OK.

    Call this at the start of tools that modify code or delegate work.
    Returns None if OK to proceed, or a JSON error string to return immediately.

    Args:
        tool_name: Name of the tool being called
        path: File or project path (resolves to project root)
    """
    project_path = resolve_project_path(path)
    tracker = get_tracker(project_path)
    error = tracker.require_context_started(project_path, tool_name)

    if error:
        return json.dumps({"result": error}, indent=2)
    return None


def check_checkpoint_gate(tool_name: str, path: str | None = None) -> str | None:
    """Check if checkpoint was called before file modification. HARD GATE.

    Call this at the start of file modification tools.
    Returns None if OK to proceed, or a JSON error string that BLOCKS the tool.

    Args:
        tool_name: Name of the tool being called
        path: File or project path (resolves to project root)
    """
    project_path = resolve_project_path(path)
    tracker = get_tracker(project_path)
    error = tracker.require_checkpoint(project_path, tool_name)

    if error:
        return json.dumps({"result": error}, indent=2)
    return None


def record_checkpoint(path: str | None = None) -> None:
    """Record that think_about_task_adherence was called.

    Call this from the think_about_task_adherence tool handler.
    """
    project_path = resolve_project_path(path)
    tracker = get_tracker(project_path)
    tracker.record_checkpoint_called(project_path)


def record_search(path: str | None = None) -> None:
    """Record a search/read operation for phase tracking.

    Call this from search tools (grep, glob, read_file, etc.)
    """
    project_path = resolve_project_path(path)
    tracker = get_tracker(project_path)
    tracker.record_search_operation(project_path)


def get_phase_injection(tool_name: str, path: str | None = None) -> str | None:
    """Get phase warning to inject into response if applicable.

    Returns warning text or None.
    """
    project_path = resolve_project_path(path)
    tracker = get_tracker(project_path)
    return tracker.get_phase_warning(project_path, tool_name)


def inject_reminder(response: str, project_path: str, tool_name: str | None = None) -> str:
    """Inject Delia Framework reminder into tool response if needed.

    This ensures agents are consistently reminded to follow the workflow.
    """
    tracker = get_tracker(project_path)
    reminder = tracker.get_dynamic_reminder(project_path, response, tool_name)
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
        from ..learning.reflector import get_reflector
        from ..learning.curator import get_curator

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
