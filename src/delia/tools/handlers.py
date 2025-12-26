# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Additional MCP tool handlers for Delia.

This module registers supplementary tools (git, bulk file ops, discovery).
Core tools are in separate modules:
- framework.py: Delia Framework core (auto_context, complete_task, etc.)
- delegation.py: Local model delegation (delegate, think, batch, etc.)
- semantic.py: Semantic search (semantic_search, get_related_files, etc.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import structlog
from fastmcp import FastMCP

from ..container import get_container
from ..context import get_project_path

# Import enforcement manager for framework admin
from .handlers_enforcement import (
    EnforcementManager,
    get_manager,
)

log = structlog.get_logger()

# Re-export for backwards compatibility
_enforcement_manager = get_manager()


def register_tool_handlers(mcp: FastMCP):
    """Register supplementary tool handlers with FastMCP.

    ADR-010: Most tools consolidated into action-based patterns:
    - git(action="log|blame|show") - Consolidated git operations
    - admin(action="tools|describe|framework_stats|framework_cleanup") - See consolidated.py
    - read_files, edit_files - Removed (use read_file/edit_file multiple times)
    """

    # =========================================================================
    # Git Operations (Consolidated - ADR-010)
    # =========================================================================

    @mcp.tool()
    async def git(
        action: Literal["log", "blame", "show"],
        file: str | None = None,
        commit: str | None = None,
        path: str = ".",
        n: int = 10,
        since: str | None = None,
        author: str | None = None,
        oneline: bool = False,
        start_line: int | None = None,
        end_line: int | None = None,
        stat: bool = False,
    ) -> str:
        """Git operations: log, blame, show.

        Actions:
        - log: Show commit history (optional: file, n, since, author, oneline)
        - blame: Show line-by-line authorship (requires file; optional: start_line, end_line)
        - show: Show commit details and diff (requires commit; optional: file, stat)
        """
        from .coding import git_log as git_log_impl
        from .coding import git_blame as git_blame_impl
        from .coding import git_show as git_show_impl

        if action == "log":
            result = await git_log_impl(path, file, n, since, author, oneline)
            return json.dumps({"result": result})

        elif action == "blame":
            if not file:
                return json.dumps({"error": "blame requires file parameter"})
            result = await git_blame_impl(file, path, start_line, end_line)
            return json.dumps({"result": result})

        elif action == "show":
            if not commit:
                return json.dumps({"error": "show requires commit parameter"})
            result = await git_show_impl(commit, file, path, stat)
            return json.dumps({"result": result})

        return json.dumps({"error": f"Unknown action: {action}"})
