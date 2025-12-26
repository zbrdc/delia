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
from typing import Any

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
    """Register supplementary tool handlers with FastMCP."""

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

        # Get all MCP tools from the server
        tools_dict = await mcp.get_tools()
        tools_info = list(tools_dict.values())

        # Group by category
        categorized: dict[str, list[dict]] = {cat: [] for cat in TOOL_CATEGORIES}

        # Categorization rules
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
            "search": ["semantic_search", "codebase_graph", "get_related_files", "explain_dependency", "project_overview"],
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

        summary = {
            "total_tools": sum(len(tools) for tools in categorized.values()),
            "categories": {
                cat: {
                    "description": TOOL_CATEGORIES[cat],
                    "count": len(tools),
                    "tools": [t["name"] for t in tools],
                }
                for cat, tools in categorized.items()
                if tools
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
