# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Global context variables for Delia.

IMPORTANT: All modules should use get_project_path() instead of Path.cwd().
This ensures proper project isolation when Delia is used across multiple projects.
"""

from contextvars import ContextVar
from pathlib import Path
from typing import Optional

# Context variable for current project path (for dynamic instructions and resources)
current_project_path: ContextVar[Optional[str]] = ContextVar("current_project_path", default=None)


def set_project_context(project_path: Optional[str]) -> None:
    """Set the current project path context."""
    current_project_path.set(project_path)


def get_project_path(explicit_path: str | Path | None = None) -> Path:
    """Get the project path with proper priority.

    This is the CANONICAL function for resolving project paths. All modules
    should use this instead of Path.cwd() to ensure proper project isolation.

    Priority:
    1. Explicit path (if provided)
    2. Context variable (set by MCP server for active project)
    3. Current working directory (fallback only)

    Args:
        explicit_path: Optional explicit path override

    Returns:
        Resolved Path object for the project
    """
    if explicit_path is not None:
        return Path(explicit_path)

    ctx_path = current_project_path.get()
    if ctx_path:
        return Path(ctx_path)

    return Path.cwd()
