# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Global context variables for Delia.
"""

from contextvars import ContextVar
from typing import Optional

# Context variable for current project path (for dynamic instructions and resources)
current_project_path: ContextVar[Optional[str]] = ContextVar("current_project_path", default=None)

def set_project_context(project_path: Optional[str]) -> None:
    """Set the current project path context."""
    current_project_path.set(project_path)
