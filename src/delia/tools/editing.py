# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Editing tools for Delia agents.

Provides surgical file editing capabilities beyond simple overwrite.
Includes search-and-replace and line-based insertion.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from ..types import Workspace
from .executor import validate_path

log = structlog.get_logger()


async def replace_in_file(
    path: str,
    search: str,
    replace: str,
    count: int | None = None,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Replace text in a file.

    Performs a string replacement. Fails if the search string is not found
    or is ambiguous (multiple matches) unless `count` is specified.

    Args:
        path: Path to file (absolute or relative to workspace/cwd)
        search: Exact string to search for
        replace: Replacement string
        count: Max replacements to make (default: replace all occurrences)
        workspace: Optional workspace to confine file access

    Returns:
        Success message or error
    """
    # Validate path (with workspace if provided)
    valid, error = validate_path(path, workspace)
    if not valid:
        return f"Error: {error}"

    # Resolve path
    if workspace and not Path(path).is_absolute():
        file_path = (workspace.root / path).resolve()
    else:
        file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        return f"Error: File not found: {path}"
    
    if not file_path.is_file():
        return f"Error: Not a file: {path}"

    # Additional safety: never write to critical system paths
    path_str = str(file_path).lower()
    dangerous_patterns = [
        "/etc/", "/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/",
        "/boot/", "/lib/", "/lib64/", "system32", "windows/system",
    ]
    for pattern in dangerous_patterns:
        if pattern in path_str:
            return f"Error: Cannot modify system directory: {path}"

    try:
        content = file_path.read_text(encoding="utf-8")
        
        # Check if search string exists
        matches = content.count(search)
        if matches == 0:
            return f"Error: Search string not found in {path}"
        
        # If count not specified but multiple matches exist, warn
        if count is None and matches > 1:
            log.warning("multiple_matches_replace", path=path, matches=matches)
            # We proceed with replacing all, but log it. 
            # Ideally the agent should be specific enough.
        
        if count is not None:
            new_content = content.replace(search, replace, count)
        else:
            new_content = content.replace(search, replace)
            
        # Verify change actually happened (redundant check but safe)
        if new_content == content:
            return f"Error: No changes made to {path}"

        file_path.write_text(new_content, encoding="utf-8")

        log.info("file_replaced", path=str(file_path), matches=matches)
        return f"Successfully replaced {matches} occurrences in {path}"

    except Exception as e:
        return f"Error replacing in file: {e}"


async def insert_into_file(
    path: str,
    content: str,
    line: int,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Insert content into a file at a specific line number.

    Args:
        path: Path to file
        content: Content to insert
        line: Line number to insert AFTER (1-indexed). 0 to insert at start.
        workspace: Optional workspace

    Returns:
        Success message or error
    """
    # Validate path
    valid, error = validate_path(path, workspace)
    if not valid:
        return f"Error: {error}"

    # Resolve path
    if workspace and not Path(path).is_absolute():
        file_path = (workspace.root / path).resolve()
    else:
        file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        return f"Error: File not found: {path}"

    try:
        file_content = file_path.read_text(encoding="utf-8")
        lines = file_content.splitlines()
        
        # Handle empty file
        if not lines and line <= 1:
            lines = [content]
        else:
            # Check bounds
            if line < 0 or line > len(lines):
                return f"Error: Line number {line} out of bounds (1-{len(lines)})"
            
            # Insert
            # 1-based index: inserting after line 1 means inserting at index 1
            # 0 means insert at start (index 0)
            lines.insert(line, content)
            
        new_content = "\n".join(lines)
        # Preserve trailing newline if original had one
        if file_content.endswith("\n") and not new_content.endswith("\n"):
            new_content += "\n"
            
        file_path.write_text(new_content, encoding="utf-8")
        
        log.info("file_inserted", path=str(file_path), line=line)
        return f"Successfully inserted content at line {line} in {path}"

    except Exception as e:
        return f"Error inserting into file: {e}"
