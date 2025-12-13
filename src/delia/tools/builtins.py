# Copyright (C) 2023 the project owner
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
Built-in tools for Delia agents.

Provides read-only tools for file operations, code search, and web fetching.
These are safe by default - no write operations.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
import structlog

from .executor import validate_path
from .registry import ToolDefinition, ToolRegistry

log = structlog.get_logger()

# Maximum file size to read (1MB)
MAX_FILE_SIZE = 1_000_000

# Maximum lines to return from file
MAX_LINES = 500

# Maximum grep results
MAX_GREP_RESULTS = 50


async def read_file(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read file contents from disk.

    Args:
        path: Path to file (absolute or relative to cwd)
        start_line: First line to read (1-indexed)
        end_line: Last line to read (None for all remaining)

    Returns:
        File contents with line numbers
    """
    # Validate path
    valid, error = validate_path(path)
    if not valid:
        return f"Error: {error}"

    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        return f"Error: File not found: {path}"

    if not file_path.is_file():
        return f"Error: Not a file: {path}"

    # Check size
    size = file_path.stat().st_size
    if size > MAX_FILE_SIZE:
        return f"Error: File too large ({size:,} bytes > {MAX_FILE_SIZE:,} limit)"

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    # Split into lines and apply range
    lines = content.splitlines()
    total_lines = len(lines)

    # Adjust indices (1-indexed to 0-indexed)
    start_idx = max(0, start_line - 1)
    end_idx = end_line if end_line else total_lines

    # Limit lines
    if end_idx - start_idx > MAX_LINES:
        end_idx = start_idx + MAX_LINES

    selected_lines = lines[start_idx:end_idx]

    # Format with line numbers
    result_lines = []
    for i, line in enumerate(selected_lines, start=start_idx + 1):
        result_lines.append(f"{i:6d}│ {line}")

    header = f"# {path} (lines {start_idx + 1}-{start_idx + len(selected_lines)} of {total_lines})\n"
    return header + "\n".join(result_lines)


async def list_directory(
    path: str = ".",
    recursive: bool = False,
    pattern: str | None = None,
) -> str:
    """List files in a directory.

    Args:
        path: Directory path (defaults to current directory)
        recursive: If True, list subdirectories recursively
        pattern: Glob pattern to filter files (e.g., "*.py")

    Returns:
        Formatted directory listing
    """
    # Validate path
    valid, error = validate_path(path)
    if not valid:
        return f"Error: {error}"

    dir_path = Path(path).expanduser().resolve()

    if not dir_path.exists():
        return f"Error: Directory not found: {path}"

    if not dir_path.is_dir():
        return f"Error: Not a directory: {path}"

    try:
        if pattern:
            glob_pattern = f"**/{pattern}" if recursive else pattern
            files = list(dir_path.glob(glob_pattern))
        elif recursive:
            files = list(dir_path.rglob("*"))
        else:
            files = list(dir_path.iterdir())

        # Sort and limit
        files = sorted(files)[:500]  # Limit to 500 entries

        result_lines = [f"# Directory: {path}"]
        if pattern:
            result_lines.append(f"# Pattern: {pattern}")
        result_lines.append("")

        for f in files:
            rel_path = f.relative_to(dir_path)
            if f.is_dir():
                result_lines.append(f"📁 {rel_path}/")
            else:
                size = f.stat().st_size
                result_lines.append(f"   {rel_path} ({size:,} bytes)")

        result_lines.append(f"\n# Total: {len(files)} items")
        return "\n".join(result_lines)

    except Exception as e:
        return f"Error listing directory: {e}"


async def search_code(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    context_lines: int = 2,
) -> str:
    """Search for a pattern in files using grep.

    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        file_pattern: Glob pattern to filter files (e.g., "*.py")
        context_lines: Number of context lines before/after match

    Returns:
        Matching lines with context
    """
    # Validate path
    valid, error = validate_path(path)
    if not valid:
        return f"Error: {error}"

    search_path = Path(path).expanduser().resolve()

    if not search_path.exists():
        return f"Error: Path not found: {path}"

    # Build grep command
    cmd = ["grep", "-rn", "--color=never"]

    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])

    if file_pattern:
        cmd.extend(["--include", file_pattern])

    # Exclude common non-code directories
    for exclude in [".git", "node_modules", "__pycache__", ".venv", "venv"]:
        cmd.extend(["--exclude-dir", exclude])

    cmd.extend([pattern, str(search_path)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 1:
            return f"No matches found for pattern: {pattern}"

        if result.returncode != 0 and result.returncode != 1:
            return f"Search error: {result.stderr}"

        # Limit results
        lines = result.stdout.splitlines()
        if len(lines) > MAX_GREP_RESULTS * (1 + 2 * context_lines):
            lines = lines[:MAX_GREP_RESULTS * (1 + 2 * context_lines)]
            lines.append(f"\n... Results truncated to {MAX_GREP_RESULTS} matches")

        header = f"# Search: {pattern} in {path}"
        if file_pattern:
            header += f" (files: {file_pattern})"
        header += "\n"

        return header + "\n".join(lines)

    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error searching: {e}"


async def web_fetch(
    url: str,
    extract_text: bool = True,
) -> str:
    """Fetch content from a URL.

    Args:
        url: URL to fetch
        extract_text: If True, extract text from HTML

    Returns:
        URL content (text extracted if HTML)
    """
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "text/html" in content_type and extract_text:
                # Basic HTML text extraction
                text = response.text
                # Remove script and style tags
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                # Limit size
                if len(text) > 50000:
                    text = text[:50000] + "\n\n... [Content truncated]"
                return f"# Content from {url}\n\n{text}"
            else:
                # Return raw content
                content = response.text
                if len(content) > 50000:
                    content = content[:50000] + "\n\n... [Content truncated]"
                return f"# Content from {url}\n\n{content}"

    except httpx.TimeoutException:
        return f"Error: Request to {url} timed out"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} from {url}"
    except Exception as e:
        return f"Error fetching URL: {e}"


def get_default_tools() -> ToolRegistry:
    """Get registry with default built-in tools.

    Returns:
        ToolRegistry with read_file, list_directory, search_code, web_fetch
    """
    registry = ToolRegistry()

    registry.register(ToolDefinition(
        name="read_file",
        description="Read contents of a file from disk. Use this to examine code, configs, or data files.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (absolute or relative to current directory)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read (1-indexed, default: 1)",
                    "default": 1
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read (default: read all)",
                }
            },
            "required": ["path"]
        },
        handler=read_file,
    ))

    registry.register(ToolDefinition(
        name="list_directory",
        description="List files and directories. Use this to explore the codebase structure.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: current directory)",
                    "default": "."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Include subdirectories (default: false)",
                    "default": False
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py')"
                }
            },
        },
        handler=list_directory,
    ))

    registry.register(ToolDefinition(
        name="search_code",
        description="Search for a pattern in files using regex. Use this to find code, functions, or text.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory)",
                    "default": "."
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py', '*.ts')"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around matches (default: 2)",
                    "default": 2
                }
            },
            "required": ["pattern"]
        },
        handler=search_code,
    ))

    registry.register(ToolDefinition(
        name="web_fetch",
        description="Fetch content from a URL. Use this to read documentation or API responses.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch (must start with http:// or https://)"
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Extract text from HTML (default: true)",
                    "default": True
                }
            },
            "required": ["url"]
        },
        handler=web_fetch,
    ))

    return registry
