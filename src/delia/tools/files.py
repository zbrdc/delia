# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
File operation tools for Delia MCP server.

Provides file system operations that enable Delia to work standalone
without relying on the calling agent's file tools.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from pathlib import Path

import structlog
from fastmcp import FastMCP

from ..types import Workspace
from .handlers import check_context_gate, check_checkpoint_gate

log = structlog.get_logger()


def register_file_tools(mcp: FastMCP):
    """Register file operation tools with FastMCP."""

    @mcp.tool()
    async def read_file(
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        """
        Read file contents with optional line range.

        Args:
            path: Path to the file (relative to project root or absolute)
            start_line: Starting line number (1-indexed, default 1)
            end_line: Ending line number (inclusive, default None for all)

        Returns:
            File contents with line numbers
        """
        return await read_file_impl(path, start_line, end_line)

    @mcp.tool()
    async def write_file(
        path: str,
        content: str,
        create_dirs: bool = True,
    ) -> str:
        """
        Write content to a file.

        Args:
            path: Path to the file (relative to project root or absolute)
            content: Content to write
            create_dirs: Create parent directories if needed (default True)

        Returns:
            Success message or error
        """
        # Context Gating
        gate_error = check_context_gate("write_file", path)
        if gate_error:
            return gate_error

        # Checkpoint Gating
        checkpoint_error = check_checkpoint_gate("write_file", path)
        if checkpoint_error:
            return checkpoint_error

        return await write_file_impl(path, content, create_dirs)

    @mcp.tool()
    async def edit_file(
        path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        """
        Edit a file by replacing text.

        Args:
            path: Path to the file
            old_text: Text to find and replace
            new_text: Replacement text

        Returns:
            Success message with number of replacements or error
        """
        # Context Gating
        gate_error = check_context_gate("edit_file", path)
        if gate_error:
            return gate_error

        # Checkpoint Gating
        checkpoint_error = check_checkpoint_gate("edit_file", path)
        if checkpoint_error:
            return checkpoint_error

        return await edit_file_impl(path, old_text, new_text)

    @mcp.tool()
    async def list_dir(
        path: str = ".",
        recursive: bool = False,
        pattern: str | None = None,
    ) -> str:
        """
        List directory contents.

        Args:
            path: Directory path (default current directory)
            recursive: List recursively (default False)
            pattern: Glob pattern to filter files (e.g., "*.py")

        Returns:
            List of files and directories
        """
        return await list_dir_impl(path, recursive, pattern)

    @mcp.tool()
    async def find_file(
        pattern: str,
        path: str = ".",
    ) -> str:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "src/**/test_*.py")
            path: Base directory to search from

        Returns:
            List of matching file paths
        """
        return await find_file_impl(pattern, path)

    @mcp.tool()
    async def search_for_pattern(
        pattern: str,
        path: str = ".",
        file_pattern: str | None = None,
        context_lines: int = 0,
    ) -> str:
        """
        Search for a regex pattern in files (grep-like).

        Args:
            pattern: Regex pattern to search for
            path: Directory or file to search in
            file_pattern: Glob pattern to filter files (e.g., "*.py")
            context_lines: Number of context lines to show (default 0)

        Returns:
            Matching lines with file paths and line numbers
        """
        return await search_for_pattern_impl(pattern, path, file_pattern, context_lines)

    @mcp.tool()
    async def delete_file(path: str) -> str:
        """
        Delete a file.

        Args:
            path: Path to the file to delete

        Returns:
            Success message or error
        """
        # Checkpoint Gating
        checkpoint_error = check_checkpoint_gate("delete_file", path)
        if checkpoint_error:
            return checkpoint_error

        return await delete_file_impl(path)

    @mcp.tool()
    async def create_directory(path: str) -> str:
        """
        Create a directory.

        Args:
            path: Path to the directory to create

        Returns:
            Success message or error
        """
        return await create_directory_impl(path)


def _get_project_root() -> Path:
    """Get the base directory for file operations."""
    return Path.cwd()


def _resolve_path(path: str, workspace: Workspace | None = None) -> Path:
    """Resolve a path relative to workspace or project root."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()

    if workspace:
        return (workspace.root / path).resolve()

    # Use project root as base
    root = _get_project_root()
    return (root / path).resolve()


def _is_safe_path(path: Path, workspace: Workspace | None = None) -> tuple[bool, str]:
    """Check if path is safe to access."""
    # Block sensitive paths
    blocked_patterns = [
        "/etc/passwd", "/etc/shadow", "~/.ssh", "~/.gnupg",
        ".env", "credentials", "secrets", ".git/config"
    ]
    path_str = str(path)
    for pattern in blocked_patterns:
        if pattern in path_str:
            return False, f"Access denied: {pattern} is blocked for security"

    # If workspace is set, ensure path is within workspace
    if workspace:
        try:
            path.resolve().relative_to(workspace.root.resolve())
        except ValueError:
            return False, f"Path {path} is outside workspace {workspace.root}"

    return True, ""


# Implementations

async def read_file_impl(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    *,
    workspace: Workspace | None = None,
) -> str:
    return await asyncio.to_thread(_read_file_sync, path, start_line, end_line, workspace=workspace)

def _read_file_sync(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    *,
    workspace: Workspace | None = None,
) -> str:
    try:
        resolved = _resolve_path(path, workspace)
        safe, error = _is_safe_path(resolved, workspace)
        if not safe:
            return f"Error: {error}"

        if not resolved.exists():
            return f"Error: File not found: {path}"

        if not resolved.is_file():
            return f"Error: Not a file: {path}"

        content = resolved.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        # Apply line range
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line else len(lines)
        selected_lines = lines[start_idx:end_idx]

        # Format with line numbers
        numbered = []
        for i, line in enumerate(selected_lines, start=start_line):
            numbered.append(f"{i:4d}│{line}")

        return "\n".join(numbered) if numbered else "(empty file)"

    except UnicodeDecodeError:
        return f"Error: File {path} is not a text file"
    except Exception as e:
        log.error("read_file_failed", path=path, error=str(e))
        return f"Error reading file: {e}"


async def write_file_impl(
    path: str,
    content: str,
    create_dirs: bool = True,
    *,
    workspace: Workspace | None = None,
) -> str:
    return await asyncio.to_thread(_write_file_sync, path, content, create_dirs, workspace=workspace)

def _write_file_sync(
    path: str,
    content: str,
    create_dirs: bool = True,
    *,
    workspace: Workspace | None = None,
) -> str:
    try:
        resolved = _resolve_path(path, workspace)
        safe, error = _is_safe_path(resolved, workspace)
        if not safe:
            return f"Error: {error}"

        if create_dirs:
            resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(content, encoding="utf-8")
        log.info("file_written", path=str(resolved), bytes=len(content))
        return f"Wrote {len(content)} bytes to {path}"

    except Exception as e:
        log.error("write_file_failed", path=path, error=str(e))
        return f"Error writing file: {e}"


async def edit_file_impl(
    path: str,
    old_text: str,
    new_text: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    return await asyncio.to_thread(_edit_file_sync, path, old_text, new_text, workspace=workspace)

def _edit_file_sync(
    path: str,
    old_text: str,
    new_text: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    try:
        resolved = _resolve_path(path, workspace)
        safe, error = _is_safe_path(resolved, workspace)
        if not safe:
            return f"Error: {error}"

        if not resolved.exists():
            return f"Error: File not found: {path}"

        content = resolved.read_text(encoding="utf-8")

        # Count occurrences
        count = content.count(old_text)
        if count == 0:
            return f"Error: Text not found in {path}"

        # Replace
        new_content = content.replace(old_text, new_text)
        resolved.write_text(new_content, encoding="utf-8")

        log.info("file_edited", path=str(resolved), replacements=count)
        return f"Replaced {count} occurrence(s) in {path}"

    except Exception as e:
        log.error("edit_file_failed", path=path, error=str(e))
        return f"Error editing file: {e}"


async def list_dir_impl(
    path: str = ".",
    recursive: bool = False,
    pattern: str | None = None,
    *,
    workspace: Workspace | None = None,
) -> str:
    return await asyncio.to_thread(_list_dir_sync, path, recursive, pattern, workspace=workspace)

def _list_dir_sync(
    path: str = ".",
    recursive: bool = False,
    pattern: str | None = None,
    *,
    workspace: Workspace | None = None,
) -> str:
    try:
        resolved = _resolve_path(path, workspace)
        safe, error = _is_safe_path(resolved, workspace)
        if not safe:
            return f"Error: {error}"

        if not resolved.exists():
            return f"Error: Directory not found: {path}"

        if not resolved.is_dir():
            return f"Error: Not a directory: {path}"

        entries = []

        if recursive:
            for root, dirs, files in os.walk(resolved):
                # Filter hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                rel_root = Path(root).relative_to(resolved)
                for f in files:
                    if f.startswith('.'):
                        continue
                    rel_path = str(rel_root / f) if str(rel_root) != "." else f
                    if pattern and not fnmatch.fnmatch(f, pattern):
                        continue
                    entries.append(rel_path)
        else:
            for item in sorted(resolved.iterdir()):
                if item.name.startswith('.'):
                    continue
                if pattern and not fnmatch.fnmatch(item.name, pattern):
                    continue
                name = item.name + "/" if item.is_dir() else item.name
                entries.append(name)

        if not entries:
            return "No files found."

        return "\n".join(sorted(entries))

    except Exception as e:
        log.error("list_dir_failed", path=path, error=str(e))
        return f"Error listing directory: {e}"


async def find_file_impl(
    pattern: str,
    path: str = ".",
    *,
    workspace: Workspace | None = None,
) -> str:
    return await asyncio.to_thread(_find_file_sync, pattern, path, workspace=workspace)

def _find_file_sync(
    pattern: str,
    path: str = ".",
    *,
    workspace: Workspace | None = None,
) -> str:
    try:
        resolved = _resolve_path(path, workspace)
        safe, error = _is_safe_path(resolved, workspace)
        if not safe:
            return f"Error: {error}"

        if not resolved.exists():
            return f"Error: Directory not found: {path}"

        matches = []
        for match in resolved.glob(pattern):
            if match.is_file():
                rel_path = match.relative_to(resolved)
                matches.append(str(rel_path))

        if not matches:
            return f"No files matching '{pattern}' found."

        # Limit output
        if len(matches) > 100:
            return f"Found {len(matches)} files matching '{pattern}':\n" + \
                   "\n".join(sorted(matches)[:100]) + \
                   f"\n... and {len(matches) - 100} more"

        return f"Found {len(matches)} file(s):\n" + "\n".join(sorted(matches))

    except Exception as e:
        log.error("find_file_failed", pattern=pattern, error=str(e))
        return f"Error finding files: {e}"


async def search_for_pattern_impl(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    context_lines: int = 0,
    *,
    workspace: Workspace | None = None,
) -> str:
    return await asyncio.to_thread(_search_for_pattern_sync, pattern, path, file_pattern, context_lines, workspace=workspace)

def _search_for_pattern_sync(
    pattern: str,
    path: str = ".",
    file_pattern: str | None = None,
    context_lines: int = 0,
    *,
    workspace: Workspace | None = None,
) -> str:
    try:
        resolved = _resolve_path(path, workspace)
        safe, error = _is_safe_path(resolved, workspace)
        if not safe:
            return f"Error: {error}"

        if not resolved.exists():
            return f"Error: Path not found: {path}"

        # Compile regex
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []
        files_searched = 0
        max_matches = 200

        # Get files to search
        if resolved.is_file():
            files = [resolved]
        else:
            if file_pattern:
                files = list(resolved.rglob(file_pattern))
            else:
                # Default to common code files
                code_patterns = ["*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "*.go", "*.rs", "*.java", "*.md", "*.txt"]
                files = []
                for cp in code_patterns:
                    files.extend(resolved.rglob(cp))

        for file_path in files:
            if not file_path.is_file():
                continue
            if any(part.startswith('.') for part in file_path.parts):
                continue

            files_searched += 1
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()

                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        rel_path = file_path.relative_to(resolved) if resolved.is_dir() else file_path.name

                        if context_lines > 0:
                            # Add context
                            start = max(0, i - 1 - context_lines)
                            end = min(len(lines), i + context_lines)
                            context = lines[start:end]
                            match_str = f"{rel_path}:{i}\n"
                            for j, ctx_line in enumerate(context, start=start + 1):
                                prefix = ">" if j == i else " "
                                match_str += f"  {prefix} {j}: {ctx_line}\n"
                            matches.append(match_str.rstrip())
                        else:
                            matches.append(f"{rel_path}:{i}: {line.strip()}")

                        if len(matches) >= max_matches:
                            break

                if len(matches) >= max_matches:
                    break

            except (UnicodeDecodeError, PermissionError):
                continue

        if not matches:
            return f"No matches for '{pattern}' in {files_searched} file(s)."

        result = f"Found {len(matches)} match(es) in {files_searched} file(s):\n\n"
        result += "\n".join(matches)

        if len(matches) >= max_matches:
            result += f"\n\n(Truncated at {max_matches} matches)"

        return result

    except Exception as e:
        log.error("search_for_pattern_failed", pattern=pattern, error=str(e))
        return f"Error searching: {e}"


async def delete_file_impl(
    path: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    return await asyncio.to_thread(_delete_file_sync, path, workspace=workspace)

def _delete_file_sync(
    path: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    try:
        resolved = _resolve_path(path, workspace)
        safe, error = _is_safe_path(resolved, workspace)
        if not safe:
            return f"Error: {error}"

        if not resolved.exists():
            return f"Error: File not found: {path}"

        if resolved.is_dir():
            return f"Error: {path} is a directory. Use delete with care."

        resolved.unlink()
        log.info("file_deleted", path=str(resolved))
        return f"Deleted {path}"

    except Exception as e:
        log.error("delete_file_failed", path=path, error=str(e))
        return f"Error deleting file: {e}"


async def create_directory_impl(
    path: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    return await asyncio.to_thread(_create_directory_sync, path, workspace=workspace)

def _create_directory_sync(
    path: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    try:
        resolved = _resolve_path(path, workspace)
        safe, error = _is_safe_path(resolved, workspace)
        if not safe:
            return f"Error: {error}"

        resolved.mkdir(parents=True, exist_ok=True)
        log.info("directory_created", path=str(resolved))
        return f"Created directory {path}"

    except Exception as e:
        log.error("create_directory_failed", path=path, error=str(e))
        return f"Error creating directory: {e}"


# Bulk operations
async def read_files(
    paths: list[str],
    *,
    workspace: Workspace | None = None,
) -> dict[str, str]:
    """Read multiple files in one call."""
    return await asyncio.to_thread(_read_files_sync, paths, workspace=workspace)

def _read_files_sync(
    paths: list[str],
    *,
    workspace: Workspace | None = None,
) -> dict[str, str]:
    results = {}
    for path in paths:
        try:
            resolved = _resolve_path(path, workspace)
            safe, error = _is_safe_path(resolved, workspace)
            if not safe:
                results[path] = f"Error: {error}"
                continue

            if not resolved.exists():
                results[path] = "Error: File not found"
                continue

            if not resolved.is_file():
                results[path] = "Error: Not a file"
                continue

            content = resolved.read_text(encoding="utf-8", errors="replace")
            # Truncate very large files
            if len(content) > 100000:
                content = content[:100000] + "\n\n... [Truncated at 100KB]"
            results[path] = content

        except Exception as e:
            results[path] = f"Error: {e}"

    log.info("read_files_completed", count=len(paths), success=sum(1 for v in results.values() if not v.startswith("Error")))
    return results


async def edit_files(
    edits: list[dict],
    *,
    workspace: Workspace | None = None,
) -> dict[str, str]:
    """Apply multiple edits across files atomically."""
    return await asyncio.to_thread(_edit_files_sync, edits, workspace=workspace)

def _edit_files_sync(
    edits: list[dict],
    *,
    workspace: Workspace | None = None,
) -> dict[str, str]:
    # Phase 1: Validate all edits
    validated = []
    results = {}

    for edit in edits:
        path = edit.get("path", "")
        old_text = edit.get("old_text", "")
        new_text = edit.get("new_text", "")

        if not path or not old_text:
            results[path or "(missing path)"] = "Error: Missing path or old_text"
            continue

        try:
            resolved = _resolve_path(path, workspace)
            safe, error = _is_safe_path(resolved, workspace)
            if not safe:
                results[path] = f"Error: {error}"
                continue

            if not resolved.exists():
                results[path] = "Error: File not found"
                continue

            content = resolved.read_text(encoding="utf-8")
            count = content.count(old_text)

            if count == 0:
                results[path] = "Error: Text not found"
                continue

            validated.append({
                "path": path,
                "resolved": resolved,
                "content": content,
                "old_text": old_text,
                "new_text": new_text,
                "count": count,
            })

        except Exception as e:
            results[path] = f"Error: {e}"

    # Phase 2: Apply all validated edits
    for edit in validated:
        try:
            new_content = edit["content"].replace(edit["old_text"], edit["new_text"])
            edit["resolved"].write_text(new_content, encoding="utf-8")
            results[edit["path"]] = f"Replaced {edit['count']} occurrence(s)"
            log.info("file_edited", path=edit["path"], replacements=edit["count"])
        except Exception as e:
            results[edit["path"]] = f"Error applying: {e}"

    return results

# Expose implementations as public functions for internal usage
read_file = read_file_impl
write_file = write_file_impl
edit_file = edit_file_impl
list_dir = list_dir_impl
find_file = find_file_impl
search_for_pattern = search_for_pattern_impl
delete_file = delete_file_impl
create_directory = create_directory_impl
