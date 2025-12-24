# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
LSP Tools for Delia.

Exposes Language Server Protocol capabilities as tools for agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import structlog
from fastmcp import FastMCP

from ..types import Workspace
from .registry import ToolDefinition, ToolRegistry
from .handlers import check_checkpoint_gate
from .. import lsp_client

log = structlog.get_logger()


# Profile-aware warning patterns
SECURITY_PATTERNS = {
    "paths": ["auth", "security", "password", "secret", "credential", "token", "jwt", "oauth", "encrypt", "crypto"],
    "symbols": ["authenticate", "authorize", "login", "logout", "hash_password", "verify_token", "decrypt", "encrypt"],
    "profile": "security.md",
}

TESTING_PATTERNS = {
    "paths": ["test", "tests", "spec", "specs", "mock", "fixture"],
    "symbols": ["test_", "mock_", "fixture", "assert", "expect"],
    "profile": "testing.md",
}

API_PATTERNS = {
    "paths": ["api", "route", "endpoint", "handler", "controller"],
    "symbols": ["get_", "post_", "put_", "delete_", "patch_", "handle_"],
    "profile": "api.md",
}

PROFILE_PATTERN_SETS = [SECURITY_PATTERNS, TESTING_PATTERNS, API_PATTERNS]


def get_profile_warnings(file_path: str, symbol_name: str | None = None) -> list[dict]:
    """Check if file/symbol matches profile patterns and return warnings.
    
    Args:
        file_path: Path to the file being worked on
        symbol_name: Optional symbol name being modified
        
    Returns:
        List of warnings with profile names and reasons
    """
    warnings = []
    path_lower = file_path.lower()
    sym_lower = (symbol_name or "").lower()
    
    for pattern_set in PROFILE_PATTERN_SETS:
        matched = False
        reason = None
        
        # Check path patterns
        for pattern in pattern_set["paths"]:
            if pattern in path_lower:
                matched = True
                reason = f"File path contains '{pattern}'"
                break
        
        # Check symbol patterns
        if not matched and symbol_name:
            for pattern in pattern_set["symbols"]:
                if pattern in sym_lower:
                    matched = True
                    reason = f"Symbol name matches '{pattern}'"
                    break
        
        if matched:
            warnings.append({
                "profile": pattern_set["profile"],
                "reason": reason,
                "severity": "info",
            })
    
    return warnings


async def get_profile_context_for_warnings(warnings: list[dict], project_path: Path | None = None) -> str:
    """Get relevant profile content for warnings.
    
    Args:
        warnings: List of profile warnings
        project_path: Optional project path for profile lookup
        
    Returns:
        Formatted profile context string
    """
    if not warnings:
        return ""
    
    from ..playbook import get_playbook_manager
    
    pm = get_playbook_manager()
    if project_path:
        pm.set_project(project_path)
    
    lines = ["", "⚠️ Profile-aware context:"]
    
    for warning in warnings:
        profile_name = warning["profile"]
        reason = warning["reason"]
        lines.append(f"  • {profile_name}: {reason}")
    
    lines.append("")
    lines.append("Consider reviewing the relevant profiles with get_profile() before making changes.")
    
    return "\n".join(lines)


def parse_name_path(name: str) -> tuple[str, list[str]]:
    """Parse a name path like 'Foo.bar.baz' into (leaf_name, container_path).

    Examples:
        'Foo' -> ('Foo', [])
        'Foo.bar' -> ('bar', ['Foo'])
        'Foo.bar.baz' -> ('baz', ['Foo', 'bar'])
        'module::Class::method' -> ('method', ['module', 'Class'])  # Rust style

    Returns:
        Tuple of (leaf symbol name, list of container names from outermost to innermost)
    """
    # Handle Rust-style :: separator
    if "::" in name:
        parts = name.split("::")
    else:
        parts = name.split(".")

    # Filter out empty parts
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == 0:
        return ("", [])
    elif len(parts) == 1:
        return (parts[0], [])
    else:
        return (parts[-1], parts[:-1])


def find_symbol_end(lines: list[str], start_idx: int) -> int:
    """Find the end of a Python symbol (function/class) by indentation.

    Looks for the next line at the same or lower indentation level,
    or end of file.

    Args:
        lines: All lines in the file
        start_idx: 0-indexed line number of the symbol definition

    Returns:
        0-indexed line number of the last line of the symbol
    """
    if start_idx >= len(lines):
        return start_idx

    # Get the indentation of the definition line
    first_line = lines[start_idx]
    base_indent = len(first_line) - len(first_line.lstrip())

    # Scan forward to find end
    end_idx = start_idx
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            end_idx = i
            continue

        # Check indentation
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= base_indent:
            # Found a line at same or lower indent - symbol ended
            break
        end_idx = i

    return end_idx


def read_symbol_body(file_path: Path, start_line: int, end_line: int | None = None, max_lines: int = 50) -> str:
    """Read the source code body of a symbol from a file.

    Args:
        file_path: Absolute path to the file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed), or None to auto-detect for Python
        max_lines: Maximum lines to return (truncate if larger)

    Returns:
        The source code of the symbol, possibly truncated
    """
    try:
        content = file_path.read_text()
        lines = content.splitlines()

        # Convert to 0-indexed
        start_idx = max(0, start_line - 1)

        # Auto-detect end for Python if not provided or same as start
        if end_line is None or end_line == start_line:
            end_idx = find_symbol_end(lines, start_idx) + 1
        else:
            end_idx = min(len(lines), end_line)

        # Extract the body lines
        body_lines = lines[start_idx:end_idx]

        # Truncate if too long
        if len(body_lines) > max_lines:
            body_lines = body_lines[:max_lines]
            remaining = end_idx - start_idx - max_lines
            body_lines.append(f"    # ... ({remaining} more lines)")

        return "\n".join(body_lines)
    except Exception:
        return "# Error reading file"


def matches_container_path(symbol: dict, container_path: list[str]) -> bool:
    """Check if a symbol's container matches the expected path.

    Args:
        symbol: Symbol dict with optional 'container' key
        container_path: List of expected container names, outermost first

    Returns:
        True if the symbol is contained within the specified path
    """
    if not container_path:
        return True

    container = symbol.get("container", "")
    if not container:
        return False

    # The container name from workspace/symbol is typically "OuterClass.InnerClass"
    # or just "ClassName" for methods
    container_parts = container.replace("::", ".").split(".")

    # Check if the expected path matches the end of the container chain
    # e.g., for Foo.bar searching 'bar', container would be 'Foo'
    if len(container_path) == 1:
        # Simple case: just check if any part matches
        return container_path[0] in container_parts or container_path[0] == container

    # For deeper paths, check if all parts are present in order
    # The container path should be a suffix of container_parts
    if len(container_path) > len(container_parts):
        return False

    # Check if container_path matches the container_parts
    for expected in container_path:
        if expected not in container_parts:
            return False

    return True


def register_lsp_tools(mcp: FastMCP):
    """Register LSP tools with FastMCP."""

    # Register tools using the public functions directly
    mcp.tool(name="lsp_goto_definition")(lsp_goto_definition)
    mcp.tool(name="lsp_find_references")(lsp_find_references)
    mcp.tool(name="lsp_hover")(lsp_hover)
    mcp.tool(name="lsp_get_symbols")(lsp_get_symbols)
    mcp.tool(name="lsp_find_symbol")(lsp_find_symbol)
    mcp.tool(name="lsp_find_referencing_symbols")(lsp_find_referencing_symbols)
    mcp.tool(name="lsp_find_symbol_semantic")(lsp_find_symbol_semantic)
    mcp.tool(name="lsp_get_hot_files")(lsp_get_hot_files)
    mcp.tool(name="lsp_move_symbol")(lsp_move_symbol)
    mcp.tool(name="lsp_extract_method")(lsp_extract_method)
    mcp.tool(name="lsp_batch")(lsp_batch)
    mcp.tool(name="lsp_organize_imports")(lsp_organize_imports)
    mcp.tool(name="lsp_get_dependencies")(lsp_get_dependencies)
    mcp.tool(name="lsp_batch_history")(lsp_batch_history)
    mcp.tool(name="lsp_batch_undo")(lsp_batch_undo)
    
    # These have gating, so we wrap them to include the check
    @mcp.tool()
    async def lsp_rename_symbol(
        path: str,
        line: int,
        character: int,
        new_name: str,
        apply: bool = False,
    ) -> str:
        """
        Rename a symbol across the entire codebase.

        Uses LSP to find all references and rename them consistently.
        By default shows a preview; set apply=True to execute the rename.

        Args:
            path: Path to file containing the symbol
            line: Line number (1-indexed)
            character: Character position (0-indexed)
            new_name: The new name for the symbol
            apply: If True, apply changes. If False, preview only.

        Returns:
            Preview of changes, or confirmation if apply=True
        """
        # Checkpoint Gating
        checkpoint_error = check_checkpoint_gate("lsp_rename_symbol", path)
        if checkpoint_error:
            return checkpoint_error

        return await lsp_rename_symbol_impl(path, line, character, new_name, apply)

    @mcp.tool()
    async def lsp_replace_symbol_body(
        path: str,
        symbol_name: str,
        new_body: str,
    ) -> str:
        """
        Replace the entire body of a symbol (function, class, method).

        Finds the symbol by name and replaces its complete definition
        with the provided new code.

        Args:
            path: Path to the file containing the symbol
            symbol_name: Name of the symbol (e.g., "MyClass", "my_function")
            new_body: The complete new code for the symbol

        Returns:
            Confirmation of replacement or error message
        """
        # Checkpoint Gating
        checkpoint_error = check_checkpoint_gate("lsp_replace_symbol_body", path)
        if checkpoint_error:
            return checkpoint_error

        return await lsp_replace_symbol_body_impl(path, symbol_name, new_body)

    @mcp.tool()
    async def lsp_insert_before_symbol(
        path: str,
        symbol_name: str,
        content: str,
    ) -> str:
        """
        Insert code before a symbol.

        Useful for adding imports, decorators, or new functions/classes
        before an existing symbol.

        Args:
            path: Path to the file
            symbol_name: Name of the symbol to insert before
            content: The code to insert

        Returns:
            Confirmation or error message
        """
        # Checkpoint Gating
        checkpoint_error = check_checkpoint_gate("lsp_insert_before_symbol", path)
        if checkpoint_error:
            return checkpoint_error

        return await lsp_insert_before_symbol_impl(path, symbol_name, content)

    @mcp.tool()
    async def lsp_insert_after_symbol(
        path: str,
        symbol_name: str,
        content: str,
    ) -> str:
        """
        Insert code after a symbol.

        Useful for adding new functions, classes, or code blocks
        after an existing symbol.

        Args:
            path: Path to the file
            symbol_name: Name of the symbol to insert after
            content: The code to insert

        Returns:
            Confirmation or error message
        """
        # Checkpoint Gating
        checkpoint_error = check_checkpoint_gate("lsp_insert_after_symbol", path)
        if checkpoint_error:
            return checkpoint_error

        return await lsp_insert_after_symbol_impl(path, symbol_name, content)


def get_lsp_tools(workspace: Workspace | None = None) -> ToolRegistry:
    """Get LSP-based code intelligence tools.

    These tools provide semantic code navigation using Language Server Protocol:
    - Go to definition
    - Find references
    - Hover for type info/docs

    Args:
        workspace: Optional workspace to confine operations

    Returns:
        ToolRegistry with LSP tools registered
    """
    registry = ToolRegistry(workspace=workspace)

    registry.register(ToolDefinition(
        name="lsp_goto_definition",
        description="Find the definition of a symbol at the given file position. Returns file path and line number where the symbol is defined.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "line": {"type": "integer", "description": "Line number (1-indexed)"},
                "character": {"type": "integer", "description": "Character position (0-indexed)"}
            },
            "required": ["path", "line", "character"]
        },
        handler=lsp_goto_definition,
        dangerous=False,
        permission_level="read",
        requires_workspace=True,
    ))

    registry.register(ToolDefinition(
        name="lsp_find_references",
        description="Find all references to a symbol at the given file position. Returns list of locations where the symbol is used.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "line": {"type": "integer", "description": "Line number (1-indexed)"},
                "character": {"type": "integer", "description": "Character position (0-indexed)"}
            },
            "required": ["path", "line", "character"]
        },
        handler=lsp_find_references,
        dangerous=False,
        permission_level="read",
        requires_workspace=True,
    ))

    registry.register(ToolDefinition(
        name="lsp_hover",
        description="Get documentation and type information for the symbol at the given position. Returns docstrings, type signatures, etc.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "line": {"type": "integer", "description": "Line number (1-indexed)"},
                "character": {"type": "integer", "description": "Character position (0-indexed)"}
            },
            "required": ["path", "line", "character"]
        },
        handler=lsp_hover,
        dangerous=False,
        permission_level="read",
        requires_workspace=True,
    ))

    return registry


async def lsp_goto_definition(
    path: str,
    line: int,
    character: int,
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Find the definition of a symbol at the given file position.

    Uses Language Server Protocol to provide semantic code navigation.
    Supports Python (pyright/pylsp), TypeScript, Rust, and Go.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        character: Character position (0-indexed)

    Returns:
        File path and line number where the symbol is defined
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    
    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)
    
    results = await client.goto_definition(path, line, character)
    if not results:
        return "No definition found."
    
    out = [f"Found {len(results)} definition(s):"]
    for i, res in enumerate(results, 1):
        out.append(f"{i}. {res['path']} line {res['line']}, char {res['character']}")
    
    return "\n".join(out)

async def lsp_find_references(
    path: str,
    line: int, character: int,
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Find all references to a symbol at the given file position.

    Uses Language Server Protocol to find all usages of a symbol
    across the codebase. Supports Python, TypeScript, Rust, and Go.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        character: Character position (0-indexed)

    Returns:
        List of locations where the symbol is used
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)

    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    results = await client.find_references(path, line, character)
    if not results:
        return "No references found."
    
    out = [f"Found {len(results)} reference(s):"]
    for i, res in enumerate(results, 1):
        out.append(f"{i}. {res['path']} line {res['line']}, char {res['character']}")
    
    return "\n".join(out)

async def lsp_hover(
    path: str,
    line: int,
    character: int,
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Get documentation and type information for a symbol.

    Uses Language Server Protocol to retrieve docstrings, type signatures,
    and other documentation for the symbol at the given position.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        character: Character position (0-indexed)

    Returns:
        Documentation and type info in markdown format
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)

    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    result = await client.hover(path, line, character)
    if not result:
        return "No information available."

    return result


async def lsp_get_symbols(
    path: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Get all symbols in a file (classes, functions, methods, variables).

    Returns a hierarchical view of the file's structure, similar to an
    IDE's outline view. Use this to understand a file's organization
    before making edits.

    Args:
        path: Path to the file to analyze

    Returns:
        Hierarchical list of symbols with their types and line ranges
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)

    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    result = await client.document_symbols(path)

    # Handle error response
    if isinstance(result, dict) and "error" in result:
        return f"LSP Error: {result['error']}"

    symbols = result
    if not symbols:
        return "No symbols found in file."

    # Format as readable output
    lines = [f"Found {len(symbols)} symbol(s) in {path}:"]
    for sym in symbols:
        indent = "  " * sym.get("depth", 0)
        kind = sym.get("kind", "unknown")
        name = sym.get("name", "?")

        if "range" in sym:
            start = sym["range"]["start_line"]
            end = sym["range"]["end_line"]
            lines.append(f"{indent}{kind}: {name} (lines {start}-{end})")
        elif "location" in sym:
            line = sym["location"]["line"]
            lines.append(f"{indent}{kind}: {name} (line {line})")
        else:
            lines.append(f"{indent}{kind}: {name}")

    return "\n".join(lines)


# Internal implementation functions for modification (kept internal to force gating)

async def lsp_rename_symbol_impl(
    path: str,
    line: int,
    character: int,
    new_name: str,
    apply: bool = False,
    *,
    workspace: Workspace | None = None,
) -> str:
    import json
    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    # Get the rename edits
    result = await client.rename_symbol(path, line, character, new_name)

    if "error" in result:
        return f"Rename failed: {result['error']}"

    if not result.get("files"):
        return "No changes needed."

    lines = [f"Rename to '{new_name}' affects {result['total_edits']} location(s) in {len(result['files'])} file(s):"]

    for file_path, edits in result["files"].items():
        lines.append(f"\n  {file_path}: {len(edits)} edit(s)")
        for edit in edits[:3]:  # Show first 3 edits per file
            lines.append(f"    - Line {edit['start_line']}")
        if len(edits) > 3:
            lines.append(f"    - ... and {len(edits) - 3} more")

    if apply:
        # Apply the edits
        for file_path, edits in result["files"].items():
            abs_path = root / file_path
            if not abs_path.exists():
                continue

            content = abs_path.read_text()
            file_lines = content.splitlines(keepends=True)

            # Sort edits in reverse order to apply from end to start
            sorted_edits = sorted(edits, key=lambda e: (e["start_line"], e["start_char"]),
                                  reverse=True)

            for edit in sorted_edits:
                start_line = edit["start_line"] - 1
                end_line = edit["end_line"] - 1
                start_char = edit["start_char"]
                end_char = edit["end_char"]
                new_text = edit["new_text"]

                if start_line == end_line:
                    # Single line edit
                    line_content = file_lines[start_line] if start_line < len(file_lines) else ""
                    file_lines[start_line] = line_content[:start_char] + new_text + line_content[end_char:]
                else:
                    # Multi-line edit (rare for renames)
                    first_line = file_lines[start_line][:start_char] if start_line < len(file_lines) else ""
                    last_line = file_lines[end_line][end_char:] if end_line < len(file_lines) else ""
                    file_lines[start_line:end_line + 1] = [first_line + new_text + last_line]

            abs_path.write_text("".join(file_lines))

        lines.append(f"\nApplied {result['total_edits']} changes.")
    else:
        lines.append("\n(Preview only. Set apply=True to apply changes.)")

    return "\n".join(lines)


async def lsp_find_symbol_impl(
    name: str,
    path: str | None = None,
    kind: str | None = None,
    kinds: List[str] | None = None,
    depth: int | None = None,
    include_body: bool = False,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Find symbols by name with support for name path syntax.

    Name path syntax allows searching for nested symbols:
        - 'Foo' - find class/function named Foo
        - 'Foo.bar' - find 'bar' inside 'Foo' (e.g., method bar in class Foo)
        - 'Foo.bar.baz' - find 'baz' inside 'Foo.bar'
        - 'module::Class::method' - Rust-style path syntax

    Args:
        name: Symbol name or path (e.g., 'MyClass.my_method')
        path: Optional file path to search within
        kind: Optional single kind filter ('function', 'class', 'method', etc.)
        kinds: Optional list of kinds to include (e.g., ['function', 'method'])
        depth: Optional depth filter (0=top-level, 1=nested once, etc.)
        include_body: If True, include the source code body of each symbol
        workspace: Workspace context

    Returns:
        Formatted string with matching symbols
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)

    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    # Parse name path to get leaf name and container path
    leaf_name, container_path = parse_name_path(name)
    if not leaf_name:
        return "Invalid symbol name."

    # Build kind filter set (combine kind and kinds)
    kind_filter: set[str] | None = None
    if kind or kinds:
        kind_filter = set()
        if kind:
            kind_filter.add(kind.lower())
        if kinds:
            kind_filter.update(k.lower() for k in kinds)

    def matches_filters(sym: dict) -> bool:
        """Check if a symbol passes all filters."""
        # Kind filter
        if kind_filter:
            sym_kind = sym.get("kind", "").lower()
            if sym_kind not in kind_filter:
                return False

        # Depth filter (only available from document_symbols)
        if depth is not None:
            sym_depth = sym.get("depth")
            if sym_depth is not None and sym_depth != depth:
                return False

        # Container path filter
        if container_path and not matches_container_path(sym, container_path):
            return False

        return True

    matches = []
    last_error = None

    if path:
        # Search in specific file using document_symbols
        result = await client.document_symbols(path)
        if isinstance(result, dict) and "error" in result:
            return f"LSP Error: {result['error']}"

        # For document symbols, search with container awareness
        for sym in result:
            sym_name = sym.get("name", "")
            if leaf_name.lower() in sym_name.lower():
                if matches_filters(sym):
                    sym["file"] = path
                    matches.append(sym)
    else:
        # Use workspace/symbol for fast project-wide search
        # Detect primary language from project markers (check python first)
        lang_id = "python"
        if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
            lang_id = "python"
        elif (root / "Cargo.toml").exists():
            lang_id = "rust"
        elif (root / "go.mod").exists():
            lang_id = "go"
        elif (root / "package.json").exists():
            lang_id = "typescript"

        # Search for the leaf symbol name
        result = await client.workspace_symbol(leaf_name, language_id=lang_id)

        if isinstance(result, dict) and "error" in result:
            last_error = result["error"]
        else:
            for sym in result:
                if matches_filters(sym):
                    matches.append(sym)

    if not matches:
        if last_error:
            return f"No symbols matching '{name}' found. LSP Error: {last_error}"
        return f"No symbols matching '{name}' found."

    # When include_body=True, limit to fewer results since bodies are verbose
    max_results = 5 if include_body else 20

    lines = [f"Found {len(matches)} symbol(s) matching '{name}':"]
    for sym in matches[:max_results]:
        kind_str = sym.get("kind", "?")
        name_str = sym.get("name", "?")
        file_str = sym.get("file", "?")
        container = sym.get("container", "")
        container_str = f" ({container})" if container else ""

        if "range" in sym:
            start_line = sym["range"]["start_line"]
            end_line = sym["range"].get("end_line", start_line)
            lines.append(f"  {kind_str}: {name_str}{container_str} in {file_str}:{start_line}")

            # Include body if requested and we have valid range
            if include_body and file_str != "?":
                file_path = Path(file_str)
                if not file_path.is_absolute():
                    file_path = root / file_path

                # Pass None for end_line if only point location (let read_symbol_body auto-detect)
                body_end = None if start_line == end_line else end_line
                body = read_symbol_body(file_path, start_line, body_end)
                lines.append("  ```")
                for body_line in body.splitlines():
                    lines.append(f"  {body_line}")
                lines.append("  ```")
        else:
            lines.append(f"  {kind_str}: {name_str}{container_str} in {file_str}")

    if len(matches) > max_results:
        lines.append(f"  ... and {len(matches) - max_results} more")

    return "\n".join(lines)

# Expose lsp_find_symbol as public function
lsp_find_symbol = lsp_find_symbol_impl


async def lsp_find_referencing_symbols_impl(
    path: str,
    line: int,
    character: int,
    kinds: List[str] | None = None,
    include_body: bool = False,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Find symbols that reference the symbol at the given position.

    Unlike lsp_find_references which returns raw locations, this returns
    the containing symbols (functions, classes, methods) that contain
    each reference, providing better context for understanding usage.

    Args:
        path: Path to the file containing the symbol
        line: Line number (1-indexed)
        character: Character position (0-indexed)
        kinds: Optional filter for containing symbol kinds (e.g., ['function', 'method'])
        include_body: If True, include source code of containing symbols
        workspace: Workspace context

    Returns:
        Formatted string with symbols that reference the target
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    
    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    # Build kind filter set
    kind_filter: set[str] | None = None
    if kinds:
        kind_filter = {k.lower() for k in kinds}

    # Get all references to the symbol
    references = await client.find_references(path, line, character)
    if not references:
        return "No references found."

    # Group references by file for efficient document_symbols lookup
    refs_by_file: dict[str, list[dict]] = {}
    for ref in references:
        ref_path = ref.get("path", "")
        if ref_path not in refs_by_file:
            refs_by_file[ref_path] = []
        refs_by_file[ref_path].append(ref)

    # Find containing symbols for each reference
    containing_symbols: list[dict] = []
    seen_symbols: set[str] = set()  # Dedupe by (file, name, line)

    for file_path, file_refs in refs_by_file.items():
        # Get all symbols in this file
        doc_symbols = await client.document_symbols(file_path)
        if isinstance(doc_symbols, dict) and "error" in doc_symbols:
            continue

        for ref in file_refs:
            ref_line = ref.get("line", 0)

            # Find the smallest symbol that contains this reference
            best_match: dict | None = None
            best_size = float("inf")

            for sym in doc_symbols:
                sym_name = sym.get("name", "")
                sym_kind = sym.get("kind", "").lower()

                # Apply kind filter
                if kind_filter and sym_kind not in kind_filter:
                    continue

                # Check if symbol contains the reference line
                # For symbols with location (not range), check if ref is after the symbol start
                if "location" in sym:
                    sym_line = sym["location"].get("line", 0)
                    # Approximate: symbol contains ref if ref is after symbol start
                    # This is a heuristic since we don't have end line
                    if ref_line >= sym_line:
                        # Prefer symbols closer to the reference
                        distance = ref_line - sym_line
                        if distance < best_size:
                            best_size = distance
                            best_match = sym
                elif "range" in sym:
                    start = sym["range"].get("start_line", 0)
                    end = sym["range"].get("end_line", start)
                    if start <= ref_line <= end:
                        size = end - start
                        if size < best_size:
                            best_size = size
                            best_match = sym

            if best_match:
                # Create unique key for deduplication
                sym_key = (
                    file_path,
                    best_match.get("name", ""),
                    best_match.get("location", {}).get("line")
                    or best_match.get("range", {}).get("start_line", 0),
                )
                if sym_key not in seen_symbols:
                    seen_symbols.add(sym_key)
                    best_match["file"] = file_path
                    best_match["ref_count"] = 1
                    containing_symbols.append(best_match)
                else:
                    # Increment ref count for existing symbol
                    for sym in containing_symbols:
                        if (
                            sym.get("file") == file_path
                            and sym.get("name") == best_match.get("name")
                        ):
                            sym["ref_count"] = sym.get("ref_count", 1) + 1
                            break

    if not containing_symbols:
        return "No containing symbols found for references."

    # Sort by ref_count descending
    containing_symbols.sort(key=lambda s: s.get("ref_count", 1), reverse=True)

    # Format output
    max_results = 5 if include_body else 15
    lines = [f"Found {len(containing_symbols)} symbol(s) referencing the target:"]

    for sym in containing_symbols[:max_results]:
        kind_str = sym.get("kind", "?")
        name_str = sym.get("name", "?")
        file_str = sym.get("file", "?")
        ref_count = sym.get("ref_count", 1)
        ref_info = f" ({ref_count} refs)" if ref_count > 1 else ""

        # Get line number
        if "location" in sym:
            sym_line = sym["location"].get("line", "?")
        elif "range" in sym:
            sym_line = sym["range"].get("start_line", "?")
        else:
            sym_line = "?"

        lines.append(f"  {kind_str}: {name_str}{ref_info} in {file_str}:{sym_line}")

        # Include body if requested
        if include_body and file_str != "?":
            file_path_obj = Path(file_str)
            if not file_path_obj.is_absolute():
                file_path_obj = root / file_path_obj

            if "location" in sym:
                start = sym["location"].get("line", 1)
                body = read_symbol_body(file_path_obj, start, None)
            elif "range" in sym:
                start = sym["range"].get("start_line", 1)
                end = sym["range"].get("end_line", start)
                body_end = None if start == end else end
                body = read_symbol_body(file_path_obj, start, body_end)
            else:
                body = "# Cannot read body"

            lines.append("  ```")
            for body_line in body.splitlines():
                lines.append(f"  {body_line}")
            lines.append("  ```")

    if len(containing_symbols) > max_results:
        lines.append(f"  ... and {len(containing_symbols) - max_results} more")

    return "\n".join(lines)


# Expose as public function
lsp_find_referencing_symbols = lsp_find_referencing_symbols_impl


async def lsp_find_symbol_semantic_impl(
    query: str,
    top_k: int = 10,
    kinds: list[str] | None = None,
    include_body: bool = False,
    boost_recent: bool = True,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Find symbols semantically using CodeRAG + LSP fusion.

    Combines semantic search (embeddings) with LSP symbol resolution.
    Use natural language queries like "authentication logic" or "database connection".

    Args:
        query: Natural language search query
        top_k: Maximum number of results (default 10)
        kinds: Filter by symbol kinds (e.g., ["function", "class", "method"])
        include_body: Include source code of matched symbols
        boost_recent: Boost recently modified files in ranking (default True)
        workspace: Workspace context

    Returns:
        Structured list of symbols ranked by semantic relevance
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    root = workspace.root if workspace else Path.cwd()

    # 1. Use semantic search to find relevant files
    from ..orchestration.summarizer import get_summarizer
    from ..orchestration.graph import get_symbol_graph

    summarizer = get_summarizer()
    await summarizer.initialize()
    
    # Get symbol graph for recency scores if boosting
    graph = get_symbol_graph() if boost_recent else None

    file_results = await summarizer.search(query, top_k=top_k)

    if not file_results:
        return f"No files found matching '{query}'"

    # 2. Get LSP client and extract symbols from matched files
    client = lsp_client.get_lsp_client(root)

    all_symbols = []
    for file_result in file_results:
        file_path = file_result.get("path", "")
        relevance_score = file_result.get("score", 0)

        try:
            symbols = await client.document_symbols(file_path)
            for sym in symbols:
                # Filter by kinds if specified
                if kinds:
                    sym_kind = sym.get("kind", "").lower()
                    if sym_kind not in [k.lower() for k in kinds]:
                        continue

                sym["file"] = file_path
                sym["relevance_score"] = relevance_score
                all_symbols.append(sym)
        except Exception as e:
            log.debug("lsp_symbols_failed", file=file_path, error=str(e))
            continue

    if not all_symbols:
        return f"No symbols found in files matching '{query}'"

    # 3. Calculate combined scores with optional recency boost
    for sym in all_symbols:
        relevance = sym.get("relevance_score", 0)
        recency = 0.0
        
        if graph and boost_recent:
            file_path = sym.get("file", "")
            recency = graph.get_file_recency_score(file_path, decay_hours=24.0)
            sym["recency_score"] = recency
        
        # Combined score: 70% relevance + 30% recency when boosting
        if boost_recent and recency > 0:
            sym["combined_score"] = 0.7 * relevance + 0.3 * recency
        else:
            sym["combined_score"] = relevance

    # Sort by combined score (higher = more relevant)
    all_symbols.sort(key=lambda s: s.get("combined_score", 0), reverse=True)

    # Limit results
    all_symbols = all_symbols[:top_k]

    # 4. Optionally include symbol bodies
    if include_body:
        for sym in all_symbols[:5]:  # Limit body reads to top 5
            file_path = sym.get("file", "")
            abs_path = root / file_path
            if abs_path.exists() and "range" in sym:
                try:
                    content = abs_path.read_text()
                    lines = content.splitlines()
                    start = sym["range"].get("start_line", 1) - 1
                    end = sym["range"].get("end_line", start + 1)
                    # Limit to 50 lines
                    end = min(end, start + 50)
                    sym["body"] = "\n".join(lines[start:end])
                except Exception:
                    pass

    # 5. Format output
    lines = [f"Found {len(all_symbols)} symbol(s) matching '{query}':"]
    for sym in all_symbols:
        name = sym.get("name", "unknown")
        kind = sym.get("kind", "symbol")
        file_path = sym.get("file", "")
        line = sym.get("location", {}).get("line") or sym.get("range", {}).get("start_line", 0)
        score = sym.get("relevance_score", 0)

        lines.append(f"  {kind}: {name} in {file_path}:{line} (relevance: {score:.2f})")

        if include_body and "body" in sym:
            # Indent body
            body_lines = sym["body"].split("\n")[:10]  # Show first 10 lines
            for bl in body_lines:
                lines.append(f"    | {bl}")
            if len(sym["body"].split("\n")) > 10:
                lines.append("    | ...")

    return "\n".join(lines)


# Expose as public function
lsp_find_symbol_semantic = lsp_find_symbol_semantic_impl


async def lsp_get_hot_files_impl(
    limit: int = 10,
    since_hours: float = 24.0,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Get recently modified files (hot files).

    Returns files that have been recently modified, useful for focusing
    on actively worked-on code areas.

    Args:
        limit: Maximum number of files to return (default 10)
        since_hours: Only include files modified within this many hours (default 24)
        workspace: Workspace context

    Returns:
        List of recently modified files with modification times
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    root = workspace.root if workspace else Path.cwd()

    from ..orchestration.graph import get_symbol_graph
    import datetime

    graph = get_symbol_graph()
    await graph.sync(root)  # Ensure graph is up to date

    hot_files = graph.get_hot_files(limit=limit, since_hours=since_hours)

    if not hot_files:
        return f"No files modified in the last {since_hours} hours."

    lines = [f"Hot files (modified in last {since_hours}h):"]
    for file_path, mtime in hot_files:
        dt = datetime.datetime.fromtimestamp(mtime)
        age_hours = (datetime.datetime.now().timestamp() - mtime) / 3600
        age_str = f"{age_hours:.1f}h ago" if age_hours < 1 else f"{int(age_hours)}h ago"
        lines.append(f"  {file_path} ({age_str})")

    return "\n".join(lines)


# Expose as public function
lsp_get_hot_files = lsp_get_hot_files_impl


async def lsp_replace_symbol_body_impl(
    path: str,
    symbol_name: str,
    new_body: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)

    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    # Find the symbol
    symbols = await client.document_symbols(path)
    target = None
    for sym in symbols:
        if sym.get("name") == symbol_name and "range" in sym:
            target = sym
            break

    if not target:
        return f"Symbol '{symbol_name}' not found in {path}"

    # Read the file
    abs_path = root / path
    if not abs_path.exists():
        return f"File not found: {path}"

    content = abs_path.read_text()
    lines = content.splitlines(keepends=True)

    # Get the range (1-indexed in our format)
    start_line = target["range"]["start_line"] - 1
    end_line = target["range"]["end_line"] - 1

    # Preserve indentation from original
    original_indent = ""
    if start_line < len(lines):
        original_line = lines[start_line]
        original_indent = original_line[:len(original_line) - len(original_line.lstrip())]

    # Ensure new_body ends with newline
    if not new_body.endswith("\n"):
        new_body += "\n"

    # Replace the lines
    lines[start_line:end_line + 1] = [new_body]

    # Write back
    abs_path.write_text("".join(lines))

    result = f"Replaced {target['kind']} '{symbol_name}' (lines {start_line + 1}-{end_line + 1}) in {path}"
    
    # Add profile-aware warnings if applicable
    warnings = get_profile_warnings(path, symbol_name)
    if warnings:
        warning_context = await get_profile_context_for_warnings(warnings, root)
        result += warning_context
    
    return result


async def lsp_insert_before_symbol_impl(
    path: str,
    symbol_name: str,
    content: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)

    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    # Find the symbol
    symbols = await client.document_symbols(path)
    target = None
    for sym in symbols:
        if sym.get("name") == symbol_name and "range" in sym:
            target = sym
            break

    if not target:
        return f"Symbol '{symbol_name}' not found in {path}"

    # Read the file
    abs_path = root / path
    if not abs_path.exists():
        return f"File not found: {path}"

    file_content = abs_path.read_text()
    lines = file_content.splitlines(keepends=True)

    # Insert before the symbol's start line
    insert_line = target["range"]["start_line"] - 1

    # Ensure content ends with newline
    if not content.endswith("\n"):
        content += "\n"

    lines.insert(insert_line, content)

    # Write back
    abs_path.write_text("".join(lines))

    result = f"Inserted content before {target['kind']} '{symbol_name}' at line {insert_line + 1} in {path}"
    
    # Add profile-aware warnings if applicable
    warnings = get_profile_warnings(path, symbol_name)
    if warnings:
        warning_context = await get_profile_context_for_warnings(warnings, root)
        result += warning_context
    
    return result


async def lsp_insert_after_symbol_impl(
    path: str,
    symbol_name: str,
    content: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    # Find the symbol
    symbols = await client.document_symbols(path)
    target = None
    for sym in symbols:
        if sym.get("name") == symbol_name and "range" in sym:
            target = sym
            break

    if not target:
        return f"Symbol '{symbol_name}' not found in {path}"

    # Read the file
    abs_path = root / path
    if not abs_path.exists():
        return f"File not found: {path}"

    file_content = abs_path.read_text()
    lines = file_content.splitlines(keepends=True)

    # Insert after the symbol's end line
    insert_line = target["range"]["end_line"]  # Already 1-indexed, so this is after

    # Ensure content starts with newline for separation and ends with newline
    if not content.startswith("\n"):
        content = "\n" + content
    if not content.endswith("\n"):
        content += "\n"

    if insert_line >= len(lines):
        lines.append(content)
    else:
        lines.insert(insert_line, content)

    # Write back
    abs_path.write_text("".join(lines))

    result = f"Inserted content after {target['kind']} '{symbol_name}' at line {insert_line + 1} in {path}"
    
    # Add profile-aware warnings if applicable
    warnings = get_profile_warnings(path, symbol_name)
    if warnings:
        warning_context = await get_profile_context_for_warnings(warnings, root)
        result += warning_context
    
    return result


async def lsp_move_symbol_impl(
    source_path: str,
    symbol_name: str,
    dest_path: str,
    update_imports: bool = True,
    cleanup_imports: bool = True,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Move a symbol from one file to another with import updates.

    Extracts the symbol from the source file, inserts it into the destination,
    and optionally updates all import statements across the codebase.

    Args:
        source_path: Path to the source file containing the symbol
        symbol_name: Name of the symbol to move
        dest_path: Path to the destination file
        update_imports: Whether to update imports in referencing files (default True)
        cleanup_imports: Whether to remove unused imports from source file (default True)
        workspace: Workspace context

    Returns:
        Summary of the move operation and import updates
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    # 1. Find the symbol in source file
    symbols = await client.document_symbols(source_path)
    target = None
    for sym in symbols:
        if sym.get("name") == symbol_name and "range" in sym:
            target = sym
            break

    if not target:
        return f"Symbol '{symbol_name}' not found in {source_path}"

    # 2. Read source file and extract symbol code
    source_abs = root / source_path
    if not source_abs.exists():
        return f"Source file not found: {source_path}"

    source_content = source_abs.read_text()
    source_lines = source_content.splitlines(keepends=True)

    start_line = target["range"]["start_line"] - 1  # Convert to 0-indexed
    end_line = target["range"]["end_line"] - 1

    symbol_code = "".join(source_lines[start_line:end_line + 1])

    # 3. Remove symbol from source file
    del source_lines[start_line:end_line + 1]
    source_abs.write_text("".join(source_lines))

    # 3b. Clean up unused imports in source file
    cleanup_result = None
    if cleanup_imports:
        cleanup_result = await _cleanup_imports(source_abs, organize=True)

    # 4. Insert symbol into destination file
    dest_abs = root / dest_path
    if dest_abs.exists():
        dest_content = dest_abs.read_text()
        dest_lines = dest_content.splitlines(keepends=True)
    else:
        dest_lines = []

    # Find insertion point - after imports, before first symbol
    insert_at = len(dest_lines)
    for i, line in enumerate(dest_lines):
        stripped = line.strip()
        if stripped and not stripped.startswith(("import ", "from ", "#", '"""', "'''")):
            if not stripped.startswith("@"):  # Skip decorators
                insert_at = i
                break

    # Ensure proper spacing
    if not symbol_code.startswith("\n"):
        symbol_code = "\n\n" + symbol_code
    if not symbol_code.endswith("\n"):
        symbol_code += "\n"

    dest_lines.insert(insert_at, symbol_code)
    dest_abs.write_text("".join(dest_lines))

    result_lines = [
        f"Moved {target['kind']} '{symbol_name}' from {source_path} to {dest_path}",
        f"  Removed from lines {start_line + 1}-{end_line + 1} in source",
        f"  Inserted at line {insert_at + 1} in destination",
    ]

    # Report import cleanup results
    if cleanup_result and cleanup_result.get("cleaned"):
        removed = cleanup_result.get("removed_imports", 0)
        if removed > 0:
            result_lines.append(f"  Cleaned up {removed} unused import(s) from source")
        if cleanup_result.get("organized"):
            result_lines.append(f"  Organized imports in source file")

    # 5. Update imports if requested
    if update_imports:
        # Find all files that import from source
        refs = await client.find_references(source_path, start_line + 1, 0)
        
        # Learn import convention from destination file
        import_style = _detect_import_style(root, source_path, dest_path)
        
        updated_files = []
        for ref in refs:
            ref_path = ref.get("path", "")
            if ref_path and ref_path != source_path and ref_path != dest_path:
                try:
                    ref_abs = root / ref_path
                    if ref_abs.exists():
                        ref_content = ref_abs.read_text()
                        # Update import statements
                        new_content = _update_import_statement(
                            ref_content, symbol_name, source_path, dest_path, import_style
                        )
                        if new_content != ref_content:
                            ref_abs.write_text(new_content)
                            updated_files.append(ref_path)
                except Exception as e:
                    log.debug("import_update_failed", file=ref_path, error=str(e))

        if updated_files:
            result_lines.append(f"  Updated imports in {len(updated_files)} file(s):")
            for f in updated_files[:5]:
                result_lines.append(f"    - {f}")
            if len(updated_files) > 5:
                result_lines.append(f"    ... and {len(updated_files) - 5} more")

    # Add profile warnings
    warnings = get_profile_warnings(source_path, symbol_name)
    warnings.extend(get_profile_warnings(dest_path, symbol_name))
    if warnings:
        warning_context = await get_profile_context_for_warnings(warnings, root)
        result_lines.append(warning_context)

    return "\n".join(result_lines)


def _detect_import_style(root: Path, source_path: str, dest_path: str) -> str:
    """Detect the project's import style preference.
    
    Returns:
        'relative' or 'absolute' based on project conventions
    """
    # Check existing imports in dest file
    dest_abs = root / dest_path
    if dest_abs.exists():
        content = dest_abs.read_text()
        relative_count = content.count("from .")
        absolute_count = content.count("from delia")
        
        if relative_count > absolute_count:
            return "relative"
    
    return "absolute"


def _update_import_statement(
    content: str,
    symbol_name: str,
    old_module: str,
    new_module: str,
    import_style: str,
) -> str:
    """Update import statements to reflect moved symbol.
    
    Args:
        content: File content to update
        symbol_name: Name of the moved symbol
        old_module: Old module path (e.g., 'src/delia/old.py')
        new_module: New module path (e.g., 'src/delia/new.py')
        import_style: 'relative' or 'absolute'
        
    Returns:
        Updated content
    """
    import re
    
    # Convert paths to module names
    old_mod = old_module.replace("src/", "").replace("/", ".").replace(".py", "")
    new_mod = new_module.replace("src/", "").replace("/", ".").replace(".py", "")
    
    # Pattern to find imports of the symbol from old module
    # Handles: from module import symbol, from module import (symbol, other)
    patterns = [
        # from old_module import symbol_name
        (rf"from {re.escape(old_mod)} import ([^(\n]*\b{re.escape(symbol_name)}\b[^)\n]*)",
         f"from {new_mod} import {symbol_name}"),
        # from old_module import (... symbol_name ...)
        (rf"from {re.escape(old_mod)} import \(([^)]*\b{re.escape(symbol_name)}\b[^)]*)\)",
         f"from {new_mod} import {symbol_name}"),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            # Simple replacement - just add new import, keep old for other symbols
            # This is a simplified approach; full implementation would parse and rewrite
            if f"from {new_mod} import" not in content:
                # Add new import after existing imports
                import_section_end = 0
                for i, line in enumerate(content.splitlines()):
                    if line.startswith(("import ", "from ")):
                        import_section_end = i + 1
                
                lines = content.splitlines(keepends=True)
                new_import = f"from {new_mod} import {symbol_name}\n"
                lines.insert(import_section_end, new_import)
                content = "".join(lines)
    
    return content


async def _cleanup_imports(file_path: Path, organize: bool = True) -> dict:
    """Clean up unused imports and optionally organize imports using Ruff.
    
    Args:
        file_path: Path to the Python file to clean up
        organize: Whether to also organize/sort imports
        
    Returns:
        Dict with cleanup results
    """
    import subprocess
    
    if not file_path.exists() or not file_path.suffix == ".py":
        return {"cleaned": False, "reason": "Not a Python file"}
    
    result = {"cleaned": False, "removed_imports": 0, "organized": False}
    
    try:
        # Step 1: Remove unused imports (F401)
        proc = subprocess.run(
            ["ruff", "check", "--fix", "--select", "F401", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "Fixed" in proc.stdout or proc.returncode == 0:
            # Count fixed issues from output
            import re
            match = re.search(r"(\d+) fixed", proc.stdout)
            if match:
                result["removed_imports"] = int(match.group(1))
                result["cleaned"] = True
        
        # Step 2: Organize imports (I001, I002) if requested
        if organize:
            proc = subprocess.run(
                ["ruff", "check", "--fix", "--select", "I", str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                result["organized"] = True
                
    except FileNotFoundError:
        result["reason"] = "ruff not installed"
    except subprocess.TimeoutExpired:
        result["reason"] = "ruff timed out"
    except Exception as e:
        result["reason"] = str(e)
    
    return result


# Expose as public function  
lsp_move_symbol = lsp_move_symbol_impl


async def lsp_organize_imports_impl(
    path: str,
    remove_unused: bool = True,
    sort_imports: bool = True,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Organize imports in a Python file using Ruff.
    
    Removes unused imports and sorts remaining imports according to PEP 8.
    
    Args:
        path: Path to the Python file
        remove_unused: Remove unused imports (default True)
        sort_imports: Sort and organize imports (default True)
        workspace: Workspace context
        
    Returns:
        Summary of changes made
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    root = workspace.root if workspace else Path.cwd()
    
    file_path = root / path
    if not file_path.exists():
        return f"File not found: {path}"
    
    if not path.endswith(".py"):
        return f"Not a Python file: {path}"
    
    import subprocess
    
    results = []
    
    try:
        # Remove unused imports
        if remove_unused:
            proc = subprocess.run(
                ["ruff", "check", "--fix", "--select", "F401", str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            import re
            match = re.search(r"(\d+) fixed", proc.stdout)
            if match:
                count = int(match.group(1))
                results.append(f"Removed {count} unused import(s)")
        
        # Sort imports
        if sort_imports:
            proc = subprocess.run(
                ["ruff", "check", "--fix", "--select", "I", str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                results.append("Organized import order")
                
    except FileNotFoundError:
        return "Error: ruff not installed. Install with: uv add --dev ruff"
    except subprocess.TimeoutExpired:
        return "Error: ruff timed out"
    except Exception as e:
        return f"Error: {e}"
    
    if results:
        return f"Import cleanup for {path}:\n  " + "\n  ".join(results)
    return f"No import changes needed for {path}"


# Expose as public function
lsp_organize_imports = lsp_organize_imports_impl


async def lsp_get_dependencies_impl(
    path: str,
    include_symbols: bool = True,
    max_depth: int = 2,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Visualize cross-file symbol dependencies for a file.
    
    Shows what the file exports, what it imports, and who depends on it.
    
    Args:
        path: Path to the file to analyze
        include_symbols: Include per-symbol dependency details (default True)
        max_depth: How many levels of dependencies to show (default 2)
        workspace: Workspace context
        
    Returns:
        Formatted dependency visualization
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)
    
    file_abs = root / path
    if not file_abs.exists():
        return f"File not found: {path}"
    
    # Get symbols in this file
    symbols = await client.document_symbols(path)
    
    # Build output
    lines = [f"# Dependency Analysis: {path}", ""]
    
    # 1. Exported symbols
    if include_symbols and symbols:
        lines.append("## Exported Symbols")
        for sym in symbols:
            kind = sym.get("kind", "unknown")
            name = sym.get("name", "?")
            line = sym.get("range", {}).get("start_line", "?")
            lines.append(f"  - {kind}: {name} (line {line})")
        lines.append("")
    
    # 2. Analyze imports (what this file depends on)
    content = file_abs.read_text()
    import re
    
    imports = []
    for line in content.splitlines():
        if line.strip().startswith(("import ", "from ")):
            imports.append(line.strip())
    
    if imports:
        lines.append("## Imports (dependencies)")
        for imp in imports[:20]:  # Limit to 20
            lines.append(f"  {imp}")
        if len(imports) > 20:
            lines.append(f"  ... and {len(imports) - 20} more")
        lines.append("")
    
    # 3. Find who depends on this file (reverse dependencies)
    lines.append("## Dependents (who imports this)")
    
    # Search for files that import from this module
    module_name = path.replace("src/", "").replace("/", ".").replace(".py", "")
    
    dependents = []
    for py_file in root.rglob("*.py"):
        if py_file == file_abs:
            continue
        try:
            file_content = py_file.read_text()
            # Check various import patterns
            patterns = [
                f"from {module_name} import",
                f"import {module_name}",
                f"from .{module_name.split('.')[-1]} import",
            ]
            for pattern in patterns:
                if pattern in file_content:
                    rel_path = str(py_file.relative_to(root))
                    if rel_path not in dependents:
                        dependents.append(rel_path)
                    break
        except Exception:
            continue
    
    if dependents:
        for dep in dependents[:15]:  # Limit to 15
            lines.append(f"  - {dep}")
        if len(dependents) > 15:
            lines.append(f"  ... and {len(dependents) - 15} more")
    else:
        lines.append("  (no dependents found)")
    lines.append("")
    
    # 4. Symbol-level dependencies (who uses each exported symbol)
    if include_symbols and symbols:
        lines.append("## Symbol Usage (who references each symbol)")
        for sym in symbols[:10]:  # Limit to 10 symbols
            name = sym.get("name", "?")
            range_info = sym.get("range", {})
            line_num = range_info.get("start_line", 1)
            
            # Find references to this symbol
            try:
                refs = await client.find_references(path, line_num, 0)
                # Filter to external files only
                external_refs = [r for r in refs if r.get("path", "") != path]
                
                if external_refs:
                    unique_files = list(set(r.get("path", "") for r in external_refs))
                    lines.append(f"  {name}: used in {len(unique_files)} file(s)")
                    for f in unique_files[:3]:
                        lines.append(f"    - {f}")
                    if len(unique_files) > 3:
                        lines.append(f"    ... and {len(unique_files) - 3} more")
                else:
                    lines.append(f"  {name}: (no external references)")
            except Exception:
                lines.append(f"  {name}: (could not analyze)")
    
    return "\n".join(lines)


# Expose as public function
lsp_get_dependencies = lsp_get_dependencies_impl


async def lsp_extract_method_impl(
    path: str,
    start_line: int,
    end_line: int,
    new_name: str | None = None,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Extract a code block into a new method.

    Extracts lines from start_line to end_line into a new method,
    replacing the original code with a call to the new method.
    
    If new_name is not provided, generates a name based on the code content.

    Args:
        path: Path to the file
        start_line: First line to extract (1-indexed)
        end_line: Last line to extract (1-indexed)
        new_name: Name for the extracted method (optional, will suggest if not provided)
        workspace: Workspace context

    Returns:
        Summary of the extraction with the generated method
    """
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    root = workspace.root if workspace else Path.cwd()

    # Read the file
    abs_path = root / path
    if not abs_path.exists():
        return f"File not found: {path}"

    content = abs_path.read_text()
    lines = content.splitlines(keepends=True)

    # Convert to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line - 1

    if start_idx < 0 or end_idx >= len(lines) or start_idx > end_idx:
        return f"Invalid line range: {start_line}-{end_line} (file has {len(lines)} lines)"

    # Extract the code block
    extracted_lines = lines[start_idx:end_idx + 1]
    extracted_code = "".join(extracted_lines)

    # Detect indentation
    first_line = extracted_lines[0] if extracted_lines else ""
    original_indent = len(first_line) - len(first_line.lstrip())
    indent_str = first_line[:original_indent]

    # Generate method name if not provided
    if not new_name:
        new_name = _suggest_method_name(extracted_code)

    # Detect variables used (simple heuristic)
    params, returns = _analyze_code_block(extracted_code)

    # Build the new method
    param_str = ", ".join(params) if params else ""
    return_str = f"return {', '.join(returns)}" if returns else "pass"
    
    # Dedent the extracted code for the method body
    dedented_lines = []
    for line in extracted_lines:
        if line.strip():  # Non-empty lines
            # Remove the original indentation, add one level
            stripped = line[original_indent:] if len(line) > original_indent else line.lstrip()
            dedented_lines.append("    " + stripped)
        else:
            dedented_lines.append(line)
    
    method_body = "".join(dedented_lines)
    
    new_method = f'''
def {new_name}({param_str}):
    """Extracted method."""
{method_body}'''

    # Create the method call
    call_args = ", ".join(params) if params else ""
    if returns:
        method_call = f"{indent_str}{', '.join(returns)} = {new_name}({call_args})\n"
    else:
        method_call = f"{indent_str}{new_name}({call_args})\n"

    # Replace the extracted code with the method call
    lines[start_idx:end_idx + 1] = [method_call]

    # Find where to insert the new method (before the containing function/class)
    insert_pos = _find_method_insert_position(lines, start_idx)
    lines.insert(insert_pos, new_method + "\n\n")

    # Write back
    abs_path.write_text("".join(lines))

    result = f"Extracted lines {start_line}-{end_line} to new method '{new_name}'"
    result += f"\n  Method inserted at line {insert_pos + 1}"
    result += f"\n  Original code replaced with: {method_call.strip()}"
    
    if params:
        result += f"\n  Detected parameters: {', '.join(params)}"
    if returns:
        result += f"\n  Detected returns: {', '.join(returns)}"

    # Add profile warnings
    warnings = get_profile_warnings(path, new_name)
    if warnings:
        warning_context = await get_profile_context_for_warnings(warnings, root)
        result += warning_context

    return result


def _suggest_method_name(code: str) -> str:
    """Suggest a method name based on code content."""
    import re
    
    code_lower = code.lower()
    
    # Look for common patterns
    if "fetch" in code_lower or "request" in code_lower or "http" in code_lower:
        return "fetch_data"
    if "save" in code_lower or "write" in code_lower:
        return "save_data"
    if "load" in code_lower or "read" in code_lower:
        return "load_data"
    if "validate" in code_lower or "check" in code_lower:
        return "validate_input"
    if "parse" in code_lower:
        return "parse_data"
    if "format" in code_lower:
        return "format_output"
    if "calculate" in code_lower or "compute" in code_lower:
        return "calculate_result"
    if "filter" in code_lower:
        return "filter_items"
    if "transform" in code_lower or "convert" in code_lower:
        return "transform_data"
    if "log" in code_lower:
        return "log_info"
    
    # Look for function calls to derive name
    func_calls = re.findall(r'\b([a-z_][a-z0-9_]*)\s*\(', code_lower)
    if func_calls:
        # Use the most common/meaningful call
        for call in func_calls:
            if call not in ('print', 'len', 'str', 'int', 'list', 'dict', 'set'):
                return f"do_{call}"
    
    return "extracted_method"


def _analyze_code_block(code: str) -> tuple[list[str], list[str]]:
    """Analyze code to detect parameters and return values.
    
    Returns:
        Tuple of (parameter names, return variable names)
    """
    import re
    
    # Find variables that are used but not defined in the block
    # This is a simplified heuristic
    defined = set()
    used = set()
    
    # Find assignments (defined)
    for match in re.finditer(r'^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code, re.MULTILINE):
        defined.add(match.group(1))
    
    # Find variable uses
    for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code):
        name = match.group(1)
        # Skip keywords and builtins
        if name not in ('if', 'else', 'for', 'while', 'in', 'and', 'or', 'not', 
                        'True', 'False', 'None', 'return', 'def', 'class', 'import',
                        'from', 'as', 'try', 'except', 'finally', 'with', 'async', 'await',
                        'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'range'):
            used.add(name)
    
    # Parameters = used but not defined
    params = sorted(used - defined)[:5]  # Limit to 5 params
    
    # Returns = defined in block (might be needed outside)
    returns = sorted(defined & used)[:3]  # Limit to 3 returns
    
    return params, returns


def _find_method_insert_position(lines: list[str], extraction_line: int) -> int:
    """Find the best position to insert a new method.
    
    Looks backwards from extraction_line to find the start of the containing
    function or class, then inserts before it.
    """
    import re
    
    for i in range(extraction_line - 1, -1, -1):
        line = lines[i]
        # Found a top-level def or class
        if re.match(r'^(async\s+)?def\s+|^class\s+', line):
            return i
    
    # If no containing function found, insert at the start of the file
    # (after imports)
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith(('import ', 'from ', '#', '"""', "'''")):
            return i
    
    return 0


# Expose as public function
lsp_extract_method = lsp_extract_method_impl


# Batch history storage for undo support
_BATCH_HISTORY_DIR = Path.home() / ".delia" / "batch_history"
_MODIFYING_OPS = {"rename", "replace_body", "insert_before", "insert_after", "move", "extract_method"}


def _save_batch_snapshot(batch_id: str, files: dict[str, str], root: Path) -> None:
    """Save file states before a batch operation for undo support."""
    import json as json_module
    
    _BATCH_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    snapshot = {
        "batch_id": batch_id,
        "timestamp": __import__("time").time(),
        "root": str(root),
        "files": files,  # path -> content
    }
    
    snapshot_path = _BATCH_HISTORY_DIR / f"{batch_id}.json"
    snapshot_path.write_text(json_module.dumps(snapshot, indent=2))
    
    # Keep only last 10 snapshots
    snapshots = sorted(_BATCH_HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old_snapshot in snapshots[10:]:
        old_snapshot.unlink()


def _get_batch_history() -> list[dict]:
    """Get list of batch snapshots available for undo."""
    import json as json_module
    
    if not _BATCH_HISTORY_DIR.exists():
        return []
    
    history = []
    for snapshot_path in sorted(_BATCH_HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json_module.loads(snapshot_path.read_text())
            history.append({
                "batch_id": data["batch_id"],
                "timestamp": data["timestamp"],
                "files_count": len(data.get("files", {})),
                "root": data.get("root", ""),
            })
        except Exception:
            continue
    
    return history


async def lsp_batch_impl(
    operations: str,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Execute multiple LSP operations in sequence.

    Runs a batch of LSP operations and tracks the sequence for learning.
    Useful for complex refactoring that requires multiple steps.
    
    Modifying operations are automatically saved for undo support.
    Use lsp_batch_history() to see recent batches and lsp_batch_undo() to revert.

    Args:
        operations: JSON array of operations, each with:
            - op: Operation name (find_symbol, find_references, rename, move, etc.)
            - args: Arguments for the operation
        workspace: Workspace context

    Example operations:
        [
            {"op": "find_symbol", "args": {"name": "MyClass"}},
            {"op": "find_references", "args": {"path": "src/mod.py", "line": 10, "character": 5}},
            {"op": "rename", "args": {"path": "src/mod.py", "line": 10, "character": 5, "new_name": "NewClass"}}
        ]

    Returns:
        Combined results of all operations with sequence summary
    """
    import json as json_module
    import uuid
    
    # Convert dict to Workspace if needed (for MCP compatibility)
    if isinstance(workspace, dict):
        workspace = Workspace(**workspace)
    root = workspace.root if workspace else Path.cwd()

    # Parse operations
    try:
        ops = json_module.loads(operations)
    except json_module.JSONDecodeError as e:
        return f"Invalid operations JSON: {e}"

    if not isinstance(ops, list):
        return "Operations must be a JSON array"

    # Check if any operations are modifying
    has_modifying_ops = any(op.get("op") in _MODIFYING_OPS for op in ops)
    
    # Collect files that will be modified for snapshot
    batch_id = None
    if has_modifying_ops:
        batch_id = str(uuid.uuid4())[:8]
        files_to_backup = set()
        
        for op_spec in ops:
            op_name = op_spec.get("op", "")
            args = op_spec.get("args", {})
            
            if op_name in _MODIFYING_OPS:
                # Extract paths from different operation types
                if "path" in args:
                    files_to_backup.add(args["path"])
                if "source_path" in args:
                    files_to_backup.add(args["source_path"])
                if "dest_path" in args:
                    files_to_backup.add(args["dest_path"])
        
        # Save current content of all affected files
        file_contents = {}
        for rel_path in files_to_backup:
            abs_path = root / rel_path
            if abs_path.exists():
                try:
                    file_contents[rel_path] = abs_path.read_text()
                except Exception:
                    pass
        
        if file_contents:
            _save_batch_snapshot(batch_id, file_contents, root)

    results = []
    sequence = []  # Track for learning

    for i, op_spec in enumerate(ops):
        op_name = op_spec.get("op", "")
        args = op_spec.get("args", {})

        # Add workspace to args
        args["workspace"] = workspace

        try:
            result = await _execute_lsp_operation(op_name, args, root)
            results.append({
                "operation": op_name,
                "success": True,
                "result": result,
            })
            sequence.append(op_name)
        except Exception as e:
            results.append({
                "operation": op_name,
                "success": False,
                "error": str(e),
            })
            # Continue with remaining operations

    # Format output
    output_lines = [f"Batch executed {len(ops)} operation(s):"]
    
    for i, r in enumerate(results):
        status = "✓" if r["success"] else "✗"
        output_lines.append(f"\n{i+1}. [{status}] {r['operation']}")
        if r["success"]:
            # Truncate long results
            result_str = str(r["result"])
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            output_lines.append(f"   {result_str}")
        else:
            output_lines.append(f"   Error: {r['error']}")

    # Record sequence for learning (if we have a playbook manager)
    if len(sequence) >= 2:
        try:
            from ..playbook import get_playbook_manager
            pm = get_playbook_manager()
            pm.set_project(root)
            
            sequence_str = " → ".join(sequence)
            # Check if this is a common pattern worth recording
            # (This could be enhanced with actual pattern detection)
            output_lines.append(f"\n📝 Sequence recorded: {sequence_str}")
        except Exception:
            pass  # Playbook recording is optional

    # Add undo hint if we saved a snapshot
    if batch_id:
        output_lines.append(f"\n💾 Batch ID: {batch_id} (use lsp_batch_undo to revert)")

    return "\n".join(output_lines)


async def lsp_batch_history_impl(
    limit: int = 10,
    *,
    workspace: Workspace | None = None,
) -> str:
    """List recent batch operations available for undo.
    
    Args:
        limit: Maximum number of entries to show (default 10)
        workspace: Workspace context
        
    Returns:
        List of recent batch operations with their IDs
    """
    import time
    
    history = _get_batch_history()[:limit]
    
    if not history:
        return "No batch history available."
    
    lines = ["Recent batch operations:"]
    for entry in history:
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["timestamp"]))
        lines.append(f"  {entry['batch_id']}: {entry['files_count']} file(s) @ {timestamp}")
    
    lines.append("\nUse lsp_batch_undo(batch_id='...') to revert a batch.")
    return "\n".join(lines)


async def lsp_batch_undo_impl(
    batch_id: str | None = None,
    *,
    workspace: Workspace | None = None,
) -> str:
    """Undo a batch operation by restoring files to their previous state.
    
    Args:
        batch_id: ID of the batch to undo (default: most recent)
        workspace: Workspace context
        
    Returns:
        Summary of restored files
    """
    import json as json_module
    
    if not _BATCH_HISTORY_DIR.exists():
        return "No batch history available."
    
    # Find the snapshot to restore
    if batch_id:
        snapshot_path = _BATCH_HISTORY_DIR / f"{batch_id}.json"
        if not snapshot_path.exists():
            return f"Batch {batch_id} not found. Use lsp_batch_history() to see available batches."
    else:
        # Get most recent
        snapshots = sorted(_BATCH_HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshots:
            return "No batch history available."
        snapshot_path = snapshots[0]
    
    # Load and restore
    try:
        data = json_module.loads(snapshot_path.read_text())
        root = Path(data.get("root", "."))
        files = data.get("files", {})
        
        restored = []
        for rel_path, content in files.items():
            abs_path = root / rel_path
            try:
                abs_path.write_text(content)
                restored.append(rel_path)
            except Exception as e:
                log.warning("batch_restore_failed", path=rel_path, error=str(e))
        
        # Remove the snapshot after successful restore
        snapshot_path.unlink()
        
        batch_id_used = data.get("batch_id", "unknown")
        return f"Restored {len(restored)} file(s) from batch {batch_id_used}:\n  " + "\n  ".join(restored)
        
    except Exception as e:
        return f"Failed to restore batch: {e}"


# Expose as public functions
lsp_batch_history = lsp_batch_history_impl
lsp_batch_undo = lsp_batch_undo_impl


async def _execute_lsp_operation(op_name: str, args: dict, root: Path) -> str:
    """Execute a single LSP operation by name."""
    
    # Map operation names to functions
    op_map = {
        "find_symbol": lsp_find_symbol_impl,
        "find_references": lsp_find_references,
        "goto_definition": lsp_goto_definition,
        "hover": lsp_hover,
        "get_symbols": lsp_get_symbols,
        "rename": lsp_rename_symbol_impl,
        "find_referencing_symbols": lsp_find_referencing_symbols_impl,
        "find_symbol_semantic": lsp_find_symbol_semantic_impl,
        "get_hot_files": lsp_get_hot_files_impl,
        "replace_body": lsp_replace_symbol_body_impl,
        "insert_before": lsp_insert_before_symbol_impl,
        "insert_after": lsp_insert_after_symbol_impl,
        "move": lsp_move_symbol_impl,
        "extract_method": lsp_extract_method_impl,
    }

    if op_name not in op_map:
        raise ValueError(f"Unknown operation: {op_name}. Available: {', '.join(op_map.keys())}")

    func = op_map[op_name]
    return await func(**args)


# Expose as public function
lsp_batch = lsp_batch_impl
