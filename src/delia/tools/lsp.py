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
