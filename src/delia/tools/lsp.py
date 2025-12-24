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


def register_lsp_tools(mcp: FastMCP):
    """Register LSP tools with FastMCP."""

    # Register tools using the public functions directly
    mcp.tool(name="lsp_goto_definition")(lsp_goto_definition)
    mcp.tool(name="lsp_find_references")(lsp_find_references)
    mcp.tool(name="lsp_hover")(lsp_hover)
    mcp.tool(name="lsp_get_symbols")(lsp_get_symbols)
    mcp.tool(name="lsp_find_symbol")(lsp_find_symbol)
    
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
    *,
    workspace: Workspace | None = None,
) -> str:
    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)

    matches = []
    last_error = None

    if path:
        # Search in specific file
        result = await client.document_symbols(path)
        if isinstance(result, dict) and "error" in result:
            return f"LSP Error: {result['error']}"
        for sym in result:
            if name.lower() in sym.get("name", "").lower():
                if kind is None or sym.get("kind", "").lower() == kind.lower():
                    sym["file"] = path
                    matches.append(sym)
    else:
        # Search across common source files
        from pathlib import Path as P
        code_extensions = { ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}
        src_dirs = ["src", "app", "lib", "."]

        for src_dir in src_dirs:
            src_path = root / src_dir
            if not src_path.exists():
                continue
            for code_file in src_path.rglob("*"):
                if code_file.is_file() and code_file.suffix in code_extensions:
                    rel_path = str(code_file.relative_to(root))
                    result = await client.document_symbols(rel_path)
                    # Check for error on first file to give feedback
                    if isinstance(result, dict) and "error" in result:
                        last_error = result["error"]
                        continue
                    for sym in result:
                        if name.lower() in sym.get("name", "").lower():
                            if kind is None or sym.get("kind", "").lower() == kind.lower():
                                sym["file"] = rel_path
                                matches.append(sym)
            if matches:
                break  # Found matches in first src dir

    if not matches:
        if last_error:
            return f"No symbols matching '{name}' found. LSP Error: {last_error}"
        return f"No symbols matching '{name}' found."

    lines = [f"Found {len(matches)} symbol(s) matching '{name}':"]
    for sym in matches[:20]:  # Limit to 20
        kind_str = sym.get("kind", "?")
        name_str = sym.get("name", "?")
        file_str = sym.get("file", "?")
        if "range" in sym:
            line = sym["range"]["start_line"]
            lines.append(f"  {kind_str}: {name_str} in {file_str}:{line}")
        else:
            lines.append(f"  {kind_str}: {name_str} in {file_str}")

    if len(matches) > 20:
        lines.append(f"  ... and {len(matches) - 20} more")

    return "\n".join(lines)

# Expose lsp_find_symbol as public function
lsp_find_symbol = lsp_find_symbol_impl


async def lsp_replace_symbol_body_impl(
    path: str,
    symbol_name: str,
    new_body: str,
    *,
    workspace: Workspace | None = None,
) -> str:
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

    return f"Replaced {target['kind']} '{symbol_name}' (lines {start_line + 1}-{end_line + 1}) in {path}"


async def lsp_insert_before_symbol_impl(
    path: str,
    symbol_name: str,
    content: str,
    *,
    workspace: Workspace | None = None,
) -> str:
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

    return f"Inserted content before {target['kind']} '{symbol_name}' at line {insert_line + 1} in {path}"


async def lsp_insert_after_symbol_impl(
    path: str,
    symbol_name: str,
    content: str,
    *,
    workspace: Workspace | None = None,
) -> str:
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

    return f"Inserted content after {target['kind']} '{symbol_name}' at line {insert_line + 1} in {path}"
