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

from ..types import Workspace
from .registry import ToolDefinition, ToolRegistry
from .. import lsp_client

log = structlog.get_logger()


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
    """Find the definition of the symbol at the given position.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        character: Character position (0-indexed)
        workspace: Workspace context

    Returns:
        List of definitions found
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
    """Find all references to the symbol at the given position.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        character: Character position (0-indexed)
        workspace: Workspace context

    Returns:
        List of references found
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
    """Get documentation and type information for the symbol at the given position.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        character: Character position (0-indexed)
        workspace: Workspace context

    Returns:
        Hover information (markdown)
    """
    root = workspace.root if workspace else Path.cwd()
    client = lsp_client.get_lsp_client(root)
    
    result = await client.hover(path, line, character)
    if not result:
        return "No information available."
    
    return result

