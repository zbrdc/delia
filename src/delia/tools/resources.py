# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Resource-oriented MCP tools for Delia.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastmcp import FastMCP

from ..container import get_container

log = structlog.get_logger()


def register_resource_tools(mcp: FastMCP):
    """Register resource-oriented tools with FastMCP."""
    
    container = get_container()

    @mcp.tool()
    async def project_memories(reload: bool = False) -> str:
        """List project memories (DELIA.md files) loaded into context."""
        from ..project_memory import get_project_memory, reload_project_memories, list_project_memories
        if reload: reload_project_memories()
        memories = list_project_memories()
        pm = get_project_memory()
        state = pm._state
        result = {
            "memories": memories,
            "total_size": state.total_size if state else 0,
            "hierarchy": [
                "1. ~/.delia/DELIA.md (user defaults)",
                "2. ./DELIA.md (project instructions)",
                "3. ./.delia/DELIA.md (project config)",
                "4. ./.delia/rules/*.md (modular rules)",
                "5. ./DELIA.local.md (local overrides)",
            ],
        }
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def codebase_graph(file_path: str | None = None, depth: int = 1, query: str | None = None, top_k: int = 10) -> str:
        """Query the project's dependency graph (GraphRAG)."""
        from ..orchestration.graph import get_symbol_graph
        from ..orchestration.summarizer import get_summarizer
        if query:
            summarizer = get_summarizer(); await summarizer.initialize()
            results = await summarizer.search(query, top_k=top_k)
            return json.dumps({"query": query, "results": results}, indent=2)
        graph = get_symbol_graph(); await graph.initialize()
        if file_path:
            if file_path not in graph.nodes:
                matches = [p for p in graph.nodes if file_path in p]
                if not matches: return json.dumps({"error": f"File not found: {file_path}"})
                file_path = matches[0]
            node = graph.nodes[file_path]
            return json.dumps({"file": file_path, "imports": list(node.imports), "symbol_count": len(node.symbols)}, indent=2)
        return json.dumps({"total_files": len(graph.nodes)}, indent=2)

    @mcp.tool()
    async def project_overview() -> str:
        """Get a hierarchical summary of the entire project structure."""
        from ..orchestration.summarizer import get_summarizer
        summarizer = get_summarizer(); await summarizer.initialize()
        overview = summarizer.get_project_overview()
        if not overview or overview == "Project overview not yet generated.":
            return json.dumps({"error": "Project not yet indexed"}, indent=2)
        return overview
