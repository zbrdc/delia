# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Semantic search tools - ALWAYS registered regardless of profile.

These tools provide intelligent code navigation using embeddings and
the dependency graph. They are essential for understanding codebases.

Tools:
- semantic_search: Find code by meaning, not just text matching
- codebase_graph: Query the project dependency structure
  - Also handles: related files (depth>1), explain dependency (explain_source/target)

NOTE: get_related_files, explain_dependency, project_overview have been consolidated
per ADR-010 into codebase_graph and project(action="overview").
"""

from __future__ import annotations

import json
import fnmatch

import structlog
from fastmcp import FastMCP

log = structlog.get_logger()


def register_semantic_tools(mcp: FastMCP):
    """Register semantic search tools - ALWAYS registered."""

    @mcp.tool()
    async def semantic_search(
        query: str,
        top_k: int = 10,
        file_pattern: str | None = None,
    ) -> str:
        """Search codebase by meaning (embeddings). Use natural language queries."""
        from ..orchestration.summarizer import get_summarizer

        summarizer = get_summarizer()
        await summarizer.initialize()

        results = await summarizer.search(query, top_k=top_k)

        # Filter by pattern if specified
        if file_pattern and results:
            results = [
                r for r in results
                if fnmatch.fnmatch(r.get("path", ""), file_pattern)
            ]

        return json.dumps({
            "query": query,
            "file_pattern": file_pattern,
            "result_count": len(results),
            "results": results,
        }, indent=2)

    @mcp.tool()
    async def codebase_graph(
        file_path: str | None = None,
        depth: int = 1,
        explain_source: str | None = None,
        explain_target: str | None = None,
    ) -> str:
        """Query dependency graph for a file.

        Consolidated from get_related_files, explain_dependency per ADR-010.

        Usage modes:
        - No args: Returns graph summary (top importers, most imported)
        - file_path only: Returns file info (imports, symbol count)
        - file_path + depth > 1: Returns related files (N hops away)
        - explain_source + explain_target: Explains dependency between files

        Use semantic_search() for meaning-based code search.
        """
        from ..orchestration.graph import get_symbol_graph

        graph = get_symbol_graph()
        await graph.initialize()

        # Mode: Explain dependency between two files
        if explain_source and explain_target:
            # Normalize paths
            for path_var, name in [(explain_source, "source"), (explain_target, "target")]:
                if path_var not in graph.nodes:
                    matches = [p for p in graph.nodes if path_var in p]
                    if not matches:
                        return json.dumps({"error": f"{name} file not found: {path_var}"})

            explanation = await graph.explain_dependency(explain_source, explain_target)
            return json.dumps({
                "mode": "explain_dependency",
                "source": explain_source,
                "target": explain_target,
                "explanation": explanation,
            }, indent=2)

        # Mode: File-specific queries
        if file_path:
            if file_path not in graph.nodes:
                matches = [p for p in graph.nodes if file_path in p]
                if not matches:
                    return json.dumps({"error": f"File not found: {file_path}"})
                file_path = matches[0]

            # Mode: Related files (depth > 1)
            if depth > 1:
                related = graph.get_related_files(file_path, max_depth=depth)
                return json.dumps({
                    "mode": "related_files",
                    "file": file_path,
                    "depth": depth,
                    "related_count": len(related),
                    "related": related,
                }, indent=2)

            # Mode: Single file info
            node = graph.nodes[file_path]
            return json.dumps({
                "mode": "file_info",
                "file": file_path,
                "imports": list(node.imports),
                "symbol_count": len(node.symbols),
            }, indent=2)

        # Mode: Graph summary (no file specified)
        import_counts: dict[str, int] = {}
        imported_by_counts: dict[str, int] = {}
        for path, node in graph.nodes.items():
            import_counts[path] = len(node.imports)
            for imp in node.imports:
                imported_by_counts[imp] = imported_by_counts.get(imp, 0) + 1

        top_importers = sorted(import_counts.items(), key=lambda x: -x[1])[:5]
        top_imported = sorted(imported_by_counts.items(), key=lambda x: -x[1])[:5]

        return json.dumps({
            "mode": "summary",
            "total_files": len(graph.nodes),
            "top_importers": [{"file": f, "import_count": c} for f, c in top_importers],
            "most_imported": [{"file": f, "imported_by_count": c} for f, c in top_imported],
            "hint": "Pass file_path to see file dependencies, depth>1 for related files, or explain_source+explain_target for dependency explanation",
        }, indent=2)

    # NOTE: get_related_files has been consolidated into codebase_graph(file_path, depth>1) per ADR-010.
    # NOTE: explain_dependency has been consolidated into codebase_graph(explain_source, explain_target) per ADR-010.
    # NOTE: project_overview has been consolidated into project(action="overview") per ADR-010.
