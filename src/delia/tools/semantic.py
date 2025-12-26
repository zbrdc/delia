# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Semantic search tools - ALWAYS registered regardless of profile.

These tools provide intelligent code navigation using embeddings and
the dependency graph. They are essential for understanding codebases.

Tools:
- semantic_search: Find code by meaning, not just text matching
- get_related_files: Find files connected via imports/dependencies
- codebase_graph: Query the project dependency structure
- explain_dependency: Understand why files are related
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
    async def get_related_files(
        file_path: str,
        depth: int = 2,
    ) -> str:
        """Get files related via imports/dependencies (N hops in dependency graph)."""
        from ..orchestration.graph import get_symbol_graph

        graph = get_symbol_graph()
        await graph.initialize()

        # Normalize path
        if file_path not in graph.nodes:
            matches = [p for p in graph.nodes if file_path in p]
            if not matches:
                return json.dumps({"error": f"File not found in graph: {file_path}"})
            file_path = matches[0]

        related = graph.get_related_files(file_path, max_depth=depth)

        return json.dumps({
            "file": file_path,
            "depth": depth,
            "related_count": len(related),
            "related": related,
        }, indent=2)

    @mcp.tool()
    async def codebase_graph(
        file_path: str | None = None,
        depth: int = 1,
    ) -> str:
        """Query dependency graph for a file. Use semantic_search() for code search."""
        from ..orchestration.graph import get_symbol_graph

        graph = get_symbol_graph()
        await graph.initialize()

        if file_path:
            if file_path not in graph.nodes:
                matches = [p for p in graph.nodes if file_path in p]
                if not matches:
                    return json.dumps({"error": f"File not found: {file_path}"})
                file_path = matches[0]
            node = graph.nodes[file_path]
            return json.dumps({
                "file": file_path,
                "imports": list(node.imports),
                "symbol_count": len(node.symbols),
            }, indent=2)

        # No file specified - return summary with top importers/imported
        import_counts: dict[str, int] = {}
        imported_by_counts: dict[str, int] = {}
        for path, node in graph.nodes.items():
            import_counts[path] = len(node.imports)
            for imp in node.imports:
                imported_by_counts[imp] = imported_by_counts.get(imp, 0) + 1

        top_importers = sorted(import_counts.items(), key=lambda x: -x[1])[:5]
        top_imported = sorted(imported_by_counts.items(), key=lambda x: -x[1])[:5]

        return json.dumps({
            "total_files": len(graph.nodes),
            "top_importers": [{"file": f, "import_count": c} for f, c in top_importers],
            "most_imported": [{"file": f, "imported_by_count": c} for f, c in top_imported],
            "hint": "Pass file_path to see specific file dependencies",
        }, indent=2)

    @mcp.tool()
    async def explain_dependency(
        source: str,
        target: str,
    ) -> str:
        """Explain why source file depends on target file.

        Uses the dependency graph to analyze the relationship between files.

        Args:
            source: Path to the source file
            target: Path to the target file (dependency)

        Returns:
            Explanation of the dependency relationship
        """
        from ..orchestration.graph import get_symbol_graph

        graph = get_symbol_graph()
        await graph.initialize()

        # Normalize paths
        for path_var, name in [(source, "source"), (target, "target")]:
            if path_var not in graph.nodes:
                matches = [p for p in graph.nodes if path_var in p]
                if not matches:
                    return json.dumps({"error": f"{name} file not found: {path_var}"})

        explanation = await graph.explain_dependency(source, target)

        return json.dumps({
            "source": source,
            "target": target,
            "explanation": explanation,
        }, indent=2)

    @mcp.tool()
    async def project_overview() -> str:
        """Get a hierarchical summary of the entire project structure.

        Returns a high-level view of the codebase organization,
        useful for onboarding to a new project.
        """
        from ..orchestration.summarizer import get_summarizer

        summarizer = get_summarizer()
        await summarizer.initialize()
        overview = summarizer.get_project_overview()

        if not overview or overview == "Project overview not yet generated.":
            return json.dumps({"error": "Project not yet indexed. Run project(action='scan') first."}, indent=2)

        return overview
