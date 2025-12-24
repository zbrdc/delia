# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Resource-oriented MCP tools for Delia.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from fastmcp import FastMCP

from ..container import get_container
from ..backend_manager import backend_manager
from ..context import current_project_path
from ..file_helpers import MEMORY_DIR
from ..config import config

log = structlog.get_logger()


def register_resource_tools(mcp: FastMCP):
    """Register resource-oriented tools with FastMCP."""
    
    container = get_container()
    stats_service = container.stats_service

    @mcp.resource("delia://file/{path}")
    async def resource_file(path: str) -> str:
        """
        Read a file from disk as an MCP resource.

        Enables other MCP servers/clients to read files through Delia.
        Useful for cross-server workflows where external MCP tools
        need to pass file content to Delia without serialization overhead.

        Args:
            path: File path (absolute or relative to cwd)

        Returns:
            File content as text, or error message if file not found/readable
        """
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path

        if not file_path.exists():
            return f"Error: File not found: {path}"

        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        try:
            size = file_path.stat().st_size
            max_size = config.max_file_size  # 500KB default
            if size > max_size:
                return f"Error: File too large ({size // 1024}KB > {max_size // 1024}KB): {path}"

            content = file_path.read_text(encoding="utf-8")
            log.info("resource_file_read", path=path, size_kb=size // 1024)
            return content
        except Exception as e:
            log.warning("resource_file_failed", path=path, error=str(e))
            return f"Error reading file: {e}"

    @mcp.resource("delia://stats", name="Usage Statistics", description="Current Delia usage statistics")
    async def resource_stats() -> str:
        """
        Get current usage statistics as JSON.

        Returns token counts, call counts, and estimated cost savings
        across all model tiers.
        """
        model_usage, task_stats, _, recent_calls = stats_service.get_snapshot()
        stats = {
            "model_usage": model_usage,
            "task_stats": task_stats,
            "recent_calls_count": len(recent_calls),
        }
        return json.dumps(stats, indent=2)

    @mcp.resource("delia://backends", name="Backend Status", description="Health and configuration of all backends")
    async def resource_backends() -> str:
        """
        Get backend health status as JSON.

        Returns configuration and availability status for all configured
        backends, useful for monitoring and cross-server coordination.
        """
        status = await backend_manager.get_health_status()
        return json.dumps(status, indent=2)

    @mcp.resource("delia://config", name="Configuration", description="Current Delia configuration")
    async def resource_config() -> str:
        """
        Get current configuration as JSON.

        Returns routing settings, model tiers, and system configuration.
        Sensitive fields (API keys) are redacted.
        """
        config_data = {
            "routing": backend_manager.routing_config,
            "system": backend_manager.system_config,
            "backends": [
                {
                    "id": b.id,
                    "name": b.name,
                    "provider": b.provider,
                    "type": b.type,
                    "url": b.url,
                    "enabled": b.enabled,
                    "models": b.models,
                    # Redact API key for security
                    "has_api_key": bool(b.api_key),
                }
                for b in backend_manager.backends.values()
            ],
        }
        return json.dumps(config_data, indent=2)

    @mcp.resource("delia://playbook", name="Active Playbook", description="Current project playbook bullets - READ THIS to stay on track")
    async def resource_playbook() -> str:
        """
        Get current playbook bullets for all task types.

        This resource provides dynamic access to project-specific
        playbook guidance. Read this regularly to ensure you're
        following project patterns.
        """
        from ..playbook import get_playbook_manager

        pm = get_playbook_manager()
        path = current_project_path.get() or str(Path.cwd())
        pm.set_project(Path(path))

        all_bullets = {}
        for task_type in ["coding", "testing", "architecture", "debugging", "git", "project"]:
            bullets = pm.get_top_bullets(task_type, limit=5)
            if bullets:
                all_bullets[task_type] = [
                    {"id": b.id, "content": b.content, "utility": b.utility_score}
                    for b in bullets
                ]

        return json.dumps({
            "project_path": path,
            "task_types": list(all_bullets.keys()),
            "bullets": all_bullets,
            "reminder": "Apply these patterns to your work. Call report_feedback() when done.",
        }, indent=2)

    @mcp.resource("delia://memories", name="Available Memories", description="List of Delia's memory files")
    async def resource_memories() -> str:
        """
        List available Delia memory files.

        Returns a JSON list of memory names that can be loaded via
        the `context` parameter in delegate/plant tools.
        """
        memories = []
        if MEMORY_DIR.exists():
            for f in MEMORY_DIR.glob("*.md"):
                memories.append(f.stem)
        return json.dumps({"memories": sorted(memories)}, indent=2)

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

    @mcp.tool()
    async def semantic_search(
        query: str,
        top_k: int = 10,
        file_pattern: str | None = None,
    ) -> str:
        """Search codebase semantically using embeddings.

        Finds files/code by meaning, not just text matching.
        Use this for natural language queries like "authentication logic"
        or "database connection handling".

        Args:
            query: Natural language search query
            top_k: Maximum number of results (default 10)
            file_pattern: Optional glob filter (e.g., "*.py", "tests/**")

        Returns:
            Ranked results with paths, scores, and content summaries
        """
        import fnmatch
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
        """Get files related to this file via imports/dependencies.

        Returns files within N hops in the dependency graph.
        Useful for understanding what code might be affected by changes.

        Args:
            file_path: Path to the file to analyze
            depth: How many dependency hops to follow (default 2)

        Returns:
            List of related files with their relationship type
        """
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