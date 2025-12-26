# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MCP Resource definitions for Delia.

Resources expose data via the MCP resource protocol (delia://...) for
cross-server workflows. Only registered in FULL profile.

Resources:
- delia://file/{path}: Read files as MCP resources
- delia://stats: Usage statistics
- delia://backends: Backend health status
- delia://config: Current configuration
- delia://playbook: Active playbook bullets
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from fastmcp import FastMCP

from ..container import get_container
from ..backend_manager import backend_manager
from ..context import current_project_path, get_project_path
from ..config import config

log = structlog.get_logger()


def register_resource_tools(mcp: FastMCP):
    """Register MCP resources with FastMCP."""

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
            file_path = get_project_path() / file_path

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
        path = current_project_path.get() or str(get_project_path())
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
            "reminder": "Apply these patterns to your work. Call complete_task() when done.",
        }, indent=2)
