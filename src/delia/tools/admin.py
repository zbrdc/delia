# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Administrative MCP tools for Delia.
"""

from __future__ import annotations

import json
from typing import Any

import humanize
import structlog
from fastmcp import FastMCP

from ..container import get_container
from ..config import get_backend_health, VALID_MODELS
from ..routing import BackendScorer

log = structlog.get_logger()


def register_admin_tools(mcp: FastMCP):
    """Register administrative tools with FastMCP."""
    
    container = get_container()

    @mcp.tool()
    async def health() -> str:
        """
        Check health status of Delia and all configured GPU backends.
        """
        health_status = await container.backend_manager.get_health_status()
        weights = container.backend_manager.get_scoring_weights()
        scorer = BackendScorer(weights=weights)
        backend_lookup = {b.id: b for b in container.backend_manager.backends.values()}
        
        for backend_info in health_status["backends"]:
            backend_obj = backend_lookup.get(backend_info["id"])
            if backend_obj and backend_info["enabled"]:
                backend_info["score"] = round(scorer.score(backend_obj), 3)
                from ..config import get_backend_metrics
                metrics = get_backend_metrics(backend_info["id"])
                if metrics.total_requests > 0:
                    backend_info["metrics"] = {
                        "success_rate": f"{metrics.success_rate * 100:.1f}%",
                        "latency_p50_ms": round(metrics.latency_p50, 1),
                        "throughput_tps": round(metrics.tokens_per_second, 1),
                        "total_requests": metrics.total_requests,
                    }

        model_usage, _, _, _ = container.stats_service.get_snapshot()
        total_quick_tokens = model_usage["quick"]["tokens"]
        total_coder_tokens = model_usage["coder"]["tokens"]
        total_moe_tokens = model_usage["moe"]["tokens"]
        total_thinking_tokens = model_usage["thinking"]["tokens"]
        local_tokens = total_quick_tokens + total_coder_tokens + total_moe_tokens + total_thinking_tokens
        local_calls = sum(model_usage[t]["calls"] for t in ("quick", "coder", "moe", "thinking"))

        local_savings = (local_tokens / 1000) * container.config.gpt4_cost_per_1k_tokens
        from ..voting_stats import get_voting_stats_tracker
        voting_stats = get_voting_stats_tracker().get_stats()

        status = {
            "status": health_status["status"],
            "active_backend": health_status["active_backend"],
            "backends": health_status["backends"],
            "routing": health_status["routing"],
            "usage": {
                "total_calls": humanize.intcomma(local_calls),
                "total_tokens": humanize.intword(local_tokens),
                "estimated_savings": f"${local_savings:,.2f}",
            },
            "voting": voting_stats,
        }
        return json.dumps(status, indent=2)

    @mcp.tool()
    async def models() -> str:
        """
        List all configured models across all GPU backends.
        """
        backends = container.backend_manager.get_enabled_backends()
        from ..mcp_server import get_loaded_models
        loaded = await get_loaded_models()
        
        result = {"backends": [], "currently_loaded": loaded}
        for b in backends:
            result["backends"].append({"id": b.id, "name": b.name, "provider": b.provider, "models": b.models})
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def switch_backend(backend_id: str) -> str:
        """
        Switch the active LLM backend.
        """
        if container.backend_manager.set_active_backend(backend_id):
            return f"Switched to backend: {backend_id}"
        return f"Error: Backend '{backend_id}' not found or disabled."

    @mcp.tool()
    async def switch_model(tier: str, model_name: str) -> str:
        """
        Switch the model for a specific tier at runtime.
        """
        if tier not in VALID_MODELS:
            return f"Error: Invalid tier '{tier}'"
        return "Model switching implemented via settings.json"

    @mcp.tool()
    async def queue_status() -> str:
        """
        Get current status of the model queue system.
        """
        return json.dumps({"status": "active"}, indent=2)

    @mcp.tool()
    async def get_model_info_tool(model_name: str) -> str:
        """
        Get detailed information about a specific model.
        """
        from ..mcp_server import get_model_info
        info = get_model_info(model_name)
        return json.dumps(info, indent=2)