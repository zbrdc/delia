# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Playbook tool implementations for Delia MCP tools.

Provides implementation functions for:
- get_playbook: Retrieve strategic playbook bullets
- report_feedback: Record bullet effectiveness
- get_project_context: Get high-level project understanding
- playbook_stats: Get playbook statistics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ..context import get_project_path
from .handlers_enforcement import get_tracker

if TYPE_CHECKING:
    from typing import Any

log = structlog.get_logger()


# =============================================================================
# Playbook Tool Implementations
# =============================================================================


async def get_playbook_impl(
    task_type: str = "general", limit: int = 15, path: str | None = None
) -> str:
    """Get strategic playbook bullets for a task type."""
    from ..playbook import playbook_manager

    # Set project path if provided (ensures project-specific playbooks)
    project_path = path or str(get_project_path())
    if path:
        playbook_manager.set_project(Path(path))

    # Record playbook query for framework enforcement
    get_tracker().record_playbook_query(project_path)

    bullets = playbook_manager.get_top_bullets(task_type, limit)
    if not bullets:
        # Try project-level playbook if task-specific is empty
        bullets = playbook_manager.get_top_bullets("project", limit)

    if not bullets:
        return json.dumps(
            {
                "status": "empty",
                "message": f"No playbook bullets found for '{task_type}'. Run 'delia index --summarize' to generate.",
                "bullets": [],
            }
        )

    result = {
        "task_type": task_type,
        "bullet_count": len(bullets),
        "bullets": [
            {
                "id": b.id,
                "content": b.content,
                "section": b.section,
                "utility_score": round(b.utility_score, 3),
                "helpful_count": b.helpful_count,
                "harmful_count": b.harmful_count,
            }
            for b in bullets
        ],
    }
    return json.dumps(result, indent=2)


async def report_feedback_impl(
    bullet_id: str, task_type: str, helpful: bool, path: str | None = None
) -> str:
    """Report whether a playbook bullet was helpful for a task."""
    from ..playbook import playbook_manager

    # Set project path if provided
    if path:
        playbook_manager.set_project(Path(path))

    success = playbook_manager.record_feedback(bullet_id, task_type, helpful)

    if success:
        log.info(
            "playbook_feedback_recorded",
            bullet_id=bullet_id,
            task_type=task_type,
            helpful=helpful,
        )
        return json.dumps(
            {
                "status": "recorded",
                "bullet_id": bullet_id,
                "helpful": helpful,
                "message": f"Feedback recorded. Bullet {'helped' if helpful else 'did not help'} with task.",
            }
        )
    else:
        return json.dumps(
            {
                "status": "not_found",
                "bullet_id": bullet_id,
                "message": f"Bullet '{bullet_id}' not found in playbook '{task_type}'.",
            }
        )


async def get_project_context_impl(path: str | None = None) -> str:
    """Get high-level project understanding from playbook and summaries."""
    from ..playbook import playbook_manager
    from ..orchestration.summarizer import get_summarizer
    from ..mcp_server import set_project_context

    # Set project path for both playbooks and MCP context
    if path:
        playbook_manager.set_project(Path(path))
        set_project_context(path)

    # Get project playbook bullets
    bullets = playbook_manager.get_top_bullets("project", limit=10)

    # Get project overview from summarizer
    summarizer = get_summarizer()
    overview = (
        summarizer.get_project_overview()
        if hasattr(summarizer, "get_project_overview")
        else None
    )

    result = {
        "playbook_bullets": [
            {"id": b.id, "content": b.content, "section": b.section} for b in bullets
        ],
        "project_overview": overview,
        "playbook_stats": playbook_manager.get_stats(),
    }
    return json.dumps(result, indent=2)


async def playbook_stats_impl(task_type: str | None = None) -> str:
    """Get playbook statistics and utility scores."""
    from ..playbook import playbook_manager

    if task_type:
        bullets = playbook_manager.load_playbook(task_type)
        return json.dumps(
            {
                "task_type": task_type,
                "bullet_count": len(bullets),
                "bullets": [
                    {
                        "id": b.id,
                        "content": b.content[:100] + "..."
                        if len(b.content) > 100
                        else b.content,
                        "utility_score": round(b.utility_score, 3),
                        "helpful": b.helpful_count,
                        "harmful": b.harmful_count,
                    }
                    for b in bullets
                ],
            },
            indent=2,
        )
    else:
        return json.dumps(playbook_manager.get_stats(), indent=2)
