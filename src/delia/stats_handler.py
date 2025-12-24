# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Stats handling logic for Delia.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import structlog

from . import paths
from .backend_manager import backend_manager
from .config import (
    config,
    get_backend_health,
    save_affinity,
    save_backend_metrics,
    save_prewarm,
)
from .container import get_container
from .orchestration.background import schedule_background_task

log = structlog.get_logger()

# Circuit breaker stats file (for dashboard)
CIRCUIT_BREAKER_FILE = paths.CIRCUIT_BREAKER_FILE


def save_circuit_breaker_stats():
    """Save circuit breaker status to disk for dashboard."""
    try:
        active_backend = backend_manager.get_active_backend()
        data = {
            "ollama": get_backend_health("ollama").get_status(),
            "llamacpp": get_backend_health("llamacpp").get_status(),
            "active_backend": {
                "id": active_backend.id,
                "name": active_backend.name,
                "provider": active_backend.provider,
                "type": active_backend.type,
            }
            if active_backend
            else None,
            "timestamp": datetime.now().isoformat(),
        }
        temp_file = CIRCUIT_BREAKER_FILE.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(CIRCUIT_BREAKER_FILE)  # Atomic on POSIX
    except Exception as e:
        log.warning("circuit_breaker_save_failed", error=str(e))


def update_stats_sync(
    model_tier: str,
    task_type: str,
    original_task: str,
    tokens: int,
    elapsed_ms: int,
    content_preview: str,
    enable_thinking: bool,
    backend: str = "ollama",
) -> None:
    """
    Thread-safe update of all in-memory stats via StatsService.

    This wrapper maintains the same interface for provider callbacks
    while delegating to the StatsService.
    """
    container = get_container()
    stats_service = container.stats_service
    
    # Determine backend type from config
    backend_type = config.get_backend_type(backend)

    # Delegate to stats service
    stats_service.record_call(
        model_tier=model_tier,
        task_type=task_type,
        original_task=original_task,
        tokens=tokens,
        elapsed_ms=elapsed_ms,
        content_preview=content_preview,
        enable_thinking=enable_thinking,
        backend=backend,
        backend_type=backend_type,
    )


async def save_all_stats_async():
    """
    Save all stats asynchronously via StatsService.

    Saves:
    - Model usage and task stats via stats_service
    - Live logs and circuit breaker status
    - Backend performance metrics
    - Task-backend affinity scores
    """
    container = get_container()
    stats_service = container.stats_service
    logging_service = container.logging_service
    
    # Save model/task stats via service
    await stats_service.save_all()

    # Save other data (live logs, circuit breaker, backend metrics, affinity)
    await logging_service.save_live_logs_async()
    await asyncio.to_thread(save_circuit_breaker_stats)
    await asyncio.to_thread(save_backend_metrics)
    await asyncio.to_thread(save_affinity)
    await asyncio.to_thread(save_prewarm)


def save_stats_background() -> None:
    """Schedule stats saving as a background task (non-blocking)."""
    schedule_background_task(save_all_stats_async())
