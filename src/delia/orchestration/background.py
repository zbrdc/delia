# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Background task management for Delia.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from ..backend_manager import backend_manager
from ..config import get_prewarm_tracker
from ..container import get_container

log = structlog.get_logger()

# Background tasks set to prevent garbage collection of fire-and-forget tasks
_background_tasks: set[asyncio.Task[Any]] = set()


def schedule_background_task(coro: Any) -> None:
    """Schedule a fire-and-forget background task, preventing garbage collection."""
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except RuntimeError:
        pass  # No running loop


# Pre-warming state
_prewarm_task_started = False


async def _prewarm_check_loop() -> None:
    """
    Background task that periodically checks and pre-warms predicted models.

    Runs until cancelled, checking every N minutes if models should be pre-loaded
    based on hourly usage patterns learned by PrewarmTracker.
    """
    global _prewarm_task_started
    _prewarm_task_started = True
    
    container = get_container()
    model_queue = container.model_queue

    while True:
        try:
            # Get prewarm config
            prewarm_config = backend_manager.routing_config.get("prewarm", {})
            if not prewarm_config.get("enabled", False):
                # If disabled, wait a bit then check again (in case config changes)
                await asyncio.sleep(60)
                continue

            interval_minutes = prewarm_config.get("check_interval_minutes", 5)

            # Get predicted tiers for current hour
            tracker = get_prewarm_tracker()
            predicted_tiers = tracker.get_predicted_tiers()

            if predicted_tiers:
                # Get active backend to resolve tier -> model name
                active_backend = backend_manager.get_active_backend()
                if active_backend:
                    for tier in predicted_tiers:
                        model_name = active_backend.models.get(tier)
                        if model_name:
                            try:
                                # Trigger model loading via queue
                                await model_queue.acquire_model(
                                    model_name,
                                    task_type="prewarm",
                                    content_length=0,
                                    provider_name=active_backend.provider,
                                )
                                log.debug(
                                    "prewarm_model_acquired",
                                    tier=tier,
                                    model=model_name,
                                    backend=active_backend.id,
                                )
                            except Exception as e:
                                log.debug("prewarm_model_failed", tier=tier, error=str(e))

            # Wait for next check interval
            await asyncio.sleep(interval_minutes * 60)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("prewarm_loop_error", error=str(e))
            await asyncio.sleep(60)  # Wait before retrying


def start_prewarm_task() -> None:
    """Start the pre-warming background task if not already running."""
    global _prewarm_task_started
    if not _prewarm_task_started:
        schedule_background_task(_prewarm_check_loop())
