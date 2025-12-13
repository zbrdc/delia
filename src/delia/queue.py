# Copyright (C) 2023 the project owner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Model queue system for GPU memory management."""

import asyncio
import heapq
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from .messages import StatusEvent, get_display_event, get_status_message

log = structlog.get_logger()


@dataclass(order=True)
class QueuedRequest:
    """Represents a queued LLM request with priority."""

    priority: int  # Lower number = higher priority
    timestamp: datetime
    request_id: str
    model_name: str
    task_type: str
    content_length: int
    future: asyncio.Future = field(compare=False)

    def __post_init__(self):
        # Tie-breaker: earlier requests get priority
        self.timestamp = self.timestamp or datetime.now()


class ModelQueue:
    """
    Intelligent queue system for GPU memory management.

    Prevents concurrent model loading and manages GPU memory efficiently.
    Queues requests when models need loading, prioritizes by model size and urgency.
    """

    def __init__(self):
        self.loaded_models: dict[str, dict[str, Any]] = {}  # model_name -> metadata
        self.loading_models: set[str] = set()  # Models currently being loaded
        self.request_queues: dict[str, list[QueuedRequest]] = {}  # model_name -> priority queue
        self.lock = asyncio.Lock()
        self.request_counter = 0

        # Model size estimates (in GB VRAM, rough estimates)
        self.model_sizes = {
            "Qwen3-4B": 4.0,
            "Qwen3-14B": 14.0,
            "Qwen3-30B": 30.0,
            "Qwen3-4B-Q4_K_M": 2.5,  # Quantized versions
            "Qwen3-14B-Q4_K_M": 8.0,
            "Qwen3-30B-Q4_K_M": 16.0,
        }

        # GPU memory limit (assume 24GB for RTX 3090/4090, adjust as needed)
        self.gpu_memory_limit_gb = 24.0
        self.memory_buffer_gb = 2.0  # Keep 2GB free

        # Queue health metrics
        self.total_queued = 0
        self.total_processed = 0
        self.max_queue_depth = 0
        self.queue_timeouts = 0

    def get_model_size(self, model_name: str) -> float:
        """Get estimated VRAM usage for a model."""
        # Try exact match first
        if model_name in self.model_sizes:
            return self.model_sizes[model_name]

        # Try partial matches
        for key, size in self.model_sizes.items():
            if key.lower() in model_name.lower():
                return size

        # Default estimate based on model name patterns
        if "30b" in model_name.lower():
            return 16.0  # Quantized 30B
        elif "14b" in model_name.lower():
            return 8.0  # Quantized 14B
        else:
            return 4.0  # Default 4B model

    def get_available_memory(self) -> float:
        """Calculate available GPU memory."""
        used_memory = sum(self.get_model_size(model) for model in self.loaded_models)
        return max(0, self.gpu_memory_limit_gb - used_memory - self.memory_buffer_gb)

    def can_load_model(self, model_name: str) -> bool:
        """Check if a model can be loaded given current memory."""
        model_size = self.get_model_size(model_name)
        return self.get_available_memory() >= model_size

    def calculate_priority(self, task_type: str, content_length: int, model_name: str) -> int:
        """Calculate request priority (lower = higher priority)."""
        priority = 0

        # Task urgency (thinking tasks are most urgent)
        if task_type in ("think", "thinking"):
            priority -= 100
        elif task_type in ("plan", "analyze"):
            priority -= 50
        elif task_type in ("review", "critique"):
            priority -= 25

        # Content size (smaller = higher priority to avoid timeouts)
        if content_length < 1000:
            priority -= 20
        elif content_length > 50000:
            priority += 20  # Large content can wait

        # Model size (smaller models = higher priority)
        model_size = self.get_model_size(model_name)
        if model_size <= 4.0:
            priority -= 10
        elif model_size > 16.0:
            priority += 10

        return priority

    async def acquire_model(
        self, model_name: str, task_type: str = "unknown", content_length: int = 0
    ) -> tuple[bool, asyncio.Future | None]:
        """
        Acquire a model for use.

        Returns:
            - (True, None) if model is immediately available
            - (False, Future) if request is queued (caller must await the Future)

        The Future will be resolved when the model finishes loading and the request is processed.
        """
        async with self.lock:
            # Model already loaded and not loading
            if model_name in self.loaded_models and model_name not in self.loading_models:
                self.loaded_models[model_name]["last_used"] = datetime.now()
                return (True, None)

            # Model is currently loading
            if model_name in self.loading_models:
                # Queue the request
                request_id = f"req_{self.request_counter}"
                self.request_counter += 1

                priority = self.calculate_priority(task_type, content_length, model_name)
                request_future: asyncio.Future[tuple[bool, str | None]] = asyncio.Future()
                queued_request = QueuedRequest(
                    priority=priority,
                    timestamp=datetime.now(),
                    request_id=request_id,
                    model_name=model_name,
                    task_type=task_type,
                    content_length=content_length,
                    future=request_future,
                )

                if model_name not in self.request_queues:
                    self.request_queues[model_name] = []

                heapq.heappush(self.request_queues[model_name], queued_request)

                # Track queue stats
                queue_length = len(self.request_queues[model_name])
                self.total_queued += 1
                self.max_queue_depth = max(self.max_queue_depth, queue_length)

                log.info(
                    get_display_event("model_queued"),
                    model=model_name,
                    queue_length=queue_length,
                    priority=priority,
                    task_type=task_type,
                    request_id=request_id,
                    status_msg=get_status_message(StatusEvent.REQUEST_RECEIVED),
                    log_type="QUEUE",
                )

                return (False, request_future)

            # Model not loaded - check if we can load it
            if not self.can_load_model(model_name):
                # Need to unload some models first
                await self._make_room_for_model(model_name)

            # Start loading the model
            self.loading_models.add(model_name)

            log.info(
                get_display_event("model_loading_start"),
                model=model_name,
                available_memory_gb=round(self.get_available_memory(), 1),
                status_msg=get_status_message(StatusEvent.MODEL_LOADING),
            )

            return (True, None)

    async def _make_room_for_model(self, new_model_name: str) -> None:
        """Unload least recently used models to make room for new model."""
        new_model_size = self.get_model_size(new_model_name)
        available_memory = self.get_available_memory()

        if available_memory >= new_model_size:
            return  # No need to unload

        # Sort loaded models by last used time (oldest first)
        unload_candidates = sorted(self.loaded_models.items(), key=lambda x: x[1]["last_used"])

        freed_memory: float = 0.0
        to_unload: list[str] = []

        for model_name, _metadata in unload_candidates:
            if freed_memory >= (new_model_size - available_memory):
                break
            freed_memory += self.get_model_size(model_name)
            to_unload.append(model_name)

        # Unload the selected models
        for model_name in to_unload:
            del self.loaded_models[model_name]
            log.info(
                get_display_event("model_unloaded"),
                model=model_name,
                reason="memory_pressure",
                status_msg=get_status_message(StatusEvent.CLEANUP),
            )

    async def release_model(self, model_name: str, success: bool = True) -> None:
        """Release a model after use and process queued requests."""
        async with self.lock:
            if model_name in self.loading_models:
                self.loading_models.remove(model_name)

                if success:
                    # Mark model as loaded
                    self.loaded_models[model_name] = {
                        "loaded_at": datetime.now(),
                        "last_used": datetime.now(),
                        "size_gb": self.get_model_size(model_name),
                    }

                    log.info(
                        get_display_event("model_loaded"),
                        model=model_name,
                        loaded_count=len(self.loaded_models),
                        available_memory_gb=round(self.get_available_memory(), 1),
                        status_msg=get_status_message(StatusEvent.READY),
                    )

                    # Process queued requests for this model
                    await self._process_queue(model_name)
                else:
                    # Loading failed - fail all queued requests
                    if model_name in self.request_queues:
                        for queued_request in self.request_queues[model_name]:
                            if not queued_request.future.done():
                                queued_request.future.set_exception(Exception(f"Failed to load model {model_name}"))
                        del self.request_queues[model_name]

    async def _process_queue(self, model_name: str) -> None:
        """Process queued requests for a newly loaded model."""
        if model_name not in self.request_queues:
            return

        queue = self.request_queues[model_name]
        processed = 0

        while queue and processed < 5:  # Process up to 5 requests at once
            queued_request = heapq.heappop(queue)

            if not queued_request.future.done():
                # Wake up the waiting request
                queued_request.future.set_result(True)
                processed += 1
                self.total_processed += 1

                wait_time_ms = (datetime.now() - queued_request.timestamp).total_seconds() * 1000
                log.info(
                    get_display_event("queue_processed"),
                    model=model_name,
                    request_id=queued_request.request_id,
                    wait_time_ms=round(wait_time_ms, 1),
                    remaining=len(queue),
                    status_msg="Request ready for processing!",
                    log_type="QUEUE",
                )

        if not queue:
            del self.request_queues[model_name]

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status for monitoring."""
        current_queue_depth = sum(len(q) for q in self.request_queues.values())
        return {
            "loaded_models": list(self.loaded_models.keys()),
            "loading_models": list(self.loading_models),
            "queued_requests": {model: len(queue) for model, queue in self.request_queues.items()},
            "available_memory_gb": round(self.get_available_memory(), 1),
            "total_loaded_gb": round(sum(m["size_gb"] for m in self.loaded_models.values()), 1),
            # Queue health metrics
            "queue_stats": {
                "total_queued": self.total_queued,
                "total_processed": self.total_processed,
                "current_queue_depth": current_queue_depth,
                "max_queue_depth": self.max_queue_depth,
                "queue_timeouts": self.queue_timeouts,
            },
        }
