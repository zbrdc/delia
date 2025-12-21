# Copyright (C) 2024 Delia Contributors
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

"""Model queue system for GPU memory management.

This module provides intelligent queue management for GPU-based LLM inference.
It coordinates model loading/unloading across providers to efficiently manage
limited GPU memory while handling concurrent requests.
"""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import structlog

from .messages import StatusEvent, get_display_event, get_status_message

if TYPE_CHECKING:
    from .providers.base import LLMProvider

# Type alias for provider getter callback
ProviderGetter = Callable[[str], "LLMProvider | None"]

log = structlog.get_logger()

# Maximum queue depth per model to prevent unbounded growth and OOM
MAX_QUEUE_DEPTH = 100


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

    def __init__(self, provider_getter: ProviderGetter | None = None):
        """Initialize the model queue.

        Args:
            provider_getter: Callable that returns a provider instance given a provider name.
                           Used to call provider-specific load/unload methods.
                           If None, lifecycle methods are not called (soft tracking only).
        """
        self.loaded_models: dict[str, dict[str, Any]] = {}
        self.loading_models: set[str] = set()
        self.request_queues: dict[str, list[QueuedRequest]] = {}
        self.lock = asyncio.Lock()
        self.request_counter = 0

        # Provider lifecycle integration
        self._get_provider = provider_getter

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
        self.rejected_requests = 0  # Track requests rejected due to queue overflow

    def get_model_size(self, model_name: str) -> float:
        """Get estimated VRAM usage for a model."""
        model_lower = model_name.lower()
        
        # Try exact match first
        if model_name in self.model_sizes:
            return self.model_sizes[model_name]

        # Handle MoE models (Mixtral, Nemotron MoE, etc.)
        # These are often named like 8x7b or have \u0027moe\u0027 and \u0027a3b\u0027 (active params)
        if "8x7b" in model_lower:
            return 28.0  # Mixtral 8x7B (quantized)
        if "a3b" in model_lower or "a2b" in model_lower:
            # Active parameters vs total parameters
            if "30b" in model_lower:
                return 21.0  # Nemotron-3 30B MoE (quantized)
            if "70b" in model_lower or "72b" in model_lower:
                return 42.0

        # Try partial matches
        for key, size in self.model_sizes.items():
            if key.lower() in model_lower:
                return size

        # Default estimate based on model name patterns
        if "70b" in model_lower or "72b" in model_lower:
            return 40.0
        elif "30b" in model_lower or "32b" in model_lower:
            return 18.0
        elif "14b" in model_lower:
            return 9.0
        elif "7b" in model_lower or "8b" in model_lower:
            return 5.0
        else:
            return 4.0  # Default small model

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
        if task_type in ("think", "thinking", "reflection", "curation"):
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
        self,
        model_name: str,
        task_type: str = "unknown",
        content_length: int = 0,
        provider_name: str = "ollama",
    ) -> tuple[bool, asyncio.Future | None]:
        """
        Acquire a model for use.

        Args:
            model_name: Name of the model to acquire
            task_type: Type of task for priority calculation
            content_length: Size of content for priority calculation
            provider_name: Name of the provider (used for lifecycle callbacks)

        Returns:
            - (True, None) if model is immediately available
            - (False, Future) if request is queued (caller must await the Future)

        The Future will be resolved when the model finishes loading and the request is processed.
        """
        async with self.lock:
            log.debug("queue_acquire_attempt", model=model_name, loading=model_name in self.loading_models, loaded=model_name in self.loaded_models)
            # Model already loaded and not loading
            if model_name in self.loaded_models and model_name not in self.loading_models:
                self.loaded_models[model_name]["last_used"] = datetime.now()
                return (True, None)

            # Create queue if it doesn't exist
            if model_name not in self.request_queues:
                self.request_queues[model_name] = []

            # Model is currently loading - queue the request
            if model_name in self.loading_models:
                current_queue_length = len(self.request_queues[model_name])
                if current_queue_length >= MAX_QUEUE_DEPTH:
                    # Queue is full - reject the request
                    self.rejected_requests += 1
                    log.warning(
                        get_display_event("queue_full"),
                        model=model_name,
                        queue_length=current_queue_length,
                        max_depth=MAX_QUEUE_DEPTH,
                        rejected_count=self.rejected_requests,
                        status_msg="Queue full - request rejected",
                        log_type="QUEUE",
                    )
                    error_future: asyncio.Future[bool] = asyncio.Future()
                    error_future.set_exception(
                        RuntimeError(
                            f"Queue full for model {model_name}: "
                            f"{current_queue_length}/{MAX_QUEUE_DEPTH} requests queued."
                        )
                    )
                    return (False, error_future)

                # Queue the request
                request_id = f"req_{self.request_counter}"
                self.request_counter += 1

                priority = self.calculate_priority(task_type, content_length, model_name)
                request_future: asyncio.Future[bool] = asyncio.Future()
                queued_request = QueuedRequest(
                    priority=priority,
                    timestamp=datetime.now(),
                    request_id=request_id,
                    model_name=model_name,
                    task_type=task_type,
                    content_length=content_length,
                    future=request_future,
                )

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
                    status_msg="Waiting for model to load...",
                    log_type="QUEUE",
                )

                return (False, request_future)

            # Model not loaded - check if we can load it
            if not self.can_load_model(model_name):
                # Need to unload some models first
                models_to_unload = self._get_unload_candidates(model_name, provider_name)
                if models_to_unload:
                    log.info("memory_pressure_trigger_unload", to_unload=models_to_unload, for_model=model_name)
                    # We release the lock because unloading takes time
                    self.lock.release()
                    try:
                        await self._do_unload_models(models_to_unload)
                    finally:
                        await self.lock.acquire()
                    
                    # Re-check after unloading (in case something else loaded in between)
                    if not self.can_load_model(model_name):
                        log.warning("memory_still_insufficient_after_unload", model=model_name)

            # Re-check if someone else started loading while we released the lock
            if model_name in self.loading_models:
                log.debug("queue_load_raced", model=model_name)
                # Recurse or just handle as loading?
                # Simplest is to just call acquire again (it will hit the loading check)
                return await self.acquire_model(model_name, task_type, content_length, provider_name)

            # Start loading the model
            self.loading_models.add(model_name)
            
            log.info(
                get_display_event("model_loading_start"),
                model=model_name,
                provider=provider_name,
                available_memory_gb=round(self.get_available_memory(), 1),
                status_msg=get_status_message(StatusEvent.MODEL_LOADING),
                log_type="QUEUE",
            )

            # Trigger provider-specific model preloading
            # RELEASE LOCK during load to avoid blocking other requests
            if self._get_provider:
                provider = self._get_provider(provider_name)
                if provider and hasattr(provider, "load_model"):
                    log.debug("queue_releasing_lock_for_preload", model=model_name)
                    self.lock.release()
                    try:
                        await provider.load_model(model_name)
                    finally:
                        await self.lock.acquire()
                        log.debug("queue_reacquired_lock_after_preload", model=model_name)
            
            # Model loading complete (either via load_model or it will load on first call)
            # Mark it as loaded so subsequent requests see it
            self.loading_models.discard(model_name)
            self.loaded_models[model_name] = {
                "loaded_at": datetime.now(),
                "last_used": datetime.now(),
                "size_gb": self.get_model_size(model_name),
                "provider": provider_name,
            }

            # Wake up any requests that queued during the lock release window
            await self._process_queue(model_name)

            # Return True so initiating request proceeds immediately
            return (True, None)

    def _get_unload_candidates(self, new_model_name: str, provider_name: str) -> list[tuple[str, str]]:
        """Identify models that need to be unloaded. Returns list of (model_name, provider_name)."""
        new_model_size = self.get_model_size(new_model_name)
        available_memory = self.get_available_memory()

        if available_memory >= new_model_size:
            return []

        unload_candidates = sorted(self.loaded_models.items(), key=lambda x: x[1]["last_used"])
        freed_memory = 0.0
        to_unload = []

        for model_name, metadata in unload_candidates:
            if freed_memory >= (new_model_size - available_memory):
                break
            freed_memory += self.get_model_size(model_name)
            to_unload.append((model_name, metadata.get("provider", provider_name)))
            
        return to_unload

    async def _do_unload_models(self, to_unload: list[tuple[str, str]]) -> None:
        """Perform the actual unloading of models."""
        for model_name, model_provider in to_unload:
            if self._get_provider:
                provider = self._get_provider(model_provider)
                if provider and hasattr(provider, "unload_model"):
                    try:
                        await provider.unload_model(model_name)
                        log.info(get_display_event("model_unloaded"), model=model_name, provider=model_provider, reason="memory_pressure")
                    except Exception as e:
                        log.warning("model_unload_error", model=model_name, error=str(e))
            
            # Remove from tracking even if provider call failed
            async with self.lock:
                if model_name in self.loaded_models:
                    del self.loaded_models[model_name]

    async def release_model(
        self,
        model_name: str,
        success: bool = True,
        provider_name: str = "ollama",
    ) -> None:
        """Release a model after use and process queued requests.

        Args:
            model_name: Name of the model to release
            success: Whether the model call was successful
            provider_name: Provider that handled this model (for tracking)
        """
        async with self.lock:
            if model_name in self.loading_models:
                self.loading_models.remove(model_name)

                # Use stored provider if available, otherwise use passed one
                actual_provider = getattr(self, "_pending_provider", provider_name)
                if hasattr(self, "_pending_provider"):
                    delattr(self, "_pending_provider")

                if success:
                    # Mark model as loaded with provider info
                    self.loaded_models[model_name] = {
                        "loaded_at": datetime.now(),
                        "last_used": datetime.now(),
                        "size_gb": self.get_model_size(model_name),
                        "provider": actual_provider,  # Track which provider loaded this
                    }

                    log.info(
                        get_display_event("model_loaded"),
                        model=model_name,
                        provider=actual_provider,
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
        
        # Wake up ALL requests waiting for this model
        # They will then proceed to re-verify availability
        while queue:
            queued_request = heapq.heappop(queue)

            if not queued_request.future.done():
                # Wake up the waiting request
                queued_request.future.set_result(True)
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
                "max_queue_limit": MAX_QUEUE_DEPTH,
                "queue_timeouts": self.queue_timeouts,
                "rejected_requests": self.rejected_requests,
            },
        }

    async def sync_with_backend(self, provider_name: str = "ollama") -> dict[str, Any]:
        """Synchronize queue state with actual backend loaded models.

        This method queries the backend to find which models are actually loaded,
        then updates the queue's internal state to match reality. Useful for:
        - Initial startup synchronization
        - Recovery after queue state corruption
        - Periodic reconciliation

        Args:
            provider_name: Provider to sync with

        Returns:
            Dict with sync results (added, removed, unchanged counts)
        """
        if not self._get_provider:
            return {"error": "No provider getter configured", "synced": False}

        provider = self._get_provider(provider_name)
        if not provider or not hasattr(provider, "list_loaded_models"):
            return {"error": f"Provider {provider_name} does not support list_loaded_models", "synced": False}

        try:
            actual_loaded = await provider.list_loaded_models()
        except Exception as e:
            return {"error": f"Failed to query backend: {e}", "synced": False}

        async with self.lock:
            added = 0
            removed = 0
            unchanged = 0

            # Remove models from tracking that are no longer loaded
            tracked_models = list(self.loaded_models.keys())
            for model in tracked_models:
                # Only remove if this model was loaded by this provider
                model_provider = self.loaded_models[model].get("provider", provider_name)
                if model_provider == provider_name and model not in actual_loaded:
                    del self.loaded_models[model]
                    removed += 1
                    log.info(
                        "model_sync_removed",
                        model=model,
                        provider=provider_name,
                        reason="not_in_backend",
                    )

            # Add models that are loaded but not tracked
            for model in actual_loaded:
                if model not in self.loaded_models:
                    self.loaded_models[model] = {
                        "loaded_at": datetime.now(),
                        "last_used": datetime.now(),
                        "size_gb": self.get_model_size(model),
                        "provider": provider_name,
                    }
                    added += 1
                    log.info(
                        "model_sync_added",
                        model=model,
                        provider=provider_name,
                        reason="found_in_backend",
                    )
                else:
                    unchanged += 1

            return {
                "synced": True,
                "provider": provider_name,
                "added": added,
                "removed": removed,
                "unchanged": unchanged,
                "total_loaded": len(self.loaded_models),
            }
