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

"""LLM calling infrastructure.

This module provides the unified LLM call dispatcher that routes requests
to the appropriate backend (Ollama, llama.cpp, Gemini).

Key components:
- Provider factory with lazy initialization and caching
- call_llm: Non-streaming LLM calls with queue management
- call_llm_stream: Streaming LLM calls
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

import structlog

from .backend_manager import BackendConfig, backend_manager
from .config import config
from .errors import BackendError, InitError, QueueError
from .providers.base import StreamChunk
from .providers.gemini import GeminiProvider
from .providers.llamacpp import LlamaCppProvider
from .providers.ollama import OllamaProvider

if TYPE_CHECKING:
    from .queue import ModelQueue

log = structlog.get_logger()

# ============================================================
# PROVIDER FACTORY (lazy initialization)
# ============================================================

# Cache for provider instances (created lazily on first use)
_provider_cache: dict[str, OllamaProvider | LlamaCppProvider | GeminiProvider] = {}

# Provider name to class mapping (includes aliases)
_PROVIDER_CLASS_MAP: dict[str, type[OllamaProvider | LlamaCppProvider | GeminiProvider]] = {
    "ollama": OllamaProvider,
    "llamacpp": LlamaCppProvider,
    "llama.cpp": LlamaCppProvider,
    "lmstudio": LlamaCppProvider,
    "vllm": LlamaCppProvider,
    "openai": LlamaCppProvider,
    "custom": LlamaCppProvider,
    "gemini": GeminiProvider,
}

# Callbacks set during initialization
_stats_callback: Callable[..., None] | None = None
_save_stats_callback: Callable[[], None] | None = None
_model_queue: ModelQueue | None = None


def init_llm_module(
    stats_callback: Callable[..., None],
    save_stats_callback: Callable[[], None],
    model_queue: "ModelQueue",
) -> None:
    """Initialize the LLM module with required callbacks.

    Must be called before using call_llm/call_llm_stream.

    Args:
        stats_callback: Function to update stats synchronously
        save_stats_callback: Function to save stats asynchronously
        model_queue: ModelQueue instance for managing model loading
    """
    global _stats_callback, _save_stats_callback, _model_queue
    _stats_callback = stats_callback
    _save_stats_callback = save_stats_callback
    _model_queue = model_queue
    # Wire up provider getter to model queue
    model_queue._get_provider = get_provider


def get_provider(provider_name: str) -> OllamaProvider | LlamaCppProvider | GeminiProvider | None:
    """Get or create a provider instance for the given provider name.

    Uses lazy initialization - providers are created on first use and cached.

    Args:
        provider_name: Provider type (ollama, llamacpp, gemini, etc.)

    Returns:
        Provider instance or None if provider not found
    """
    if provider_name not in _PROVIDER_CLASS_MAP:
        return None

    # Return cached instance if available
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    # Create new instance with dependencies
    if _stats_callback is None or _save_stats_callback is None:
        raise InitError("LLM module not initialized. Call init_llm_module first.")

    provider_class = _PROVIDER_CLASS_MAP[provider_name]
    provider = provider_class(
        config=config,
        backend_manager=backend_manager,
        stats_callback=_stats_callback,
        save_stats_callback=_save_stats_callback,
    )
    _provider_cache[provider_name] = provider
    return provider


async def call_llm(
    model: str,
    prompt: str,
    system: str | None = None,
    enable_thinking: bool = False,
    task_type: str = "unknown",
    original_task: str = "unknown",
    language: str = "unknown",
    content_preview: str = "",
    backend: str | BackendConfig | None = None,
    backend_obj: BackendConfig | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Unified LLM call dispatcher that routes to the appropriate backend.

    Args:
        model: Model name/tier
        prompt: The prompt to send
        system: Optional system prompt
        enable_thinking: Enable thinking mode for supported models
        task_type: Type of task for tracking
        language: Detected language for tracking
        content_preview: Preview for logging
        backend: Override backend ID or provider name
        backend_obj: Optional BackendConfig object to use directly
        max_tokens: Maximum response tokens (limits verbosity)
        tools: Optional list of OpenAI-format tool schemas for native tool calling
        tool_choice: Optional tool choice strategy ("auto", "none", or specific tool name)
        temperature: Sampling temperature (0.0-2.0, higher = more random/diverse)

    Returns:
        Dict with 'success', 'response', 'tokens', 'error' keys
    """
    if _model_queue is None:
        raise InitError("LLM module not initialized. Call init_llm_module first.")

    # Determine backend to use first (needed for provider_name in queue)
    active_backend = None

    # 1. Use passed object if available
    if backend_obj:
        active_backend = backend_obj
    # 2. Resolve by ID or provider name
    elif backend:
        if isinstance(backend, BackendConfig):
            active_backend = backend
        else:
            active_backend = backend_manager.get_backend(backend)
            if not active_backend:
                for b in backend_manager.get_enabled_backends():
                    if b.provider == backend:
                        active_backend = b
                        break
    # 3. Use default active backend
    else:
        active_backend = backend_manager.get_active_backend()

    if not active_backend:
        return {"success": False, "error": "No active backend found"}

    provider_name = active_backend.provider

    # Acquire model from queue (prevents concurrent loading)
    content_length = len(prompt) + len(system or "")
    is_available, queue_future = await _model_queue.acquire_model(
        model, task_type, content_length, provider_name=provider_name
    )

    # If model is not immediately available, wait for it
    if not is_available and queue_future:
        try:
            await asyncio.wait_for(queue_future, timeout=300)
        except TimeoutError:
            _model_queue.queue_timeouts += 1
            await _model_queue.release_model(model, success=False, provider_name=provider_name)
            log.warning(
                "queue_timeout",
                model=model,
                provider=provider_name,
                wait_seconds=300,
                total_timeouts=_model_queue.queue_timeouts,
                log_type="QUEUE",
            )
            return {"success": False, "error": f"Timeout waiting for model {model} to load (waited 5 minutes)"}
        except Exception as e:
            await _model_queue.release_model(model, success=False, provider_name=provider_name)
            log.error("queue_error", model=model, provider=provider_name, error=str(e), log_type="QUEUE")
            return {"success": False, "error": f"Error waiting for model {model}: {e!s}"}

    try:
        provider = get_provider(provider_name)
        if not provider:
            await _model_queue.release_model(model, success=False, provider_name=provider_name)
            return {"success": False, "error": f"Unsupported provider: {provider_name}"}

        response = await provider.call(
            model=model,
            prompt=prompt,
            system=system,
            enable_thinking=enable_thinking,
            task_type=task_type,
            original_task=original_task,
            language=language,
            content_preview=content_preview,
            backend_obj=active_backend,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            messages=messages,
        )
        result = response.to_dict()

        await _model_queue.release_model(
            model, success=result.get("success", False), provider_name=provider_name
        )
        return result
    except Exception:
        await _model_queue.release_model(model, success=False, provider_name=provider_name)
        raise


async def call_llm_stream(
    model: str,
    prompt: str,
    system: str | None = None,
    enable_thinking: bool = False,
    task_type: str = "unknown",
    original_task: str = "unknown",
    language: str = "unknown",
    content_preview: str = "",
    backend: str | BackendConfig | None = None,
    backend_obj: BackendConfig | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[StreamChunk]:
    """
    Unified LLM streaming dispatcher that routes to the appropriate backend.

    Yields StreamChunk objects as the model generates tokens.

    Args:
        model: Model name/tier
        prompt: The prompt to send
        system: Optional system prompt
        enable_thinking: Enable thinking mode for supported models
        task_type: Type of task for tracking
        language: Detected language for tracking
        content_preview: Preview for logging
        backend: Override backend ID or provider name
        backend_obj: Optional BackendConfig object to use directly
        max_tokens: Maximum response tokens

    Yields:
        StreamChunk objects with incremental text and final metadata
    """
    # Note: For streaming, we skip the queue for now to reduce complexity

    # Determine backend to use
    active_backend = None

    if backend_obj:
        active_backend = backend_obj
    elif backend:
        if isinstance(backend, BackendConfig):
            active_backend = backend
        else:
            active_backend = backend_manager.get_backend(backend)
            if not active_backend:
                for b in backend_manager.get_enabled_backends():
                    if b.provider == backend:
                        active_backend = b
                        break
    else:
        active_backend = backend_manager.get_active_backend()

    if not active_backend:
        yield StreamChunk(done=True, error="No active backend found")
        return

    provider = get_provider(active_backend.provider)
    if not provider:
        yield StreamChunk(done=True, error=f"Unsupported provider: {active_backend.provider}")
        return

    # Check if provider supports streaming
    if hasattr(provider, "call_stream"):
        async for chunk in provider.call_stream(
            model=model,
            prompt=prompt,
            system=system,
            enable_thinking=enable_thinking,
            task_type=task_type,
            original_task=original_task,
            language=language,
            content_preview=content_preview,
            backend_obj=active_backend,
            max_tokens=max_tokens,
        ):
            yield chunk
    else:
        # Fallback to non-streaming call
        response = await provider.call(
            model=model,
            prompt=prompt,
            system=system,
            enable_thinking=enable_thinking,
            task_type=task_type,
            original_task=original_task,
            language=language,
            content_preview=content_preview,
            backend_obj=active_backend,
            max_tokens=max_tokens,
        )
        if response.success:
            yield StreamChunk(
                text=response.response,
                done=True,
                tokens=response.tokens,
                metadata=response.metadata or {},
            )
        else:
            yield StreamChunk(done=True, error=response.error)
