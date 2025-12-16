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

"""llama.cpp (OpenAI-compatible) LLM provider implementation.

This module provides the LlamaCppProvider class that implements the LLMProvider protocol
for OpenAI-compatible backends including llama.cpp, vLLM, LM Studio, and others.
"""

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable

import httpx
import humanize
import structlog
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..backend_manager import BackendConfig
from ..config import detect_model_tier, get_backend_health, get_backend_metrics
from ..messages import (
    StatusEvent,
    format_completion_stats,
    get_display_event,
    get_status_message,
    get_tier_message,
)
from ..tokens import count_tokens
from .base import (
    LLMResponse,
    ModelLoadResult,
    StreamChunk,
    create_error_response,
    create_success_response,
)
from .models import LlamaCppError, LlamaCppResponse
from .ollama import log_thinking_and_response

if TYPE_CHECKING:
    from ..config import Config

log = structlog.get_logger()

# Type for stats callback
StatsCallback = Callable[
    [str, str, str, int, int, str, bool, str],
    None,
]


class LlamaCppProvider:
    """OpenAI-compatible LLM provider (llama.cpp, vLLM, LM Studio, etc.).

    This class implements the LLMProvider protocol for OpenAI-compatible backends.
    """

    # Providers that use this OpenAI-compatible interface
    COMPATIBLE_PROVIDERS = ("llamacpp", "lmstudio", "vllm", "openai", "custom")

    def __init__(
        self,
        config: "Config",
        backend_manager: Any,
        stats_callback: StatsCallback | None = None,
        save_stats_callback: Callable[[], Any] | None = None,
    ):
        """Initialize the llama.cpp provider.

        Args:
            config: Configuration object with model tiers and timeouts
            backend_manager: Backend manager for finding available backends
            stats_callback: Optional callback for recording stats
            save_stats_callback: Optional callback for persisting stats to disk
        """
        self.config = config
        self.backend_manager = backend_manager
        self.stats_callback = stats_callback
        self.save_stats_callback = save_stats_callback

    async def call(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        enable_thinking: bool = False,
        task_type: str = "unknown",
        original_task: str = "unknown",
        language: str = "unknown",
        content_preview: str = "",
        backend_obj: BackendConfig | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """Call OpenAI-compatible API with Pydantic validation, retry, and circuit breaker."""
        start_time = time.time()

        # Resolve backend (skip embeddings-only backends)
        if not backend_obj:
            for b in self.backend_manager.get_enabled_backends():
                if b.provider in self.COMPATIBLE_PROVIDERS:
                    if await self._is_embeddings_backend(b):
                        continue
                    backend_obj = b
                    break

        if not backend_obj:
            return create_error_response("No enabled OpenAI-compatible backend found (or all are embeddings-only)")

        # Circuit breaker check
        health = get_backend_health(backend_obj.id)
        if not health.is_available():
            wait_time = health.time_until_available()
            log.warning("circuit_open", backend=backend_obj.id, wait_seconds=round(wait_time, 1))
            return create_error_response(
                f"Backend circuit breaker open. Too many failures. Retry in {wait_time:.0f}s.",
                circuit_breaker=True,
            )

        # Context size check and potential reduction
        content_size = len(prompt) + len(system or "")
        should_reduce, recommended_size = health.should_reduce_context(content_size)
        if should_reduce:
            log.info(
                "context_reduction",
                backend=backend_obj.id,
                original_kb=content_size // 1024,
                recommended_kb=recommended_size // 1024,
            )
            if len(prompt) > recommended_size:
                prompt = prompt[:recommended_size] + "\n\n[Content truncated due to previous timeout]"

        # For thinking mode with qwen models, add /think prefix
        if enable_thinking and "qwen" in model.lower():
            prompt = f"/think\n{prompt}"

        # Auto-select context size based on model tier
        model_tier = detect_model_tier(model)
        if model_tier == "moe":
            num_ctx = self.config.model_moe.num_ctx
        elif model_tier == "coder":
            num_ctx = self.config.model_coder.num_ctx
        else:
            num_ctx = self.config.model_quick.num_ctx

        # Temperature based on thinking mode
        temperature = self.config.temperature_thinking if enable_thinking else self.config.temperature_normal

        # Build messages for OpenAI-compatible API
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens else num_ctx,
            "stream": False,
        }

        # Add tool calling support if tools are provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        client = backend_obj.get_client()

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            retry=retry_if_exception_type(httpx.ConnectError),
            reraise=True,
        )
        async def _make_request():
            return await client.post(backend_obj.chat_endpoint, json=payload)

        data: dict[str, Any] = {}
        try:
            log.info(
                get_display_event("model_starting"),
                log_type="MODEL",
                backend=backend_obj.name,
                task=task_type,
                thinking=enable_thinking,
                model=model_tier,
                status_msg=get_tier_message(model_tier, "start"),
            )

            response = await _make_request()
            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                try:
                    data = response.json()
                    validated = LlamaCppResponse.model_validate(data)

                    if not validated.choices:
                        return create_error_response("Backend returned no choices")

                    message = validated.choices[0].message
                    # Use content, or fall back to reasoning_content (Qwen3 extended thinking)
                    response_text = message.content or message.reasoning_content or ""
                    tool_calls = message.tool_calls

                    if validated.usage:
                        tokens = validated.usage.completion_tokens + validated.usage.prompt_tokens
                    else:
                        tokens = count_tokens(response_text)

                except json.JSONDecodeError:
                    return create_error_response("Backend returned non-JSON response")
                except ValidationError as e:
                    log.warning("llamacpp_validation_failed", error=str(e))
                    choices = data.get("choices", [])
                    if not choices:
                        return create_error_response("Backend returned no choices")
                    msg = choices[0].get("message", {})
                    # Use content, or fall back to reasoning_content (Qwen3 extended thinking)
                    response_text = msg.get("content", "") or msg.get("reasoning_content", "")
                    usage = data.get("usage", {})
                    tokens = usage.get("completion_tokens", 0) + usage.get("prompt_tokens", 0)
                    if tokens == 0:
                        tokens = count_tokens(response_text)

                # Stats tracking
                if self.stats_callback:
                    self.stats_callback(
                        model_tier,
                        task_type,
                        original_task,
                        tokens,
                        elapsed_ms,
                        content_preview,
                        enable_thinking,
                        "llamacpp",
                    )

                log_thinking_and_response(response_text, model_tier, tokens)

                log.info(
                    get_display_event("model_completed"),
                    log_type="INFO",
                    elapsed=humanize.naturaldelta(elapsed_ms / 1000),
                    elapsed_ms=elapsed_ms,
                    tokens=humanize.intcomma(tokens),
                    model=model_tier,
                    backend="llamacpp",
                    status_msg=format_completion_stats(tokens, elapsed_ms, model_tier),
                )

                health.record_success(content_size)
                get_backend_metrics(backend_obj.id).record_success(
                    elapsed_ms=float(elapsed_ms), tokens=tokens
                )

                if self.save_stats_callback:
                    self.save_stats_callback()

                # Build metadata, including tool_calls if present
                metadata = {"backend": "llamacpp", "model": model, "tier": model_tier}
                if tool_calls:
                    metadata["tool_calls"] = [tc.model_dump() for tc in tool_calls]

                return create_success_response(
                    response_text=response_text,
                    tokens=tokens,
                    elapsed_ms=elapsed_ms,
                    metadata=metadata,
                )

            # HTTP error handling with context-specific messages
            error_msg = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    try:
                        err = LlamaCppError.model_validate(error_data["error"])
                        if err.type == "exceed_context_size_error":
                            error_msg = f"Context exceeded: {err.n_prompt_tokens} tokens > limit."
                        else:
                            error_msg += f": {err.message}"
                    except ValidationError:
                        err = error_data["error"]
                        error_msg += f": {err.get('message', str(err))}"
            except (json.JSONDecodeError, KeyError):
                error_text = response.text[:500] if len(response.text) > 500 else response.text
                error_msg += f": {error_text}"
            http_error_elapsed_ms = int((time.time() - start_time) * 1000)
            health.record_failure("http_error", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(http_error_elapsed_ms))
            return create_error_response(error_msg)

        except httpx.TimeoutException:
            timeout_elapsed_ms = int((time.time() - start_time) * 1000)
            log.error("llamacpp_timeout", timeout_seconds=self.config.llamacpp_timeout_seconds)
            health.record_failure("timeout", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(timeout_elapsed_ms))
            return create_error_response(
                f"Timeout after {self.config.llamacpp_timeout_seconds}s. Model may be loading or prompt too large."
            )
        except httpx.ConnectError:
            log.error("llamacpp_connection_refused", base_url=backend_obj.url)
            health.record_failure("connection", content_size)
            # Connection errors don't have meaningful latency - record as 0
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=0)
            return create_error_response(f"Cannot connect to {backend_obj.url}. Is the server running?")
        except Exception as e:
            exc_elapsed_ms = int((time.time() - start_time) * 1000)
            log.error("llamacpp_error", error=str(e))
            health.record_failure("exception", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(exc_elapsed_ms))
            return create_error_response(f"Error: {e!s}")

    async def call_stream(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        enable_thinking: bool = False,
        task_type: str = "unknown",
        original_task: str = "unknown",
        language: str = "unknown",
        content_preview: str = "",
        backend_obj: BackendConfig | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from OpenAI-compatible API token by token.

        Uses Server-Sent Events (SSE) format. Yields StreamChunk objects
        as the model generates tokens.
        """
        start_time = time.time()

        # Resolve backend (skip embeddings-only backends)
        if not backend_obj:
            for b in self.backend_manager.get_enabled_backends():
                if b.provider in self.COMPATIBLE_PROVIDERS:
                    if await self._is_embeddings_backend(b):
                        continue
                    backend_obj = b
                    break

        if not backend_obj:
            yield StreamChunk(done=True, error="No enabled OpenAI-compatible backend found (or all are embeddings-only)")
            return

        # Circuit breaker check
        health = get_backend_health(backend_obj.id)
        if not health.is_available():
            wait_time = health.time_until_available()
            yield StreamChunk(
                done=True,
                error=f"Backend circuit breaker open. Retry in {wait_time:.0f}s.",
            )
            return

        # Context size check and potential reduction
        content_size = len(prompt) + len(system or "")
        should_reduce, recommended_size = health.should_reduce_context(content_size)
        if should_reduce and len(prompt) > recommended_size:
            prompt = prompt[:recommended_size] + "\n\n[Content truncated due to previous timeout]"

        if enable_thinking and "qwen" in model.lower():
            prompt = f"/think\n{prompt}"

        # Auto-select context size based on model tier
        model_tier = detect_model_tier(model)
        if model_tier == "moe":
            num_ctx = self.config.model_moe.num_ctx
        elif model_tier == "coder":
            num_ctx = self.config.model_coder.num_ctx
        else:
            num_ctx = self.config.model_quick.num_ctx

        temperature = self.config.temperature_thinking if enable_thinking else self.config.temperature_normal

        # Build messages for OpenAI-compatible API
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens else num_ctx,
            "stream": True,  # Enable streaming
        }

        client = backend_obj.get_client()

        log.info(
            get_display_event("model_starting"),
            log_type="MODEL",
            backend=backend_obj.name,
            task=task_type,
            thinking=enable_thinking,
            model=model_tier,
            streaming=True,
            status_msg=get_tier_message(model_tier, "start"),
        )

        total_tokens = 0
        full_response = ""

        try:
            async with client.stream("POST", backend_obj.chat_endpoint, json=payload) as response:
                if response.status_code != 200:
                    error_text = ""
                    async for chunk in response.aiter_text():
                        error_text += chunk
                    stream_error_ms = int((time.time() - start_time) * 1000)
                    health.record_failure("http_error", content_size)
                    get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(stream_error_ms))
                    yield StreamChunk(
                        done=True,
                        error=f"HTTP {response.status_code}: {error_text[:500]}",
                    )
                    return

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    # OpenAI SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str == "[DONE]":
                            # Stream complete
                            elapsed_ms = int((time.time() - start_time) * 1000)

                            # Estimate tokens if not provided
                            if total_tokens == 0:
                                total_tokens = count_tokens(full_response)

                            # Stats tracking
                            if self.stats_callback:
                                self.stats_callback(
                                    model_tier,
                                    task_type,
                                    original_task,
                                    total_tokens,
                                    elapsed_ms,
                                    content_preview,
                                    enable_thinking,
                                    "llamacpp",
                                )

                            health.record_success(content_size)
                            get_backend_metrics(backend_obj.id).record_success(
                                elapsed_ms=float(elapsed_ms), tokens=total_tokens
                            )

                            if self.save_stats_callback:
                                self.save_stats_callback()

                            log.info(
                                get_display_event("model_completed"),
                                log_type="INFO",
                                elapsed=humanize.naturaldelta(elapsed_ms / 1000),
                                elapsed_ms=elapsed_ms,
                                tokens=humanize.intcomma(total_tokens),
                                model=model_tier,
                                backend="llamacpp",
                                streaming=True,
                                status_msg=format_completion_stats(total_tokens, elapsed_ms, model_tier),
                            )

                            yield StreamChunk(
                                done=True,
                                tokens=total_tokens,
                                metadata={
                                    "backend": "llamacpp",
                                    "model": model,
                                    "tier": model_tier,
                                    "elapsed_ms": elapsed_ms,
                                },
                            )
                            return

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Extract delta content from choices
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            # Use content, or fall back to reasoning_content (Qwen3)
                            text_chunk = delta.get("content", "") or delta.get("reasoning_content", "")
                            if text_chunk:
                                full_response += text_chunk
                                yield StreamChunk(text=text_chunk)

                        # Track usage if provided
                        usage = data.get("usage")
                        if usage:
                            total_tokens = usage.get("total_tokens", 0)

        except httpx.TimeoutException:
            stream_timeout_ms = int((time.time() - start_time) * 1000)
            health.record_failure("timeout", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(stream_timeout_ms))
            yield StreamChunk(
                done=True,
                error=f"Timeout after {self.config.llamacpp_timeout_seconds}s.",
            )
        except httpx.ConnectError:
            health.record_failure("connection", content_size)
            # Connection errors don't have meaningful latency - record as 0
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=0)
            yield StreamChunk(
                done=True,
                error=f"Cannot connect to {backend_obj.url}. Is the server running?",
            )
        except Exception as e:
            stream_exc_ms = int((time.time() - start_time) * 1000)
            health.record_failure("exception", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(stream_exc_ms))
            yield StreamChunk(done=True, error=f"Streaming error: {e!s}")

    def _find_compatible_backend(self) -> BackendConfig | None:
        """Find the first enabled OpenAI-compatible backend."""
        for b in self.backend_manager.get_enabled_backends():
            if b.provider in self.COMPATIBLE_PROVIDERS:
                return b
        return None

    async def _is_embeddings_backend(self, backend: BackendConfig) -> bool:
        """Check if a backend is running an embeddings-only model.

        Detects embeddings models by checking if loaded model names contain
        'embed' (e.g., nomic-embed-text, bge-embed, etc.).
        """
        try:
            loaded = await self.list_loaded_models(backend)
            for model_name in loaded:
                if "embed" in model_name.lower():
                    log.debug("skipping_embeddings_backend", backend=backend.id, model=model_name)
                    return True
        except Exception:
            pass
        return False

    async def _get_available_model_ids(self, backend_obj: BackendConfig) -> list[str]:
        """Get all available model IDs from the backend.

        Queries /v1/models to discover what models are configured/available.
        This is used to provide helpful error messages when a requested
        model doesn't exist.

        Returns:
            List of model IDs available in the backend, or empty list on error.
        """
        try:
            client = backend_obj.get_client()
            response = await client.get(backend_obj.models_endpoint, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        except Exception as e:
            log.debug("get_available_models_failed", error=str(e))
        return []

    async def load_model(
        self,
        model: str,
        backend_obj: BackendConfig | None = None,
    ) -> ModelLoadResult:
        """Load a model in llama.cpp router mode.

        For llama.cpp in router mode (started without -m flag), uses
        POST /models/load to load a specific model.
        For single-model mode or non-llama.cpp backends, this is a no-op.

        When model loading fails, queries available models and returns
        a helpful error message showing what models ARE available.
        """
        start_time = time.time()

        if not backend_obj:
            backend_obj = self._find_compatible_backend()
        if not backend_obj:
            return ModelLoadResult(
                success=False,
                model=model,
                action="load",
                error="No enabled OpenAI-compatible backend found",
            )

        client = backend_obj.get_client()

        try:
            # Try llama.cpp router mode endpoint
            payload = {"model": model}
            response = await client.post("/models/load", json=payload, timeout=300.0)
            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                log.info(
                    "model_preloaded",
                    model=model,
                    elapsed_ms=elapsed_ms,
                    backend="llamacpp",
                )
                return ModelLoadResult(
                    success=True,
                    model=model,
                    action="load",
                    elapsed_ms=elapsed_ms,
                    metadata={"backend": "llamacpp", "mode": "router"},
                )
            elif response.status_code == 404:
                # 404 could mean:
                # 1. Router mode endpoint doesn't exist (single-model mode)
                # 2. Router mode exists but model ID not found
                # Query /v1/models to distinguish these cases
                available = await self._get_available_model_ids(backend_obj)
                
                if available:
                    # Router mode is active - check if model exists
                    if model in available:
                        # Model exists but not loaded - try to proceed anyway
                        log.info(
                            "model_available_not_loaded",
                            model=model,
                            available=available,
                        )
                        return ModelLoadResult(
                            success=True,
                            model=model,
                            action="load",
                            elapsed_ms=elapsed_ms,
                            metadata={"backend": "llamacpp", "note": "model_available"},
                        )
                    else:
                        # Model doesn't exist in router - return helpful error
                        log.warning(
                            "model_not_found_in_router",
                            requested=model,
                            available=available,
                        )
                        return ModelLoadResult(
                            success=False,
                            model=model,
                            action="load",
                            error=f"Model '{model}' not found. Available models: {', '.join(available)}",
                            elapsed_ms=elapsed_ms,
                            metadata={"available_models": available},
                        )
                else:
                    # Can't query models - assume single-model mode (no router)
                    return ModelLoadResult(
                        success=True,
                        model=model,
                        action="load",
                        elapsed_ms=elapsed_ms,
                        metadata={"backend": "llamacpp", "note": "router_mode_unavailable"},
                    )
            elif response.status_code == 400:
                # 400 often means model not found in router mode
                available = await self._get_available_model_ids(backend_obj)
                if available and model not in available:
                    return ModelLoadResult(
                        success=False,
                        model=model,
                        action="load",
                        error=f"Model '{model}' not found. Available models: {', '.join(available)}",
                        elapsed_ms=elapsed_ms,
                        metadata={"available_models": available},
                    )
                return ModelLoadResult(
                    success=False,
                    model=model,
                    action="load",
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    elapsed_ms=elapsed_ms,
                )
            else:
                return ModelLoadResult(
                    success=False,
                    model=model,
                    action="load",
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    elapsed_ms=elapsed_ms,
                )
        except httpx.TimeoutException:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ModelLoadResult(
                success=False,
                model=model,
                action="load",
                error="Timeout loading model (waited 5 minutes)",
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            # Connection errors mean backend is unreachable
            if "ConnectError" in type(e).__name__:
                return ModelLoadResult(
                    success=False,
                    model=model,
                    action="load",
                    error=f"Cannot connect to backend: {e}",
                    elapsed_ms=elapsed_ms,
                )
            return ModelLoadResult(
                success=False,
                model=model,
                action="load",
                error=str(e),
                elapsed_ms=elapsed_ms,
            )

    async def unload_model(
        self,
        model: str,
        backend_obj: BackendConfig | None = None,
    ) -> ModelLoadResult:
        """Unload a model in llama.cpp router mode.

        For llama.cpp in router mode, uses POST /models/unload.
        For single-model mode or non-llama.cpp backends, this is a no-op.
        """
        start_time = time.time()

        if not backend_obj:
            backend_obj = self._find_compatible_backend()
        if not backend_obj:
            return ModelLoadResult(
                success=False,
                model=model,
                action="unload",
                error="No enabled OpenAI-compatible backend found",
            )

        client = backend_obj.get_client()

        try:
            # Try llama.cpp router mode endpoint
            payload = {"model": model}
            response = await client.post("/models/unload", json=payload, timeout=30.0)
            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                log.info(
                    "model_unloaded_llamacpp",
                    model=model,
                    elapsed_ms=elapsed_ms,
                    backend="llamacpp",
                )
                return ModelLoadResult(
                    success=True,
                    model=model,
                    action="unload",
                    elapsed_ms=elapsed_ms,
                    metadata={"backend": "llamacpp", "mode": "router"},
                )
            elif response.status_code == 404:
                # Router mode not available - cannot unload programmatically
                # Return success as we cannot do anything
                return ModelLoadResult(
                    success=True,
                    model=model,
                    action="unload",
                    elapsed_ms=elapsed_ms,
                    metadata={"backend": "llamacpp", "note": "router_mode_unavailable"},
                )
            else:
                return ModelLoadResult(
                    success=False,
                    model=model,
                    action="unload",
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    elapsed_ms=elapsed_ms,
                )
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            # Connection errors or 404s mean router mode isn't available
            if "404" in str(e) or "ConnectError" in type(e).__name__:
                return ModelLoadResult(
                    success=True,
                    model=model,
                    action="unload",
                    elapsed_ms=elapsed_ms,
                    metadata={"backend": "llamacpp", "note": "router_mode_unavailable"},
                )
            return ModelLoadResult(
                success=False,
                model=model,
                action="unload",
                error=str(e),
                elapsed_ms=elapsed_ms,
            )

    async def list_loaded_models(
        self,
        backend_obj: BackendConfig | None = None,
    ) -> list[str]:
        """Get list of loaded models from llama.cpp or OpenAI-compatible backend.

        For llama.cpp router mode, uses GET /models to find loaded models.
        Falls back to /v1/models for standard OpenAI-compatible backends.
        """
        if not backend_obj:
            backend_obj = self._find_compatible_backend()
        if not backend_obj:
            return []

        client = backend_obj.get_client()

        # Try llama.cpp router mode /models endpoint first
        try:
            response = await client.get("/models", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                # Filter to only loaded models (router mode includes status)
                loaded = []
                for m in data.get("data", []):
                    status = m.get("status", {})
                    if isinstance(status, dict) and status.get("value") == "loaded":
                        loaded.append(m.get("id", ""))
                    elif not isinstance(status, dict):
                        # Non-router mode - all returned models are available
                        loaded.append(m.get("id", ""))
                if loaded:
                    return loaded
        except Exception:
            pass

        # Fallback: try standard /v1/models endpoint
        try:
            response = await client.get(backend_obj.models_endpoint, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        except Exception as e:
            log.warning("list_loaded_models_failed", backend="llamacpp", error=str(e))

        return []
