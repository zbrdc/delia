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
"""llama.cpp (OpenAI-compatible) LLM provider implementation.

This module provides the LlamaCppProvider class that implements the LLMProvider protocol
for OpenAI-compatible backends including llama.cpp, vLLM, LM Studio, and others.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Callable

import httpx
import humanize
import structlog
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..backend_manager import BackendConfig
from ..config import detect_model_tier, get_backend_health
from ..messages import (
    StatusEvent,
    format_completion_stats,
    get_display_event,
    get_status_message,
    get_tier_message,
)
from ..tokens import count_tokens
from .base import LLMResponse, create_error_response, create_success_response
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
    ) -> LLMResponse:
        """Call OpenAI-compatible API with Pydantic validation, retry, and circuit breaker."""
        start_time = time.time()

        # Resolve backend
        if not backend_obj:
            for b in self.backend_manager.get_enabled_backends():
                if b.provider in self.COMPATIBLE_PROVIDERS:
                    backend_obj = b
                    break

        if not backend_obj:
            return create_error_response("No enabled OpenAI-compatible backend found")

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

                    response_text = validated.choices[0].message.content

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
                    response_text = choices[0].get("message", {}).get("content", "")
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

                if self.save_stats_callback:
                    self.save_stats_callback()

                return create_success_response(
                    response_text=response_text,
                    tokens=tokens,
                    elapsed_ms=elapsed_ms,
                    metadata={"backend": "llamacpp", "model": model, "tier": model_tier},
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
            health.record_failure("http_error", content_size)
            return create_error_response(error_msg)

        except httpx.TimeoutException:
            log.error("llamacpp_timeout", timeout_seconds=self.config.llamacpp_timeout_seconds)
            health.record_failure("timeout", content_size)
            return create_error_response(
                f"Timeout after {self.config.llamacpp_timeout_seconds}s. Model may be loading or prompt too large."
            )
        except httpx.ConnectError:
            log.error("llamacpp_connection_refused", base_url=backend_obj.url)
            health.record_failure("connection", content_size)
            return create_error_response(f"Cannot connect to {backend_obj.url}. Is the server running?")
        except Exception as e:
            log.error("llamacpp_error", error=str(e))
            health.record_failure("exception", content_size)
            return create_error_response(f"Error: {e!s}")
