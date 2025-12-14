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
"""Ollama LLM provider implementation.

This module provides the OllamaProvider class that implements the LLMProvider protocol
for Ollama backends. It handles:
- Ollama-specific API request formatting
- Pydantic validation of responses
- Circuit breaker integration
- Stats tracking callbacks
- Retry logic for transient failures
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
from ..config import detect_model_tier, get_backend_health
from ..messages import (
    StatusEvent,
    format_completion_stats,
    get_display_event,
    get_status_message,
    get_tier_message,
)
from .base import LLMResponse, StreamChunk, create_error_response, create_success_response
from .models import OllamaResponse

if TYPE_CHECKING:
    from ..config import Config

log = structlog.get_logger()


def extract_thinking_content(response_text: str) -> str | None:
    """Extract thinking content from LLM response.

    Returns the content between <think> and </think> tags, or None if not present.
    """
    if "<think>" not in response_text:
        return None

    start = response_text.find("<think>") + 7
    end = response_text.find("</think>")

    if end < start:
        return None

    return response_text[start:end].strip()


def log_thinking_and_response(response_text: str, model_tier: str, tokens: int) -> None:
    """Log thinking content and response preview."""
    thinking = extract_thinking_content(response_text)
    if thinking:
        thinking_preview = thinking[:200] + "..." if len(thinking) > 200 else thinking
        log.info(
            get_display_event("model_thinking"),
            log_type="THINK",
            preview=thinking_preview.replace("\n", " "),
            model=model_tier,
            status_msg=get_status_message(StatusEvent.PROCESSING),
        )

    response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
    log.info(
        get_display_event("model_response"),
        log_type="RESPONSE",
        preview=response_preview.replace("\n", " ").strip(),
        model=model_tier,
        tokens=tokens,
        status_msg=get_status_message(StatusEvent.COMPLETED),
    )


# Type for stats callback
StatsCallback = Callable[
    [str, str, str, int, int, str, bool, str],  # model_tier, task_type, original_task, tokens, elapsed_ms, preview, thinking, backend
    None,
]


class OllamaProvider:
    """Ollama LLM provider with circuit breaker and retry logic.

    This class implements the LLMProvider protocol for Ollama backends.
    It uses dependency injection for configuration and stats tracking.
    """

    def __init__(
        self,
        config: "Config",
        backend_manager: Any,
        stats_callback: StatsCallback | None = None,
        save_stats_callback: Callable[[], Any] | None = None,
    ):
        """Initialize the Ollama provider.

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
        """Call Ollama API with Pydantic validation, retry logic, and circuit breaker."""
        start_time = time.time()

        # Resolve backend
        if not backend_obj:
            for b in self.backend_manager.get_enabled_backends():
                if b.provider == "ollama":
                    backend_obj = b
                    break

        if not backend_obj:
            return create_error_response("No enabled Ollama backend found")

        # Circuit breaker check
        health = get_backend_health(backend_obj.id)
        if not health.is_available():
            wait_time = health.time_until_available()
            log.warning("circuit_open", backend=backend_obj.id, wait_seconds=round(wait_time, 1))
            return create_error_response(
                f"Ollama circuit breaker open. Too many failures. Retry in {wait_time:.0f}s.",
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

        options: dict[str, float | int] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        client = backend_obj.get_client()

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            retry=retry_if_exception_type(httpx.ConnectError),
            reraise=True,
        )
        async def _make_request():
            return await client.post("/api/generate", json=payload)

        data: dict[str, Any] = {}
        try:
            log.info(
                get_display_event("model_starting"),
                log_type="MODEL",
                model=model,
                task=task_type,
                thinking=enable_thinking,
                backend=backend_obj.name,
                status_msg=get_tier_message(model_tier, "start"),
            )

            response = await _make_request()
            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                try:
                    data = response.json()
                    validated = OllamaResponse.model_validate(data)
                except json.JSONDecodeError:
                    return create_error_response("Ollama returned non-JSON response")
                except ValidationError as e:
                    log.warning("ollama_validation_failed", error=str(e))
                    validated = None

                if validated:
                    tokens = validated.eval_count
                    response_text = validated.response
                else:
                    tokens = data.get("eval_count", 0)
                    response_text = data.get("response", "")

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
                        "ollama",
                    )

                log_thinking_and_response(response_text, model_tier, tokens)

                log.info(
                    get_display_event("model_completed"),
                    log_type="INFO",
                    elapsed=humanize.naturaldelta(elapsed_ms / 1000),
                    elapsed_ms=elapsed_ms,
                    tokens=humanize.intcomma(tokens),
                    model=model_tier,
                    backend="ollama",
                    status_msg=format_completion_stats(tokens, elapsed_ms, model_tier),
                )

                health.record_success(content_size)

                if self.save_stats_callback:
                    self.save_stats_callback()

                return create_success_response(
                    response_text=response_text,
                    tokens=tokens,
                    elapsed_ms=elapsed_ms,
                    metadata={"backend": "ollama", "model": model, "tier": model_tier},
                )

            # HTTP error handling
            error_msg = f"Ollama HTTP {response.status_code}"
            if response.status_code == 404:
                error_msg += f": Model '{model}' not found. Run: ollama pull {model}"
            elif response.status_code == 500:
                error_msg += ": Internal server error. Check Ollama logs."
            elif response.status_code == 503:
                error_msg += ": Ollama service unavailable. Is it running?"
            else:
                error_text = response.text[:500] if len(response.text) > 500 else response.text
                error_msg += f": {error_text}"
            health.record_failure("http_error", content_size)
            return create_error_response(error_msg)

        except httpx.TimeoutException:
            log.error("ollama_timeout", model=model, timeout_seconds=self.config.ollama_timeout_seconds)
            health.record_failure("timeout", content_size)
            return create_error_response(
                f"Ollama timeout after {self.config.ollama_timeout_seconds}s. Model may be loading or prompt too large."
            )
        except httpx.ConnectError:
            log.error("ollama_connection_refused", base_url=backend_obj.url)
            health.record_failure("connection", content_size)
            return create_error_response(f"Cannot connect to Ollama at {backend_obj.url}. Is Ollama running?")
        except Exception as e:
            log.error("ollama_error", model=model, error=str(e))
            health.record_failure("exception", content_size)
            return create_error_response(f"Ollama error: {e!s}")

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
        """Stream response from Ollama API token by token.

        Yields StreamChunk objects as the model generates tokens.
        The final chunk has done=True and includes total token count.
        """
        start_time = time.time()

        # Resolve backend
        if not backend_obj:
            for b in self.backend_manager.get_enabled_backends():
                if b.provider == "ollama":
                    backend_obj = b
                    break

        if not backend_obj:
            yield StreamChunk(done=True, error="No enabled Ollama backend found")
            return

        # Circuit breaker check
        health = get_backend_health(backend_obj.id)
        if not health.is_available():
            wait_time = health.time_until_available()
            yield StreamChunk(
                done=True,
                error=f"Ollama circuit breaker open. Retry in {wait_time:.0f}s.",
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

        options: dict[str, float | int] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,  # Enable streaming
            "options": options,
        }
        if system:
            payload["system"] = system

        client = backend_obj.get_client()

        log.info(
            get_display_event("model_starting"),
            log_type="MODEL",
            model=model,
            task=task_type,
            thinking=enable_thinking,
            backend=backend_obj.name,
            streaming=True,
            status_msg=get_tier_message(model_tier, "start"),
        )

        total_tokens = 0
        full_response = ""

        try:
            async with client.stream("POST", "/api/generate", json=payload) as response:
                if response.status_code != 200:
                    error_text = ""
                    async for chunk in response.aiter_text():
                        error_text += chunk
                    health.record_failure("http_error", content_size)
                    yield StreamChunk(
                        done=True,
                        error=f"Ollama HTTP {response.status_code}: {error_text[:500]}",
                    )
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract text chunk
                    text_chunk = data.get("response", "")
                    if text_chunk:
                        full_response += text_chunk
                        yield StreamChunk(text=text_chunk)

                    # Check if done
                    if data.get("done", False):
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        total_tokens = data.get("eval_count", 0)

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
                                "ollama",
                            )

                        health.record_success(content_size)

                        if self.save_stats_callback:
                            self.save_stats_callback()

                        log.info(
                            get_display_event("model_completed"),
                            log_type="INFO",
                            elapsed=humanize.naturaldelta(elapsed_ms / 1000),
                            elapsed_ms=elapsed_ms,
                            tokens=humanize.intcomma(total_tokens),
                            model=model_tier,
                            backend="ollama",
                            streaming=True,
                            status_msg=format_completion_stats(total_tokens, elapsed_ms, model_tier),
                        )

                        # Final chunk with metadata
                        yield StreamChunk(
                            done=True,
                            tokens=total_tokens,
                            metadata={
                                "backend": "ollama",
                                "model": model,
                                "tier": model_tier,
                                "elapsed_ms": elapsed_ms,
                            },
                        )
                        return

        except httpx.TimeoutException:
            health.record_failure("timeout", content_size)
            yield StreamChunk(
                done=True,
                error=f"Ollama timeout after {self.config.ollama_timeout_seconds}s.",
            )
        except httpx.ConnectError:
            health.record_failure("connection", content_size)
            yield StreamChunk(
                done=True,
                error=f"Cannot connect to Ollama at {backend_obj.url}. Is Ollama running?",
            )
        except Exception as e:
            health.record_failure("exception", content_size)
            yield StreamChunk(done=True, error=f"Ollama streaming error: {e!s}")
