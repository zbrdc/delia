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
"""Google Gemini LLM provider implementation.

This module provides the GeminiProvider class that implements the LLMProvider protocol
for Google's Gemini API. It handles the synchronous-to-async bridge since the Gemini
SDK is synchronous.
"""

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any, Callable

import humanize
import structlog

from ..backend_manager import BackendConfig
from ..config import get_backend_health
from ..messages import format_completion_stats, get_display_event
from ..tokens import count_tokens
from .base import LLMResponse, create_error_response, create_success_response

if TYPE_CHECKING:
    from ..config import Config

log = structlog.get_logger()

# Type for stats callback
StatsCallback = Callable[
    [str, str, str, int, int, str, bool, str],
    None,
]


class GeminiProvider:
    """Google Gemini LLM provider.

    This class implements the LLMProvider protocol for Google's Gemini API.
    The Gemini SDK is synchronous, so calls are wrapped in asyncio.to_thread.
    """

    def __init__(
        self,
        config: "Config",
        backend_manager: Any,
        stats_callback: StatsCallback | None = None,
        save_stats_callback: Callable[[], Any] | None = None,
    ):
        """Initialize the Gemini provider.

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
        """Call Google Gemini API with stats tracking and circuit breaker."""
        start_time = time.time()

        # Dependency Check - import at call time since it's optional
        try:
            import google.generativeai as genai
            from google.api_core import exceptions as google_exceptions
        except ImportError:
            return create_error_response("Gemini dependency missing. Please run: uv add google-generativeai")

        # Resolve Backend
        if not backend_obj:
            for b in self.backend_manager.get_enabled_backends():
                if b.provider == "gemini":
                    backend_obj = b
                    break

        if not backend_obj:
            return create_error_response("No enabled Gemini backend found")

        # Circuit Breaker Check
        health = get_backend_health(backend_obj.id)
        if not health.is_available():
            wait_time = health.time_until_available()
            return create_error_response(
                f"Gemini circuit breaker open. Retry in {wait_time:.0f}s.",
                circuit_breaker=True,
            )

        # Configuration (API Key)
        api_key = backend_obj.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return create_error_response("Missing GEMINI_API_KEY")

        genai.configure(api_key=api_key)

        # Model Configuration
        # Strip tier prefix if present (e.g. "gemini:gemini-2.0-flash" -> "gemini-2.0-flash")
        model_name = model.split(":")[-1] if ":" in model else model
        # Default to flash if tier name passed directly
        if model_name in ["quick", "coder", "moe", "thinking"]:
            model_name = "gemini-2.0-flash"

        # Generation Config
        generation_config: dict[str, Any] = {
            "temperature": self.config.temperature_thinking if enable_thinking else self.config.temperature_normal,
        }

        try:
            log.info(
                get_display_event("model_starting"),
                log_type="MODEL",
                backend="gemini",
                task=task_type,
                model=model_name,
                status_msg="Sending request to Gemini...",
            )

            # Instantiate model
            gen_model = genai.GenerativeModel(model_name=model_name, system_instruction=system)

            # Run in thread executor because the SDK is synchronous
            response = await asyncio.to_thread(
                gen_model.generate_content,
                prompt,
                generation_config=generation_config,  # type: ignore[arg-type]
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Extract text and usage
            response_text = response.text

            # Usage metadata
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata"):
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count

            total_tokens = prompt_tokens + completion_tokens
            if total_tokens == 0:
                total_tokens = count_tokens(prompt) + count_tokens(response_text)

            # Stats tracking
            if self.stats_callback:
                self.stats_callback(
                    "moe",  # Treat Gemini as MoE/High-end tier for stats
                    task_type,
                    original_task,
                    total_tokens,
                    elapsed_ms,
                    content_preview,
                    enable_thinking,
                    "gemini",
                )

            log.info(
                get_display_event("model_completed"),
                log_type="INFO",
                elapsed=humanize.naturaldelta(elapsed_ms / 1000),
                elapsed_ms=elapsed_ms,
                tokens=humanize.intcomma(total_tokens),
                model=model_name,
                backend="gemini",
                status_msg=format_completion_stats(total_tokens, elapsed_ms, "moe"),
            )

            health.record_success(len(prompt))

            if self.save_stats_callback:
                self.save_stats_callback()

            return create_success_response(
                response_text=response_text,
                tokens=total_tokens,
                elapsed_ms=elapsed_ms,
                metadata={"backend": "gemini", "model": model_name, "tier": "moe"},
            )

        except google_exceptions.ResourceExhausted:
            health.record_failure("rate_limit", len(prompt))
            return create_error_response("Gemini rate limit exceeded (429).")
        except Exception as e:
            log.error("gemini_error", error=str(e))
            health.record_failure("exception", len(prompt))
            return create_error_response(f"Gemini error: {e!s}")
