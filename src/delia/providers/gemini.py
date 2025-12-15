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

"""Google Gemini LLM provider implementation.

This module provides the GeminiProvider class that implements the LLMProvider protocol
for Google's Gemini API. It handles the synchronous-to-async bridge since the Gemini
SDK is synchronous.
"""

import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable

import humanize
import structlog

from ..backend_manager import BackendConfig
from ..config import get_backend_health
from ..messages import format_completion_stats, get_display_event
from ..tokens import count_tokens
from .base import (
    LLMResponse,
    ModelLoadResult,
    StreamChunk,
    create_error_response,
    create_success_response,
)

if TYPE_CHECKING:
    from ..config import Config

log = structlog.get_logger()


def _convert_openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[Any]:
    """Convert OpenAI-format tools to Gemini function declarations.

    OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {...}  # JSON Schema
        }
    }

    Gemini format uses FunctionDeclaration objects.
    """
    try:
        from google.generativeai.types import FunctionDeclaration
    except ImportError:
        return []

    declarations = []
    for tool in tools:
        if tool.get("type") != "function":
            continue

        func = tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})

        # Gemini expects parameters in a specific format
        declarations.append(
            FunctionDeclaration(
                name=name,
                description=description,
                parameters=parameters,
            )
        )

    return declarations


def _extract_gemini_tool_calls(response: Any) -> list[dict[str, Any]] | None:
    """Extract tool calls from Gemini response in OpenAI-compatible format."""
    try:
        # Check if response has function calls
        if not hasattr(response, "candidates") or not response.candidates:
            return None

        candidate = response.candidates[0]
        if not hasattr(candidate, "content") or not candidate.content.parts:
            return None

        tool_calls = []
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(
                    {
                        "id": f"call_{fc.name}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": dict(fc.args) if fc.args else {},
                        },
                    }
                )

        return tool_calls if tool_calls else None
    except Exception:
        return None


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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """Call Google Gemini API with stats tracking and circuit breaker.

        When tools are provided, converts from OpenAI format to Gemini function
        calling format and returns tool_calls in response metadata.
        """
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

            # Instantiate model with optional tools
            model_kwargs: dict[str, Any] = {
                "model_name": model_name,
                "system_instruction": system,
            }

            # Convert and add tools if provided
            gemini_tools = None
            if tools:
                gemini_tools = _convert_openai_tools_to_gemini(tools)
                if gemini_tools:
                    model_kwargs["tools"] = gemini_tools

            gen_model = genai.GenerativeModel(**model_kwargs)

            # Run in thread executor because the SDK is synchronous
            response = await asyncio.to_thread(
                gen_model.generate_content,
                prompt,
                generation_config=generation_config,  # type: ignore[arg-type]
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Extract text - may be empty if tool call response
            try:
                response_text = response.text
            except ValueError:
                # Response may not have text if it's a function call
                response_text = ""

            # Extract tool calls if any
            tool_calls = _extract_gemini_tool_calls(response)

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

            metadata: dict[str, Any] = {"backend": "gemini", "model": model_name, "tier": "moe"}
            if tool_calls:
                metadata["tool_calls"] = tool_calls

            return create_success_response(
                response_text=response_text,
                tokens=total_tokens,
                elapsed_ms=elapsed_ms,
                metadata=metadata,
            )

        except google_exceptions.ResourceExhausted:
            health.record_failure("rate_limit", len(prompt))
            return create_error_response("Gemini rate limit exceeded (429).")
        except Exception as e:
            log.error("gemini_error", error=str(e))
            health.record_failure("exception", len(prompt))
            return create_error_response(f"Gemini error: {e!s}")

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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from Gemini API.

        Uses the Gemini SDK's streaming mode with async wrapper.

        Note: tools/tool_choice are accepted for signature compatibility but
        tool calling with streaming is not currently supported.
        """
        start_time = time.time()

        # Dependency Check
        try:
            import google.generativeai as genai
            from google.api_core import exceptions as google_exceptions
        except ImportError:
            yield StreamChunk(done=True, error="Gemini dependency missing. Please run: uv add google-generativeai")
            return

        # Resolve Backend
        if not backend_obj:
            for b in self.backend_manager.get_enabled_backends():
                if b.provider == "gemini":
                    backend_obj = b
                    break

        if not backend_obj:
            yield StreamChunk(done=True, error="No enabled Gemini backend found")
            return

        # Circuit Breaker Check
        health = get_backend_health(backend_obj.id)
        if not health.is_available():
            wait_time = health.time_until_available()
            yield StreamChunk(
                done=True,
                error=f"Gemini circuit breaker open. Retry in {wait_time:.0f}s.",
            )
            return

        # Configuration
        api_key = backend_obj.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            yield StreamChunk(done=True, error="Missing GEMINI_API_KEY")
            return

        genai.configure(api_key=api_key)

        model_name = model.split(":")[-1] if ":" in model else model
        if model_name in ["quick", "coder", "moe", "thinking"]:
            model_name = "gemini-2.0-flash"

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
                streaming=True,
                status_msg="Streaming from Gemini...",
            )

            gen_model = genai.GenerativeModel(model_name=model_name, system_instruction=system)

            # Gemini SDK streaming is synchronous, so we use a queue-based approach
            # to convert it to async
            full_response = ""
            total_tokens = 0

            def _stream_sync():
                """Synchronous streaming generator."""
                nonlocal full_response, total_tokens
                response = gen_model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    stream=True,
                )
                for chunk in response:
                    if hasattr(chunk, "text") and chunk.text:
                        full_response += chunk.text
                        yield chunk.text

                # Get final usage metadata
                if hasattr(response, "usage_metadata"):
                    total_tokens = (
                        response.usage_metadata.prompt_token_count
                        + response.usage_metadata.candidates_token_count
                    )

            # Convert sync generator to async by running in thread
            # and using a queue for chunks
            import queue
            import threading

            chunk_queue: queue.Queue[str | None | Exception] = queue.Queue()

            def _run_stream():
                try:
                    for text in _stream_sync():
                        chunk_queue.put(text)
                    chunk_queue.put(None)  # Signal completion
                except Exception as e:
                    chunk_queue.put(e)

            thread = threading.Thread(target=_run_stream, daemon=True)
            thread.start()

            # Yield chunks as they arrive
            while True:
                try:
                    # Use asyncio-friendly wait
                    item = await asyncio.to_thread(chunk_queue.get, timeout=60)

                    if item is None:
                        # Stream complete
                        break
                    elif isinstance(item, Exception):
                        yield StreamChunk(done=True, error=str(item))
                        return
                    else:
                        yield StreamChunk(text=item)

                except Exception as e:
                    yield StreamChunk(done=True, error=f"Queue error: {e!s}")
                    return

            elapsed_ms = int((time.time() - start_time) * 1000)

            if total_tokens == 0:
                total_tokens = count_tokens(prompt) + count_tokens(full_response)

            # Stats tracking
            if self.stats_callback:
                self.stats_callback(
                    "moe",
                    task_type,
                    original_task,
                    total_tokens,
                    elapsed_ms,
                    content_preview,
                    enable_thinking,
                    "gemini",
                )

            health.record_success(len(prompt))

            if self.save_stats_callback:
                self.save_stats_callback()

            log.info(
                get_display_event("model_completed"),
                log_type="INFO",
                elapsed=humanize.naturaldelta(elapsed_ms / 1000),
                elapsed_ms=elapsed_ms,
                tokens=humanize.intcomma(total_tokens),
                model=model_name,
                backend="gemini",
                streaming=True,
                status_msg=format_completion_stats(total_tokens, elapsed_ms, "moe"),
            )

            yield StreamChunk(
                done=True,
                tokens=total_tokens,
                metadata={
                    "backend": "gemini",
                    "model": model_name,
                    "tier": "moe",
                    "elapsed_ms": elapsed_ms,
                },
            )

        except google_exceptions.ResourceExhausted:
            health.record_failure("rate_limit", len(prompt))
            yield StreamChunk(done=True, error="Gemini rate limit exceeded (429).")
        except Exception as e:
            log.error("gemini_stream_error", error=str(e))
            health.record_failure("exception", len(prompt))
            yield StreamChunk(done=True, error=f"Gemini streaming error: {e!s}")

    async def load_model(
        self,
        model: str,
        backend_obj: BackendConfig | None = None,
    ) -> ModelLoadResult:
        """No-op for cloud provider - Gemini models are always available.

        Cloud providers don't require model loading as the infrastructure
        handles model management transparently.
        """
        return ModelLoadResult(
            success=True,
            model=model,
            action="load",
            metadata={"backend": "gemini", "note": "cloud_provider_noop"},
        )

    async def unload_model(
        self,
        model: str,
        backend_obj: BackendConfig | None = None,
    ) -> ModelLoadResult:
        """No-op for cloud provider - no local resources to release.

        Cloud providers don't have GPU memory to free, so unloading
        is a no-op that always succeeds.
        """
        return ModelLoadResult(
            success=True,
            model=model,
            action="unload",
            metadata={"backend": "gemini", "note": "cloud_provider_noop"},
        )

    async def list_loaded_models(
        self,
        backend_obj: BackendConfig | None = None,
    ) -> list[str]:
        """Return empty list for cloud provider - all models always available.

        For cloud providers, the concept of "loaded" models doesn't apply.
        All available models are ready to use at any time.
        """
        return []
