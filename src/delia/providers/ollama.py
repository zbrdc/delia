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

"""Ollama LLM provider implementation."""

import json
import re
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
from .base import (
    LLMResponse,
    ModelLoadResult,
    StreamChunk,
    create_error_response,
    create_success_response,
)
from .models import OllamaResponse

if TYPE_CHECKING:
    from ..config import Config

log = structlog.get_logger()

def extract_thinking_content(response_text: str) -> str | None:
    """Extract thinking content from LLM response."""
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

class OllamaProvider:
    """Ollama LLM provider with full support for streaming, native tool calling and conversation history."""

    def __init__(
        self,
        config: "Config",
        backend_manager: Any,
        stats_callback: Any | None = None,
        save_stats_callback: Callable[[], Any] | None = None,
    ):
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
        temperature: float | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Call Ollama /api/chat or /api/generate."""
        start_time = time.time()
        if not backend_obj:
            backend_obj = self.backend_manager.get_active_backend()
        if not backend_obj:
            return create_error_response("No backend")

        use_chat = (messages is not None) or (tools is not None)
        endpoint = "/api/chat" if use_chat else "/api/generate"
        
        # Build payload
        payload = {"model": model, "stream": False}
        if use_chat:
            history = messages.copy() if messages else []
            if system and not any(m["role"] == "system" for m in history):
                history.insert(0, {"role": "system", "content": system})
            if not any(m["role"] == "user" for m in history) and prompt:
                history.append({"role": "user", "content": prompt})
            payload["messages"] = history
            if tools: payload["tools"] = tools
        else:
            payload["prompt"] = prompt
            if system: payload["system"] = system

        options = {"temperature": temperature or (self.config.temperature_thinking if enable_thinking else self.config.temperature_normal)}
        if max_tokens: options["num_predict"] = max_tokens
        payload["options"] = options

        try:
            client = backend_obj.get_client()
            response = await client.post(endpoint, json=payload, timeout=self.config.ollama_timeout_seconds)
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                data = response.json()
                if use_chat:
                    msg = data.get("message", {})
                    text = msg.get("content", "")
                    tool_calls = msg.get("tool_calls")
                else:
                    text = data.get("response", "")
                    tool_calls = None
                
                tokens = data.get("eval_count", 0) or (data.get("prompt_eval_count", 0) + data.get("eval_count", 0))
                
                # Internal logging
                model_tier = detect_model_tier(model)
                log_thinking_and_response(text, model_tier, tokens)
                
                return create_success_response(text, tokens, elapsed_ms, {"tool_calls": tool_calls})
            return create_error_response(f"Ollama error {response.status_code}: {response.text}")
        except Exception as e:
            return create_error_response(str(e))

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
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion from Ollama."""
        start_time = time.time()
        if not backend_obj:
            backend_obj = self.backend_manager.get_active_backend()
        
        use_chat = (messages is not None)
        endpoint = "/api/chat" if use_chat else "/api/generate"
        
        payload = {"model": model, "stream": True}
        if use_chat:
            history = messages.copy()
            if system and not any(m["role"] == "system" for m in history):
                history.insert(0, {"role": "system", "content": system})
            payload["messages"] = history
        else:
            payload["prompt"] = prompt
            if system: payload["system"] = system

        try:
            client = backend_obj.get_client()
            async with client.stream("POST", endpoint, json=payload, timeout=self.config.ollama_timeout_seconds) as response:
                if response.status_code != 200:
                    yield StreamChunk(done=True, error=f"Ollama error {response.status_code}")
                    return

                async for line in response.aiter_lines():
                    if not line: continue
                    data = json.loads(line)
                    
                    if use_chat:
                        chunk_text = data.get("message", {}).get("content", "")
                    else:
                        chunk_text = data.get("response", "")
                    
                    if chunk_text:
                        yield StreamChunk(text=chunk_text)
                    
                    if data.get("done"):
                        tokens = data.get("eval_count", 0)
                        yield StreamChunk(done=True, tokens=tokens)
        except Exception as e:
            yield StreamChunk(done=True, error=str(e))

    async def load_model(self, model: str, backend_obj: BackendConfig | None = None) -> ModelLoadResult:
        return ModelLoadResult(True, model, "load")

    async def unload_model(self, model: str, backend_obj: BackendConfig | None = None) -> ModelLoadResult:
        return ModelLoadResult(True, model, "unload")

    async def list_loaded_models(self, backend_obj: BackendConfig | None = None) -> list[str]:
        return []
