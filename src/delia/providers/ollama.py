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

# Track models that have been auto-pulled this session to avoid repeated attempts
_auto_pulled_models: set[str] = set()


async def auto_pull_model(model: str, base_url: str) -> bool:
    """Auto-pull a model from Ollama if not available.

    Args:
        model: Model name to pull
        base_url: Ollama API base URL

    Returns:
        True if model was successfully pulled, False otherwise
    """
    if model in _auto_pulled_models:
        log.debug("model_already_pulled_this_session", model=model)
        return False

    log.info(
        "auto_pull_starting",
        model=model,
        log_type="PULL",
        status_msg=f"Auto-pulling model {model}... This may take a while.",
    )

    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
            # Use Ollama's pull endpoint with streaming to track progress
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": model, "stream": True},
            ) as response:
                if response.status_code != 200:
                    log.error("auto_pull_failed", model=model, status=response.status_code)
                    return False

                last_status = ""
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if status != last_status:
                            # Log progress updates
                            if "pulling" in status.lower():
                                completed = data.get("completed", 0)
                                total = data.get("total", 0)
                                if total > 0:
                                    pct = (completed / total) * 100
                                    log.info(
                                        "auto_pull_progress",
                                        model=model,
                                        status=status,
                                        progress=f"{pct:.1f}%",
                                        log_type="PULL",
                                    )
                            elif status:
                                log.info("auto_pull_status", model=model, status=status, log_type="PULL")
                            last_status = status
                    except json.JSONDecodeError:
                        continue

        _auto_pulled_models.add(model)
        log.info(
            "auto_pull_complete",
            model=model,
            log_type="PULL",
            status_msg=f"Model {model} is now available.",
        )
        return True

    except httpx.TimeoutException:
        log.error("auto_pull_timeout", model=model)
        return False
    except Exception as e:
        log.error("auto_pull_error", model=model, error=str(e))
        return False


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


def _gemma_escape_json(obj: Any) -> str:
    """Recursive formatter for FunctionGemma's peculiar JSON-like DSL."""
    if isinstance(obj, str):
        return f"<escape>{obj}<escape>"
    if isinstance(obj, list):
        items = [_gemma_escape_json(i) for i in obj]
        return "[" + ",".join(items) + "]"
    if isinstance(obj, dict):
        # Keys are NOT quoted in FunctionGemma DSL
        pairs = [f"{k}:{_gemma_escape_json(v)}" for k, v in obj.items()]
        return "{" + ",".join(pairs) + "}"
    # For numbers/booleans, standard JSON is fine
    return json.dumps(obj)


def _format_function_gemma_tools(tools: list[dict[str, Any]]) -> str:
    """Format tools for FunctionGemma using strict XML-like tags and DSL.
    
    See: https://docs.unsloth.ai/models/functiongemma
    """
    declarations = []
    for tool in tools:
        if "function" not in tool:
            continue
        func = tool["function"]
        name = func["name"]
        desc = func.get("description", "")
        params = _gemma_escape_json(func.get("parameters", {}))
        
        # FunctionGemma expects: <start_function_declaration>declaration:name{description:<escape>desc<escape>,parameters:params}<end_function_declaration>
        # Note: No space after colon, no quotes on top-level keys.
        decl = (
            f"\ndeclaration:{name}{{"
            f"description:<escape>{desc}<escape>,"
            f"parameters:{params}}}"
        )
        declarations.append(decl)
    
    return "<start_function_declaration>" + "".join(declarations) + "\n<end_function_declaration>"


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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Call Ollama API with Pydantic validation, retry logic, and circuit breaker.

        When tools are provided, uses /api/chat endpoint for native tool calling support.
        Otherwise uses /api/generate for standard completions.
        """
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
        elif model_tier == "dispatcher":
            num_ctx = self.config.model_dispatcher.num_ctx
        else:
            num_ctx = self.config.model_quick.num_ctx

        # Temperature: use override if provided, otherwise based on thinking mode
        if temperature is not None:
            temp_value = temperature
        else:
            temp_value = self.config.temperature_thinking if enable_thinking else self.config.temperature_normal

        options: dict[str, Any] = {
            "temperature": temp_value,
            "num_ctx": num_ctx,
        }
        
        # CRITICAL: Force Dispatcher to CPU to save VRAM for 30B models
        if model_tier == "dispatcher":
            options["num_gpu"] = 0
            log.debug("forcing_cpu_for_dispatcher", model=model)
            
        if max_tokens:
            options["num_predict"] = max_tokens

        # Determine endpoint and payload format based on whether messages or tools are provided
        # /api/chat supports tools and message history, /api/generate is for single completions
        use_chat_endpoint = (tools is not None) or (messages is not None)
        is_function_gemma = "functiongemma" in model.lower()

        if use_chat_endpoint:
            if is_function_gemma:
                # FunctionGemma logic (stays the same, handles messages internally)
                tool_declarations = _format_function_gemma_tools(tools) if tools else ""
                gemma_system = ""
                if tool_declarations:
                    gemma_system = (
                        f"You are a model that can do function calling with the following functions\n"
                        f"{tool_declarations}"
                    )
                if system:
                    gemma_system = f"{system}\n\n{gemma_system}" if gemma_system else system
                
                full_prompt = f"<start_of_turn>developer\n{gemma_system}\n<end_of_turn>\n"
                
                if messages:
                    for msg in messages:
                        role = msg.get("role")
                        content = msg.get("content")
                        if role == "system": continue 
                        full_prompt += f"<start_of_turn>{role}\n{content}\n<end_of_turn>\n"
                else:
                    full_prompt += f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n"
                
                full_prompt += "<start_of_turn>model\n"
                
                payload: dict[str, Any] = {
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "raw": True,
                    "options": options,
                }
                endpoint = "/api/generate"
            else:
                # Standard /api/chat endpoint
                if messages:
                    chat_messages = messages.copy()
                    # Ensure system prompt is present in history
                    if system and not any(m.get("role") == "system" for m in chat_messages):
                        chat_messages.insert(0, {"role": "system", "content": system})
                else:
                    chat_messages = []
                    if system:
                        chat_messages.append({"role": "system", "content": system})
                    chat_messages.append({"role": "user", "content": prompt})

                payload: dict[str, Any] = {
                    "model": model,
                    "messages": chat_messages,
                    "stream": False,
                    "options": options,
                }
                if tools:
                    payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice
                endpoint = "/api/chat"
        else:
            # Standard single completion
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }
            if system:
                payload["system"] = system
            endpoint = "/api/generate"

        client = backend_obj.get_client()

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            retry=retry_if_exception_type(httpx.ConnectError),
            reraise=True,
        )
        async def _make_request():
            return await client.post(endpoint, json=payload)

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
                except json.JSONDecodeError:
                    return create_error_response("Ollama returned non-JSON response")

                # Handle response based on endpoint used
                tool_calls = None
                if endpoint == "/api/chat":
                    # Chat endpoint response format
                    message = data.get("message", {})
                    response_text = message.get("content", "")
                    tool_calls = message.get("tool_calls")
                    # Token counts in chat response
                    tokens = data.get("eval_count", 0)
                    if tokens == 0:
                        tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                else:
                    # Generate endpoint response format
                    response_text = data.get("response", "")
                    tokens = data.get("eval_count", 0)
                    if tokens == 0:
                        tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)

                # If it's FunctionGemma and we don't have tool_calls yet, parse them from text
                if is_function_gemma and not tool_calls and "<start_function_call>" in response_text:
                    tool_calls = []
                    # Regex for <start_function_call>call:name{args}<end_function_call>
                    pattern = r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
                    for match in re.finditer(pattern, response_text, re.DOTALL):
                        func_name = match.group(1)
                        args_str = match.group(2)
                        
                        try:
                            # Clean up the string to make it valid JSON
                            # FunctionGemma output is often: path:<escape>val<escape>
                            # We need to convert it to "path":"val"
                            cleaned_args = args_str.replace("<escape>", '"')
                            # Handle unquoted keys: key:value -> "key":value
                            cleaned_args = re.sub(r'(\w+):', r'"\1":', cleaned_args)
                            
                            args = json.loads(f"{{{cleaned_args}}}")
                        except Exception as e:
                            log.debug("gemma_parse_args_failed", error=str(e), raw=args_str)
                            args = {"raw_args": args_str}
                            
                        tool_calls.append({
                            "id": f"call_{int(time.time())}_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(args)
                            }
                        })
                    
                    # Strip tool calls from the text for a cleaner response
                    response_text = re.sub(pattern, "", response_text, flags=re.DOTALL).strip()

                # Stats tracking
                if self.stats_callback:
                    self.stats_callback(
                        model_tier=model_tier,
                        task_type=task_type,
                        original_task=original_task,
                        tokens=tokens,
                        elapsed_ms=elapsed_ms,
                        content_preview=content_preview,
                        enable_thinking=enable_thinking,
                        backend="ollama",
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
                get_backend_metrics(backend_obj.id).record_success(
                    elapsed_ms=float(elapsed_ms), tokens=tokens
                )

                if self.save_stats_callback:
                    self.save_stats_callback()

                metadata: dict[str, Any] = {"backend": "ollama", "model": model, "tier": model_tier}
                if tool_calls:
                    metadata["tool_calls"] = tool_calls

                return create_success_response(
                    response_text=response_text,
                    tokens=tokens,
                    elapsed_ms=elapsed_ms,
                    metadata=metadata,
                )

            # HTTP error handling
            error_elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Ollama HTTP {response.status_code}"

            if response.status_code == 404:
                # Model not found - try to auto-pull it
                if model not in _auto_pulled_models:
                    log.info("model_not_found_attempting_pull", model=model)
                    pull_success = await auto_pull_model(model, backend_obj.url)
                    if pull_success:
                        # Retry the request after successful pull
                        log.info("retrying_after_auto_pull", model=model)
                        response = await _make_request()
                        if response.status_code == 200:
                            # Re-process successful response (same logic as above)
                            try:
                                data = response.json()
                            except json.JSONDecodeError:
                                return create_error_response("Ollama returned non-JSON response")

                            tool_calls = None
                            if use_chat_endpoint:
                                message = data.get("message", {})
                                response_text = message.get("content", "")
                                tool_calls = message.get("tool_calls")
                                tokens = data.get("eval_count", 0)
                                if tokens == 0:
                                    tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                            else:
                                try:
                                    validated = OllamaResponse.model_validate(data)
                                    tokens = validated.eval_count
                                    response_text = validated.response
                                except ValidationError as e:
                                    log.warning("ollama_validation_failed", error=str(e))
                                    tokens = data.get("eval_count", 0)
                                    response_text = data.get("response", "")

                            retry_elapsed_ms = int((time.time() - start_time) * 1000)

                            if self.stats_callback:
                                self.stats_callback(
                                    model_tier=model_tier,
                                    task_type=task_type,
                                    original_task=original_task,
                                    tokens=tokens,
                                    elapsed_ms=retry_elapsed_ms,
                                    content_preview=content_preview,
                                    enable_thinking=enable_thinking,
                                    backend="ollama",
                                )

                            log_thinking_and_response(response_text, model_tier, tokens)
                            health.record_success(content_size)
                            get_backend_metrics(backend_obj.id).record_success(
                                elapsed_ms=float(retry_elapsed_ms), tokens=tokens
                            )

                            if self.save_stats_callback:
                                self.save_stats_callback()

                            metadata = {"backend": "ollama", "model": model, "tier": model_tier, "auto_pulled": True}
                            if tool_calls:
                                metadata["tool_calls"] = tool_calls

                            return create_success_response(
                                response_text=response_text,
                                tokens=tokens,
                                elapsed_ms=retry_elapsed_ms,
                                metadata=metadata,
                            )
                        # If retry also failed, return error with retry status
                        retry_error_msg = f"Ollama HTTP {response.status_code} (after auto-pull)"
                        if response.status_code == 500:
                            retry_error_msg += ": Internal server error. Check Ollama logs."
                        else:
                            retry_error_text = response.text[:200] if len(response.text) > 200 else response.text
                            retry_error_msg += f": {retry_error_text}"
                        health.record_failure("http_error", content_size)
                        get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(error_elapsed_ms))
                        return create_error_response(retry_error_msg)

                error_msg += f": Model '{model}' not found. Auto-pull failed or already attempted."
            elif response.status_code == 500:
                error_msg += ": Internal server error. Check Ollama logs."
            elif response.status_code == 503:
                error_msg += ": Ollama service unavailable. Is it running?"
            else:
                error_text = response.text[:500] if len(response.text) > 500 else response.text
                error_msg += f": {error_text}"
            health.record_failure("http_error", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(error_elapsed_ms))
            return create_error_response(error_msg)

        except httpx.TimeoutException:
            timeout_elapsed_ms = int((time.time() - start_time) * 1000)
            log.error("ollama_timeout", model=model, timeout_seconds=self.config.ollama_timeout_seconds)
            health.record_failure("timeout", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(timeout_elapsed_ms))
            return create_error_response(
                f"Ollama timeout after {self.config.ollama_timeout_seconds}s. Model may be loading or prompt too large."
            )
        except httpx.ConnectError:
            log.error("ollama_connection_refused", base_url=backend_obj.url)
            health.record_failure("connection", content_size)
            # Connection errors don't have meaningful latency - record as 0
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=0)
            return create_error_response(f"Cannot connect to Ollama at {backend_obj.url}. Is Ollama running?")
        except Exception as e:
            exc_elapsed_ms = int((time.time() - start_time) * 1000)
            log.error("ollama_error", model=model, error=str(e))
            health.record_failure("exception", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(exc_elapsed_ms))
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from Ollama API token by token.

        Yields StreamChunk objects as the model generates tokens.
        The final chunk has done=True and includes total token count.

        Note: tools/tool_choice are accepted for signature compatibility but
        tool calling with streaming is not currently supported.
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
        elif model_tier == "dispatcher":
            num_ctx = self.config.model_dispatcher.num_ctx
        else:
            num_ctx = self.config.model_quick.num_ctx

        temperature = self.config.temperature_thinking if enable_thinking else self.config.temperature_normal

        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }

        # Force Dispatcher to CPU
        if model_tier == "dispatcher":
            options["num_gpu"] = 0

        if max_tokens:
            options["num_predict"] = max_tokens

        is_function_gemma = "functiongemma" in model.lower()
        
        if is_function_gemma:
            # Wrap in strict Gemma turn markers for streaming too
            full_prompt = ""
            if system:
                full_prompt += f"<start_of_turn>developer\n{system}\n<end_of_turn>\n"
            full_prompt += f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n"
            full_prompt += "<start_of_turn>model\n"
            
            payload: dict[str, Any] = {
                "model": model,
                "prompt": full_prompt,
                "stream": True,
                "raw": True, # Exact control
                "options": options,
            }
        else:
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
                    stream_error_ms = int((time.time() - start_time) * 1000)
                    health.record_failure("http_error", content_size)
                    get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(stream_error_ms))
                    yield StreamChunk(
                        done=True,
                        error=f"Ollama HTTP {response.status_code}: {error_text[:500]}",
                    )
                    return

                last_emitted_thinking_len = 0
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
                        
                        # Detect thinking in real-time
                        from ..text_utils import extract_thinking, extract_answer
                        current_thinking = extract_thinking(full_response)
                        
                        if current_thinking:
                            # If thinking has grown, emit the NEW part as a thinking event
                            new_thinking = current_thinking[last_emitted_thinking_len:]
                            if new_thinking:
                                yield StreamChunk(text="", thinking=new_thinking)
                                last_emitted_thinking_len = len(current_thinking)
                        
                        # Only yield actual answer text to the UI
                        # extract_answer returns "" if only thinking is present
                        current_answer = extract_answer(full_response)
                        if current_answer:
                            # We only want to yield the NEW part of the answer
                            # full_response includes thinking + answer.
                            # text_chunk is just the latest delta.
                            # If current_answer is shorter than text_chunk, it means
                            # text_chunk was part of thinking.
                            # If current_answer exists, we yield the delta of the answer.
                            
                            # Logic: find where the answer starts in full_response
                            answer_start_idx = full_response.rfind("</think>")
                            if answer_start_idx != -1:
                                answer_start_idx += 8 # len("</think>")
                                # If the latest chunk crossed the </think> boundary:
                                if len(full_response) - len(text_chunk) < answer_start_idx:
                                    # Yield only the part of text_chunk that is answer
                                    offset = answer_start_idx - (len(full_response) - len(text_chunk))
                                    answer_delta = text_chunk[offset:]
                                    if answer_delta:
                                        yield StreamChunk(text=answer_delta)
                                else:
                                    # Latest chunk is entirely answer
                                    yield StreamChunk(text=text_chunk)
                            else:
                                # No thinking tags at all, whole chunk is answer
                                yield StreamChunk(text=text_chunk)

                    # Check if done
                    if data.get("done", False):
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        total_tokens = data.get("eval_count", 0)

                        # Stats tracking
                        if self.stats_callback:
                            self.stats_callback(
                                model_tier=model_tier,
                                task_type=task_type,
                                original_task=original_task,
                                tokens=total_tokens,
                                elapsed_ms=elapsed_ms,
                                content_preview=content_preview,
                                enable_thinking=enable_thinking,
                                backend="ollama",
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
            stream_timeout_ms = int((time.time() - start_time) * 1000)
            health.record_failure("timeout", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(stream_timeout_ms))
            yield StreamChunk(
                done=True,
                error=f"Ollama timeout after {self.config.ollama_timeout_seconds}s.",
            )
        except httpx.ConnectError:
            health.record_failure("connection", content_size)
            # Connection errors don't have meaningful latency - record as 0
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=0)
            yield StreamChunk(
                done=True,
                error=f"Cannot connect to Ollama at {backend_obj.url}. Is Ollama running?",
            )
        except Exception as e:
            stream_exc_ms = int((time.time() - start_time) * 1000)
            health.record_failure("exception", content_size)
            get_backend_metrics(backend_obj.id).record_failure(elapsed_ms=float(stream_exc_ms))
            yield StreamChunk(done=True, error=f"Ollama streaming error: {e!s}")

    def _find_ollama_backend(self) -> BackendConfig | None:
        """Find the first enabled Ollama backend."""
        for b in self.backend_manager.get_enabled_backends():
            if b.provider == "ollama":
                return b
        return None

    async def load_model(
        self,
        model: str,
        backend_obj: BackendConfig | None = None,
    ) -> ModelLoadResult:
        """Preload a model into Ollama's GPU memory.

        Uses Ollama's keep_alive=-1 to load and keep the model indefinitely.
        This reduces latency on the first inference request.
        """
        start_time = time.time()

        if not backend_obj:
            backend_obj = self._find_ollama_backend()
        if not backend_obj:
            return ModelLoadResult(
                success=False,
                model=model,
                action="load",
                error="No enabled Ollama backend found",
            )

        client = backend_obj.get_client()

        try:
            # Empty prompt with keep_alive=-1 preloads model indefinitely
            payload = {"model": model, "keep_alive": -1}
            response = await client.post("/api/generate", json=payload, timeout=300.0)
            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                log.info(
                    "model_preloaded",
                    model=model,
                    elapsed_ms=elapsed_ms,
                    backend="ollama",
                )
                return ModelLoadResult(
                    success=True,
                    model=model,
                    action="load",
                    elapsed_ms=elapsed_ms,
                    metadata={"backend": "ollama"},
                )
            elif response.status_code == 404:
                return ModelLoadResult(
                    success=False,
                    model=model,
                    action="load",
                    error=f"Model '{model}' not found. Run: ollama pull {model}",
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
        """Unload a model from Ollama's GPU memory.

        Uses Ollama's keep_alive=0 to immediately unload the model,
        freeing GPU memory for other models.
        """
        start_time = time.time()

        if not backend_obj:
            backend_obj = self._find_ollama_backend()
        if not backend_obj:
            return ModelLoadResult(
                success=False,
                model=model,
                action="unload",
                error="No enabled Ollama backend found",
            )

        client = backend_obj.get_client()

        try:
            # Empty prompt with keep_alive=0 unloads model immediately
            payload = {"model": model, "keep_alive": 0}
            response = await client.post("/api/generate", json=payload, timeout=30.0)
            elapsed_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                # Verify unload succeeded (Ollama returns done_reason: "unload")
                done_reason = data.get("done_reason", "")
                log.info(
                    "model_unloaded_ollama",
                    model=model,
                    elapsed_ms=elapsed_ms,
                    done_reason=done_reason,
                    backend="ollama",
                )
                return ModelLoadResult(
                    success=True,
                    model=model,
                    action="unload",
                    elapsed_ms=elapsed_ms,
                    metadata={"backend": "ollama", "done_reason": done_reason},
                )
            elif response.status_code == 404:
                # Model not found - consider unload successful
                return ModelLoadResult(
                    success=True,
                    model=model,
                    action="unload",
                    elapsed_ms=elapsed_ms,
                    metadata={"backend": "ollama", "note": "model_not_found"},
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
        """Get list of currently loaded models in Ollama.

        Uses Ollama's /api/ps endpoint to query running models.
        """
        if not backend_obj:
            backend_obj = self._find_ollama_backend()
        if not backend_obj:
            return []

        client = backend_obj.get_client()

        try:
            response = await client.get("/api/ps", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                # Extract model names, stripping :latest suffix for consistency
                return [
                    m.get("name", "").replace(":latest", "")
                    for m in models
                    if m.get("name")
                ]
        except Exception as e:
            log.warning("list_loaded_models_failed", backend="ollama", error=str(e))

        return []
