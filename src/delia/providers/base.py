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

"""Base provider interface for LLM orchestration.

This module defines the protocol-based interface that all LLM providers must implement.
It provides a unified abstraction layer similar to LiteLLM's approach, where each provider
handles its own request/response transformation.

Design principles:
- Protocol-based interface (not ABC) for more Pythonic duck typing
- Standardized response format across all providers
- Circuit breaker and retry logic handled at provider level
- Dependency injection ready for stats tracking and configuration
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..backend_manager import BackendConfig


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response.

    Attributes:
        text: The text content of this chunk (may be empty for metadata-only chunks)
        done: Whether this is the final chunk
        tokens: Running token count (only set on final chunk)
        error: Error message if streaming failed mid-stream
        metadata: Provider-specific chunk metadata
        thinking: The thinking/reasoning content of this chunk (optional)
    """

    text: str = ""
    done: bool = False
    tokens: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    thinking: str | None = None


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider.

    This dataclass provides a uniform interface for all provider responses,
    making it easy to switch between providers without changing calling code.

    Attributes:
        success: Whether the call succeeded
        response: The text response from the LLM (empty string if failed)
        tokens: Total tokens used (prompt + completion)
        elapsed_ms: Time taken in milliseconds
        error: Error message if success=False, None otherwise
        metadata: Provider-specific metadata (usage breakdown, model info, etc.)
        circuit_breaker: Whether failure was due to circuit breaker (for retry logic)
    """

    success: bool
    response: str
    tokens: int
    elapsed_ms: int
    error: str | None = None
    metadata: dict[str, Any] | None = None
    circuit_breaker: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary format (legacy compatibility).

        This matches the existing dict-based return format from call_ollama, etc.
        Allows gradual migration from dict returns to typed responses.
        """
        result: dict[str, Any] = {
            "success": self.success,
            "response": self.response,
            "tokens": self.tokens,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.error:
            result["error"] = self.error
        if self.circuit_breaker:
            result["circuit_breaker"] = True
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMResponse":
        """Create LLMResponse from dictionary (for migration/testing)."""
        return cls(
            success=data.get("success", False),
            response=data.get("response", ""),
            tokens=data.get("tokens", 0),
            elapsed_ms=data.get("elapsed_ms", 0),
            error=data.get("error"),
            metadata=data.get("metadata"),
            circuit_breaker=data.get("circuit_breaker", False),
        )


@dataclass
class ModelLoadResult:
    """Result of a model load/unload operation.

    Attributes:
        success: Whether the operation succeeded
        model: Model name that was loaded/unloaded
        action: The operation performed ("load" or "unload")
        error: Error message if success=False
        elapsed_ms: Time taken in milliseconds
        metadata: Provider-specific metadata
    """

    success: bool
    model: str
    action: str  # "load" | "unload"
    error: str | None = None
    elapsed_ms: int = 0
    metadata: dict[str, Any] | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface all LLM providers must implement.

    Using Protocol instead of ABC allows for structural subtyping - any class
    that implements the required methods is considered a valid provider without
    explicit inheritance.

    Each provider handles:
    - Request formatting (provider-specific API schema)
    - Response parsing and validation
    - Error handling and retries
    - Circuit breaker integration
    - Stats tracking callbacks

    The protocol matches the signature of existing call_ollama/call_llamacpp/call_gemini
    functions to enable gradual refactoring.
    """

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
    ) -> LLMResponse:
        """Call the LLM provider with the given parameters.

        Args:
            model: Model name to use (provider-specific format)
            prompt: The user prompt to send
            system: Optional system prompt/instructions
            enable_thinking: Enable extended reasoning mode for supported models
            task_type: Type of task for stats tracking (quick/review/analyze/etc.)
            original_task: Original task string from user
            language: Detected programming language for context
            content_preview: Short preview of content for logging
            backend_obj: Backend configuration to use (or None to auto-select)
            max_tokens: Optional limit on response length
            tools: Optional list of OpenAI-format tool schemas for native tool calling
            tool_choice: Optional tool choice strategy ("auto", "none", or specific tool name)

        Returns:
            LLMResponse with success status, response text, tokens, timing, and metadata.
            If tools were used, metadata will include a "tool_calls" field with the calls.

        Raises:
            This method should NOT raise exceptions - all errors should be captured
            in the LLMResponse.error field with success=False
        """
        ...

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
        """Stream response from the LLM provider token by token.

        This method provides incremental output as the model generates tokens,
        enabling real-time feedback to users. The primary use case is UX/latency
        optimization - getting tokens to users as quickly as possible.

        Args:
            model: Model name to use (provider-specific format)
            prompt: The user prompt to send
            system: Optional system prompt/instructions
            enable_thinking: Enable extended reasoning mode for supported models
            task_type: Type of task for stats tracking
            original_task: Original task string from user
            language: Detected programming language for context
            content_preview: Short preview of content for logging
            backend_obj: Backend configuration to use (or None to auto-select)
            max_tokens: Optional limit on response length
            tools: Optional list of OpenAI-format tool schemas. Tool support in
                streaming mode is provider-dependent and best-effort.
            tool_choice: Optional tool choice strategy ("auto", "none", or
                specific tool name). Only meaningful if tools are provided.

        Yields:
            StreamChunk objects containing:
            - text: Incremental text as it's generated
            - done: True on the final chunk
            - tokens: Total token count (only on final chunk)
            - error: Error message if streaming fails mid-stream
            - metadata: May contain "tool_calls" on final chunk if tools were used

        Tool Calling Behavior:
            Streaming tool support is provider-dependent and may be handled in
            one of the following ways:

            1. **Ignored**: Provider streams text normally, tools have no effect
            2. **Final chunk**: Tool calls accumulated and returned in the final
               chunk's metadata["tool_calls"] field
            3. **Fallback**: Provider internally calls non-streaming call() method
               which fully supports tools, then yields result as single chunk

            Incremental tool-call deltas (partial JSON fragments mid-stream) are
            NOT currently supported. For guaranteed tool execution, use the
            non-streaming call() method.

        Note:
            Implementations should handle errors gracefully by yielding a final
            chunk with done=True and error set, rather than raising exceptions.
        """
        # Default implementation: fall back to non-streaming call
        # This fallback DOES support tools since call() supports them
        response = await self.call(
            model=model,
            prompt=prompt,
            system=system,
            enable_thinking=enable_thinking,
            task_type=task_type,
            original_task=original_task,
            language=language,
            content_preview=content_preview,
            backend_obj=backend_obj,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
        if response.success:
            yield StreamChunk(
                text=response.response,
                done=True,
                tokens=response.tokens,
                metadata=response.metadata or {},
            )
        else:
            yield StreamChunk(
                done=True,
                error=response.error,
            )

    async def load_model(
        self,
        model: str,
        backend_obj: BackendConfig | None = None,
    ) -> ModelLoadResult:
        """Preload a model into GPU memory.

        This method triggers the backend to load a model before it's needed,
        reducing latency on the first inference request.

        Args:
            model: Model name to load (provider-specific format)
            backend_obj: Backend configuration (or None to auto-select)

        Returns:
            ModelLoadResult indicating success/failure with timing info

        Note:
            - For cloud providers (Gemini), this should be a no-op returning success
            - For local providers (Ollama, llama.cpp), this triggers actual model loading
            - Implementations should NOT raise exceptions
        """
        # Default: no-op returning success (for cloud providers)
        return ModelLoadResult(
            success=True,
            model=model,
            action="load",
            metadata={"note": "default_noop"},
        )

    async def unload_model(
        self,
        model: str,
        backend_obj: BackendConfig | None = None,
    ) -> ModelLoadResult:
        """Unload a model from GPU memory.

        This method releases GPU memory by unloading a model that's no longer needed.
        Called by ModelQueue when making room for other models.

        Args:
            model: Model name to unload
            backend_obj: Backend configuration (or None to auto-select)

        Returns:
            ModelLoadResult indicating success/failure

        Note:
            - For cloud providers (Gemini), this should be a no-op returning success
            - For local providers, this triggers actual model unloading
            - Safe to call even if model is not currently loaded
        """
        # Default: no-op returning success (for cloud providers)
        return ModelLoadResult(
            success=True,
            model=model,
            action="unload",
            metadata={"note": "default_noop"},
        )

    async def list_loaded_models(
        self,
        backend_obj: BackendConfig | None = None,
    ) -> list[str]:
        """Get list of currently loaded models.

        Returns the models currently resident in GPU memory for this provider.
        Used by ModelQueue to synchronize its internal state with actual backend state.

        Args:
            backend_obj: Backend configuration (or None to auto-select)

        Returns:
            List of model names currently loaded in GPU memory

        Note:
            - For cloud providers, return empty list (all models always available)
            - For local providers, query the backend for actual loaded models
        """
        # Default: empty list (for cloud providers)
        return []


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def create_error_response(
    error_message: str,
    circuit_breaker: bool = False,
    metadata: dict[str, Any] | None = None,
) -> LLMResponse:
    """Create a standardized error response.

    Helper function to ensure consistent error response format across all providers.

    Args:
        error_message: Human-readable error description
        circuit_breaker: Whether this error triggered the circuit breaker
        metadata: Optional provider-specific error details

    Returns:
        LLMResponse with success=False and error details
    """
    return LLMResponse(
        success=False,
        response="",
        tokens=0,
        elapsed_ms=0,
        error=error_message,
        metadata=metadata,
        circuit_breaker=circuit_breaker,
    )


def create_success_response(
    response_text: str,
    tokens: int,
    elapsed_ms: int,
    metadata: dict[str, Any] | None = None,
) -> LLMResponse:
    """Create a standardized success response.

    Helper function to ensure consistent success response format across all providers.

    Args:
        response_text: The LLM's text response
        tokens: Total tokens used (prompt + completion)
        elapsed_ms: Time taken in milliseconds
        metadata: Optional provider-specific details (usage breakdown, etc.)

    Returns:
        LLMResponse with success=True and response details
    """
    return LLMResponse(
        success=True,
        response=response_text,
        tokens=tokens,
        elapsed_ms=elapsed_ms,
        metadata=metadata,
    )
