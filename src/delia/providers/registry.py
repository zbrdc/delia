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
"""Provider registry for routing requests to the appropriate LLM backend.

This module implements a registry pattern that maps provider names to provider
implementations. It serves as the central dispatcher, similar to LiteLLM's router.

The registry:
- Maintains a mapping of provider name -> provider instance
- Handles provider lookup and delegation
- Supports runtime provider registration
- Enables dependency injection (stats callbacks, config, etc.)

Design principles:
- Single responsibility: routing only, not implementation
- Extensible: easy to add new providers
- Type-safe: uses Protocol for provider validation
- Testable: dependency injection for all stateful components
"""

from typing import Any, Callable

from ..backend_manager import BackendConfig
from .base import LLMProvider, LLMResponse, create_error_response


class ProviderRegistry:
    """Central registry for routing LLM requests to provider implementations.

    This class acts as a facade/dispatcher, delegating calls to the appropriate
    provider based on the backend configuration. It decouples the calling code
    from specific provider implementations.

    Example:
        registry = ProviderRegistry()
        registry.register("ollama", OllamaProvider())
        registry.register("llamacpp", LlamaCppProvider())

        response = await registry.call(
            backend=my_backend_config,
            model="qwen2.5:14b",
            prompt="Hello world",
        )
    """

    def __init__(self) -> None:
        """Initialize empty provider registry."""
        self._providers: dict[str, LLMProvider] = {}
        self._stats_callback: Callable[..., None] | None = None

    def register(self, provider_name: str, provider: LLMProvider) -> None:
        """Register a provider implementation.

        Args:
            provider_name: Unique identifier for the provider (e.g., "ollama", "gemini")
            provider: Provider instance implementing the LLMProvider protocol

        Raises:
            ValueError: If provider_name is already registered or if provider
                       doesn't implement LLMProvider protocol
        """
        if provider_name in self._providers:
            raise ValueError(f"Provider '{provider_name}' is already registered")

        # Runtime protocol check
        if not isinstance(provider, LLMProvider):
            raise ValueError(f"Provider must implement LLMProvider protocol, got {type(provider)}")

        self._providers[provider_name] = provider

    def unregister(self, provider_name: str) -> None:
        """Remove a provider from the registry.

        Args:
            provider_name: Name of provider to remove

        Raises:
            KeyError: If provider_name is not registered
        """
        if provider_name not in self._providers:
            raise KeyError(f"Provider '{provider_name}' is not registered")

        del self._providers[provider_name]

    def get_provider(self, provider_name: str) -> LLMProvider | None:
        """Get a registered provider by name.

        Args:
            provider_name: Name of the provider to retrieve

        Returns:
            Provider instance if found, None otherwise
        """
        return self._providers.get(provider_name)

    def list_providers(self) -> list[str]:
        """Get list of all registered provider names.

        Returns:
            Sorted list of provider names
        """
        return sorted(self._providers.keys())

    def set_stats_callback(self, callback: Callable[..., None]) -> None:
        """Set a callback for stats tracking.

        The callback will be invoked by providers after successful completions
        to update usage statistics. This enables dependency injection for
        observability without coupling providers to specific stats implementations.

        Args:
            callback: Function to call with stats data (model_tier, task_type,
                     tokens, elapsed_ms, etc.)
        """
        self._stats_callback = callback

    async def call(
        self,
        backend: BackendConfig,
        model: str,
        prompt: str,
        system: str | None = None,
        enable_thinking: bool = False,
        task_type: str = "unknown",
        original_task: str = "unknown",
        language: str = "unknown",
        content_preview: str = "",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Route request to the appropriate provider based on backend configuration.

        This is the main entry point for making LLM calls through the registry.
        It looks up the provider based on backend.provider and delegates to it.

        Args:
            backend: Backend configuration containing provider type and settings
            model: Model name to use
            prompt: User prompt
            system: Optional system prompt
            enable_thinking: Enable extended reasoning mode
            task_type: Task type for stats tracking
            original_task: Original task string from user
            language: Detected programming language
            content_preview: Short preview for logging
            max_tokens: Optional response length limit

        Returns:
            LLMResponse from the provider, or error response if provider not found

        Note:
            This method never raises exceptions - all errors are returned as
            LLMResponse objects with success=False
        """
        # Look up provider
        provider = self._providers.get(backend.provider)
        if provider is None:
            return create_error_response(
                error_message=f"No provider registered for '{backend.provider}'. "
                f"Available providers: {', '.join(self.list_providers())}",
                metadata={"backend_id": backend.id, "provider": backend.provider},
            )

        # Delegate to provider
        try:
            response = await provider.call(
                model=model,
                prompt=prompt,
                system=system,
                enable_thinking=enable_thinking,
                task_type=task_type,
                original_task=original_task,
                language=language,
                content_preview=content_preview,
                backend_obj=backend,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            # Catch any unexpected exceptions from providers (shouldn't happen
            # per protocol contract, but defensive programming)
            return create_error_response(
                error_message=f"Provider '{backend.provider}' raised unexpected exception: {e!s}",
                metadata={
                    "backend_id": backend.id,
                    "provider": backend.provider,
                    "exception_type": type(e).__name__,
                },
            )


# ============================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================

# Singleton instance for application-wide use
# Providers will be registered during application startup
_global_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry instance.

    This function provides lazy initialization of the global registry.
    It ensures a single registry exists for the entire application.

    Returns:
        The global ProviderRegistry instance

    Example:
        registry = get_registry()
        await registry.call(backend=my_backend, model="...", prompt="...")
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ProviderRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (mainly for testing).

    This function clears the singleton instance, allowing tests to
    start with a clean registry state.
    """
    global _global_registry
    _global_registry = None
