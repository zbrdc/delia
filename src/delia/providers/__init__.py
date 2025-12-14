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
"""LLM provider implementations for Delia.

This package contains provider-specific implementations for calling LLMs:
- ollama: Ollama API client
- llamacpp: OpenAI-compatible API client (llama.cpp, vLLM, LM Studio, etc.)
- gemini: Google Gemini API client
- router: Unified dispatcher that routes to the appropriate provider

The provider abstraction layer follows LiteLLM patterns:
- base: Protocol-based interface and response types
- registry: Central dispatcher for routing to providers
- models: Pydantic models for API responses
"""

from .base import (
    LLMProvider,
    LLMResponse,
    StreamChunk,
    create_error_response,
    create_success_response,
)
from .gemini import GeminiProvider
from .llamacpp import LlamaCppProvider
from .models import (
    LlamaCppChoice,
    LlamaCppError,
    LlamaCppMessage,
    LlamaCppResponse,
    LlamaCppUsage,
    OllamaResponse,
)
from .ollama import OllamaProvider
from .registry import ProviderRegistry, get_registry, reset_registry

__all__ = [
    # Base provider interface
    "LLMProvider",
    "LLMResponse",
    "StreamChunk",
    "create_error_response",
    "create_success_response",
    # Provider implementations
    "OllamaProvider",
    "LlamaCppProvider",
    "GeminiProvider",
    # Registry
    "ProviderRegistry",
    "get_registry",
    "reset_registry",
    # API response models
    "OllamaResponse",
    "LlamaCppMessage",
    "LlamaCppChoice",
    "LlamaCppUsage",
    "LlamaCppResponse",
    "LlamaCppError",
]
