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
"""Pydantic models for LLM API responses."""

from pydantic import BaseModel


class OllamaResponse(BaseModel):
    """Ollama /api/generate response model."""

    response: str = ""
    eval_count: int = 0
    done: bool = True


class ToolCall(BaseModel):
    """OpenAI-compatible tool call."""

    id: str
    type: str = "function"
    function: dict[str, str]  # Contains "name" and "arguments" keys


class LlamaCppMessage(BaseModel):
    """OpenAI-compatible message."""

    role: str = "assistant"
    content: str | None = ""
    tool_calls: list[ToolCall] | None = None


class LlamaCppChoice(BaseModel):
    """OpenAI-compatible choice."""

    message: LlamaCppMessage
    index: int = 0
    finish_reason: str = "stop"


class LlamaCppUsage(BaseModel):
    """OpenAI-compatible usage stats."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LlamaCppResponse(BaseModel):
    """llama.cpp /v1/chat/completions response model."""

    choices: list[LlamaCppChoice]
    usage: LlamaCppUsage | None = None
    model: str = ""
    id: str = ""


class LlamaCppError(BaseModel):
    """llama.cpp error response."""

    type: str = ""
    message: str = ""
    n_prompt_tokens: int | None = None
    n_ctx: int | None = None
