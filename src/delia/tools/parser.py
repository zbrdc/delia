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
"""
Tool call parser for LLM responses.

Handles both OpenAI-native tool calling format and text-based
fallback format using XML-like markers.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()

# Pattern for text-based tool calls: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL
)


@dataclass
class ParsedToolCall:
    """A parsed tool call from LLM response.

    Attributes:
        id: Unique identifier for this call (for tracking)
        name: Tool name to execute
        arguments: Arguments to pass to the tool
    """
    id: str
    name: str
    arguments: dict[str, Any]


def parse_tool_calls(
    response: str | dict[str, Any],
    native_mode: bool = False,
) -> list[ParsedToolCall]:
    """Parse tool calls from LLM response.

    Supports two formats:
    1. OpenAI-native format (from tool_calls in response)
    2. Text-based format using <tool_call> XML markers

    Args:
        response: Raw LLM response (string for text mode, dict for native)
        native_mode: If True, expect OpenAI JSON format with tool_calls.
                    If False, parse <tool_call> markers from text.

    Returns:
        List of parsed tool calls (empty if none found)

    Examples:
        # Native mode (from OpenAI-compatible API)
        >>> parse_tool_calls({"tool_calls": [...]}, native_mode=True)

        # Text mode (from raw LLM output)
        >>> parse_tool_calls('<tool_call>{"name": "read_file", ...}</tool_call>')
    """
    if native_mode:
        return _parse_native_format(response)
    else:
        return _parse_text_format(response if isinstance(response, str) else "")


def _parse_native_format(response: dict[str, Any]) -> list[ParsedToolCall]:
    """Parse OpenAI-native tool calling format.

    Expected format:
    {
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "arguments": '{"arg": "value"}'
                }
            }
        ]
    }
    """
    tool_calls = []

    # Handle both direct tool_calls and nested in message
    raw_calls = response.get("tool_calls", [])
    if not raw_calls and "message" in response:
        raw_calls = response["message"].get("tool_calls", [])
    if not raw_calls and "choices" in response:
        # OpenAI format: choices[0].message.tool_calls
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            raw_calls = message.get("tool_calls", [])

    for call in raw_calls:
        try:
            func = call.get("function", {})
            call_id = call.get("id", f"call_{uuid.uuid4().hex[:8]}")
            name = func.get("name", "")

            # Arguments may be string or dict
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)

            if name:
                tool_calls.append(ParsedToolCall(
                    id=call_id,
                    name=name,
                    arguments=args,
                ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("tool_call_parse_error", call=call, error=str(e))

    return tool_calls


def _parse_text_format(text: str) -> list[ParsedToolCall]:
    """Parse text-based tool call format using XML markers.

    Expected format:
    <tool_call>
    {"name": "tool_name", "arguments": {"arg": "value"}}
    </tool_call>
    """
    tool_calls = []

    matches = TOOL_CALL_PATTERN.findall(text)
    for match in matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})

            # Handle case where arguments is a string
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if name:
                tool_calls.append(ParsedToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=name,
                    arguments=arguments,
                ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("tool_call_parse_error", match=match[:100], error=str(e))

    return tool_calls


def has_tool_calls(response: str | dict[str, Any]) -> bool:
    """Quick check if response contains tool calls.

    More efficient than parsing when you just need to know if
    tool calls are present.

    Args:
        response: LLM response (string or dict)

    Returns:
        True if tool calls detected
    """
    if isinstance(response, dict):
        # Check native format
        if response.get("tool_calls"):
            return True
        if response.get("message", {}).get("tool_calls"):
            return True
        choices = response.get("choices", [])
        if choices and choices[0].get("message", {}).get("tool_calls"):
            return True
        # Check finish_reason
        if choices and choices[0].get("finish_reason") == "tool_calls":
            return True

    if isinstance(response, str):
        # Quick text check
        return "<tool_call>" in response

    return False


def format_tool_result(tool_call_id: str, tool_name: str, result: str) -> dict[str, Any]:
    """Format tool execution result for conversation history.

    Creates a message in OpenAI-compatible format that can be
    appended to conversation history.

    Args:
        tool_call_id: ID of the tool call this result is for
        tool_name: Name of the tool that was called
        result: Result string from tool execution

    Returns:
        Message dict in OpenAI tool result format
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": result,
    }
