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
        calls = _parse_native_format(response)
        # Fall back to text parsing if no native calls found but response has text
        if not calls and isinstance(response, dict):
            response_text = response.get("response", "")
            if response_text and isinstance(response_text, str):
                calls = _parse_text_format(response_text)
        return calls
    else:
        # For text mode, check both string response and dict's response field
        if isinstance(response, dict):
            response = response.get("response", "")
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

    # Handle both direct tool_calls and nested in message or metadata
    raw_calls = response.get("tool_calls", [])
    if not raw_calls and "message" in response:
        raw_calls = response["message"].get("tool_calls", [])
    if not raw_calls and "metadata" in response:
        # Delia LLMResponse format: metadata.tool_calls
        raw_calls = response["metadata"].get("tool_calls", [])
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

            # Validate name is a string (fuzzing found this bug)
            if not isinstance(name, str):
                log.warning("tool_call_invalid_name_type", name_type=type(name).__name__)
                continue

            # Arguments may be string or dict
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)

            # Validate arguments is a dict
            if not isinstance(args, dict):
                log.warning("tool_call_invalid_arguments_type", args_type=type(args).__name__)
                args = {}

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
    """Parse text-based tool call format.

    Supports two formats:
    1. XML markers: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. Raw JSON fallback: {"name": "...", "arguments": {...}}

    The XML format is preferred; raw JSON is used as fallback when XML markers
    are not present.
    """
    tool_calls = []

    # Try XML-wrapped format first (preferred)
    matches = TOOL_CALL_PATTERN.findall(text)
    for match in matches:
        parsed = _parse_tool_json(match)
        if parsed:
            tool_calls.append(parsed)

    # If no XML-wrapped calls found, try raw JSON fallback
    if not tool_calls:
        tool_calls = _parse_raw_json_tools(text)

    return tool_calls


def _parse_tool_json(json_str: str) -> ParsedToolCall | None:
    """Parse a single tool call JSON string.

    Args:
        json_str: JSON string with "name" and "arguments" keys

    Returns:
        ParsedToolCall if valid, None otherwise
    """
    try:
        data = json.loads(json_str)
        name = data.get("name", "")
        arguments = data.get("arguments", {})

        # Validate name is a string (fuzzing found this bug)
        if not isinstance(name, str):
            log.warning("tool_call_invalid_name_type", name_type=type(name).__name__)
            return None

        # Handle case where arguments is a string
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        # Validate arguments is a dict
        if not isinstance(arguments, dict):
            log.warning("tool_call_invalid_arguments_type", args_type=type(arguments).__name__)
            arguments = {}

        if name:
            return ParsedToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=arguments,
            )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.warning("tool_call_parse_error", json_str=json_str[:100], error=str(e))

    return None


def _parse_raw_json_tools(text: str) -> list[ParsedToolCall]:
    """Parse raw JSON tool calls without XML wrapper.

    This is a fallback for models that output tool calls as plain JSON
    without the <tool_call> wrapper tags.

    Uses a robust approach by searching for '{' and attempting to decode
    valid JSON objects that look like tool calls.
    """
    tool_calls = []
    decoder = json.JSONDecoder()
    pos = 0
    
    while True:
        pos = text.find('{', pos)
        if pos == -1:
            break
            
        try:
            # Attempt to decode a JSON object starting at this position
            obj, end = decoder.raw_decode(text[pos:])
            
            # Check if it looks like a tool call
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                name = obj.get("name")
                arguments = obj.get("arguments")
                
                if isinstance(name, str) and name:
                    # Arguments can be dict or JSON string
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            pass # Keep as string if not valid JSON

                    if isinstance(arguments, dict):
                        tool_calls.append(ParsedToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=name,
                            arguments=arguments,
                        ))
                        log.debug("raw_json_tool_parsed", name=name)
            
            # Move position forward to after this object
            pos += end
        except json.JSONDecodeError:
            # Not a valid JSON object starting here, move to next '{'
            pos += 1

    return tool_calls


def has_tool_calls(response: str | dict[str, Any]) -> bool:
    """Quick check if response contains tool calls.

    More efficient than parsing when you just need to know if
    tool calls are present. Checks for:
    1. Native OpenAI format (tool_calls in dict)
    2. XML-wrapped format (<tool_call> tags)
    3. Raw JSON format ({"name": "...", "arguments": {...}})

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
        # Check metadata (Delia LLM response format)
        if response.get("metadata", {}).get("tool_calls"):
            return True
        choices = response.get("choices", [])
        if choices and choices[0].get("message", {}).get("tool_calls"):
            return True
        # Check finish_reason
        if choices and choices[0].get("finish_reason") == "tool_calls":
            return True
        # Check response text field for text-based tool calls
        response_text = response.get("response", "")
        if response_text and isinstance(response_text, str):
            if "<tool_call>" in response_text:
                return True
            if '"name"' in response_text and '"arguments"' in response_text:
                if _parse_raw_json_tools(response_text):
                    return True

    if isinstance(response, str):
        # Check XML-wrapped format first (fast string check)
        if "<tool_call>" in response:
            return True
        # Check raw JSON format (quick heuristic then robust parser)
        if '"name"' in response and '"arguments"' in response:
            return bool(_parse_raw_json_tools(response))

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
