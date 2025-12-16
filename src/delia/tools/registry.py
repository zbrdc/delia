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
Tool registry for managing available tools.

Provides a central registry for tool definitions that can be used
by the agentic loop to execute tool calls from LLMs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class ToolDefinition:
    """Definition of a tool available to the agent.

    Attributes:
        name: Unique tool identifier (e.g., "read_file")
        description: Human-readable description for LLM
        parameters: JSON Schema for parameters
        handler: Async function to execute the tool
        permission_level: Required permission ("read", "write", "exec")
        dangerous: If True, requires user confirmation before execution
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[str]]
    permission_level: str = "read"  # "read" | "write" | "exec"
    dangerous: bool = False  # Requires confirmation prompt

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        This format is supported by Ollama, llama.cpp, and most
        OpenAI-compatible backends.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class ToolRegistry:
    """Registry of available tools for agent execution.

    Provides methods to register, retrieve, and list tools.
    Tools are stored by name for quick lookup during execution.

    Example:
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="read_file",
            description="Read file contents",
            parameters={"type": "object", "properties": {...}},
            handler=read_file_handler
        ))

        # Get schemas for LLM prompt
        schemas = registry.get_openai_schemas()

        # Execute a tool
        tool = registry.get("read_file")
        result = await tool.handler(path="/some/file.py")
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool.

        Args:
            tool: Tool definition to register

        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        """Get tool by name.

        Args:
            name: Tool name to look up

        Returns:
            Tool definition or None if not found
        """
        return self._tools.get(name)

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas in OpenAI format for LLM prompt.

        Returns:
            List of tool schemas in OpenAI function calling format
        """
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_tool_prompt(self) -> str:
        """Get tool descriptions for text-based prompting.

        Use this for models that don't support native tool calling.
        The prompt includes instructions for the XML-based fallback format.

        Returns:
            Text prompt describing available tools
        """
        lines = [
            "You have access to the following tools:",
            "",
        ]

        for tool in self._tools.values():
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}")
            lines.append(f"Parameters: {json.dumps(tool.parameters, indent=2)}")
            lines.append("")

        lines.extend([
            "To use a tool, respond with ONLY a tool call in this format:",
            "<tool_call>",
            '{"name": "tool_name", "arguments": {"arg1": "value1"}}',
            "</tool_call>",
            "",
            "Wait for the tool result before continuing.",
        ])

        return "\n".join(lines)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def filter(self, tool_names: list[str]) -> "ToolRegistry":
        """Create a new registry with only the specified tools.

        Args:
            tool_names: Names of tools to include

        Returns:
            New registry containing only specified tools
        """
        filtered = ToolRegistry()
        for name in tool_names:
            if tool := self._tools.get(name):
                filtered._tools[name] = tool
        return filtered

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
