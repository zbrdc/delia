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
import functools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import Workspace

class TrustLevel(IntEnum):
    """Security trust levels for tool execution."""
    READ_ONLY = 0      # Safe to execute (e.g., read_file)
    MUTATION = 1       # Modifies state (e.g., write_file)
    EXECUTION = 2      # Arbitrary code execution (e.g., shell_exec)
    DANGEROUS = 3      # High-risk operations

# Tool categories for discoverability
TOOL_CATEGORIES = {
    "file_ops": "File operations (read, write, edit, search)",
    "lsp": "Language Server Protocol code intelligence",
    "git": "Git version control operations",
    "testing": "Test execution and analysis",
    "ace": "ACE Framework (playbooks, context, feedback)",
    "orchestration": "LLM delegation and workflows",
    "admin": "System administration and configuration",
    "search": "Code search and discovery",
    "general": "General purpose tools",
}


@dataclass
class ToolDefinition:
    """Definition of a tool available to Delia."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    dangerous: bool = False  # If True, requires explicit user confirmation
    permission_level: str = "read"  # 'read', 'write', or 'exec'
    requires_session: bool = False
    requires_workspace: bool = False
    # Discoverability fields
    category: str = "general"  # Key from TOOL_CATEGORIES
    return_type: str = "string"  # 'string', 'json', 'structured'
    examples: list[dict[str, Any]] = field(default_factory=list)  # [{"input": {...}, "output": "..."}]

    def to_openai_schema(self) -> Dict[str, Any]:
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

    def __init__(self, workspace: Workspace | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self.workspace = workspace

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool.

        Args:
            tool: Tool definition to register

        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        
        # If registry has a workspace and tool requires it, wrap handler
        if self.workspace and tool.requires_workspace:
            original_handler = tool.handler
            tool.handler = functools.partial(original_handler, workspace=self.workspace)
            
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
            "You are an AI assistant with access to tools. You MUST use tools to complete tasks.",
            "",
            "CRITICAL RULES - VIOLATIONS WILL CAUSE FAILURES:",
            "",
            "1. USE TOOLS - NOT CODE EXAMPLES",
            "   [X] WRONG: 'Here's how to read a file: open(path).read()'",
            "   [OK] RIGHT: Use the read_file tool to actually read the file",
            "",
            "2. USE REAL PATHS - NOT PLACEHOLDERS",
            "   [X] WRONG: read_file(path='/path/to/file.py')",
            "   [X] WRONG: read_file(path='<filepath>')",
            "   [OK] RIGHT: read_file(path='src/config.py')  # Use actual paths from context",
            "",
            "3. DO NOT HALLUCINATE FILE CONTENTS",
            "   [X] WRONG: 'The file contains: class Config...' (without reading it)",
            "   [OK] RIGHT: Call read_file first, then describe what you actually saw",
            "",
            "4. ONE TOOL CALL AT A TIME",
            "   Call a tool, wait for the result, then decide next action.",
            "   Do not assume tool results before receiving them.",
            "",
            "5. IF INFORMATION IS MISSING, USE A TOOL TO GET IT",
            "   Don't guess or make up file contents, directory structures, or search results.",
            "",
            "Available tools:",
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
            "REMEMBER: Use ACTUAL paths/values from context. Never use placeholder paths like '/path/to/...'.",
            "Wait for tool results before continuing. Do not fabricate results.",
        ])

        return "\n".join(lines)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_by_category(self, category: str) -> list[ToolDefinition]:
        """Get all tools in a specific category.

        Args:
            category: Category key (e.g., 'file_ops', 'lsp', 'ace')

        Returns:
            List of tools in that category
        """
        return [t for t in self._tools.values() if t.category == category]

    def get_categories(self) -> dict[str, list[str]]:
        """Get all categories with their tool names.

        Returns:
            Dict mapping category -> list of tool names
        """
        categories: dict[str, list[str]] = {}
        for tool in self._tools.values():
            if tool.category not in categories:
                categories[tool.category] = []
            categories[tool.category].append(tool.name)
        return categories

    def describe_tool(self, name: str) -> dict[str, Any] | None:
        """Get full description of a tool including examples.

        Args:
            name: Tool name

        Returns:
            Dict with name, description, parameters, category, return_type, examples
        """
        tool = self._tools.get(name)
        if not tool:
            return None
        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "return_type": tool.return_type,
            "parameters": tool.parameters,
            "examples": tool.examples,
            "dangerous": tool.dangerous,
            "permission_level": tool.permission_level,
        }

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
