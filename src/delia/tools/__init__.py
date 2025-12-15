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
Tool use module for Delia.

Provides agentic tool calling capabilities for local LLMs, enabling them
to read files, search code, and execute commands to complete tasks.

Usage:
    from delia.tools import ToolRegistry, run_agent_loop, get_default_tools

    registry = get_default_tools()
    result = await run_agent_loop(call_llm, prompt, system, registry, model)
"""

from .registry import ToolDefinition, ToolRegistry
from .parser import ParsedToolCall, parse_tool_calls
from .executor import execute_tool, ToolResult
from .builtins import get_default_tools
from .agent import run_agent_loop, AgentConfig, AgentResult
from .mcp_client import (
    MCPClient,
    MCPClientManager,
    MCPServerConfig,
    MCPTool,
    MCPError,
)

__all__ = [
    # Registry
    "ToolDefinition",
    "ToolRegistry",
    # Parser
    "ParsedToolCall",
    "parse_tool_calls",
    # Executor
    "execute_tool",
    "ToolResult",
    # Builtins
    "get_default_tools",
    # Agent
    "run_agent_loop",
    "AgentConfig",
    "AgentResult",
    # MCP Client
    "MCPClient",
    "MCPClientManager",
    "MCPServerConfig",
    "MCPTool",
    "MCPError",
]
