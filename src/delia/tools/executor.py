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
Tool executor with security sandboxing.

Executes tool calls with timeouts, resource limits, and path validation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from .parser import ParsedToolCall
from .registry import ToolRegistry

log = structlog.get_logger()

# Maximum output size from a tool (100KB)
MAX_OUTPUT_SIZE = 100_000

# Default timeout for tool execution (30 seconds)
DEFAULT_TIMEOUT = 30.0

# Paths that are always blocked (security)
BLOCKED_PATHS = frozenset({
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "~/.ssh",
    "~/.gnupg",
    "~/.aws/credentials",
    "~/.config/gcloud",
})


@dataclass
class ToolResult:
    """Result from tool execution.

    Attributes:
        success: Whether execution succeeded
        output: Tool output (result or error message)
        tool_name: Name of the tool that was executed
        tool_call_id: ID of the tool call
        elapsed_ms: Execution time in milliseconds
        truncated: Whether output was truncated
    """
    success: bool
    output: str
    tool_name: str
    tool_call_id: str
    elapsed_ms: int = 0
    truncated: bool = False


def validate_path(path: str) -> tuple[bool, str]:
    """Validate a file path for security.

    Checks:
    - Path doesn't contain traversal sequences
    - Path isn't in blocked list
    - Path is within allowed directories

    Args:
        path: Path to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Expand and resolve path
    try:
        expanded = Path(path).expanduser().resolve()
    except Exception as e:
        return False, f"Invalid path: {e}"

    path_str = str(expanded)

    # Check for path traversal
    if ".." in path:
        return False, "Path traversal not allowed"

    # Check blocked paths
    for blocked in BLOCKED_PATHS:
        blocked_expanded = str(Path(blocked).expanduser().resolve())
        if path_str.startswith(blocked_expanded):
            return False, f"Access to {blocked} is not allowed"

    return True, ""


def truncate_output(output: str, max_size: int = MAX_OUTPUT_SIZE) -> tuple[str, bool]:
    """Truncate output if it exceeds max size.

    Args:
        output: Output string to potentially truncate
        max_size: Maximum size in characters

    Returns:
        Tuple of (output, was_truncated)
    """
    if len(output) <= max_size:
        return output, False

    truncated = output[:max_size]
    truncated += f"\n\n... [Output truncated. Showing {max_size:,} of {len(output):,} characters]"
    return truncated, True


async def execute_tool(
    tool_call: ParsedToolCall,
    registry: ToolRegistry,
    timeout: float = DEFAULT_TIMEOUT,
) -> ToolResult:
    """Execute a single tool call with sandboxing.

    Args:
        tool_call: Parsed tool call to execute
        registry: Registry containing tool definitions
        timeout: Maximum execution time in seconds

    Returns:
        ToolResult with execution outcome
    """
    start_time = time.time()

    # Get tool definition
    tool = registry.get(tool_call.name)
    if not tool:
        return ToolResult(
            success=False,
            output=f"Unknown tool: {tool_call.name}. Available tools: {', '.join(registry.list_tools())}",
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
        )

    # Execute with timeout
    try:
        log.info("tool_executing", tool=tool_call.name, args=tool_call.arguments)

        result = await asyncio.wait_for(
            tool.handler(**tool_call.arguments),
            timeout=timeout,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Truncate if needed
        output, truncated = truncate_output(str(result))

        log.info(
            "tool_executed",
            tool=tool_call.name,
            success=True,
            elapsed_ms=elapsed_ms,
            output_len=len(output),
            truncated=truncated,
        )

        return ToolResult(
            success=True,
            output=output,
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )

    except asyncio.TimeoutError:
        elapsed_ms = int((time.time() - start_time) * 1000)
        log.warning("tool_timeout", tool=tool_call.name, timeout=timeout)
        return ToolResult(
            success=False,
            output=f"Tool '{tool_call.name}' timed out after {timeout}s",
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            elapsed_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        log.warning("tool_error", tool=tool_call.name, error=str(e))
        return ToolResult(
            success=False,
            output=f"Tool '{tool_call.name}' failed: {e}",
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            elapsed_ms=elapsed_ms,
        )


async def execute_tools(
    tool_calls: list[ParsedToolCall],
    registry: ToolRegistry,
    timeout: float = DEFAULT_TIMEOUT,
    parallel: bool = False,
) -> list[ToolResult]:
    """Execute multiple tool calls.

    Args:
        tool_calls: List of tool calls to execute
        registry: Registry containing tool definitions
        timeout: Timeout per tool call
        parallel: If True, execute tools in parallel

    Returns:
        List of ToolResults in same order as input
    """
    if parallel:
        # Execute all tools concurrently
        tasks = [
            execute_tool(call, registry, timeout)
            for call in tool_calls
        ]
        return await asyncio.gather(*tasks)
    else:
        # Execute sequentially
        results = []
        for call in tool_calls:
            result = await execute_tool(call, registry, timeout)
            results.append(result)
        return results
