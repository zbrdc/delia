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

from ..types import Workspace
from ..security import get_security_manager
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
        is_final: Whether this result should terminate the agent loop
                  (skip reloading the orchestrating model)
    """
    success: bool
    output: str
    tool_name: str
    tool_call_id: str
    elapsed_ms: int = 0
    truncated: bool = False
    is_final: bool = False


def validate_path(
    path: str,
    workspace: Workspace | None = None,
) -> tuple[bool, str]:
    """Validate a file path for security.

    Checks:
    - Path doesn't contain traversal sequences (unless workspace allows it)
    - Path isn't in blocked list
    - If workspace provided, path must be within workspace boundaries

    Args:
        path: Path to validate
        workspace: Optional workspace to confine path within

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Expand and resolve path
    try:
        if workspace and not Path(path).is_absolute():
            # Relative paths are relative to workspace root
            expanded = (workspace.root / path).resolve()
        else:
            expanded = Path(path).expanduser().resolve()
    except Exception as e:
        return False, f"Invalid path: {e}"

    path_str = str(expanded).lower()

    # Check for path traversal (unless workspace allows it)
    if ".." in path:
        if workspace and not workspace.allow_parent_traversal:
            return False, "Path traversal (..) not allowed in workspace"
        elif not workspace:
            return False, "Path traversal not allowed"

    # CRITICAL SECURITY: Block sensitive directory patterns
    sensitive_patterns = [
        "/.ssh", "/.aws", "/.config/gcloud", "/etc/passwd", "/etc/shadow",
        "/.bash_history", "/.zsh_history", "/.env", "/.git",
        "/bin/", "/usr/bin", "/dev/", "/proc/", "/sys/", "/boot/", "/lib",
        "/var/log", 
        "c:\\windows", "c:\\users", # Windows sensitive
    ]
    for pattern in sensitive_patterns:
        if pattern in path_str:
            return False, f"Access to sensitive pattern '{pattern}' is not allowed"

    # Block dangerous extensions (binaries and databases, NOT source code)
    dangerous_extensions = [
        ".sql", ".db", ".sqlite", ".bin", ".exe", ".dll", ".so",
        ".pem", ".key", ".p12", ".pfx",  # certificates/keys
    ]
    if any(path_str.endswith(ext) for ext in dangerous_extensions):
        return False, "Access to restricted file type not allowed"

    # Check for command injection in path
    if any(char in path for char in ["$", "`", "|", "&", ";", ">", "<"]):
        return False, "Potential command injection in path"
        
    # Check for administrative keywords in path (shell injection)
    admin_keywords = ["sudo ", "chmod ", "chown ", "rm -rf", "mkfs", "doas "]
    if any(kw in path_str for kw in admin_keywords):
        return False, "Potential command injection in path"

    # Check blocked paths (always blocked, even within workspace)
    for blocked in BLOCKED_PATHS:
        try:
            blocked_expanded = str(Path(blocked).expanduser().resolve()).lower()
            if path_str.startswith(blocked_expanded):
                return False, f"Access to {blocked} is not allowed"
        except Exception:
            continue

    # If workspace provided, enforce boundary
    if workspace:
        if not workspace.contains(expanded):
            return False, f"Path '{path}' is outside workspace '{workspace.root}'"

    return True, ""


def validate_path_in_workspace(
    path: str,
    workspace: Workspace,
) -> tuple[bool, str, Path | None]:
    """Validate and resolve a path within a workspace.

    This is a convenience function that validates the path and returns
    the resolved absolute path if valid.

    Args:
        path: Path to validate (absolute or relative to workspace)
        workspace: Workspace to confine path within

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
        resolved_path is None if invalid
    """
    valid, error = validate_path(path, workspace)
    if not valid:
        return False, error, None

    try:
        resolved = workspace.resolve_path(path)
        return True, "", resolved
    except ValueError as e:
        return False, str(e), None


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
    session_id: str | None = None,
) -> ToolResult:
    """Execute a single tool call with sandboxing.

    Args:
        tool_call: Parsed tool call to execute
        registry: Registry containing tool definitions
        timeout: Maximum execution time in seconds
        session_id: Optional session ID for audit/undo tracking

    Returns:
        ToolResult with execution outcome
    """
    start_time = time.time()
    security = get_security_manager()

    # Get tool definition
    tool = registry.get(tool_call.name)
    if not tool:
        return ToolResult(
            success=False,
            output=f"Unknown tool: {tool_call.name}. Available tools: {', '.join(registry.list_tools())}",
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
        )

    # Check if approval is needed
    needs_approval = security.needs_approval(
        tool_call.name, tool.permission_level, tool.dangerous
    )

    if needs_approval:
        approved, method = await security.request_approval(
            tool_call.name,
            tool_call.arguments,
            tool.permission_level,
        )
        if not approved:
            security.audit(
                operation="tool_call",
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                permission_level=tool.permission_level,
                approved=False,
                approval_method=method,
                result="denied",
                session_id=session_id,
            )
            return ToolResult(
                success=False,
                output=f"Operation denied: {tool_call.name} requires approval",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
            )
    else:
        method = "auto"

    # Backup file if this is a write operation
    if tool.permission_level == "write" and "path" in tool_call.arguments:
        security.backup_file(tool_call.arguments["path"], session_id)

    # Execute with timeout
    try:
        log.info("tool_executing", tool=tool_call.name, args=tool_call.arguments)

        result = await asyncio.wait_for(
            tool.handler(**tool_call.arguments),
            timeout=timeout,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Check for final flag (used by delegate with final=True)
        # This signals the agent loop to return this result directly
        # without reloading the orchestrating model (saves GPU swap)
        is_final = False
        if isinstance(result, dict) and result.get("__is_final__"):
            is_final = True
            result = result.get("__result__", result)
            log.info("tool_result_is_final", tool=tool_call.name)

        # Truncate if needed
        output, truncated = truncate_output(str(result))

        log.info(
            "tool_executed",
            tool=tool_call.name,
            success=True,
            elapsed_ms=elapsed_ms,
            output_len=len(output),
            truncated=truncated,
            is_final=is_final,
        )

        # Audit successful execution
        security.audit(
            operation="tool_call",
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            permission_level=tool.permission_level,
            approved=True,
            approval_method=method,
            result="success",
            session_id=session_id,
            duration_ms=elapsed_ms,
        )

        return ToolResult(
            success=True,
            output=output,
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            elapsed_ms=elapsed_ms,
            truncated=truncated,
            is_final=is_final,
        )

    except asyncio.TimeoutError:
        elapsed_ms = int((time.time() - start_time) * 1000)
        log.warning("tool_timeout", tool=tool_call.name, timeout=timeout)

        security.audit(
            operation="tool_call",
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            permission_level=tool.permission_level,
            approved=True,
            approval_method=method,
            result="error",
            error_message=f"Timeout after {timeout}s",
            session_id=session_id,
            duration_ms=elapsed_ms,
        )

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

        security.audit(
            operation="tool_call",
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            permission_level=tool.permission_level,
            approved=True,
            approval_method=method,
            result="error",
            error_message=str(e),
            session_id=session_id,
            duration_ms=elapsed_ms,
        )

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
    session_id: str | None = None,
) -> list[ToolResult]:
    """Execute multiple tool calls.

    Args:
        tool_calls: List of tool calls to execute
        registry: Registry containing tool definitions
        timeout: Timeout per tool call
        parallel: If True, execute tools in parallel
        session_id: Optional session ID for audit/undo tracking

    Returns:
        List of ToolResults in same order as input
    """
    if parallel:
        # Execute all tools concurrently
        tasks = [
            execute_tool(call, registry, timeout, session_id)
            for call in tool_calls
        ]
        return await asyncio.gather(*tasks)
    else:
        # Execute sequentially
        results = []
        for call in tool_calls:
            result = await execute_tool(call, registry, timeout, session_id)
            results.append(result)
        return results
