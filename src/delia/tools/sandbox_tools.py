# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Sandboxed execution tools for Delia agents.

Provides tools for executing code and shell commands in isolated Docker containers.
Requires optional dependency: pip install 'delia[sandbox]'
"""

from __future__ import annotations

import structlog

from ..sandbox import (
    ExecutionResult,
    execute_code_sandboxed,
    execute_shell_sandboxed,
    is_sandbox_available,
)

log = structlog.get_logger()


def _format_sandbox_result(result: ExecutionResult) -> str:
    """Format sandbox execution result for display."""
    status = "Success" if result.success else "Failed"

    lines = [
        f"# Sandbox Execution ({result.language})",
        f"**Status:** {status}",
        f"**Exit Code:** {result.exit_code}",
        f"**Duration:** {result.execution_time_ms}ms",
        "",
    ]

    if result.stdout:
        lines.extend([
            "## Output",
            "```",
            result.stdout[:5000],  # Truncate long output
            "```",
        ])

    if result.stderr:
        lines.extend([
            "",
            "## Errors",
            "```",
            result.stderr[:2000],  # Truncate errors
            "```",
        ])

    if result.error:
        lines.extend([
            "",
            f"**Error:** {result.error}",
        ])

    return "\n".join(lines)


async def shell_exec_sandboxed(
    command: str,
    timeout: int = 30,
) -> str:
    """
    Execute a shell command in an isolated Docker container.

    This is safe for untrusted commands as they run in complete isolation
    with no access to the host system.

    Args:
        command: Shell command to execute
        timeout: Execution timeout in seconds

    Returns:
        Formatted execution result with stdout, stderr, and exit code
    """
    if not is_sandbox_available():
        return "Error: Sandbox not available. Install with: pip install 'delia[sandbox]'"

    log.info("sandbox_shell_start", command=command[:100], timeout=timeout)

    result = await execute_shell_sandboxed(command, timeout)

    return _format_sandbox_result(result)


async def code_execute(
    code: str,
    language: str = "python",
    libraries: list[str] | None = None,
    timeout: int = 30,
) -> str:
    """
    Execute code in an isolated Docker container.

    Supports multiple programming languages with automatic dependency management.

    Args:
        code: Code to execute
        language: Programming language (python, javascript, java, cpp, go, r)
        libraries: Libraries to install before execution
        timeout: Execution timeout in seconds

    Returns:
        Formatted execution result with stdout, stderr, and exit code
    """
    if not is_sandbox_available():
        return "Error: Sandbox not available. Install with: pip install 'delia[sandbox]'"

    log.info(
        "sandbox_code_start",
        language=language,
        code_length=len(code),
        libraries=libraries,
        timeout=timeout,
    )

    result = await execute_code_sandboxed(
        code=code,
        language=language,
        libraries=libraries,
        timeout=timeout,
    )

    return _format_sandbox_result(result)
