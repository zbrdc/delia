# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Sandboxed code execution using llm-sandbox.

Provides secure, isolated execution environments for LLM-generated code.
Requires optional dependency: pip install 'delia[sandbox]'
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from llm_sandbox import SandboxSession

log = structlog.get_logger()

# Check if llm-sandbox is available
try:
    from llm_sandbox import SandboxSession as _SandboxSession

    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    _SandboxSession = None


class SandboxLanguage(str, Enum):
    """Supported sandbox languages (per llm-sandbox)."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    R = "r"
    # Note: bash is not directly supported by llm-sandbox
    # Shell commands are executed via Python subprocess wrapper


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    enabled: bool = True
    backend: str = "docker"  # docker, podman, k8s
    timeout_seconds: int = 30
    memory_limit_mb: int = 512
    cpu_limit: float = 1.0
    network_enabled: bool = False
    default_language: str = "python"
    # Custom images per language (optional)
    images: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "SandboxConfig":
        """Create config from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            backend=data.get("backend", "docker"),
            timeout_seconds=data.get("timeout_seconds", 30),
            memory_limit_mb=data.get("memory_limit_mb", 512),
            cpu_limit=data.get("cpu_limit", 1.0),
            network_enabled=data.get("network_enabled", False),
            default_language=data.get("default_language", "python"),
            images=data.get("images", {}),
        )


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution."""

    stdout: str
    stderr: str
    exit_code: int
    success: bool
    language: str
    execution_time_ms: int = 0
    error: str | None = None


class SandboxExecutor:
    """
    Executes code in isolated Docker/Podman containers.

    Uses llm-sandbox for secure, isolated code execution.
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if sandbox backend is available."""
        if not SANDBOX_AVAILABLE:
            log.warning(
                "sandbox_not_available",
                hint="Install with: pip install 'delia[sandbox]'",
            )

    @property
    def available(self) -> bool:
        """Check if sandbox execution is available."""
        return SANDBOX_AVAILABLE and self.config.enabled

    def _get_session_kwargs(self, language: str) -> dict:
        """Get session kwargs for language."""
        kwargs = {
            "lang": language,
            "verbose": False,
        }

        # Add custom image if configured
        if language in self.config.images:
            kwargs["image"] = self.config.images[language]

        return kwargs

    async def execute_code(
        self,
        code: str,
        language: str | None = None,
        libraries: list[str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Execute code in a sandboxed container.

        Args:
            code: The code to execute
            language: Programming language (python, javascript, java, cpp, go, r)
            libraries: List of libraries to install before execution
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with stdout, stderr, and exit_code
        """
        if not self.available:
            return ExecutionResult(
                stdout="",
                stderr="Sandbox not available. Install with: pip install 'delia[sandbox]'",
                exit_code=1,
                success=False,
                language=language or self.config.default_language,
                error="sandbox_not_available",
            )

        lang = language or self.config.default_language
        timeout_sec = timeout or self.config.timeout_seconds
        libs = libraries or []

        log.info(
            "sandbox_execute_start",
            language=lang,
            code_length=len(code),
            libraries=libs,
            timeout=timeout_sec,
        )

        # Run in thread pool since llm-sandbox is synchronous
        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self._execute_sync(code, lang, libs),
                ),
                timeout=timeout_sec + 10,  # Extra buffer for container startup
            )
            return result
        except asyncio.TimeoutError:
            log.warning("sandbox_timeout", language=lang, timeout=timeout_sec)
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {timeout_sec} seconds",
                exit_code=124,
                success=False,
                language=lang,
                error="timeout",
            )
        except Exception as e:
            log.error("sandbox_error", language=lang, error=str(e))
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
                success=False,
                language=lang,
                error=str(e),
            )

    def _execute_sync(
        self, code: str, language: str, libraries: list[str]
    ) -> ExecutionResult:
        """Synchronous execution in sandbox."""
        import time

        start = time.monotonic()

        try:
            session_kwargs = self._get_session_kwargs(language)

            with _SandboxSession(**session_kwargs) as session:
                result = session.run(code, libraries=libraries if libraries else None)

            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Handle result - llm-sandbox returns ConsoleOutput object
            stdout = result.stdout if hasattr(result, "stdout") else str(result)
            stderr = result.stderr if hasattr(result, "stderr") else ""
            exit_code = result.exit_code if hasattr(result, "exit_code") else 0

            success = exit_code == 0

            log.info(
                "sandbox_execute_complete",
                language=language,
                success=success,
                exit_code=exit_code,
                elapsed_ms=elapsed_ms,
            )

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                success=success,
                language=language,
                execution_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            log.error("sandbox_execute_failed", language=language, error=str(e))
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
                success=False,
                language=language,
                execution_time_ms=elapsed_ms,
                error=str(e),
            )

    async def execute_shell(
        self,
        command: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        Execute a shell command in a sandboxed container.

        Uses Python subprocess to execute shell commands since llm-sandbox
        doesn't support bash directly.

        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with stdout, stderr, and exit_code
        """
        # Wrap command in Python subprocess call
        # Escape the command string for Python
        escaped_command = command.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
        python_code = f'''
import subprocess
import sys

try:
    result = subprocess.run(
        "{escaped_command}",
        shell=True,
        capture_output=True,
        text=True,
        timeout={timeout or self.config.timeout_seconds}
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    print("Command timed out", file=sys.stderr)
    sys.exit(124)
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
        return await self.execute_code(
            code=python_code,
            language="python",
            timeout=timeout,
        )


# Singleton instance
_executor: SandboxExecutor | None = None


def get_sandbox_executor(config: SandboxConfig | None = None) -> SandboxExecutor:
    """Get or create the sandbox executor singleton."""
    global _executor
    if _executor is None:
        # Load config from settings if not provided
        if config is None:
            config = _load_sandbox_config()
        _executor = SandboxExecutor(config)
    return _executor


def _load_sandbox_config() -> SandboxConfig:
    """Load sandbox config from settings.json via BackendManager."""
    try:
        from .backend_manager import get_backend_manager

        manager = get_backend_manager()
        if manager.sandbox_config:
            return SandboxConfig.from_dict(manager.sandbox_config)
    except Exception as e:
        log.debug("sandbox_config_load_fallback", error=str(e))

    return SandboxConfig()


def is_sandbox_available() -> bool:
    """Check if sandbox execution is available."""
    return SANDBOX_AVAILABLE


async def execute_code_sandboxed(
    code: str,
    language: str = "python",
    libraries: list[str] | None = None,
    timeout: int = 30,
) -> ExecutionResult:
    """
    Convenience function to execute code in sandbox.

    Args:
        code: Code to execute
        language: Programming language
        libraries: Libraries to install
        timeout: Execution timeout

    Returns:
        ExecutionResult
    """
    executor = get_sandbox_executor()
    return await executor.execute_code(code, language, libraries, timeout)


async def execute_shell_sandboxed(
    command: str,
    timeout: int = 30,
) -> ExecutionResult:
    """
    Convenience function to execute shell command in sandbox.

    Args:
        command: Shell command to execute
        timeout: Execution timeout

    Returns:
        ExecutionResult
    """
    executor = get_sandbox_executor()
    return await executor.execute_shell(command, timeout)
