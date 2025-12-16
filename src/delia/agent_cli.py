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
Standalone Agent CLI for Delia.

Provides `delia agent` command for running autonomous agents from the command line.
Supports single-shot tasks and interactive chat mode with optional k-voting
for mathematically-guaranteed reliability.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from .tools.agent import AgentConfig, AgentResult, run_agent_loop
from .tools.builtins import get_default_tools
from .tools.registry import ToolRegistry
from .types import Workspace

log = structlog.get_logger()

# Try to import Rich for pretty output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


@dataclass
class AgentCLIConfig:
    """Configuration for agent CLI execution."""

    model: str | None = None
    workspace: str | None = None
    max_iterations: int = 10
    tools: list[str] | None = None
    backend_type: str | None = None
    verbose: bool = False
    voting_enabled: bool = False
    voting_k: int | None = None
    allow_write: bool = False
    allow_exec: bool = False


@dataclass
class AgentCLIResult:
    """Result from agent CLI execution."""

    success: bool
    response: str
    iterations: int
    tool_calls: list[str]
    elapsed_ms: int
    model: str
    backend: str
    voting_used: bool = False
    voting_k: int | None = None


def print_header(task: str) -> None:
    """Print agent task header."""
    if RICH_AVAILABLE and console:
        console.print()
        console.print(Panel(task, title="[bold blue]Task[/bold blue]", border_style="blue"))
    else:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")


def print_thinking() -> None:
    """Print thinking indicator."""
    if RICH_AVAILABLE and console:
        console.print("[dim]Thinking...[/dim]")


def print_tool_call(name: str, args: dict[str, Any]) -> None:
    """Print tool call information."""
    if RICH_AVAILABLE and console:
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
        console.print(Panel(
            f"[cyan]{args_str}[/cyan]" if args_str else "[dim]no args[/dim]",
            title=f"[bold yellow]:wrench: {name}[/bold yellow]",
            border_style="yellow",
            expand=False,
        ))
    else:
        print(f"\n[Tool: {name}]")
        for k, v in args.items():
            print(f"  {k}: {repr(v)[:80]}")


def print_tool_result(name: str, result: str, elapsed_ms: int, success: bool = True) -> None:
    """Print tool result (truncated)."""
    if RICH_AVAILABLE and console:
        status = "[green]:heavy_check_mark:[/green]" if success else "[red]:x:[/red]"
        # Truncate long results
        preview = result[:500] + "..." if len(result) > 500 else result
        console.print(Panel(
            f"[dim]{preview}[/dim]",
            title=f"{status} [bold]{name}[/bold] ({elapsed_ms}ms)",
            border_style="green" if success else "red",
            expand=False,
        ))
    else:
        status = "OK" if success else "FAIL"
        preview = result[:200] + "..." if len(result) > 200 else result
        print(f"[{status}] {name} ({elapsed_ms}ms): {preview}")


def print_response(response: str) -> None:
    """Print agent response with markdown formatting."""
    if RICH_AVAILABLE and console:
        console.print()
        console.print(Markdown(response))
    else:
        print(f"\n{response}")


def print_footer(result: AgentCLIResult) -> None:
    """Print execution summary footer."""
    if RICH_AVAILABLE and console:
        status = "[green]:heavy_check_mark:[/green]" if result.success else "[yellow]:warning:[/yellow]"

        parts = [
            f"Model: [cyan]{result.model}[/cyan]",
            f"Backend: [cyan]{result.backend}[/cyan]",
            f"Iterations: {result.iterations}",
            f"Time: {result.elapsed_ms}ms",
        ]

        if result.tool_calls:
            tools_summary = ", ".join(result.tool_calls)
            parts.append(f"Tools: {tools_summary}")

        if result.voting_used:
            parts.append(f"Voting: k={result.voting_k} :heavy_check_mark:")

        console.print()
        console.print(Panel(
            " | ".join(parts),
            title=f"{status} Agent Completed",
            border_style="dim",
        ))
    else:
        status = "OK" if result.success else "WARNING"
        print(f"\n{'='*60}")
        print(f"[{status}] Agent Completed")
        print(f"Model: {result.model} | Backend: {result.backend}")
        print(f"Iterations: {result.iterations} | Time: {result.elapsed_ms}ms")
        if result.tool_calls:
            print(f"Tools: {', '.join(result.tool_calls)}")
        if result.voting_used:
            print(f"Voting: k={result.voting_k}")
        print(f"{'='*60}")


async def run_agent_cli(
    task: str,
    config: AgentCLIConfig,
) -> AgentCLIResult:
    """
    Run an autonomous agent from the CLI.

    Args:
        task: The task/prompt for the agent
        config: Agent CLI configuration

    Returns:
        AgentCLIResult with response and metadata
    """
    # Import here to avoid circular imports
    from .mcp_server import call_llm, select_model, _select_optimal_backend_v2

    start_time = time.time()

    # Print header
    print_header(task)

    # Select backend
    if config.verbose:
        print_thinking()

    try:
        backend_provider, backend_obj = await _select_optimal_backend_v2(
            task, None, "analyze", config.backend_type
        )
    except Exception as e:
        return AgentCLIResult(
            success=False,
            response=f"Failed to select backend: {e}",
            iterations=0,
            tool_calls=[],
            elapsed_ms=int((time.time() - start_time) * 1000),
            model="unknown",
            backend="unknown",
        )

    backend_name = backend_obj.name if backend_obj else "unknown"

    # Select model
    try:
        selected_model = await select_model(
            task_type="analyze",
            content_size=len(task),
            model_override=config.model,
            content=task,
        )
    except Exception as e:
        return AgentCLIResult(
            success=False,
            response=f"Failed to select model: {e}",
            iterations=0,
            tool_calls=[],
            elapsed_ms=int((time.time() - start_time) * 1000),
            model="unknown",
            backend=backend_name,
        )

    # Set up workspace
    workspace_obj = Workspace(root=config.workspace) if config.workspace else None

    # Set up tool registry
    registry = get_default_tools(workspace=workspace_obj)

    # Filter tools if specified
    if config.tools:
        registry = registry.filter(config.tools)

    # Detect native tool calling support
    use_native = backend_obj.supports_native_tool_calling if backend_obj else False

    # Track tool calls for result
    tool_calls_made: list[str] = []

    # Create LLM callable wrapper
    async def agent_llm_call(
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> str:
        # Convert messages to a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "tool")
                prompt_parts.append(f"[Tool Result - {tool_name}]\n{content}")

        combined_prompt = "\n\n".join(prompt_parts)

        result = await call_llm(
            model=selected_model,
            prompt=combined_prompt,
            system=system,
            task_type="agent",
            original_task="agent",
            language="unknown",
            backend_obj=backend_obj,
        )

        if result.get("success"):
            return result.get("response", "")
        else:
            raise RuntimeError(result.get("error", "LLM call failed"))

    # Create agent config
    agent_config = AgentConfig(
        max_iterations=config.max_iterations,
        timeout_per_tool=30.0,
        total_timeout=300.0,
        parallel_tools=True,
        native_tool_calling=use_native,
    )

    # Run the agent loop
    try:
        if RICH_AVAILABLE and console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Running agent...", total=None)
                result = await run_agent_loop(
                    call_llm=agent_llm_call,
                    prompt=task,
                    system_prompt=None,
                    registry=registry,
                    model=selected_model,
                    config=agent_config,
                )
        else:
            print("Running agent...")
            result = await run_agent_loop(
                call_llm=agent_llm_call,
                prompt=task,
                system_prompt=None,
                registry=registry,
                model=selected_model,
                config=agent_config,
            )
    except Exception as e:
        log.error("agent_cli_error", error=str(e))
        return AgentCLIResult(
            success=False,
            response=f"Agent error: {e}",
            iterations=0,
            tool_calls=[],
            elapsed_ms=int((time.time() - start_time) * 1000),
            model=selected_model,
            backend=backend_name,
        )

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Extract tool call names
    if result.tool_calls:
        tool_calls_made = [tc.name for tc in result.tool_calls]

    # Print response
    print_response(result.response)

    # Build result
    cli_result = AgentCLIResult(
        success=result.success,
        response=result.response,
        iterations=result.iterations,
        tool_calls=tool_calls_made,
        elapsed_ms=elapsed_ms,
        model=selected_model,
        backend=backend_name,
        voting_used=config.voting_enabled,
        voting_k=config.voting_k,
    )

    # Print footer
    print_footer(cli_result)

    return cli_result


def run_agent_sync(task: str, config: AgentCLIConfig) -> AgentCLIResult:
    """Synchronous wrapper for run_agent_cli."""
    return asyncio.run(run_agent_cli(task, config))
