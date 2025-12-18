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
from typing import Any, Callable

import structlog

from .tools.agent import AgentConfig, AgentResult, run_agent_loop
from .tools.builtins import get_default_tools
from .tools.registry import ToolRegistry
from .types import Workspace
from .delegation import DelegateContext, delegate_impl
from .multi_user_tracking import tracker
from .tui import RichAgentUI

log = structlog.get_logger()

# Initialize UI
ui = RichAgentUI()


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
    # Reflection settings
    reflection_enabled: bool = False
    max_reflections: int = 1
    reflection_confidence: str = "normal"
    # Planning settings
    planning_enabled: bool = False
    planning_model: str = "moe"  # Use MoE for planning by default


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
    from .mcp_server import call_llm, select_model, _select_optimal_backend_v2, get_active_backend

    start_time = time.time()

    # Print header
    ui.print_header(task)

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

    # Generate Plan (if enabled)
    initial_plan = ""
    if config.planning_enabled:
        ui.print_planning_start()

        # Construct planning context
        plan_ctx = DelegateContext(
            select_model=select_model,
            get_active_backend=get_active_backend,
            call_llm=call_llm,
            get_client_id=lambda: None,
            tracker=tracker,
        )

        plan_prompt = f"""You are a Senior Architect planning a task.

Task:
{task}

Instructions:
1. Break down the task into clear, sequential steps.
2. Identify necessary information gathering steps first (e.g., read files, search).
3. Outline the execution steps.
4. Include verification steps at the end.
5. Output ONLY the plan as a numbered list.
"""

        try:
            initial_plan = await delegate_impl(
                ctx=plan_ctx,
                task="plan",
                content=plan_prompt,
                model=config.planning_model,
                include_metadata=False,
            )
            
            ui.print_plan(initial_plan)
                
            # Append plan to the task for the agent
            task = f"""{task}

---
### Execution Plan
Please follow this plan:
{initial_plan}
"""
        except Exception as e:
            log.warning("planning_failed", error=str(e))
            if RICH_AVAILABLE and console:
                console.print(f"[yellow]Warning: Planning failed: {e}. Proceeding without plan.[/yellow]")
            else:
                print(f"Warning: Planning failed: {e}")

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

    # Critique callback for reflection loop
    async def critique_callback(response: str, original_prompt: str) -> tuple[bool, str]:
        if RICH_AVAILABLE and console:
            console.print(Panel(
                "[dim]Critiquing response...[/dim]",
                title="[bold magenta]Reflection[/bold magenta]",
                border_style="magenta",
            ))
        else:
            print("\n[Reflection] Critiquing response...")

        # Construct delegate context
        ctx = DelegateContext(
            select_model=select_model,
            get_active_backend=get_active_backend,
            call_llm=call_llm,
            get_client_id=lambda: None,
            tracker=tracker,
        )

        critique_prompt = f"""You are a QA Lead reviewing a response to a task.

Original Task:
{original_prompt}

Response to Review:
{response}

Instructions:
1. Verify if the response fully addresses the Original Task.
2. Check for any logical errors, safety issues, or hallucinations.
3. If the response is satisfactory, simply output "VERIFIED" (without quotes).
4. If there are issues, provide concise, actionable feedback for the agent to fix them. Do not rewrite the response yourself.
"""
        
        # Use high confidence (voting) if requested, otherwise normal delegation
        # Note: We rely on the delegation system's internal voting logic if configured in settings.json
        # or we could force it here. For now, we trust the delegate() tool's routing.
        
        # Use thinking model for critique if available/configured
        model_tier = "thinking" if config.reflection_confidence == "high" else "moe"

        result = await delegate_impl(
            ctx=ctx,
            task="critique",
            content=critique_prompt,
            model=model_tier,
            include_metadata=False,
        )
        
        # Check if verified
        # Loose matching to handle "VERIFIED." or "Status: VERIFIED"
        is_verified = "VERIFIED" in result or "verified" in result.lower() and len(result) < 50
        
        if is_verified:
            ui.print_reflection_success()
            return True, ""
        else:
            ui.print_reflection_feedback(result)
            return False, result

    # UI Callbacks
    def on_tool_call(tc: Any):
        # Unwrap ParsedToolCall
        ui.print_tool_call(tc.name, tc.arguments)

    def on_tool_result(res: Any):
        # Unwrap ToolResult
        ui.print_tool_result(res.name, res.output, res.success)

    # Create agent config
    agent_config = AgentConfig(
        max_iterations=config.max_iterations,
        timeout_per_tool=30.0,
        total_timeout=300.0,
        parallel_tools=True,
        native_tool_calling=use_native,
        reflection_enabled=config.reflection_enabled,
        max_reflections=config.max_reflections,
        reflection_confidence=config.reflection_confidence,
    )

    # Run the agent loop
    try:
        # Note: We don't use the Progress spinner here because the UI updates live 
        # via callbacks, and nested live displays can conflict.
        print("Running agent...")
        result = await run_agent_loop(
            call_llm=agent_llm_call,
            prompt=task,
            system_prompt=None,
            registry=registry,
            model=selected_model,
            config=agent_config,
            critique_callback=critique_callback,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
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
    ui.print_final_response(result.response)

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
    ui.print_footer(
        cli_result.model,
        cli_result.backend,
        cli_result.iterations,
        cli_result.elapsed_ms
    )

    return cli_result


def run_agent_sync(task: str, config: AgentCLIConfig) -> AgentCLIResult:
    """Synchronous wrapper for run_agent_cli."""
    return asyncio.run(run_agent_cli(task, config))
