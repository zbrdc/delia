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
Agentic loop for tool-using LLM agents.

Implements an autonomous agent that can call tools to complete tasks.
The agent loops: LLM response -> parse tools -> execute -> feed back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import structlog

from .parser import parse_tool_calls, has_tool_calls, format_tool_result, ParsedToolCall
from .executor import execute_tools, ToolResult
from .registry import ToolRegistry

log = structlog.get_logger()

# Default configuration
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_TIMEOUT_PER_TOOL = 30.0
DEFAULT_TOTAL_TIMEOUT = 300.0  # 5 minutes total


@dataclass
class AgentConfig:
    """Configuration for the agentic loop.

    Attributes:
        max_iterations: Maximum tool call iterations before stopping
        timeout_per_tool: Timeout for each individual tool execution
        total_timeout: Total timeout for entire agent run
        parallel_tools: Whether to execute multiple tools in parallel
        native_tool_calling: If True, expect OpenAI-format tool calls
    """
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    timeout_per_tool: float = DEFAULT_TIMEOUT_PER_TOOL
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT
    parallel_tools: bool = True
    native_tool_calling: bool = False


@dataclass
class AgentResult:
    """Result from running the agentic loop.

    Attributes:
        success: Whether the agent completed successfully
        response: Final response from the agent
        iterations: Number of iterations completed
        tool_calls: List of all tool calls made
        tool_results: List of all tool results
        elapsed_ms: Total execution time in milliseconds
        stopped_reason: Why the agent stopped (completed, max_iterations, timeout, error)
    """
    success: bool
    response: str
    iterations: int
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    elapsed_ms: int = 0
    stopped_reason: str = "completed"


# Type for the LLM calling function
LLMCallable = Callable[
    [list[dict[str, Any]], str | None],  # (messages, system_prompt)
    Awaitable[str | dict[str, Any]]  # Returns response (str or dict with tool_calls)
]


def build_system_prompt(base_system: str | None, registry: ToolRegistry, native_mode: bool) -> str:
    """Build system prompt with tool information.

    Args:
        base_system: Base system prompt from caller
        registry: Tool registry with available tools
        native_mode: If True, tools passed via API; if False, include in prompt

    Returns:
        Complete system prompt
    """
    parts = []

    if base_system:
        parts.append(base_system)

    if not native_mode:
        # Include tool descriptions for text-based tool calling
        parts.append("")
        parts.append(registry.get_tool_prompt())

    return "\n".join(parts)


def build_messages(
    prompt: str,
    conversation: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build message list for LLM call.

    Args:
        prompt: User prompt/task
        conversation: Existing conversation history (optional)

    Returns:
        Messages list for LLM
    """
    if conversation:
        return list(conversation)

    return [{"role": "user", "content": prompt}]


async def run_agent_loop(
    call_llm: LLMCallable,
    prompt: str,
    system_prompt: str | None,
    registry: ToolRegistry,
    model: str,
    config: AgentConfig | None = None,
) -> AgentResult:
    """Run an agentic loop with tool calling.

    The agent repeatedly:
    1. Calls the LLM with the current conversation
    2. Checks if response contains tool calls
    3. If yes: executes tools, appends results, loops
    4. If no: returns the final response

    Args:
        call_llm: Async function to call the LLM. Signature:
                  (messages: list[dict], system: str | None) -> str | dict
        prompt: Initial user prompt/task
        system_prompt: System prompt (tool info added automatically)
        registry: Tool registry with available tools
        model: Model name (for logging)
        config: Agent configuration (uses defaults if None)

    Returns:
        AgentResult with final response and execution details

    Example:
        async def my_llm(messages, system):
            return await call_ollama(messages, system, model="qwen2.5:14b")

        result = await run_agent_loop(
            call_llm=my_llm,
            prompt="Read config.py and summarize its purpose",
            system_prompt="You are a code assistant.",
            registry=get_default_tools(),
            model="qwen2.5:14b",
        )
    """
    config = config or AgentConfig()
    start_time = time.time()

    # Build system prompt with tools
    full_system = build_system_prompt(system_prompt, registry, config.native_tool_calling)

    # Initialize conversation
    messages = build_messages(prompt)

    all_tool_calls: list[ParsedToolCall] = []
    all_tool_results: list[ToolResult] = []

    log.info(
        "agent_loop_starting",
        model=model,
        prompt_len=len(prompt),
        tools=registry.list_tools(),
        max_iterations=config.max_iterations,
    )

    for iteration in range(config.max_iterations):
        # Check total timeout
        elapsed = time.time() - start_time
        if elapsed > config.total_timeout:
            log.warning("agent_loop_timeout", iteration=iteration, elapsed=elapsed)
            return AgentResult(
                success=False,
                response="Agent timed out before completing the task.",
                iterations=iteration,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                elapsed_ms=int(elapsed * 1000),
                stopped_reason="timeout",
            )

        # Call LLM
        log.debug("agent_iteration", iteration=iteration, messages_count=len(messages))

        try:
            response = await call_llm(messages, full_system)
        except Exception as e:
            log.error("agent_llm_error", iteration=iteration, error=str(e))
            return AgentResult(
                success=False,
                response=f"LLM call failed: {e}",
                iterations=iteration,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                elapsed_ms=int((time.time() - start_time) * 1000),
                stopped_reason="error",
            )

        # Extract text content
        if isinstance(response, dict):
            # OpenAI format - extract message content
            content = ""
            if "message" in response:
                content = response["message"].get("content", "") or ""
            elif "choices" in response:
                choices = response.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "") or ""
            else:
                content = response.get("content", "") or ""
        else:
            content = str(response)

        # Check for tool calls
        if not has_tool_calls(response):
            # No tool calls - agent is done
            log.info(
                "agent_loop_completed",
                iterations=iteration + 1,
                total_tool_calls=len(all_tool_calls),
            )
            return AgentResult(
                success=True,
                response=content,
                iterations=iteration + 1,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                elapsed_ms=int((time.time() - start_time) * 1000),
                stopped_reason="completed",
            )

        # Parse tool calls
        tool_calls = parse_tool_calls(response, native_mode=config.native_tool_calling)

        if not tool_calls:
            # Has tool call markers but couldn't parse - treat as done
            log.warning("agent_tool_parse_failed", iteration=iteration)
            return AgentResult(
                success=True,
                response=content,
                iterations=iteration + 1,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                elapsed_ms=int((time.time() - start_time) * 1000),
                stopped_reason="completed",
            )

        log.info(
            "agent_executing_tools",
            iteration=iteration,
            tool_count=len(tool_calls),
            tools=[tc.name for tc in tool_calls],
        )

        # Execute tools
        results = await execute_tools(
            tool_calls,
            registry,
            timeout=config.timeout_per_tool,
            parallel=config.parallel_tools,
        )

        all_tool_calls.extend(tool_calls)
        all_tool_results.extend(results)

        # Add assistant message with tool calls to conversation
        if config.native_tool_calling:
            # OpenAI format: assistant message with tool_calls
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in tool_calls
                ]
            })
        else:
            # Text format: just add the response as-is
            messages.append({
                "role": "assistant",
                "content": content,
            })

        # Add tool results to conversation
        for tc, result in zip(tool_calls, results):
            messages.append(format_tool_result(tc.id, tc.name, result.output))

    # Hit max iterations
    log.warning(
        "agent_loop_max_iterations",
        iterations=config.max_iterations,
        total_tool_calls=len(all_tool_calls),
    )
    return AgentResult(
        success=False,
        response="Agent reached maximum iterations without completing the task.",
        iterations=config.max_iterations,
        tool_calls=all_tool_calls,
        tool_results=all_tool_results,
        elapsed_ms=int((time.time() - start_time) * 1000),
        stopped_reason="max_iterations",
    )
