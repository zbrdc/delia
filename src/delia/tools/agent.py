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
        allow_write: If True, enable file write operations (--allow-write)
        allow_exec: If True, enable shell command execution (--allow-exec)
        require_confirmation: If True, prompt before dangerous operations (disabled by --yolo)
    """
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    timeout_per_tool: float = DEFAULT_TIMEOUT_PER_TOOL
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT
    parallel_tools: bool = True
    native_tool_calling: bool = False
    # Permission flags (all disabled by default for security)
    allow_write: bool = False
    allow_exec: bool = False
    require_confirmation: bool = True  # Set to False via --yolo

    # Agentic Feedback Loop settings
    reflection_enabled: bool = False
    max_reflections: int = 1
    reflection_confidence: str = "normal"  # "normal" or "high" (triggers voting)


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
    tokens: int = 0
    stopped_reason: str = "completed"


# Type for the LLM calling function
LLMCallable = Callable[
    [list[dict[str, Any]], str | None],  # (messages, system_prompt)
    Awaitable[str | dict[str, Any]]  # Returns response (str or dict with tool_calls)
]


def build_system_prompt(base_system: str | None, registry: ToolRegistry, native_mode: bool) -> str:
    """Build system prompt with tool information.

    Includes current system time for accurate time-aware responses.

    Args:
        base_system: Base system prompt from caller
        registry: Tool registry with available tools
        native_mode: If True, tools passed via API; if False, include in prompt

    Returns:
        Complete system prompt
    """
    from ..language import get_current_time_context

    parts = []

    # Always include time context for accurate time-aware responses
    parts.append(get_current_time_context())

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
    critique_callback: Callable[[str, str], Awaitable[tuple[bool, str]]] | None = None,
    on_tool_call: Callable[[ParsedToolCall], None] | None = None,
    on_tool_result: Callable[[ToolResult], None] | None = None,
    messages: list[dict[str, Any]] | None = None,
    interruption_callback: Callable[[], Awaitable[str | None]] | None = None,
) -> AgentResult:
    """Run an agentic loop with tool calling.

    Args:
        ...
        interruption_callback: Async function called before each step. 
                               Returns None (continue), "STOP", or new instruction string.
    """
    config = config or AgentConfig()
    start_time = time.time()

    # Build system prompt with tools
    full_system = build_system_prompt(system_prompt, registry, config.native_tool_calling)

    # Initialize conversation
    if messages is None:
        messages = build_messages(prompt)
    else:
        # If history exists, append the new user prompt if not empty
        if prompt:
            messages.append({"role": "user", "content": prompt})

    all_tool_calls: list[ParsedToolCall] = []
    all_tool_results: list[ToolResult] = []
    total_tokens: int = 0
    reflection_count = 0

    log.info(
        "agent_loop_starting",
        model=model,
        prompt_len=len(prompt),
        tools=registry.list_tools(),
        max_iterations=config.max_iterations,
    )

    for iteration in range(config.max_iterations):
        # Check for interruption/injection
        if interruption_callback:
            instruction = await interruption_callback()
            if instruction == "STOP":
                log.info("agent_interrupted_by_user")
                return AgentResult(
                    success=False,
                    response="Agent stopped by user.",
                    iterations=iteration,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    elapsed_ms=int((time.time() - start_time) * 1000),
                    tokens=total_tokens,
                    stopped_reason="user_stop",
                )
            elif instruction:
                log.info("agent_instruction_injected", instruction=instruction)
                messages.append({"role": "user", "content": f"[User Intervention]: {instruction}"})

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
                tokens=total_tokens,
                stopped_reason="timeout",
            )

        # Call LLM
        log.debug("agent_iteration", iteration=iteration, messages_count=len(messages))

        try:
            response_data = await call_llm(messages, full_system)
        except Exception as e:
            log.error("agent_llm_error", iteration=iteration, error=str(e))
            return AgentResult(
                success=False,
                response=f"LLM call failed: {e}",
                iterations=iteration,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                elapsed_ms=int((time.time() - start_time) * 1000),
                tokens=total_tokens,
                stopped_reason="error",
            )

        # Extract text content and tokens
        if isinstance(response_data, dict):
            content = ""
            total_tokens += response_data.get("tokens", 0)
            
            # Delia LLMResponse format - response field
            if "response" in response_data:
                content = response_data.get("response", "") or ""
            # OpenAI format - message.content
            elif "message" in response_data:
                content = response_data["message"].get("content", "") or ""
            elif "choices" in response_data:
                choices = response_data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "") or ""
            else:
                content = response_data.get("content", "") or ""
            
            # For text analysis (reflection/logging), use the content string
            response_text = content
            
            # For tool parsing, we need the original data if in native mode
            if config.native_tool_calling:
                response_for_parsing = response_data
            else:
                response_for_parsing = response_text
        else:
            response_text = str(response_data)
            content = response_text
            response_for_parsing = response_text

        # Check for tool calls
        if not has_tool_calls(response_for_parsing):
            # No tool calls - check if we should reflect/critique
            if (
                config.reflection_enabled
                and critique_callback
                and reflection_count < config.max_reflections
            ):
                log.info("agent_reflecting", reflection_count=reflection_count + 1)
                
                # Execute critique
                # Note: critique callback tokens are not currently tracked in total_tokens
                # unless the callback returns them. For now we track agent loop tokens only.
                is_valid, feedback = await critique_callback(response_text, prompt)
                
                if not is_valid:
                    # Critique failed - add feedback and loop again
                    reflection_count += 1
                    
                    # Add agent's provisional response
                    messages.append({
                        "role": "assistant",
                        "content": response_text,
                    })
                    
                    # Add critique feedback
                    feedback_msg = f"Review Feedback: {feedback}\n\nPlease update your response based on this feedback."
                    messages.append({
                        "role": "user",
                        "content": feedback_msg,
                    })
                    
                    log.info("agent_reflection_feedback", feedback_len=len(feedback))
                    continue
                else:
                    log.info("agent_reflection_passed")

            # Agent is done and verified (or no reflection enabled)
            log.info(
                "agent_loop_completed",
                iterations=iteration + 1,
                total_tool_calls=len(all_tool_calls),
                reflections=reflection_count,
            )
            return AgentResult(
                success=True,
                response=response_text,
                iterations=iteration + 1,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                elapsed_ms=int((time.time() - start_time) * 1000),
                tokens=total_tokens,
                stopped_reason="completed",
            )

        # Parse tool calls
        tool_calls = parse_tool_calls(response_for_parsing, native_mode=config.native_tool_calling)

        if not tool_calls:
            # Has tool call markers but couldn't parse - treat as done
            log.warning("agent_tool_parse_failed", iteration=iteration)
            return AgentResult(
                success=True,
                response=response_text,
                iterations=iteration + 1,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                elapsed_ms=int((time.time() - start_time) * 1000),
                tokens=total_tokens,
                stopped_reason="completed",
            )

        log.info(
            "agent_executing_tools",
            iteration=iteration,
            tool_count=len(tool_calls),
            tools=[tc.name for tc in tool_calls],
        )
        
        # Trigger tool call callbacks
        if on_tool_call:
            for tc in tool_calls:
                try:
                    on_tool_call(tc)
                except Exception as e:
                    log.warning("tool_callback_failed", error=str(e))

        # Execute tools
        results = await execute_tools(
            tool_calls,
            registry,
            timeout=config.timeout_per_tool,
            parallel=config.parallel_tools,
        )
        
        # Trigger tool result callbacks
        if on_tool_result:
            for result in results:
                try:
                    on_tool_result(result)
                except Exception as e:
                    log.warning("result_callback_failed", error=str(e))

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
        tokens=total_tokens,
        stopped_reason="max_iterations",
    )
