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
import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, AsyncIterator

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
    """Configuration for the agentic loop."""
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    timeout_per_tool: float = DEFAULT_TIMEOUT_PER_TOOL
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT
    parallel_tools: bool = True
    native_tool_calling: bool = False
    allow_write: bool = False
    allow_exec: bool = False
    require_confirmation: bool = True
    reflection_enabled: bool = False
    max_reflections: int = 1
    reflection_confidence: str = "normal"


@dataclass
class AgentResult:
    """Result from running the agentic loop."""
    success: bool
    response: str
    iterations: int
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    elapsed_ms: int = 0
    tokens: int = 0
    stopped_reason: str = "completed"


LLMCallable = Callable[
    [list[dict[str, Any]], str | None],
    Awaitable[str | dict[str, Any]]
]


def build_system_prompt(base_system: str | None, registry: ToolRegistry, native_mode: bool) -> str:
    from ..language import get_current_time_context
    parts = [get_current_time_context()]
    if base_system: parts.append(base_system)
    if not native_mode:
        parts.append("")
        parts.append(registry.get_tool_prompt())
    return "\n".join(parts)


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
    step_callback: Callable[[int], Awaitable[bool]] | None = None,
) -> AgentResult:
    config = config or AgentConfig()
    start_time = time.time()
    
    messages = messages or []
    if not messages:
        messages.append({"role": "user", "content": prompt})

    all_tool_calls: list[ParsedToolCall] = []
    all_tool_results: list[ToolResult] = []
    total_tokens = 0
    reflection_count = 0
    full_system = build_system_prompt(system_prompt, registry, config.native_tool_calling)

    for iteration in range(config.max_iterations):
        if step_callback:
            if not await step_callback(iteration):
                return AgentResult(False, "Stopped by watchdog", iteration, all_tool_calls, all_tool_results, int((time.time() - start_time) * 1000), total_tokens, "watchdog_stop")

        if interruption_callback:
            instr = await interruption_callback()
            if instr == "STOP":
                return AgentResult(False, "Stopped by user", iteration, all_tool_calls, all_tool_results, int((time.time() - start_time) * 1000), total_tokens, "user_stop")
            elif instr:
                messages.append({"role": "user", "content": f"[User Intervention]: {instr}"})

        if (time.time() - start_time) > config.total_timeout:
            return AgentResult(False, "Timed out", iteration, all_tool_calls, all_tool_results, int((time.time() - start_time) * 1000), total_tokens, "timeout")

        log.debug("agent_iteration", iteration=iteration, messages_count=len(messages))

        try:
            response_data = await call_llm(messages, full_system)
            log.debug("agent_llm_response", iteration=iteration, response_data=str(response_data)[:200])
        except Exception as e:
            log.error("agent_llm_error", iteration=iteration, error=str(e))
            return AgentResult(False, f"LLM call failed: {e}", iteration, all_tool_calls, all_tool_results, int((time.time() - start_time) * 1000), total_tokens, "error")

        if isinstance(response_data, dict):
            total_tokens += response_data.get("tokens", 0)
            if "response" in response_data: content = response_data["response"]
            elif "choices" in response_data: content = response_data["choices"][0]["message"].get("content", "")
            else: content = response_data.get("content", "")
            response_for_parsing = response_data if config.native_tool_calling else content
        else:
            content = str(response_data)
            response_for_parsing = content

        response_text = content or ""
        log.debug("agent_extracted_content", iteration=iteration, content=response_text[:100])

        if not has_tool_calls(response_for_parsing):
            log.debug("agent_no_tool_calls", iteration=iteration)
            if config.reflection_enabled and critique_callback and reflection_count < config.max_reflections:
                is_valid, feedback = await critique_callback(response_text, prompt)
                if not is_valid:
                    reflection_count += 1
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Review Feedback: {feedback}\n\nPlease update your response based on this feedback."})
                    continue
            
            # Record the final assistant response in history
            messages.append({"role": "assistant", "content": response_text})
            
            return AgentResult(True, response_text, iteration + 1, all_tool_calls, all_tool_results, int((time.time() - start_time) * 1000), total_tokens, "completed")

        tool_calls = parse_tool_calls(response_for_parsing, native_mode=config.native_tool_calling)
        log.debug("agent_parsed_tool_calls", iteration=iteration, count=len(tool_calls))
        if not tool_calls:
            return AgentResult(True, response_text, iteration + 1, all_tool_calls, all_tool_results, int((time.time() - start_time) * 1000), total_tokens, "completed")

        if on_tool_call:
            for tc in tool_calls:
                res_call = on_tool_call(tc)
                if asyncio.iscoroutine(res_call) or inspect.isawaitable(res_call):
                    await res_call

        results = await execute_tools(tool_calls, registry, timeout=config.timeout_per_tool, parallel=config.parallel_tools)
        log.debug("agent_tool_results", iteration=iteration, count=len(results))
        
        if on_tool_result:
            for res in results: on_tool_result(res)

        all_tool_calls.extend(tool_calls)
        all_tool_results.extend(results)

        if config.native_tool_calling:
            messages.append({"role": "assistant", "content": content, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}} for tc in tool_calls]})
        else:
            messages.append({"role": "assistant", "content": content})

        for tc, res in zip(tool_calls, results):
            messages.append(format_tool_result(tc.id, tc.name, res.output))

    return AgentResult(False, "Max iterations reached", config.max_iterations, all_tool_calls, all_tool_results, int((time.time() - start_time) * 1000), total_tokens, "max_iterations")