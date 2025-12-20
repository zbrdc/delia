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

"""Tests for agentic feedback loop (reflection)."""

import pytest
from unittest.mock import AsyncMock

from delia.tools import (
    ToolDefinition,
    ToolRegistry,
    run_agent_loop,
    AgentConfig,
)

@pytest.mark.asyncio
async def test_feedback_loop_triggers_retry():
    """Test that feedback loop forces agent to retry when critique fails."""
    call_count = 0

    async def mock_llm(messages, system):
        nonlocal call_count
        call_count += 1

        # Check conversation history
        # 1st call: Initial prompt
        # 2nd call: Includes feedback
        if call_count == 1:
            return "Here is a script to read the file: import os; print(open('f.txt').read())"
        elif call_count == 2:
            # Verify feedback is in history
            assert messages[-1]["role"] == "user"
            assert "Review Feedback" in messages[-1]["content"]
            assert "You provided a script" in messages[-1]["content"]
            
            # Agent corrects itself by using the tool
            return '''<tool_call>{"name": "read_file", "arguments": {"path": "f.txt"}}</tool_call>'''
        else:
            return "File content is: Hello World"

    registry = ToolRegistry()
    async def mock_read_file(path: str, **kwargs):
        return "Hello World"
    
    registry.register(ToolDefinition(
        name="read_file",
        description="Read file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        handler=mock_read_file
    ))

    # Mock critique callback
    # First critique fails, second succeeds
    critique_calls = 0
    async def mock_critique(response: str, prompt: str) -> tuple[bool, str]:
        nonlocal critique_calls
        critique_calls += 1
        if critique_calls == 1:
            return False, "You provided a script but user wanted data."
        return True, "VERIFIED"

    config = AgentConfig(
        reflection_enabled=True, 
        max_reflections=2
    )

    result = await run_agent_loop(
        call_llm=mock_llm,
        prompt="Read f.txt",
        system_prompt=None,
        registry=registry,
        model="test",
        config=config,
        critique_callback=mock_critique,
    )

    assert result.success
    assert critique_calls == 2
    assert result.iterations == 3 # 1 (bad) + 1 (tool) + 1 (good)
    assert "Hello World" in result.response

@pytest.mark.asyncio
async def test_feedback_loop_max_reflections():
    """Test that agent stops reflecting after max_reflections limit."""
    
    async def mock_llm(messages, system):
        return "Stubborn response."

    async def mock_critique(response: str, prompt: str) -> tuple[bool, str]:
        return False, "Still bad."

    config = AgentConfig(
        reflection_enabled=True,
        max_reflections=3
    )

    result = await run_agent_loop(
        call_llm=mock_llm,
        prompt="Do task",
        system_prompt=None,
        registry=ToolRegistry(),
        model="test",
        config=config,
        critique_callback=mock_critique,
    )

    # Should succeed (return result) but stop reflecting
    assert result.success
    # Iterations: 
    # 1. Initial
    # 2. Retry 1
    # 3. Retry 2
    # 4. Retry 3 (max reached)
    assert result.iterations == 4 
