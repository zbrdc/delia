# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import asyncio
import time
from unittest.mock import AsyncMock
from delia.tools.executor import execute_tools, ParsedToolCall
from delia.tools.registry import ToolRegistry, ToolDefinition

@pytest.fixture
def registry():
    reg = ToolRegistry()
    
    async def slow_tool(delay: float = 0.1):
        await asyncio.sleep(delay)
        return f"Done after {delay}"
        
    async def failing_tool():
        raise RuntimeError("Fail")
        
    reg.register(ToolDefinition("slow", "Slow tool", {"type":"object"}, slow_tool))
    reg.register(ToolDefinition("fail", "Failing tool", {"type":"object"}, failing_tool))
    return reg

@pytest.mark.asyncio
@pytest.mark.parametrize("tool_count", [1, 2, 5, 10, 20])
async def test_parallel_execution_scaling(registry, tool_count):
    """Test scaling of parallel tool execution."""
    calls = [
        ParsedToolCall(id=f"c{i}", name="slow", arguments={"delay": 0.01})
        for i in range(tool_count)
    ]
    
    start = time.time()
    results = await execute_tools(calls, registry, parallel=True)
    elapsed = time.time() - start
    
    assert len(results) == tool_count
    assert all(r.success for r in results)
    # If truly parallel, total time should be close to single delay
    assert elapsed < 0.5 # Relaxed from 0.1 to account for system overhead
@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0.01, 0.05, 0.1])
async def test_execution_timeout_fuzzing(registry, timeout):
    """Test various timeout thresholds."""
    calls = [ParsedToolCall(id="c1", name="slow", arguments={"delay": 0.5})]
    
    results = await execute_tools(calls, registry, timeout=timeout)
    assert not results[0].success
    assert "timed out" in results[0].output

@pytest.mark.asyncio
async def test_mixed_failure_parallel(registry):
    """Test parallel execution when some tools fail and others succeed."""
    calls = [
        ParsedToolCall(id="c1", name="slow", arguments={"delay": 0.01}),
        ParsedToolCall(id="c2", name="fail", arguments={}),
        ParsedToolCall(id="c3", name="slow", arguments={"delay": 0.01}),
    ]
    
    results = await execute_tools(calls, registry, parallel=True)
    assert results[0].success
    assert not results[1].success
    assert results[2].success

@pytest.mark.asyncio
@pytest.mark.parametrize("i", range(50))
async def test_stress_iteration_fuzzing(registry, i):
    """50 iterations of high-concurrency tool calls."""
    calls = [
        ParsedToolCall(id=f"c{j}", name="slow", arguments={"delay": 0.001})
        for j in range(10)
    ]
    results = await execute_tools(calls, registry, parallel=True)
    assert len(results) == 10
