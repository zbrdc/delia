# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from delia.tools.builtins import update_milestone, check_progress, _agent_milestones
from delia.queue import ModelQueue
from delia.session_manager import SessionState, SessionMessage
from delia.orchestration.executor import OrchestrationExecutor
from delia.orchestration.intent import DetectedIntent, OrchestrationMode

@pytest.mark.asyncio
async def test_milestone_tools():
    """Test that update_milestone and check_progress work correctly."""
    session_id = "test-session-123"
    _agent_milestones.clear()
    res1 = await update_milestone("logic_gate", "in_progress", session_id=session_id)
    assert "updated to 'in_progress'" in res1
    res2 = await check_progress(session_id=session_id)
    data = json.loads(res2)
    assert data["logic_gate"] == "in_progress"

@pytest.mark.asyncio
async def test_queue_priority_ace():
    """Test that ACE task types (reflection, curation) get high priority."""
    queue = ModelQueue()
    p_think = queue.calculate_priority("think", 100, "Qwen3-14B")
    p_reflection = queue.calculate_priority("reflection", 100, "Qwen3-14B")
    assert p_reflection == p_think

@pytest.mark.asyncio
async def test_queue_acquisition_wait():
    """Test that the triggering request loads the model synchronously and returns True."""
    queue = ModelQueue()
    model = "test-load-wait"
    is_available, future = await queue.acquire_model(model)

    # Triggering request now waits for load and returns True immediately
    assert is_available is True
    assert future is None
    # Model should be loaded (not loading) after acquire completes
    assert model in queue.loaded_models
    assert model not in queue.loading_models

@pytest.mark.asyncio
async def test_token_based_context_window():
    """Test that SessionState.get_context_window uses tokens accurately."""
    session = SessionState(session_id="test-tokens")
    now = datetime.now()
    session.messages.append(SessionMessage(role="user", content="msg1", tokens=10, timestamp=now))
    session.messages.append(SessionMessage(role="assistant", content="msg2", tokens=20, timestamp=now))
    session.messages.append(SessionMessage(role="user", content="msg3", tokens=30, timestamp=now))
    
    window = session.get_context_window(max_tokens=40)
    assert "[user]: msg3" in window
    assert "[assistant]: msg2" not in window

@pytest.mark.asyncio
async def test_agentic_planning_integration():
    """Test that AGENTIC mode triggers planning and injects it into prompt."""
    executor = OrchestrationExecutor()
    intent = DetectedIntent(task_type="generate", orchestration_mode=OrchestrationMode.AGENTIC)
    
    mock_plan = MagicMock()
    mock_plan.steps = [MagicMock(id="step1")]
    mock_plan.dict.return_value = {"steps": [{"id": "step1"}]}
    executor.planner.plan = AsyncMock(return_value=mock_plan)
    
    mock_backend = MagicMock()
    mock_backend.id = "test-backend"
    mock_backend.name = "TestBackend"
    mock_backend.models = {"agentic": "m"}
    mock_backend.supports_native_tool_calling = False
    
    with patch("delia.routing.get_router") as mock_router_get, \
         patch("delia.orchestration.executor.select_model", new_callable=AsyncMock) as mock_select, \
         patch("delia.tools.agent.run_agent_loop", new_callable=AsyncMock) as mock_agent:
        
        mock_router = MagicMock()
        # FIX: select_optimal_backend must be an AsyncMock
        mock_router.select_optimal_backend = AsyncMock(return_value=(None, mock_backend))
        mock_router_get.return_value = mock_router
        
        mock_select.return_value = "m"
        mock_agent.return_value = MagicMock(success=True, response="Done", tool_calls=[], tokens=100)
        
        await executor.execute(intent, "build")
        
        executor.planner.plan.assert_called_once()
        args, kwargs = mock_agent.call_args
        assert "### EXECUTION PLAN" in kwargs["system_prompt"]

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
