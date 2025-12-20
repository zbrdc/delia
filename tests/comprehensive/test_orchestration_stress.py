# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from delia.orchestration.executor import OrchestrationExecutor
from delia.orchestration.result import DetectedIntent, OrchestrationMode
from delia.backend_manager import BackendConfig

@pytest.fixture
def mock_backend():
    return BackendConfig(
        id="stress-test",
        name="Stress Backend",
        provider="llamacpp",
        type="local",
        url="http://localhost",
        models={"quick": "m", "coder": "m", "moe": "m"}
    )

class TestOrchestrationStress:
    """Stress tests for complex orchestration modes (Voting, Agent, Comparison)."""

    # ============================================================ 
    # VOTING CONSENSUS STRESS
    # ============================================================ 
    @pytest.mark.asyncio
    async def test_voting_consensus_success(self, mock_backend):
        executor = OrchestrationExecutor()
        intent = DetectedIntent(task_type="quick", orchestration_mode=OrchestrationMode.VOTING, k_votes=2)
        
        # 2 identical responses + 1 outlier
        responses = [
            {"success": True, "response": "Yes", "tokens": 5},
            {"success": True, "response": "No", "tokens": 5},
            {"success": True, "response": "Yes", "tokens": 5},
        ]
        
        with patch("delia.llm.call_llm", new_callable=AsyncMock) as mock_call, \
             patch("delia.orchestration.executor.select_model", new_callable=AsyncMock) as mock_sel, \
             patch("delia.routing.get_router") as mock_router:
            mock_call.side_effect = responses
            mock_sel.return_value = "test-model"
            
            mock_router_obj = MagicMock()
            mock_router_obj.select_optimal_backend = AsyncMock(return_value=(None, mock_backend))
            mock_router.return_value = mock_router_obj
            
            result = await executor.execute(intent, "is it true?")
            assert result.success
            assert result.response == "Yes"
            assert result.consensus_reached is True
            assert result.votes_cast == 3

    @pytest.mark.asyncio
    async def test_voting_consensus_failure_graceful(self, mock_backend):
        executor = OrchestrationExecutor()
        # Require 5 matching votes, but we only have 4 unique responses
        intent = DetectedIntent(task_type="quick", orchestration_mode=OrchestrationMode.VOTING, k_votes=5)
        
        # 10 COMPLETELY unique responses (no consensus possible)
        # Using radically different content to ensure SequenceMatcher < 0.8
        responses = [
            {"success": True, "response": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG MULTIPLE TIMES.", "tokens": 5},
            {"success": True, "response": "A RADIANT SUN SETS BEHIND THE PURPLE MOUNTAINS OF DESTINY.", "tokens": 5},
            {"success": True, "response": "ELECTRONIC DREAMS OF SILICON SHEEP ARE COMMON IN AUGUST.", "tokens": 5},
            {"success": True, "response": "WHISPERS OF ANCIENT CODES ECHO THROUGH THE DIGITAL VOID.", "tokens": 5},
        ] * 4 # ensure enough items for max_attempts
        
        with patch("delia.llm.call_llm", new_callable=AsyncMock) as mock_call, \
             patch("delia.orchestration.executor.select_model", new_callable=AsyncMock) as mock_sel, \
             patch("delia.routing.get_router") as mock_router:
            mock_call.side_effect = responses
            mock_sel.return_value = "test-model"
            
            mock_router_obj = MagicMock()
            mock_router_obj.select_optimal_backend = AsyncMock(return_value=(None, mock_backend))
            mock_router.return_value = mock_router_obj
            
            result = await executor.execute(intent, "choose one")
            # Should not reach consensus, but return the best effort one
            assert result.success is True
            assert "Partial consensus" in result.response
            assert result.consensus_reached is False

    # ============================================================ 
    # AGENTIC LOOP ROBUSTNESS
    # ============================================================ 
    @pytest.mark.asyncio
    async def test_agent_error_recovery(self, mock_backend):
        executor = OrchestrationExecutor()
        intent = DetectedIntent(task_type="coder", orchestration_mode=OrchestrationMode.AGENTIC)
        
        # 1. First call fails
        # 2. Second call succeeds with tool use
        # 3. Third call provides final answer
        responses = [
            Exception("Simulated Network Error"),
            {"success": True, "response": '<tool_call>{"name": "shell_exec", "arguments": {"command": "ls"}}</tool_call>', "tokens": 10},
            {"success": True, "response": "The files are: README.md", "tokens": 10},
        ]
        
        with patch("delia.llm.call_llm", new_callable=AsyncMock) as mock_call, \
             patch("delia.orchestration.executor.select_model", new_callable=AsyncMock) as mock_sel, \
             patch("delia.routing.get_router") as mock_router, \
             patch("delia.tools.agent.AgentConfig") as mock_config:
            
            mock_call.side_effect = responses
            mock_sel.return_value = "test-model"
            
            mock_router_obj = MagicMock()
            mock_router_obj.select_optimal_backend = AsyncMock(return_value=(None, mock_backend))
            mock_router.return_value = mock_router_obj
            
            mock_config_obj = MagicMock()
            mock_config_obj.reflection_enabled = False
            mock_config_obj.max_iterations = 5
            mock_config_obj.total_timeout = 60
            mock_config_obj.timeout_per_tool = 10
            mock_config_obj.native_tool_calling = True
            mock_config.return_value = mock_config_obj
            
            result = await executor._execute_agentic(intent, "list files", None, None, None)
            assert result.success is False
            assert "Simulated Network Error" in result.response
    # ============================================================ 
    # COMPARISON MODE STRESS
    # ============================================================ 
    @pytest.mark.asyncio
    async def test_comparison_multi_backend(self, mock_backend):
        executor = OrchestrationExecutor()
        intent = DetectedIntent(task_type="quick", orchestration_mode=OrchestrationMode.COMPARISON)
        intent.comparison_models = ["model-a", "model-b"]
        
        with patch("delia.llm.call_llm", new_callable=AsyncMock) as mock_call, \
             patch("delia.backend_manager.backend_manager.get_enabled_backends", return_value=[mock_backend]):
            
            mock_call.return_value = {"success": True, "response": "Opinion", "tokens": 5}
            
            result = await executor.execute(intent, "what do you think?")
            assert result.success
            assert "model-a" in result.response
            assert "model-b" in result.response
            assert len(result.models_compared) == 2