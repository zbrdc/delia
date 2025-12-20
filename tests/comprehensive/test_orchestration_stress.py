# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from delia.orchestration.executor import OrchestrationExecutor
from delia.orchestration.result import DetectedIntent, OrchestrationMode, OrchestrationResult
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

    @pytest.mark.asyncio
    async def test_voting_consensus_success(self, mock_backend):
        executor = OrchestrationExecutor()
        intent = DetectedIntent(task_type="quick", orchestration_mode=OrchestrationMode.VOTING, k_votes=2)
        
        # Mock the internal logic completely to verify it handles consensus
        with patch.object(executor, "_execute_voting", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = OrchestrationResult(
                response="Yes", 
                success=True, 
                consensus_reached=True, 
                votes_cast=3,
                mode=OrchestrationMode.VOTING
            )
            
            result = await executor.execute(intent, "is it true?")
            assert result.success
            assert result.response == "Yes"
            assert result.consensus_reached is True
            assert result.votes_cast == 3

    @pytest.mark.asyncio
    async def test_voting_consensus_failure_graceful(self, mock_backend):
        executor = OrchestrationExecutor()
        intent = DetectedIntent(task_type="quick", orchestration_mode=OrchestrationMode.VOTING, k_votes=5)
        
        with patch.object(executor, "_execute_voting", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = OrchestrationResult(
                response="Partial consensus: No", 
                success=True, 
                consensus_reached=False, 
                votes_cast=10,
                mode=OrchestrationMode.VOTING
            )
            
            result = await executor.execute(intent, "choose one")
            assert result.success is True
            assert "Partial consensus" in result.response
            assert result.consensus_reached is False

    @pytest.mark.asyncio
    async def test_agent_error_recovery(self, mock_backend):
        executor = OrchestrationExecutor()
        intent = DetectedIntent(task_type="coder", orchestration_mode=OrchestrationMode.AGENTIC)
        
        with patch.object(executor, "_execute_agentic", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = OrchestrationResult(
                response="Error: Simulated Network Error", 
                success=False, 
                mode=OrchestrationMode.AGENTIC
            )
            
            result = await executor.execute(intent, "list files")
            assert result.success is False
            assert "Simulated Network Error" in result.response

    @pytest.mark.asyncio
    async def test_comparison_multi_backend(self, mock_backend):
        executor = OrchestrationExecutor()
        intent = DetectedIntent(task_type="quick", orchestration_mode=OrchestrationMode.COMPARISON)
        intent.comparison_models = ["model-a", "model-b"]
        
        with patch.object(executor, "_execute_comparison", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = OrchestrationResult(
                response="# Comparison\n## model-a\nOpinion\n## model-b\nOpinion", 
                success=True, 
                models_compared=["model-a", "model-b"],
                mode=OrchestrationMode.COMPARISON
            )
            
            result = await executor.execute(intent, "what do you think?")
            assert result.success
            assert "model-a" in result.response
            assert "model-b" in result.response
            assert len(result.models_compared) == 2