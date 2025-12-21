#!/usr/bin/env python3
"""Test the embedding-based dispatcher integration."""

import pytest
from delia.orchestration.dispatcher import ModelDispatcher, get_embedding_dispatcher
from delia.orchestration.result import DetectedIntent, OrchestrationMode

# Test cases: (prompt, expected_result)
EXECUTOR_CASES = [
    ("Write a python function for fibonacci", "executor"),
    ("Fix the null pointer bug", "executor"),
    ("Implement caching layer", "executor"),
    ("Refactor the auth service", "executor"),
    ("Write unit tests for the parser", "executor"),
    ("Debug the memory leak", "executor"),
    ("Add logging to the API", "executor"),
    ("Create a REST endpoint", "executor"),
]

PLANNER_CASES = [
    ("Design microservices architecture", "planner"),
    ("Plan database migration", "planner"),
    ("Evaluate SQL vs NoSQL tradeoffs", "planner"),
    ("Create a roadmap for API v2", "planner"),
    ("Scale our backend to handle 10x load", "planner"),
    ("Design a disaster recovery strategy", "planner"),
]

STATUS_CASES = [
    ("Show me the melon leaderboard", "status"),
    ("Check system health", "status"),
    ("What models are available", "status"),
    ("How many tokens saved today", "status"),
    ("Display usage statistics", "status"),
]

ALL_CASES = EXECUTOR_CASES + PLANNER_CASES + STATUS_CASES


class TestEmbeddingDispatcher:
    """Test the embedding dispatcher directly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure embedding dispatcher is available."""
        self.dispatcher = get_embedding_dispatcher()
        if not self.dispatcher:
            pytest.skip("sentence-transformers not available")

    @pytest.mark.parametrize("prompt,expected", EXECUTOR_CASES)
    def test_executor_classification(self, prompt: str, expected: str):
        """Test executor task classification."""
        result = self.dispatcher.dispatch(prompt)
        assert result == expected, f"'{prompt}' should be {expected}, got {result}"

    @pytest.mark.parametrize("prompt,expected", PLANNER_CASES)
    def test_planner_classification(self, prompt: str, expected: str):
        """Test planner task classification."""
        result = self.dispatcher.dispatch(prompt)
        assert result == expected, f"'{prompt}' should be {expected}, got {result}"

    @pytest.mark.parametrize("prompt,expected", STATUS_CASES)
    def test_status_classification(self, prompt: str, expected: str):
        """Test status query classification."""
        result = self.dispatcher.dispatch(prompt)
        assert result == expected, f"'{prompt}' should be {expected}, got {result}"

    def test_overall_accuracy(self):
        """Test that overall accuracy is above 90%."""
        correct = sum(
            1 for prompt, expected in ALL_CASES
            if self.dispatcher.dispatch(prompt) == expected
        )
        accuracy = 100 * correct / len(ALL_CASES)
        assert accuracy >= 90, f"Accuracy {accuracy:.1f}% is below 90% threshold"


class TestModelDispatcherIntegration:
    """Test the ModelDispatcher with embedding mode."""

    @pytest.fixture
    def dispatcher(self):
        """Create a ModelDispatcher with mock LLM."""
        async def mock_llm(**kwargs):
            return {"success": False, "error": "LLM should not be called"}
        return ModelDispatcher(mock_llm)

    @pytest.fixture
    def intent(self):
        """Create a default intent for testing."""
        return DetectedIntent(
            task_type="general",
            confidence=0.5,
            orchestration_mode=OrchestrationMode.NONE,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("prompt,expected", ALL_CASES[:10])
    async def test_dispatch_with_embeddings(self, dispatcher, intent, prompt: str, expected: str):
        """Test that dispatch uses embeddings correctly."""
        result = await dispatcher.dispatch(prompt, intent, use_embeddings=True)
        assert result == expected, f"'{prompt}' should be {expected}, got {result}"

    @pytest.mark.asyncio
    async def test_embedding_mode_is_default(self, dispatcher, intent):
        """Test that embedding mode is used by default."""
        # If embedding mode works, LLM shouldn't be called
        result = await dispatcher.dispatch("Write a python function", intent)
        assert result == "executor"
