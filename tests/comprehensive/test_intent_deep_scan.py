# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from unittest.mock import patch, MagicMock
from delia.orchestration.intent import IntentDetector
from delia.orchestration.result import OrchestrationMode, ModelRole

class TestIntentDeepScan:
    """Deep scan of intent detection logic."""

@pytest.fixture
def detector():
    # Disable ToT exploration during tests to ensure deterministic mode detection
    with patch("delia.orchestration.meta_learning.get_orchestration_learner") as mock_learner_getter:
        mock_learner = MagicMock()
        mock_learner.should_use_tot.return_value = (False, "disabled for test")
        mock_learner.get_best_mode.return_value = (None, 0.0)
        mock_learner_getter.return_value = mock_learner
        return IntentDetector()

    @pytest.mark.parametrize("msg, expected_mode", [
        ("Read src/main.py", OrchestrationMode.AGENTIC),
        ("show me the files in this folder", OrchestrationMode.AGENTIC),
        ("where are we right now?", OrchestrationMode.AGENTIC),
        ("what direcotry is this?", OrchestrationMode.AGENTIC),
        ("search for all instances of 'class Config'", OrchestrationMode.AGENTIC),
        ("run npm install and tell me if it fails", OrchestrationMode.AGENTIC),
        ("web search for the latest deepseek model", OrchestrationMode.AGENTIC),
        ("what is the news today", OrchestrationMode.AGENTIC),
        ("save this code to a new file named test.py", OrchestrationMode.AGENTIC),
        ("create a python script and put it on disk", OrchestrationMode.AGENTIC),
        ("grep -r TODO .", OrchestrationMode.AGENTIC),
        ("find definition of select_model", OrchestrationMode.AGENTIC),
    ])
    def test_agentic_triggers(self, detector, msg, expected_mode):
        intent = detector.detect(msg)
        assert intent.orchestration_mode == expected_mode

    @pytest.mark.parametrize("msg, expected_mode", [
        ("verify this code is secure", OrchestrationMode.VOTING),
        ("make sure there are no bugs in this", OrchestrationMode.VOTING),
        ("double check my logic", OrchestrationMode.VOTING),
        ("I need a highly reliable answer for this critical task", OrchestrationMode.VOTING),
        ("ensure this migration script is safe to run", OrchestrationMode.AGENTIC), # Security tool migration script triggers tool use
        ("confirm this math is correct", OrchestrationMode.VOTING),
        ("I need accuracy on this", OrchestrationMode.VOTING),
    ])
    def test_voting_triggers(self, detector, msg, expected_mode):
        intent = detector.detect(msg)
        assert intent.orchestration_mode == expected_mode

    @pytest.mark.parametrize("msg, expected_mode", [
        ("compare how different models handle this", OrchestrationMode.COMPARISON),
        ("what do multiple models think about this strategy", OrchestrationMode.COMPARISON),
        ("get a second opinion on my architectural design", OrchestrationMode.DEEP_THINKING), 
        ("compare vs deepseek vs claude", OrchestrationMode.COMPARISON),
        ("which model is best at this code?", OrchestrationMode.NONE), # Better/best isn't comparison without 'vs' or 'multiple'
    ])
    def test_comparison_triggers(self, detector, msg, expected_mode):
        intent = detector.detect(msg)
        assert intent.orchestration_mode == expected_mode

    @pytest.mark.parametrize("msg, expected_mode", [
        ("first analyze the code, then refactor it", OrchestrationMode.CHAIN),
        ("1. Read the file 2. Review it 3. Suggest fixes", OrchestrationMode.AGENTIC), # Technical sequence defaults to tools
        ("Step 1: check health. Step 2: list models.", OrchestrationMode.AGENTIC),
        ("Analyze the project structure and then generate a README", OrchestrationMode.CHAIN), # Multi-step workflow
    ])
    def test_chain_triggers(self, detector, msg, expected_mode):
        intent = detector.detect(msg)
        assert intent.orchestration_mode == expected_mode

    @pytest.mark.parametrize("msg, expected_task", [
        ("write a function", "coder"),
        ("design a system", "moe"),
        ("architect a solution", "moe"),
        ("how are you doing", "quick"),
        ("find security bugs in this snippet", "coder"),
        ("plan the migration to microservices", "moe"),
        ("summarize this text", "quick"),
        ("is this code efficient", "coder"),
    ])
    def test_task_type_detection(self, detector, msg, expected_task):
        intent = detector.detect(msg)
        assert intent.task_type == expected_task
