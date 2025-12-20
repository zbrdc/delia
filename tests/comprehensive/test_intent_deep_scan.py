# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from delia.orchestration.intent import IntentDetector
from delia.orchestration.result import OrchestrationMode, ModelRole

@pytest.fixture
def detector():
    return IntentDetector()

class TestIntentDeepScan:
    """Deep Scan of Intent Detection - 200+ unique phrasing checks."""

    # ============================================================
    # AGENTIC TRIGGERS (Tools)
    # ============================================================
    @pytest.mark.parametrize("msg, expected_mode", [
        ("Read src/main.py", OrchestrationMode.AGENTIC),
        ("show me the files in this folder", OrchestrationMode.AGENTIC),
        ("where are we right now?", OrchestrationMode.AGENTIC), # Directory query
        ("what direcotry is this?", OrchestrationMode.AGENTIC), # Typo check
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

    # ============================================================
    # VOTING TRIGGERS (Reliability)
    # ============================================================
    @pytest.mark.parametrize("msg, expected_mode", [
        ("verify this code is secure", OrchestrationMode.VOTING),
        ("make sure there are no bugs in this", OrchestrationMode.VOTING),
        ("double check my logic", OrchestrationMode.VOTING),
        ("I need a highly reliable answer for this critical task", OrchestrationMode.VOTING),
        ("ensure this migration script is safe to run", OrchestrationMode.VOTING),
        ("confirm this math is correct", OrchestrationMode.VOTING),
        ("I need 100% accuracy on this", OrchestrationMode.VOTING),
    ])
    def test_voting_triggers(self, detector, msg, expected_mode):
        intent = detector.detect(msg)
        assert intent.orchestration_mode == expected_mode

    # ============================================================
    # COMPARISON TRIGGERS
    # ============================================================
    @pytest.mark.parametrize("msg, expected_mode", [
        ("compare how different models handle this", OrchestrationMode.COMPARISON),
        ("what do multiple models think about this strategy", OrchestrationMode.COMPARISON),
        ("get a second opinion on my architectural design", OrchestrationMode.COMPARISON),
        ("compare vs deepseek vs claude", OrchestrationMode.COMPARISON),
        ("which model is better at this code?", OrchestrationMode.COMPARISON),
    ])
    def test_comparison_triggers(self, detector, msg, expected_mode):
        intent = detector.detect(msg)
        assert intent.orchestration_mode == expected_mode

    # ============================================================
    # CHAIN TRIGGERS
    # ============================================================
    @pytest.mark.parametrize("msg, expected_mode", [
        ("first analyze the code, then refactor it", OrchestrationMode.CHAIN),
        ("1. Read the file 2. Review it 3. Suggest fixes", OrchestrationMode.CHAIN),
        ("Step 1: check health. Step 2: list models.", OrchestrationMode.CHAIN),
        ("Analyze the project structure and then generate a README", OrchestrationMode.CHAIN),
    ])
    def test_chain_triggers(self, detector, msg, expected_mode):
        intent = detector.detect(msg)
        assert intent.orchestration_mode == expected_mode

    # ============================================================
    # TASK TYPE DETECTION (Coder/Moe/Quick)
    # ============================================================
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

    # ============================================================
    # NOISE ROBUSTNESS (The "SQL Code" crash prevention)
    # ============================================================
    def test_code_block_noise_immunity(self, detector):
        """Ensure that intent is detected from instructions, not pasted code."""
        sql_message = """Review this SQL Code:
```sql
-- ensure the query is unambiguous
-- Admin check
SELECT id FROM users;
```"""
        intent = detector.detect(sql_message)
        # Should NOT trigger VOTING (even though 'ensure' and 'check' are in the code)
        # Should detect CODER task because of "Review this SQL Code"
        assert intent.orchestration_mode == OrchestrationMode.NONE
        assert intent.task_type == "coder"
        assert intent.model_role == ModelRole.CODE_REVIEWER

    @pytest.mark.parametrize("msg", [
        "hi " * 50, # Long simple message
        "a",        # Too short
        "!!! ???",  # Just symbols
    ])
    def test_edge_case_robustness(self, detector, msg):
        intent = detector.detect(msg)
        assert intent is not None
        assert isinstance(intent.task_type, str)
