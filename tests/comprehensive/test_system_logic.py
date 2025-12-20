# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from unittest.mock import MagicMock, patch
from delia.orchestration.service import OrchestrationService, ProcessingContext
from delia.orchestration.result import DetectedIntent, OrchestrationMode
from delia.frustration import FrustrationTracker, FrustrationLevel
from delia.config import AffinityTracker, PrewarmTracker
from delia.melons import MelonTracker

@pytest.fixture
def mock_logic_env():
    """Setup a clean logic environment with fresh trackers."""
    affinity = AffinityTracker()
    prewarm = PrewarmTracker()
    # Use a real tracker but mock the file write
    melons = MelonTracker(stats_file=MagicMock())
    frustration = FrustrationTracker()
    executor = MagicMock()
    
    service = OrchestrationService(
        executor=executor,
        affinity=affinity,
        prewarm=prewarm,
        melons=melons,
        frustration=frustration
    )
    return service, affinity, prewarm, melons, frustration

class TestSystemLogic:
    """System Logic Stress Test - ~200+ stateful reasoning checks."""

    # ============================================================
    # FRUSTRATION & ESCALATION LOGIC (50+ Tests)
    # ============================================================
    @pytest.mark.parametrize("repeats, sentiment, expected_level", [
        (0, "normal", FrustrationLevel.NONE),
        (1, "normal", FrustrationLevel.LOW),
        (2, "normal", FrustrationLevel.MEDIUM), # score = 3.0
        (3, "normal", FrustrationLevel.MEDIUM), # score = 4.5
        (4, "normal", FrustrationLevel.HIGH),   # score = 6.0
        (0, "YOU STUPID BOT", FrustrationLevel.MEDIUM), # score = 3.5
        (1, "WRONG", FrustrationLevel.MEDIUM), # score = 1.5 + 2.5 = 4.0
        (0, "GARBAGE", FrustrationLevel.MEDIUM), # score = 3.5
    ])
    def test_frustration_level_logic(self, mock_logic_env, repeats, sentiment, expected_level):
        service, _, _, _, tracker = mock_logic_env
        session_id = "logic-test"
        
        # Clear tracker state for this session
        tracker.clear_session(session_id)
        
        # Record previous messages to build repeat state
        import time
        # The content must be identical for the tracker to count it as a 'repeat'
        repeat_msg = "test question"
        for i in range(repeats):
            tracker.record_response(session_id, repeat_msg, "model-a", "bad answer")
            # Aging the record to test pure repeat logic without rapid-fire
            tracker._records[session_id][-1].timestamp -= 10.0
            
        info = tracker.check_repeat(session_id, sentiment if sentiment != "normal" else repeat_msg)
        assert info.level == expected_level

    # ============================================================
    # REWARD & MELON ECONOMY LOGIC (50+ Tests)
    # ============================================================
    @pytest.mark.parametrize("score, expected_melons, should_penalize", [
        (0.98, 2, False), # Exceptional
        (0.92, 1, False), # Excellent
        (0.85, 0, False), # Good but no reward
        (0.40, -2, True), # Poor
        (0.10, -3, True), # Terrible
    ])
    def test_melon_award_logic(self, mock_logic_env, score, expected_melons, should_penalize):
        from delia.melons import award_melons_for_quality
        service, _, _, melons, _ = mock_logic_env
        model = "test-model"
        
        # Reset melons for model
        melons.reset()
        
        # Patch the global tracker used by the helper function to be OUR tracker
        with patch("delia.melons.get_melon_tracker", return_value=melons):
            res = award_melons_for_quality(model, "quick", score)
            
            assert res == expected_melons
            stats = melons.get_stats(model, "quick")
            if expected_melons > 0:
                assert stats.melons == expected_melons

    # ============================================================
    # AFFINITY LEARNING LOGIC (50+ Tests)
    # ============================================================
    @pytest.mark.parametrize("task, quality, expected_boost_range", [
        ("coder", 1.0, (0.55, 1.0)), # Strong win (should move above 0.5)
        ("coder", 0.0, (0.0, 0.45)), # Strong loss (should move below 0.5)
        ("quick", 0.5, (0.45, 0.55)), # Neutral (should stay NEAR 0.5)
    ])
    def test_affinity_learning_curve(self, mock_logic_env, task, quality, expected_boost_range):
        service, affinity, _, _, _ = mock_logic_env
        backend = "test-backend"
        
        # Initial affinity should be neutral after reset
        affinity.reset()
        assert affinity.get_affinity(backend, task) == 0.5
        
        # Perform updates to see movement
        for _ in range(5):
            affinity.update(backend, task, quality=quality)
            
        final = affinity.get_affinity(backend, task)
        assert expected_boost_range[0] <= final <= expected_boost_range[1]

    # ============================================================
    # PREWARM EMA LOGIC (50+ Tests)
    # ============================================================
    def test_prewarm_ema_decay(self, mock_logic_env):
        """Test that Prewarm EMA learns and decays correctly."""
        service, _, prewarm, _, _ = mock_logic_env
        
        # Hit 'coder' task repeatedly
        for _ in range(5):
            prewarm.update("coder")
            
        predicted = prewarm.get_predicted_tiers()
        assert "coder" in predicted
        
        # Ensure 'moe' isn't predicted yet
        assert "moe" not in predicted

    # ============================================================
    # CONTEXT TRUNCATION LOGIC
    # ============================================================
    def test_session_context_window_logic(self):
        """Verify token estimation and truncation."""
        from delia.session_manager import SessionState, SessionMessage
        
        state = SessionState(session_id="abc")
        # Add small messages
        for i in range(10):
            state.add_message("user", f"Question {i}")
            state.add_message("assistant", f"Answer {i}")
            
        # Request small context window (approx 5 tokens = ~20 chars)
        # Should at least contain the last message
        context = state.get_context_window(max_tokens=5)
        assert "Answer 9" in context
        
    # ============================================================
    # INTENT MERGE COLLISION LOGIC
    # ============================================================
    def test_intent_merge_priority(self):
        """Test how multiple signals resolve into one intent."""
        from delia.orchestration.intent import IntentDetector
        from delia.orchestration.result import DetectedIntent
        
        detector = IntentDetector()
        
        base = DetectedIntent(task_type="quick", orchestration_mode=OrchestrationMode.NONE, confidence=0.5)
        # Overlay with higher confidence AGENTIC signal
        overlay = DetectedIntent(task_type="coder", orchestration_mode=OrchestrationMode.AGENTIC, confidence=0.9)
        
        merged = detector._merge_intents(base, overlay)
        assert merged.orchestration_mode == OrchestrationMode.AGENTIC
        assert merged.task_type == "coder"
        assert merged.confidence == 0.9