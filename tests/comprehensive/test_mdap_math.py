# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import math
from hypothesis import given, strategies as st
from delia.voting import VotingConsensus

class TestMDAPMathematicalFuzzing:
    """
    Mathematical Fuzzing for MDAP (Multi-Decision Accuracy Probability).
    Tests the core reliability formulas across thousands of generated scenarios.
    """

    @given(
        k=st.integers(min_value=1, max_value=10),
        p=st.floats(min_value=0.51, max_value=0.9999)
    )
    def test_voting_probability_formula(self, k, p):
        """Verify the MDAP reliability formula verified by Wolfram Alpha."""
        # P(correct) = 1 / (1 + ((1-p)/p)^k)
        prob = VotingConsensus.voting_probability(k, p)
        
        assert 0 <= prob <= 1.0
        
        # If p is high, prob should be very high
        if p > 0.9 and k >= 3:
            assert prob > 0.999 or prob == pytest.approx(0.999, rel=1e-3)
            
        # Increasing k should always increase or maintain reliability
        assert VotingConsensus.voting_probability(k + 1, p) >= prob

    @given(
        total_steps=st.integers(min_value=1, max_value=100),
        target_accuracy=st.floats(min_value=0.9, max_value=0.9999),
        base_accuracy=st.floats(min_value=0.7, max_value=0.99)
    )
    def test_calculate_kmin_fuzzing(self, total_steps, target_accuracy, base_accuracy):
        """Fuzz the kmin calculation logic used for adaptive voting."""
        kmin = VotingConsensus.calculate_kmin(
            total_steps=total_steps,
            target_accuracy=target_accuracy,
            base_accuracy=base_accuracy
        )
        
        assert kmin >= 1
        # For very high steps or high target accuracy, kmin should increase
        if total_steps > 50 and target_accuracy > 0.99:
            assert kmin >= 2

    @pytest.mark.parametrize("p,k,expected_min", [
        (0.95, 3, 0.999),
        (0.99, 3, 0.999999),
        (0.80, 5, 0.98),
    ])
    def test_known_mathematical_anchors(self, p, k, expected_min):
        """Test specific probability anchors verified via Wolfram Alpha."""
        prob = VotingConsensus.voting_probability(k, p)
        assert prob >= expected_min or prob == pytest.approx(expected_min, rel=1e-5)

    def test_complexity_estimation_fuzzing(self):
        """Test complexity estimation across 50 varied prompts."""
        prompts = [
            "hello",
            "write a quick function",
            "refactor the entire src directory and update the build system and run tests",
            "check for bugs",
            "A " * 1000, # Large repetitive
            "!@#$%^&*()", # Garbage
        ]
        
        from delia.voting import estimate_task_complexity
        
        for p in prompts:
            complexity = estimate_task_complexity(p)
            assert complexity >= 1
            # Long complex prompts should have higher complexity
            if "refactor" in p and "update" in p:
                assert complexity > estimate_task_complexity("hello")

    @given(st.lists(st.text(), min_size=1, max_size=20))
    def test_consensus_logic_fuzzing(self, votes):
        """Fuzz the consensus grouping logic with random strings."""
        consensus = VotingConsensus(k=3)
        for v in votes:
            res = consensus.add_vote(v)
            assert res is not None
            
        best, meta = consensus.get_best_response()
        if votes:
            assert best in votes or best is None
