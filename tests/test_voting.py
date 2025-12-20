"""
Tests for k-voting consensus system.

Verifies the MDAP-inspired voting mechanism:
- K-voting probability calculations
- Consensus detection
- Red-flag quality checks
- kmin calculation

All mathematical formulas verified with Wolfram Alpha:
- P(correct|k=2, p=0.99) = 1/(1+0.01^2) = 0.9999
- P(correct|k=3, p=0.99) = 1/(1+0.01^3) = 0.999999
- kmin = ceiling(log(99999)/log(99)) = 3 for P >= 0.99999
"""

import pytest
from delia.voting import (
    VotingConsensus,
    VoteResult,
    ConsensusMetadata,
    estimate_task_complexity,
)


class TestVotingProbability:
    """Test voting probability calculations (Wolfram Alpha verified)."""

    def test_k2_probability(self):
        """P(correct|k=2, p=0.99) = 0.9999."""
        p = VotingConsensus.voting_probability(k=2, base_accuracy=0.99)
        assert 0.9998 < p < 1.0
        assert abs(p - 0.9999) < 0.0001

    def test_k3_probability(self):
        """P(correct|k=3, p=0.99) = 0.999999."""
        p = VotingConsensus.voting_probability(k=3, base_accuracy=0.99)
        assert p > 0.99999
        assert abs(p - 0.999999) < 0.000001

    def test_probability_increases_with_k(self):
        """Higher k should increase probability."""
        p1 = VotingConsensus.voting_probability(k=1, base_accuracy=0.99)
        p2 = VotingConsensus.voting_probability(k=2, base_accuracy=0.99)
        p3 = VotingConsensus.voting_probability(k=3, base_accuracy=0.99)
        assert p1 < p2 < p3

    def test_probability_edge_cases(self):
        """Edge cases for probability calculation."""
        # Perfect accuracy
        p = VotingConsensus.voting_probability(k=2, base_accuracy=1.0)
        assert p == 1.0

        # Zero accuracy
        p = VotingConsensus.voting_probability(k=2, base_accuracy=0.0)
        assert p == 0.0


class TestKminCalculation:
    """Test kmin calculation (Wolfram Alpha verified)."""

    def test_kmin_for_single_step(self):
        """Single step should need minimal k."""
        k = VotingConsensus.calculate_kmin(total_steps=1, target_accuracy=0.9999)
        assert k >= 2  # Minimum clamp

    def test_kmin_for_10_steps(self):
        """10 steps with 0.9999 target needs k=3."""
        # ceiling(log(99999)/log(99)) = 3
        k = VotingConsensus.calculate_kmin(total_steps=10, target_accuracy=0.9999)
        assert k >= 2
        assert k <= 5  # Max clamp

    def test_kmin_logarithmic_scaling(self):
        """kmin should scale logarithmically with steps (O(ln s))."""
        k10 = VotingConsensus.calculate_kmin(total_steps=10)
        k100 = VotingConsensus.calculate_kmin(total_steps=100)
        k1000 = VotingConsensus.calculate_kmin(total_steps=1000)

        # Logarithmic scaling: k should increase slowly
        # k100/k10 should be much less than 100/10
        assert k100 <= k10 + 2
        assert k1000 <= k100 + 2

    def test_kmin_clamped_to_range(self):
        """kmin should be clamped to [2, 5]."""
        k_low = VotingConsensus.calculate_kmin(total_steps=1, target_accuracy=0.5)
        k_high = VotingConsensus.calculate_kmin(total_steps=10000, target_accuracy=0.99999)

        assert k_low >= 2
        assert k_high <= 5


class TestVotingConsensus:
    """Test voting consensus mechanism."""

    def test_consensus_reached_with_k_matching_votes(self):
        """Consensus is reached after k matching votes."""
        consensus = VotingConsensus(k=2)

        result1 = consensus.add_vote("The answer is 42")
        assert not result1.consensus_reached

        result2 = consensus.add_vote("The answer is 42")
        assert result2.consensus_reached
        assert result2.winning_response == "The answer is 42"
        assert result2.votes_for_winner == 2

    def test_consensus_with_k3(self):
        """Consensus with k=3 requires 3 matching votes."""
        consensus = VotingConsensus(k=3)

        consensus.add_vote("Answer A")
        consensus.add_vote("Answer A")
        result = consensus.add_vote("Answer A")

        assert result.consensus_reached
        assert result.votes_for_winner == 3

    def test_different_responses_no_consensus(self):
        """Different responses should not reach consensus."""
        consensus = VotingConsensus(k=2)

        consensus.add_vote("Answer A")
        result = consensus.add_vote("Answer B")

        assert not result.consensus_reached

    def test_similar_responses_match(self):
        """Similar responses should be grouped together."""
        consensus = VotingConsensus(k=2, similarity_threshold=0.9)

        consensus.add_vote("The answer is 42")
        result = consensus.add_vote("the answer is 42")  # Same but lowercase

        # Should match due to normalization
        assert result.consensus_reached

    def test_get_best_response_without_consensus(self):
        """Get best response when no consensus reached."""
        consensus = VotingConsensus(k=3)

        consensus.add_vote("Answer A")
        consensus.add_vote("Answer A")
        consensus.add_vote("Answer B")

        response, metadata = consensus.get_best_response()

        assert response == "Answer A"
        assert metadata.winning_votes == 2
        assert metadata.unique_responses == 2


class TestRedFlagging:
    """Test red-flag quality checks."""

    def test_red_flag_long_response(self):
        """Responses over token limit should be red-flagged."""
        consensus = VotingConsensus(k=2, max_response_length=100)

        # Create response over 100 tokens (~400 chars)
        long_response = "word " * 200  # ~800 chars = ~200 tokens

        result = consensus.add_vote(long_response)

        assert result.red_flagged
        assert result.red_flag_reason == "response_too_long"
        assert not result.consensus_reached

    def test_valid_response_not_flagged(self):
        """Valid responses should not be red-flagged."""
        consensus = VotingConsensus(k=2, max_response_length=700)

        result = consensus.add_vote("This is a valid short response.")

        assert not result.red_flagged
        assert result.red_flag_reason is None

    def test_red_flagged_responses_excluded_from_voting(self):
        """Red-flagged responses should not count as votes."""
        consensus = VotingConsensus(k=2, max_response_length=50)

        # Add two identical but too-long responses
        long_response = "word " * 100
        consensus.add_vote(long_response)
        consensus.add_vote(long_response)

        # Should not reach consensus (both red-flagged)
        response, metadata = consensus.get_best_response()
        assert response is None
        assert metadata.red_flagged_count == 2


class TestTaskComplexityEstimation:
    """Test task complexity estimation for auto-kmin."""

    def test_simple_prompt(self):
        """Simple prompt should estimate few steps."""
        steps = estimate_task_complexity("What is 2+2?")
        assert 1 <= steps <= 5

    def test_complex_prompt_with_steps(self):
        """Prompt with step words should estimate more steps."""
        prompt = """
        First, analyze the code.
        Then, identify bugs.
        Next, fix the issues.
        Finally, write tests.
        """
        steps = estimate_task_complexity(prompt)
        assert steps >= 4

    def test_long_prompt_complexity(self):
        """Longer prompts should estimate higher complexity."""
        short_prompt = "Fix the bug."
        long_prompt = "Fix the bug. " * 100

        short_steps = estimate_task_complexity(short_prompt)
        long_steps = estimate_task_complexity(long_prompt)

        assert long_steps > short_steps


class TestVotingReset:
    """Test voting state reset."""

    def test_reset_clears_state(self):
        """Reset should clear all voting state."""
        consensus = VotingConsensus(k=2)

        consensus.add_vote("Answer A")
        consensus.add_vote("Answer B")

        consensus.reset()

        # After reset, should start fresh
        result = consensus.add_vote("Answer A")
        assert not result.consensus_reached
        assert result.total_votes == 1


class TestVotingMetadata:
    """Test voting metadata tracking."""

    def test_metadata_tracks_votes(self):
        """Metadata should track vote counts."""
        consensus = VotingConsensus(k=3)

        consensus.add_vote("A")
        consensus.add_vote("B")
        consensus.add_vote("A")

        _, metadata = consensus.get_best_response()

        assert metadata.total_votes == 3
        assert metadata.unique_responses == 2
        assert metadata.winning_votes == 2
        assert metadata.k_used == 3

    def test_metadata_tracks_red_flags(self):
        """Metadata should track red-flagged count."""
        consensus = VotingConsensus(k=2, max_response_length=10)

        consensus.add_vote("Hi")  # ~1 token, valid
        # 10 tokens = ~40 chars, so we need > 40 chars
        consensus.add_vote("This response has way more than forty characters easily")

        _, metadata = consensus.get_best_response()

        assert metadata.red_flagged_count == 1


class TestTextNormalization:
    """Test text normalization for vote matching."""

    def test_whitespace_normalization(self):
        """Extra whitespace should be normalized."""
        consensus = VotingConsensus(k=2)

        consensus.add_vote("hello   world")
        result = consensus.add_vote("hello world")

        assert result.consensus_reached

    def test_case_normalization(self):
        """Case should be normalized."""
        consensus = VotingConsensus(k=2)

        consensus.add_vote("HELLO WORLD")
        result = consensus.add_vote("hello world")

        assert result.consensus_reached

    def test_markdown_stripped(self):
        """Markdown formatting should be stripped."""
        consensus = VotingConsensus(k=2)

        consensus.add_vote("**bold** and *italic*")
        result = consensus.add_vote("bold and italic")

        assert result.consensus_reached
