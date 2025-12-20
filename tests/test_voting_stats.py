"""Tests for VotingStatsTracker.

Covers:
- Consensus recording and metrics
- Rejection tracking
- Tier statistics
- Persistence
- Percentile calculations (verified with Wolfram Alpha)
"""

import json
import math
import pytest
import tempfile
from pathlib import Path

from delia.voting_stats import (
    VotingStatsTracker,
    ConsensusStats,
    RejectionStats,
    TierStats,
    RejectionRecord,
    get_voting_stats_tracker,
    reset_voting_stats_tracker,
)


class TestConsensusStats:
    """Test ConsensusStats calculations."""

    def test_initial_state(self):
        """Fresh stats should have zeros."""
        stats = ConsensusStats()
        assert stats.total_attempts == 0
        assert stats.successful == 0
        assert stats.failed == 0
        assert stats.consensus_rate == 0.0
        assert stats.avg_votes == 0.0

    def test_record_success(self):
        """Recording success should update counters."""
        stats = ConsensusStats()
        stats.record(votes_needed=2, success=True)

        assert stats.total_attempts == 1
        assert stats.successful == 1
        assert stats.failed == 0
        assert stats.consensus_rate == 1.0
        assert stats.avg_votes == 2.0

    def test_record_failure(self):
        """Recording failure should update counters."""
        stats = ConsensusStats()
        stats.record(votes_needed=3, success=False)

        assert stats.total_attempts == 1
        assert stats.successful == 0
        assert stats.failed == 1
        assert stats.consensus_rate == 0.0

    def test_votes_histogram(self):
        """Votes should be tracked in histogram."""
        stats = ConsensusStats()
        stats.record(votes_needed=2, success=True)
        stats.record(votes_needed=2, success=True)
        stats.record(votes_needed=3, success=True)

        assert stats.votes_histogram["2"] == 2
        assert stats.votes_histogram["3"] == 1

    def test_percentile_p50(self):
        """p50 calculation (verified with Wolfram Alpha)."""
        stats = ConsensusStats()
        # Add 10 samples: [2,2,2,2,2,3,3,3,4,4]
        for _ in range(5):
            stats.record(2, True)
        for _ in range(3):
            stats.record(3, True)
        for _ in range(2):
            stats.record(4, True)

        # p50 for n=10: index = ceil(0.5*10)-1 = 4 (0-indexed)
        # Sorted: [2,2,2,2,2,3,3,3,4,4], index 4 = 2
        p50 = stats.percentile(0.5)
        assert p50 == 2

    def test_percentile_p95(self):
        """p95 calculation (verified with Wolfram Alpha)."""
        stats = ConsensusStats()
        # Add 100 samples: mostly 2s with some 3s and 4s
        for _ in range(80):
            stats.record(2, True)
        for _ in range(15):
            stats.record(3, True)
        for _ in range(5):
            stats.record(4, True)

        # p95 for n=100: index = ceil(0.95*100)-1 = 94
        # Should be in the 3s range
        p95 = stats.percentile(0.95)
        assert p95 == 3

    def test_percentile_empty(self):
        """Percentile of empty list should be 0."""
        stats = ConsensusStats()
        assert stats.percentile(0.5) == 0

    def test_rolling_window(self):
        """Samples should be limited to max_samples."""
        stats = ConsensusStats()
        stats.max_samples = 100

        # Add more than max samples
        for i in range(150):
            stats.record(2, True)

        assert len(stats.votes_samples) == 100

    def test_serialization_roundtrip(self):
        """Stats should survive JSON roundtrip."""
        stats = ConsensusStats()
        stats.record(2, True)
        stats.record(3, True)
        stats.record(4, False)

        data = stats.to_dict()
        restored = ConsensusStats.from_dict(data)

        assert restored.total_attempts == stats.total_attempts
        assert restored.successful == stats.successful
        assert restored.failed == stats.failed
        assert restored.votes_histogram == stats.votes_histogram


class TestRejectionStats:
    """Test RejectionStats tracking."""

    def test_initial_state(self):
        """Fresh stats should have zeros."""
        stats = RejectionStats()
        assert stats.total == 0
        assert len(stats.by_reason) == 0

    def test_record_rejection(self):
        """Recording rejection should update all counters."""
        stats = RejectionStats()
        stats.record(
            reason="too_long",
            backend_id="ollama-local",
            tier="coder",
            response_preview="This is a test...",
        )

        assert stats.total == 1
        assert stats.by_reason["too_long"] == 1
        assert stats.by_backend["ollama-local"] == 1
        assert stats.by_tier["coder"] == 1
        assert len(stats.recent) == 1

    def test_multiple_rejections(self):
        """Multiple rejections should accumulate."""
        stats = RejectionStats()
        stats.record("too_long", "backend1", "quick", "...")
        stats.record("too_long", "backend1", "coder", "...")
        stats.record("repetitive", "backend2", "quick", "...")

        assert stats.total == 3
        assert stats.by_reason["too_long"] == 2
        assert stats.by_reason["repetitive"] == 1
        assert stats.by_backend["backend1"] == 2
        assert stats.by_backend["backend2"] == 1

    def test_recent_limit(self):
        """Recent rejections should be limited to 100."""
        stats = RejectionStats()
        for i in range(150):
            stats.record(f"reason_{i}", "backend", "tier", "...")

        assert len(stats.recent) == 100

    def test_preview_truncation(self):
        """Response preview should be truncated to 100 chars."""
        stats = RejectionStats()
        long_response = "x" * 500
        stats.record("too_long", "backend", "tier", long_response)

        assert len(stats.recent[0].response_preview) == 100


class TestTierStats:
    """Test per-tier statistics."""

    def test_initial_state(self):
        """Fresh stats should have neutral values."""
        stats = TierStats()
        assert stats.calls == 0
        assert stats.quality_ema == 0.5  # Neutral starting point

    def test_record_call(self):
        """Recording a call should update stats."""
        stats = TierStats()
        stats.record_call(quality_score=0.9)

        assert stats.calls == 1
        assert stats.quality_sum == 0.9
        assert stats.avg_quality == 0.9

    def test_ema_update(self):
        """EMA should converge toward consistent values."""
        stats = TierStats()
        stats.ema_alpha = 0.2

        # Feed constant 1.0 values
        for _ in range(10):
            stats.record_call(1.0)

        # EMA should approach 1.0 (verified with Wolfram Alpha)
        # After 10 iterations with alpha=0.2: should be > 0.89
        assert stats.quality_ema > 0.89

    def test_rejection_rate(self):
        """Rejection rate calculation."""
        stats = TierStats()
        stats.record_call(0.9)
        stats.record_call(0.8)
        stats.record_rejection()

        assert stats.calls == 2
        assert stats.rejections == 1
        assert stats.rejection_rate == 0.5

    def test_consensus_rate(self):
        """Consensus rate calculation."""
        stats = TierStats()
        stats.record_consensus(True)
        stats.record_consensus(True)
        stats.record_consensus(False)

        assert stats.consensus_attempts == 3
        assert stats.consensus_successes == 2
        assert abs(stats.consensus_rate - 0.6667) < 0.01


class TestVotingStatsTracker:
    """Test the full tracker with persistence."""

    @pytest.fixture
    def temp_stats_file(self):
        """Create a temporary stats file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            yield Path(f.name)

    def test_record_consensus_success(self, temp_stats_file):
        """Recording consensus should update all relevant stats."""
        tracker = VotingStatsTracker(stats_file=temp_stats_file)

        tracker.record_consensus(
            votes_cast=2,
            k=2,
            tier="quick",
            backend_id="ollama-local",
            success=True,
        )

        stats = tracker.get_stats()
        assert stats["consensus"]["successful"] == 1
        assert stats["consensus"]["consensus_rate"] == 100.0
        assert "quick" in stats["tiers"]

    def test_record_consensus_failure(self, temp_stats_file):
        """Recording failed consensus should track it."""
        tracker = VotingStatsTracker(stats_file=temp_stats_file)

        tracker.record_consensus(
            votes_cast=5,
            k=3,
            tier="coder",
            backend_id="none",
            success=False,
        )

        stats = tracker.get_stats()
        assert stats["consensus"]["failed"] == 1
        assert stats["consensus"]["consensus_rate"] == 0.0

    def test_record_rejection(self, temp_stats_file):
        """Recording rejection should track reason and backend."""
        tracker = VotingStatsTracker(stats_file=temp_stats_file)

        tracker.record_rejection(
            reason="too_long",
            backend_id="ollama-local",
            tier="moe",
            response_preview="Very long response...",
        )

        stats = tracker.get_stats()
        assert stats["rejections"]["total"] == 1
        assert stats["rejections"]["by_reason"]["too_long"] == 1
        assert stats["rejections"]["by_tier"]["moe"] == 1

    def test_record_quality(self, temp_stats_file):
        """Recording quality should update tier stats."""
        tracker = VotingStatsTracker(stats_file=temp_stats_file)

        tracker.record_quality("quick", 0.95)
        tracker.record_quality("quick", 0.85)

        stats = tracker.get_stats()
        assert stats["tiers"]["quick"]["calls"] == 2
        assert stats["tiers"]["quick"]["avg_quality"] == 0.9

    def test_persistence(self, temp_stats_file):
        """Stats should persist across tracker instances."""
        # Create and populate first tracker
        tracker1 = VotingStatsTracker(stats_file=temp_stats_file)
        tracker1.record_consensus(2, 2, "quick", "backend1", True)
        tracker1.record_rejection("too_long", "backend1", "quick", "...")

        # Create new tracker with same file
        tracker2 = VotingStatsTracker(stats_file=temp_stats_file)
        stats = tracker2.get_stats()

        assert stats["consensus"]["successful"] == 1
        assert stats["rejections"]["total"] == 1

    def test_get_stats_format(self, temp_stats_file):
        """get_stats should return properly formatted data."""
        tracker = VotingStatsTracker(stats_file=temp_stats_file)

        # Add some data
        for _ in range(10):
            tracker.record_consensus(2, 2, "quick", "b1", True)
        for _ in range(5):
            tracker.record_consensus(3, 2, "quick", "b1", True)
        tracker.record_rejection("too_long", "b1", "quick", "...")

        stats = tracker.get_stats()

        # Check consensus format
        assert "total_attempts" in stats["consensus"]
        assert "consensus_rate" in stats["consensus"]
        assert "avg_votes" in stats["consensus"]
        assert "p50_votes" in stats["consensus"]
        assert "p95_votes" in stats["consensus"]
        assert "distribution" in stats["consensus"]

        # Check rejection format
        assert "by_reason" in stats["rejections"]
        assert "by_backend" in stats["rejections"]
        assert "recent" in stats["rejections"]

        # Check tier format
        assert "avg_quality" in stats["tiers"]["quick"]
        assert "rejection_rate" in stats["tiers"]["quick"]

    def test_distribution_format(self, temp_stats_file):
        """Distribution should show percentages by k."""
        tracker = VotingStatsTracker(stats_file=temp_stats_file)

        # 7 at k=2, 3 at k=3
        for _ in range(7):
            tracker.record_consensus(2, 2, "quick", "b1", True)
        for _ in range(3):
            tracker.record_consensus(3, 2, "quick", "b1", True)

        stats = tracker.get_stats()
        dist = stats["consensus"]["distribution"]

        assert dist["k=2"] == 70.0
        assert dist["k=3"] == 30.0

    def test_reset(self, temp_stats_file):
        """Reset should clear all stats."""
        tracker = VotingStatsTracker(stats_file=temp_stats_file)
        tracker.record_consensus(2, 2, "quick", "b1", True)
        tracker.record_rejection("too_long", "b1", "quick", "...")

        tracker.reset()

        stats = tracker.get_stats()
        assert stats["consensus"]["total_attempts"] == 0
        assert stats["rejections"]["total"] == 0
        assert len(stats["tiers"]) == 0


class TestSingleton:
    """Test singleton pattern."""

    def test_get_voting_stats_tracker_returns_same_instance(self):
        """Singleton should return same instance."""
        reset_voting_stats_tracker()

        tracker1 = get_voting_stats_tracker()
        tracker2 = get_voting_stats_tracker()

        assert tracker1 is tracker2

    def test_reset_clears_singleton(self):
        """Reset should clear the singleton."""
        tracker1 = get_voting_stats_tracker()
        reset_voting_stats_tracker()
        tracker2 = get_voting_stats_tracker()

        assert tracker1 is not tracker2


class TestMathVerification:
    """Verify math calculations match Wolfram Alpha results."""

    def test_ema_convergence_wolfram(self):
        """EMA after 4 values of 1.0 with alpha=0.2 should be 0.5904."""
        stats = TierStats()
        stats.ema_alpha = 0.2
        stats.quality_ema = 0.0  # Start from 0

        for _ in range(4):
            stats.record_call(1.0)

        # Wolfram: 0.2*1 + 0.8*(0.2*1 + 0.8*(0.2*1 + 0.8*(0.2*1 + 0.8*0))) = 0.5904
        assert abs(stats.quality_ema - 0.5904) < 0.0001

    def test_percentile_index_formula_wolfram(self):
        """Percentile index formula verified with Wolfram."""
        # For p95 with n=100: ceil(0.95*100) - 1 = 94
        n = 100
        p = 0.95
        idx = math.ceil(p * n) - 1
        assert idx == 94

        # For p50 with n=10: ceil(0.5*10) - 1 = 4
        n = 10
        p = 0.5
        idx = math.ceil(p * n) - 1
        assert idx == 4
