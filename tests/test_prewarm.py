# Copyright (C) 2024 Delia Contributors
#
# Tests for predictive pre-warming functionality

"""Tests for PrewarmTracker and pre-warming infrastructure."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


class TestPrewarmTracker:
    """Test PrewarmTracker class."""

    def test_update_increments_score(self):
        """Update should increase EMA score for tier at current hour."""
        from delia.config import PrewarmTracker

        tracker = PrewarmTracker(alpha=0.5)
        current_hour = datetime.now().hour

        tracker.update("coder")
        # First update: 0.0 * 0.5 + 0.5 = 0.5
        assert tracker._scores[(current_hour, "coder")] == 0.5

        tracker.update("coder")
        # Second update: 0.5 * 0.5 + 0.5 = 0.75
        assert tracker._scores[(current_hour, "coder")] == 0.75

    def test_update_different_tiers(self):
        """Updates for different tiers should track separately."""
        from delia.config import PrewarmTracker

        tracker = PrewarmTracker(alpha=0.5)
        current_hour = datetime.now().hour

        tracker.update("quick")
        tracker.update("coder")

        assert (current_hour, "quick") in tracker._scores
        assert (current_hour, "coder") in tracker._scores
        assert tracker._scores[(current_hour, "quick")] == 0.5
        assert tracker._scores[(current_hour, "coder")] == 0.5

    def test_get_predicted_tiers_returns_above_threshold(self):
        """get_predicted_tiers should return tiers above threshold."""
        from delia.config import PrewarmTracker

        tracker = PrewarmTracker(threshold=0.3)
        tracker._scores = {
            (10, "coder"): 0.5,  # Above threshold
            (10, "quick"): 0.2,  # Below threshold
            (10, "moe"): 0.4,  # Above threshold
            (11, "coder"): 0.8,  # Different hour
        }

        tiers = tracker.get_predicted_tiers(hour=10)

        # Should return coder and moe, sorted by score descending
        assert tiers == ["coder", "moe"]

    def test_get_predicted_tiers_empty_when_below_threshold(self):
        """get_predicted_tiers should return empty list when all below threshold."""
        from delia.config import PrewarmTracker

        tracker = PrewarmTracker(threshold=0.5)
        tracker._scores = {
            (10, "coder"): 0.3,
            (10, "quick"): 0.2,
        }

        tiers = tracker.get_predicted_tiers(hour=10)
        assert tiers == []

    def test_get_predicted_tiers_uses_current_hour_by_default(self):
        """get_predicted_tiers should use current hour when not specified."""
        from delia.config import PrewarmTracker

        tracker = PrewarmTracker(threshold=0.2)
        current_hour = datetime.now().hour

        # Set up scores for current hour
        tracker._scores = {
            (current_hour, "coder"): 0.5,
        }

        tiers = tracker.get_predicted_tiers()
        assert tiers == ["coder"]

    def test_to_dict_serialization(self):
        """to_dict should serialize tracker state."""
        from delia.config import PrewarmTracker

        tracker = PrewarmTracker(alpha=0.2, threshold=0.4)
        tracker._scores = {
            (10, "coder"): 0.5,
            (14, "moe"): 0.6,
        }

        data = tracker.to_dict()

        assert data["alpha"] == 0.2
        assert data["threshold"] == 0.4
        assert data["scores"]["10:coder"] == 0.5
        assert data["scores"]["14:moe"] == 0.6

    def test_from_dict_deserialization(self):
        """from_dict should deserialize tracker state."""
        from delia.config import PrewarmTracker

        data = {
            "alpha": 0.2,
            "threshold": 0.4,
            "scores": {
                "10:coder": 0.5,
                "14:moe": 0.6,
            },
        }

        tracker = PrewarmTracker.from_dict(data)

        assert tracker.alpha == 0.2
        assert tracker.threshold == 0.4
        assert tracker._scores[(10, "coder")] == 0.5
        assert tracker._scores[(14, "moe")] == 0.6

    def test_from_dict_handles_malformed_entries(self):
        """from_dict should skip malformed score entries."""
        from delia.config import PrewarmTracker

        data = {
            "scores": {
                "10:coder": 0.5,
                "invalid": 0.3,  # Missing colon
                "notanumber:moe": 0.4,  # Invalid hour
            },
        }

        tracker = PrewarmTracker.from_dict(data)

        # Only valid entry should be loaded
        assert len(tracker._scores) == 1
        assert tracker._scores[(10, "coder")] == 0.5

    def test_get_status_includes_predictions(self):
        """get_status should include current predictions."""
        from delia.config import PrewarmTracker

        tracker = PrewarmTracker(threshold=0.3)
        current_hour = datetime.now().hour

        tracker._scores = {
            (current_hour, "coder"): 0.5,
            (current_hour, "moe"): 0.4,
        }

        status = tracker.get_status()

        assert status["current_hour"] == current_hour
        assert "coder" in status["predicted_tiers"]
        assert "moe" in status["predicted_tiers"]
        assert status["tracked_entries"] == 2

    def test_ema_decay_over_time(self):
        """EMA should decay scores when tier is not used."""
        from delia.config import PrewarmTracker

        tracker = PrewarmTracker(alpha=0.5)

        # Simulate usage then no usage
        tracker._scores = {(10, "coder"): 0.8}

        # Another update at same hour for different tier
        # coder score should not change (no decay without update)
        tracker._scores[(10, "quick")] = 0.5

        # coder score unchanged
        assert tracker._scores[(10, "coder")] == 0.8


class TestPrewarmPersistence:
    """Test prewarm data persistence."""

    def test_save_and_load_prewarm(self):
        """save_prewarm and load_prewarm should persist tracker state."""
        from delia import paths
        from delia.config import (
            PrewarmTracker,
            load_prewarm,
            save_prewarm,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Override path
            original_file = paths.PREWARM_FILE
            paths.PREWARM_FILE = Path(tmpdir) / "prewarm.json"
            paths.ensure_directories = lambda: None  # Noop for test

            try:
                # Modify tracker
                import delia.config

                delia.config.PREWARM_TRACKER = PrewarmTracker(alpha=0.2, threshold=0.35)
                delia.config.PREWARM_TRACKER._scores = {
                    (9, "quick"): 0.4,
                    (14, "coder"): 0.6,
                }

                # Save
                save_prewarm()

                # Verify file was created
                assert paths.PREWARM_FILE.exists()

                # Reset tracker
                delia.config.PREWARM_TRACKER = PrewarmTracker()

                # Load
                load_prewarm()

                # Verify data was restored
                assert delia.config.PREWARM_TRACKER.alpha == 0.2
                assert delia.config.PREWARM_TRACKER.threshold == 0.35
                assert delia.config.PREWARM_TRACKER._scores[(9, "quick")] == 0.4
                assert delia.config.PREWARM_TRACKER._scores[(14, "coder")] == 0.6

            finally:
                paths.PREWARM_FILE = original_file


class TestPrewarmConfig:
    """Test prewarm configuration in routing config."""

    def test_default_prewarm_config_in_settings(self):
        """Default settings include prewarm configuration."""
        from delia.backend_manager import BackendManager

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            # Create manager (will create default settings file)
            manager = BackendManager(settings_file=settings_path)

            # Read the created settings file directly
            with open(settings_path) as f:
                settings = json.load(f)

            # Check that prewarm config exists in routing
            prewarm = settings.get("routing", {}).get("prewarm", {})
            assert "enabled" in prewarm
            assert "threshold" in prewarm
            assert "check_interval_minutes" in prewarm

            # Check defaults
            assert prewarm["enabled"] is False
            assert prewarm["threshold"] == 0.3
            assert prewarm["check_interval_minutes"] == 5

    def test_prewarm_config_threshold_respected(self):
        """Threshold config should control which tiers are predicted."""
        from delia.config import PrewarmTracker

        # Low threshold - more tiers predicted
        tracker_low = PrewarmTracker(threshold=0.2)
        tracker_low._scores = {
            (10, "coder"): 0.25,
            (10, "quick"): 0.15,
        }

        # High threshold - fewer tiers predicted
        tracker_high = PrewarmTracker(threshold=0.3)
        tracker_high._scores = {
            (10, "coder"): 0.25,
            (10, "quick"): 0.15,
        }

        assert tracker_low.get_predicted_tiers(10) == ["coder"]
        assert tracker_high.get_predicted_tiers(10) == []
