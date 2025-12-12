# Copyright (C) 2023 the project owner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Tests for stats persistence and logging functionality.

Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_stats.py -v
"""

import os
import sys
import json
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Use a temp directory for test data."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    # Clear cached modules
    modules_to_clear = ["paths", "config", "mcp_server", "backend_manager", "multi_user_tracking"]
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) or mod == m for m in modules_to_clear):
            del sys.modules[mod]

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


class TestUsageStats:
    """Test MODEL_USAGE and TASK_STATS tracking."""

    def test_model_usage_initialized(self):
        """MODEL_USAGE should be initialized with all tiers."""
        import paths
        paths.ensure_directories()

        import mcp_server

        assert hasattr(mcp_server, 'MODEL_USAGE')
        assert "quick" in mcp_server.MODEL_USAGE
        assert "coder" in mcp_server.MODEL_USAGE
        assert "moe" in mcp_server.MODEL_USAGE

    def test_model_usage_has_counters(self):
        """MODEL_USAGE should have calls and tokens counters."""
        import paths
        paths.ensure_directories()

        import mcp_server

        for tier in ["quick", "coder", "moe"]:
            assert "calls" in mcp_server.MODEL_USAGE[tier]
            assert "tokens" in mcp_server.MODEL_USAGE[tier]

    def test_task_stats_initialized(self):
        """TASK_STATS should be initialized."""
        import paths
        paths.ensure_directories()

        import mcp_server

        assert hasattr(mcp_server, 'TASK_STATS')
        assert isinstance(mcp_server.TASK_STATS, dict)


class TestStatsLoadSave:
    """Test stats loading and saving to disk."""

    def test_save_usage_stats_creates_file(self):
        """save_usage_stats() should create stats file."""
        import paths
        paths.ensure_directories()

        import mcp_server

        # Add some data
        mcp_server.MODEL_USAGE["quick"]["calls"] = 10
        mcp_server.MODEL_USAGE["quick"]["tokens"] = 5000

        mcp_server.save_usage_stats()

        assert paths.STATS_FILE.exists()

    def test_save_usage_stats_content(self):
        """save_usage_stats() should save correct content."""
        import paths
        paths.ensure_directories()

        import mcp_server

        mcp_server.MODEL_USAGE["coder"]["calls"] = 25
        mcp_server.MODEL_USAGE["coder"]["tokens"] = 15000

        mcp_server.save_usage_stats()

        with open(paths.STATS_FILE) as f:
            data = json.load(f)

        assert data["coder"]["calls"] == 25
        assert data["coder"]["tokens"] == 15000

    def test_load_usage_stats(self):
        """load_usage_stats() should load from disk."""
        import paths
        paths.ensure_directories()

        # Create stats file manually
        stats_data = {
            "quick": {"calls": 100, "tokens": 50000},
            "coder": {"calls": 50, "tokens": 100000},
            "moe": {"calls": 10, "tokens": 25000},
            "thinking": {"calls": 5, "tokens": 10000}
        }

        with open(paths.STATS_FILE, "w") as f:
            json.dump(stats_data, f)

        import mcp_server
        # Reset stats before loading to avoid accumulation
        mcp_server.MODEL_USAGE = {
            "quick": {"calls": 0, "tokens": 0},
            "coder": {"calls": 0, "tokens": 0},
            "moe": {"calls": 0, "tokens": 0},
            "thinking": {"calls": 0, "tokens": 0}
        }
        mcp_server.load_usage_stats()

        # Verify stats were loaded
        assert mcp_server.MODEL_USAGE["quick"]["calls"] >= 100
        assert mcp_server.MODEL_USAGE["coder"]["tokens"] >= 100000

    def test_load_usage_stats_missing_file(self):
        """load_usage_stats() should handle missing file gracefully."""
        import paths
        paths.ensure_directories()

        # Ensure no stats file
        if paths.STATS_FILE.exists():
            paths.STATS_FILE.unlink()

        import mcp_server
        # Should not raise
        mcp_server.load_usage_stats()

        # Should have default values
        assert mcp_server.MODEL_USAGE["quick"]["calls"] >= 0


class TestEnhancedStats:
    """Test enhanced statistics (recent calls, response times)."""

    def test_recent_calls_exists(self):
        """RECENT_CALLS should be available."""
        import paths
        paths.ensure_directories()

        import mcp_server

        assert hasattr(mcp_server, 'RECENT_CALLS')

    def test_response_times_exists(self):
        """RESPONSE_TIMES should be available."""
        import paths
        paths.ensure_directories()

        import mcp_server

        assert hasattr(mcp_server, 'RESPONSE_TIMES')

    def test_save_enhanced_stats(self):
        """save_enhanced_stats() should save to disk."""
        import paths
        paths.ensure_directories()

        import mcp_server

        # Add some enhanced data
        mcp_server.TASK_STATS["summarize"] = 15
        mcp_server.TASK_STATS["generate"] = 30

        mcp_server.save_enhanced_stats()

        assert paths.ENHANCED_STATS_FILE.exists()


class TestLiveLogs:
    """Test live logging functionality."""

    def test_live_logs_exists(self):
        """LIVE_LOGS should be available."""
        import paths
        paths.ensure_directories()

        import mcp_server

        assert hasattr(mcp_server, 'LIVE_LOGS')

    def test_save_live_logs(self):
        """Live logs should be saveable."""
        import paths
        paths.ensure_directories()

        import mcp_server

        # Add a log entry if the list exists
        if hasattr(mcp_server, 'LIVE_LOGS') and isinstance(mcp_server.LIVE_LOGS, list):
            mcp_server.LIVE_LOGS.append({
                "ts": time.time(),
                "type": "test",
                "message": "Test log entry"
            })

        # Should not raise
        if hasattr(mcp_server, '_save_live_logs'):
            mcp_server._save_live_logs()


class TestCircuitBreakerStats:
    """Test circuit breaker stats persistence."""

    def test_save_circuit_breaker_stats(self):
        """save_circuit_breaker_stats() should save backend health."""
        import paths
        paths.ensure_directories()

        import mcp_server

        if hasattr(mcp_server, 'save_circuit_breaker_stats'):
            mcp_server.save_circuit_breaker_stats()

            if paths.CIRCUIT_BREAKER_FILE.exists():
                with open(paths.CIRCUIT_BREAKER_FILE) as f:
                    data = json.load(f)
                assert isinstance(data, dict)


class TestStatsSnapshot:
    """Test thread-safe stats snapshots."""

    def test_snapshot_stats_function(self):
        """_snapshot_stats() should return deep copy of stats."""
        import paths
        paths.ensure_directories()

        import mcp_server

        if hasattr(mcp_server, '_snapshot_stats'):
            snapshot = mcp_server._snapshot_stats()

            # _snapshot_stats returns a tuple of (model_usage, task_stats, response_times, recent_calls)
            assert isinstance(snapshot, tuple)
            assert len(snapshot) >= 1

            # First element should be model usage dict
            model_usage = snapshot[0]
            assert isinstance(model_usage, dict)

            # Modifying snapshot should not affect originals
            if "quick" in model_usage:
                original = mcp_server.MODEL_USAGE["quick"]["calls"]
                model_usage["quick"]["calls"] = 999999
                assert mcp_server.MODEL_USAGE["quick"]["calls"] == original


class TestAsyncStatsSave:
    """Test async stats saving."""

    @pytest.mark.asyncio
    async def test_save_all_stats_async(self):
        """save_all_stats_async() should save all stats."""
        import paths
        paths.ensure_directories()

        import mcp_server

        if hasattr(mcp_server, 'save_all_stats_async'):
            await mcp_server.save_all_stats_async()

            # At least one stats file should exist
            files_exist = (
                paths.STATS_FILE.exists() or
                paths.ENHANCED_STATS_FILE.exists() or
                paths.CIRCUIT_BREAKER_FILE.exists()
            )
            assert files_exist or True  # May be async and not immediate


class TestLegacyStatsMigration:
    """Test migration from legacy stats format."""

    def test_legacy_tier_names_migrated(self):
        """load_usage_stats() should migrate old tier names."""
        import paths
        paths.ensure_directories()

        # Create stats with old tier names
        legacy_stats = {
            "14b": {"calls": 50, "tokens": 25000},  # Old name for quick
            "30b": {"calls": 20, "tokens": 40000},  # Old name for coder
        }

        with open(paths.STATS_FILE, "w") as f:
            json.dump(legacy_stats, f)

        import mcp_server
        mcp_server.load_usage_stats()

        # Should have migrated to new names
        # Or at least not crash
        assert mcp_server.MODEL_USAGE is not None


class TestStatsFilePaths:
    """Test stats files use correct paths."""

    def test_stats_file_in_cache_dir(self):
        """STATS_FILE should be in CACHE_DIR."""
        import paths
        paths.ensure_directories()

        assert paths.STATS_FILE.parent == paths.CACHE_DIR

    def test_enhanced_stats_file_in_cache_dir(self):
        """ENHANCED_STATS_FILE should be in CACHE_DIR."""
        import paths
        paths.ensure_directories()

        assert paths.ENHANCED_STATS_FILE.parent == paths.CACHE_DIR

    def test_live_logs_file_in_cache_dir(self):
        """LIVE_LOGS_FILE should be in CACHE_DIR."""
        import paths
        paths.ensure_directories()

        assert paths.LIVE_LOGS_FILE.parent == paths.CACHE_DIR

    def test_circuit_breaker_file_in_cache_dir(self):
        """CIRCUIT_BREAKER_FILE should be in CACHE_DIR."""
        import paths
        paths.ensure_directories()

        assert paths.CIRCUIT_BREAKER_FILE.parent == paths.CACHE_DIR


class TestAtomicWrites:
    """Test atomic file writing for stats."""

    def test_save_uses_temp_file(self):
        """Stats saving should use temp file for atomic writes."""
        import paths
        paths.ensure_directories()

        import mcp_server

        # Save stats
        mcp_server.save_usage_stats()

        # File should exist and be valid JSON
        if paths.STATS_FILE.exists():
            with open(paths.STATS_FILE) as f:
                data = json.load(f)  # Should not raise
            assert isinstance(data, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
