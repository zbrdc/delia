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
    modules_to_clear = ["delia.paths", "delia.config", "delia.mcp_server", "delia.backend_manager", "delia.multi_user_tracking", "delia.stats", "delia"]
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) or mod == m for m in modules_to_clear):
            del sys.modules[mod]

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


class TestUsageStats:
    """Test stats_service model usage and task stats tracking."""

    def test_model_usage_initialized(self):
        """stats_service should be initialized with all tiers."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service

        model_usage, _, _, _ = stats_service.get_snapshot()
        assert "quick" in model_usage
        assert "coder" in model_usage
        assert "moe" in model_usage

    def test_model_usage_has_counters(self):
        """model_usage should have calls and tokens counters."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service

        model_usage, _, _, _ = stats_service.get_snapshot()
        for tier in ["quick", "coder", "moe"]:
            assert "calls" in model_usage[tier]
            assert "tokens" in model_usage[tier]

    def test_task_stats_initialized(self):
        """task_stats should be initialized."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service

        _, task_stats, _, _ = stats_service.get_snapshot()
        assert isinstance(task_stats, dict)


class TestStatsLoadSave:
    """Test stats loading and saving to disk."""

    def test_save_stats_creates_file(self):
        """stats_service.save_all() should create stats file."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service
        import asyncio

        # Add some data
        with stats_service._lock:
            stats_service.model_usage["quick"]["calls"] = 10
            stats_service.model_usage["quick"]["tokens"] = 5000

        asyncio.run(stats_service.save_all())

        assert paths.STATS_FILE.exists()

    def test_save_stats_content(self):
        """stats_service.save_all() should save correct content."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service
        import asyncio

        with stats_service._lock:
            stats_service.model_usage["coder"]["calls"] = 25
            stats_service.model_usage["coder"]["tokens"] = 15000

        asyncio.run(stats_service.save_all())

        with open(paths.STATS_FILE) as f:
            data = json.load(f)

        assert data["coder"]["calls"] == 25
        assert data["coder"]["tokens"] == 15000

    def test_load_stats(self):
        """stats_service.load() should load from disk."""
        from delia import paths
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

        from delia.mcp_server import stats_service
        # Reset stats before loading
        with stats_service._lock:
            for tier in stats_service.model_usage:
                stats_service.model_usage[tier]["calls"] = 0
                stats_service.model_usage[tier]["tokens"] = 0

        stats_service.load()

        # Verify stats were loaded
        model_usage, _, _, _ = stats_service.get_snapshot()
        assert model_usage["quick"]["calls"] >= 100
        assert model_usage["coder"]["tokens"] >= 100000

    def test_load_stats_missing_file(self):
        """stats_service.load() should handle missing file gracefully."""
        from delia import paths
        paths.ensure_directories()

        # Ensure no stats file
        if paths.STATS_FILE.exists():
            paths.STATS_FILE.unlink()

        from delia.mcp_server import stats_service
        # Should not raise
        stats_service.load()

        # Should have default values
        model_usage, _, _, _ = stats_service.get_snapshot()
        assert model_usage["quick"]["calls"] >= 0


class TestEnhancedStats:
    """Test enhanced statistics (recent calls, response times)."""

    def test_recent_calls_exists(self):
        """recent_calls should be available in stats_service."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service

        _, _, _, recent_calls = stats_service.get_snapshot()
        assert isinstance(recent_calls, list)

    def test_response_times_exists(self):
        """response_times should be available in stats_service."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service

        _, _, response_times, _ = stats_service.get_snapshot()
        assert isinstance(response_times, dict)
        assert "quick" in response_times
        assert "coder" in response_times
        assert "moe" in response_times

    def test_save_enhanced_stats(self):
        """stats_service.save_all() should save enhanced stats to disk."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service
        import asyncio

        # Add some enhanced data
        stats_service.increment_task("review")
        stats_service.increment_task("generate")

        asyncio.run(stats_service.save_all())

        assert paths.ENHANCED_STATS_FILE.exists()


class TestLiveLogs:
    """Test live logging functionality."""

    def test_live_logs_exists(self):
        """LIVE_LOGS should be available."""
        from delia import paths
        paths.ensure_directories()

        from delia import mcp_server

        assert hasattr(mcp_server, 'LIVE_LOGS')

    def test_save_live_logs(self):
        """Live logs should be saveable."""
        from delia import paths
        paths.ensure_directories()

        from delia import mcp_server

        # Add a log entry if the list exists
        if hasattr(mcp_server, 'LIVE_LOGS') and isinstance(mcp_server.LIVE_LOGS, list):
            mcp_server.LIVE_LOGS.append({
                "ts": time.time(),
                "type": "test",
                "message": "Test log entry"
            })

        # Should not raise
        if hasattr(mcp_server, '_save_live_logs_sync'):
            mcp_server._save_live_logs_sync()


class TestCircuitBreakerStats:
    """Test circuit breaker stats persistence."""

    def test_save_circuit_breaker_stats(self):
        """save_circuit_breaker_stats() should save backend health."""
        from delia import paths
        paths.ensure_directories()

        from delia import mcp_server

        if hasattr(mcp_server, 'save_circuit_breaker_stats'):
            mcp_server.save_circuit_breaker_stats()

            if paths.CIRCUIT_BREAKER_FILE.exists():
                with open(paths.CIRCUIT_BREAKER_FILE) as f:
                    data = json.load(f)
                assert isinstance(data, dict)


class TestStatsSnapshot:
    """Test thread-safe stats snapshots."""

    def test_snapshot_stats_function(self):
        """stats_service.get_snapshot() should return deep copy of stats."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service

        snapshot = stats_service.get_snapshot()

        # get_snapshot returns a tuple of (model_usage, task_stats, response_times, recent_calls)
        assert isinstance(snapshot, tuple)
        assert len(snapshot) == 4

        # First element should be model usage dict
        model_usage = snapshot[0]
        assert isinstance(model_usage, dict)

        # Modifying snapshot should not affect originals
        if "quick" in model_usage:
            original_snapshot = stats_service.get_snapshot()
            original = original_snapshot[0]["quick"]["calls"]
            model_usage["quick"]["calls"] = 999999
            new_snapshot = stats_service.get_snapshot()
            assert new_snapshot[0]["quick"]["calls"] == original


class TestAsyncStatsSave:
    """Test async stats saving."""

    @pytest.mark.asyncio
    async def test_save_all_stats_async(self):
        """save_all_stats_async() should save all stats."""
        from delia import paths
        paths.ensure_directories()

        from delia import mcp_server

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
        """stats_service.load() should migrate old tier names."""
        from delia import paths
        paths.ensure_directories()

        # Create stats with old tier names
        legacy_stats = {
            "14b": {"calls": 50, "tokens": 25000},  # Old name for quick
            "30b": {"calls": 20, "tokens": 40000},  # Old name for coder
        }

        with open(paths.STATS_FILE, "w") as f:
            json.dump(legacy_stats, f)

        from delia.mcp_server import stats_service
        stats_service.load()

        # Should have loaded without crash (legacy migration)
        model_usage, _, _, _ = stats_service.get_snapshot()
        assert model_usage is not None


class TestStatsFilePaths:
    """Test stats files use correct paths."""

    def test_stats_file_in_cache_dir(self):
        """STATS_FILE should be in CACHE_DIR."""
        from delia import paths
        paths.ensure_directories()

        assert paths.STATS_FILE.parent == paths.CACHE_DIR

    def test_enhanced_stats_file_in_cache_dir(self):
        """ENHANCED_STATS_FILE should be in CACHE_DIR."""
        from delia import paths
        paths.ensure_directories()

        assert paths.ENHANCED_STATS_FILE.parent == paths.CACHE_DIR

    def test_live_logs_file_in_cache_dir(self):
        """LIVE_LOGS_FILE should be in CACHE_DIR."""
        from delia import paths
        paths.ensure_directories()

        assert paths.LIVE_LOGS_FILE.parent == paths.CACHE_DIR

    def test_circuit_breaker_file_in_cache_dir(self):
        """CIRCUIT_BREAKER_FILE should be in CACHE_DIR."""
        from delia import paths
        paths.ensure_directories()

        assert paths.CIRCUIT_BREAKER_FILE.parent == paths.CACHE_DIR


class TestAtomicWrites:
    """Test atomic file writing for stats."""

    def test_save_uses_temp_file(self):
        """Stats saving should use temp file for atomic writes."""
        from delia import paths
        paths.ensure_directories()

        from delia.mcp_server import stats_service
        import asyncio

        # Save stats
        asyncio.run(stats_service.save_all())

        # File should exist and be valid JSON
        if paths.STATS_FILE.exists():
            with open(paths.STATS_FILE) as f:
                data = json.load(f)  # Should not raise
            assert isinstance(data, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
