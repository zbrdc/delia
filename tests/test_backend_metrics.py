# Copyright (C) 2024 Delia Contributors
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
Tests for BackendMetrics class.

Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_backend_metrics.py -v
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path):
    """Use a temp directory for test data and clear module cache."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    # Clear cached modules to ensure fresh imports
    modules_to_clear = ["delia.paths", "delia.config"]
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) or mod == m for m in modules_to_clear):
            del sys.modules[mod]

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


class TestBackendMetricsBasic:
    """Test basic BackendMetrics functionality."""

    def test_create_metrics(self):
        """Create metrics instance with backend_id."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test-backend")
        assert metrics.backend_id == "test-backend"
        assert metrics.total_requests == 0
        assert metrics.total_successes == 0
        assert metrics.total_failures == 0
        assert metrics.total_tokens == 0

    def test_record_success(self):
        """Record successful request updates counters."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        metrics.record_success(elapsed_ms=150.0, tokens=100)

        assert metrics.total_requests == 1
        assert metrics.total_successes == 1
        assert metrics.total_failures == 0
        assert metrics.total_tokens == 100
        assert metrics.sample_count == 1
        assert metrics.last_request_time > 0

    def test_record_success_without_tokens(self):
        """Record success without token count."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        metrics.record_success(elapsed_ms=100.0)

        assert metrics.total_tokens == 0
        assert metrics.total_successes == 1

    def test_record_failure(self):
        """Record failed request updates counters."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        metrics.record_failure(elapsed_ms=500.0)

        assert metrics.total_requests == 1
        assert metrics.total_successes == 0
        assert metrics.total_failures == 1
        assert metrics.sample_count == 1  # Latency still recorded

    def test_record_failure_timeout(self):
        """Record timeout failure without latency."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        metrics.record_failure(elapsed_ms=0)

        assert metrics.total_requests == 1
        assert metrics.total_failures == 1
        assert metrics.sample_count == 0  # No latency recorded for timeout

    def test_multiple_records(self):
        """Multiple records accumulate correctly."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        metrics.record_success(elapsed_ms=100, tokens=50)
        metrics.record_success(elapsed_ms=200, tokens=75)
        metrics.record_failure(elapsed_ms=300)
        metrics.record_success(elapsed_ms=150, tokens=60)

        assert metrics.total_requests == 4
        assert metrics.total_successes == 3
        assert metrics.total_failures == 1
        assert metrics.total_tokens == 185
        assert metrics.sample_count == 4


class TestBackendMetricsSuccessRate:
    """Test success rate calculation."""

    def test_success_rate_no_requests(self):
        """Success rate is 1.0 with no requests (optimistic)."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        assert metrics.success_rate == 1.0

    def test_success_rate_all_success(self):
        """Success rate is 1.0 with all successes."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        for _ in range(10):
            metrics.record_success(elapsed_ms=100)

        assert metrics.success_rate == 1.0

    def test_success_rate_all_failures(self):
        """Success rate is 0.0 with all failures."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        for _ in range(10):
            metrics.record_failure(elapsed_ms=100)

        assert metrics.success_rate == 0.0

    def test_success_rate_mixed(self):
        """Success rate calculated correctly for mixed results."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        for _ in range(7):
            metrics.record_success(elapsed_ms=100)
        for _ in range(3):
            metrics.record_failure(elapsed_ms=100)

        assert metrics.success_rate == 0.7


class TestBackendMetricsLatency:
    """Test latency percentile calculations."""

    def test_latency_p50_no_samples(self):
        """P50 is 0.0 with no samples."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        assert metrics.latency_p50 == 0.0

    def test_latency_p50_single_sample(self):
        """P50 with single sample returns that sample."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        metrics.record_success(elapsed_ms=150.0)

        assert metrics.latency_p50 == 150.0

    def test_latency_p50_multiple_samples(self):
        """P50 calculates median correctly."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        # Add samples: 100, 200, 300, 400, 500
        for ms in [100, 200, 300, 400, 500]:
            metrics.record_success(elapsed_ms=float(ms))

        # Median of [100, 200, 300, 400, 500] is 300 (index 2)
        assert metrics.latency_p50 == 300.0

    def test_latency_p50_even_samples(self):
        """P50 with even number of samples."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        for ms in [100, 200, 300, 400]:
            metrics.record_success(elapsed_ms=float(ms))

        # With 4 samples, mid index is 2, value is 300
        assert metrics.latency_p50 == 300.0

    def test_latency_p95_no_samples(self):
        """P95 is 0.0 with no samples."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        assert metrics.latency_p95 == 0.0

    def test_latency_p95_calculation(self):
        """P95 calculated correctly."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        # Add 20 samples from 100 to 2000
        for i in range(20):
            metrics.record_success(elapsed_ms=float((i + 1) * 100))

        # 95th percentile of 20 samples is index 19 (0.95 * 20 = 19)
        assert metrics.latency_p95 == 2000.0


class TestBackendMetricsThroughput:
    """Test throughput (tokens per second) calculation."""

    def test_tokens_per_second_no_data(self):
        """TPS is 0.0 with no data."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        assert metrics.tokens_per_second == 0.0

    def test_tokens_per_second_no_tokens(self):
        """TPS is 0.0 when no tokens recorded."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        metrics.record_success(elapsed_ms=100.0, tokens=0)

        assert metrics.tokens_per_second == 0.0

    def test_tokens_per_second_calculation(self):
        """TPS calculated correctly."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        # 100 tokens in 100ms = 1000 tokens/sec
        metrics.record_success(elapsed_ms=100.0, tokens=100)

        assert metrics.tokens_per_second == 1000.0

    def test_tokens_per_second_multiple_requests(self):
        """TPS calculated from averages."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        # Request 1: 100 tokens in 100ms
        # Request 2: 200 tokens in 200ms
        # Average: 150 tokens / 150ms = 1000 tokens/sec
        metrics.record_success(elapsed_ms=100.0, tokens=100)
        metrics.record_success(elapsed_ms=200.0, tokens=200)

        assert metrics.tokens_per_second == 1000.0


class TestBackendMetricsRollingWindow:
    """Test rolling window behavior for latency samples."""

    def test_rolling_window_limit(self):
        """Latency samples limited to window size."""
        from delia.config import BackendMetrics, _METRICS_WINDOW_SIZE

        metrics = BackendMetrics(backend_id="test")

        # Add more than window size
        for i in range(_METRICS_WINDOW_SIZE + 50):
            metrics.record_success(elapsed_ms=float(i))

        assert metrics.sample_count == _METRICS_WINDOW_SIZE

    def test_rolling_window_fifo(self):
        """Oldest samples removed when window full (FIFO)."""
        from delia.config import BackendMetrics, _METRICS_WINDOW_SIZE

        metrics = BackendMetrics(backend_id="test")

        # Fill window with 0s, then add one more
        for _ in range(_METRICS_WINDOW_SIZE):
            metrics.record_success(elapsed_ms=0.0)
        metrics.record_success(elapsed_ms=999.0)

        # The last sample should be in the window
        assert 999.0 in list(metrics._latency_samples)


class TestBackendMetricsSerialization:
    """Test serialization and deserialization."""

    def test_to_dict(self):
        """Serialize metrics to dict."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test-backend")
        metrics.record_success(elapsed_ms=150.0, tokens=100)
        metrics.record_failure(elapsed_ms=200.0)

        data = metrics.to_dict()

        assert data["backend_id"] == "test-backend"
        assert data["total_requests"] == 2
        assert data["total_successes"] == 1
        assert data["total_failures"] == 1
        assert data["total_tokens"] == 100
        assert len(data["latency_samples"]) == 2

    def test_from_dict(self):
        """Deserialize metrics from dict."""
        from delia.config import BackendMetrics

        data = {
            "backend_id": "restored-backend",
            "latency_samples": [100.0, 200.0, 300.0],
            "total_requests": 10,
            "total_successes": 8,
            "total_failures": 2,
            "total_tokens": 500,
            "last_request_time": 1234567890.0,
        }

        metrics = BackendMetrics.from_dict(data)

        assert metrics.backend_id == "restored-backend"
        assert metrics.total_requests == 10
        assert metrics.total_successes == 8
        assert metrics.total_failures == 2
        assert metrics.total_tokens == 500
        assert metrics.sample_count == 3
        assert metrics.success_rate == 0.8

    def test_roundtrip(self):
        """Serialize and deserialize preserves all data."""
        from delia.config import BackendMetrics

        original = BackendMetrics(backend_id="test")
        original.record_success(elapsed_ms=100.0, tokens=50)
        original.record_success(elapsed_ms=200.0, tokens=75)
        original.record_failure(elapsed_ms=300.0)

        data = original.to_dict()
        restored = BackendMetrics.from_dict(data)

        assert restored.backend_id == original.backend_id
        assert restored.total_requests == original.total_requests
        assert restored.total_successes == original.total_successes
        assert restored.total_failures == original.total_failures
        assert restored.total_tokens == original.total_tokens
        assert restored.sample_count == original.sample_count
        assert restored.success_rate == original.success_rate

    def test_from_dict_missing_fields(self):
        """Handle missing fields gracefully."""
        from delia.config import BackendMetrics

        data = {"backend_id": "partial"}
        metrics = BackendMetrics.from_dict(data)

        assert metrics.backend_id == "partial"
        assert metrics.total_requests == 0
        assert metrics.total_successes == 0
        assert metrics.sample_count == 0


class TestBackendMetricsStatus:
    """Test get_status() method."""

    def test_get_status(self):
        """Get status returns correct structure."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        metrics.record_success(elapsed_ms=150.0, tokens=100)

        status = metrics.get_status()

        assert status["backend_id"] == "test"
        assert "success_rate" in status
        assert "latency_p50_ms" in status
        assert "latency_p95_ms" in status
        assert "tokens_per_second" in status
        assert "total_requests" in status
        assert "sample_count" in status

    def test_get_status_values_rounded(self):
        """Status values are rounded appropriately."""
        from delia.config import BackendMetrics

        metrics = BackendMetrics(backend_id="test")
        metrics.record_success(elapsed_ms=123.456789, tokens=100)

        status = metrics.get_status()

        # Latency rounded to 1 decimal
        assert status["latency_p50_ms"] == 123.5
        # Success rate rounded to 3 decimals
        assert status["success_rate"] == 1.0


class TestBackendMetricsGlobalState:
    """Test global BACKEND_METRICS dict and getter."""

    def test_get_backend_metrics_creates_new(self):
        """get_backend_metrics creates new instance if not exists."""
        from delia.config import BACKEND_METRICS, get_backend_metrics

        BACKEND_METRICS.clear()

        metrics = get_backend_metrics("new-backend")

        assert metrics.backend_id == "new-backend"
        assert "new-backend" in BACKEND_METRICS

    def test_get_backend_metrics_returns_existing(self):
        """get_backend_metrics returns existing instance."""
        from delia.config import BACKEND_METRICS, get_backend_metrics

        BACKEND_METRICS.clear()

        metrics1 = get_backend_metrics("backend-1")
        metrics1.record_success(elapsed_ms=100.0)

        metrics2 = get_backend_metrics("backend-1")

        assert metrics2 is metrics1
        assert metrics2.total_requests == 1


class TestBackendMetricsPersistence:
    """Test persistence (load/save) functionality."""

    def test_save_and_load(self, tmp_path: Path):
        """Save and load metrics preserves data."""
        from delia.config import (
            BACKEND_METRICS,
            get_backend_metrics,
            load_backend_metrics,
            save_backend_metrics,
        )

        BACKEND_METRICS.clear()

        # Record some metrics
        metrics = get_backend_metrics("persist-test")
        metrics.record_success(elapsed_ms=150.0, tokens=100)
        metrics.record_failure(elapsed_ms=200.0)

        # Save
        save_backend_metrics()

        # Clear and reload
        BACKEND_METRICS.clear()
        load_backend_metrics()

        # Verify
        restored = BACKEND_METRICS.get("persist-test")
        assert restored is not None
        assert restored.total_requests == 2
        assert restored.total_successes == 1
        assert restored.total_failures == 1
        assert restored.total_tokens == 100

    def test_load_missing_file(self, tmp_path: Path):
        """Load handles missing file gracefully."""
        from delia import paths
        from delia.config import BACKEND_METRICS, load_backend_metrics

        BACKEND_METRICS.clear()

        # Ensure file doesn't exist (explicit cleanup for test isolation)
        if paths.BACKEND_METRICS_FILE.exists():
            paths.BACKEND_METRICS_FILE.unlink()

        # Should not raise
        load_backend_metrics()

        assert len(BACKEND_METRICS) == 0

    def test_load_corrupt_file(self, tmp_path: Path):
        """Load handles corrupt file gracefully."""
        from delia import paths
        from delia.config import BACKEND_METRICS, load_backend_metrics

        paths.ensure_directories()
        BACKEND_METRICS.clear()

        # Write corrupt JSON
        paths.BACKEND_METRICS_FILE.write_text("not valid json {{{")

        # Should not raise
        load_backend_metrics()

        assert len(BACKEND_METRICS) == 0

    def test_save_creates_directory(self, tmp_path: Path):
        """Save creates directory if needed."""
        from delia.config import BACKEND_METRICS, get_backend_metrics, save_backend_metrics

        BACKEND_METRICS.clear()
        get_backend_metrics("test")

        # Should not raise even if directory doesn't exist
        save_backend_metrics()
