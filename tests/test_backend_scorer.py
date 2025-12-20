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
Tests for BackendScorer and ScoringWeights classes.

Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_backend_scorer.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path):
    """Use a temp directory for test data and clear module cache."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    # Clear cached modules to ensure fresh imports
    modules_to_clear = ["delia.paths", "delia.config", "delia.routing"]
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) or mod == m for m in modules_to_clear):
            del sys.modules[mod]

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


class TestScoringWeights:
    """Test ScoringWeights dataclass."""

    def test_default_weights(self):
        """Default weights are set correctly."""
        from delia.routing import ScoringWeights

        weights = ScoringWeights()
        assert weights.latency == 0.35
        assert weights.throughput == 0.15
        assert weights.reliability == 0.35
        assert weights.availability == 0.15

    def test_default_weights_sum_approximately_one(self):
        """Default weights sum to approximately 1.0."""
        from delia.routing import ScoringWeights

        weights = ScoringWeights()
        total = weights.latency + weights.throughput + weights.reliability + weights.availability
        assert abs(total - 1.0) < 0.01

    def test_custom_weights(self):
        """Custom weights can be specified."""
        from delia.routing import ScoringWeights

        weights = ScoringWeights(latency=0.5, throughput=0.2, reliability=0.2, availability=0.1)
        assert weights.latency == 0.5
        assert weights.throughput == 0.2

    def test_negative_weight_raises(self):
        """Negative weights raise ValueError."""
        from delia.routing import ScoringWeights

        with pytest.raises(ValueError, match="latency"):
            ScoringWeights(latency=-0.1)

        with pytest.raises(ValueError, match="reliability"):
            ScoringWeights(reliability=-0.5)


class TestBackendScorerLatencyScoring:
    """Test latency scoring curve."""

    def test_zero_latency_score_one(self):
        """Zero latency (no data) returns 1.0."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        assert scorer._score_latency(0.0) == 1.0

    def test_reference_latency_score_half(self):
        """Reference latency (1000ms) returns 0.5."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        assert scorer._score_latency(1000.0) == 0.5

    def test_low_latency_high_score(self):
        """Low latency results in high score."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        score_100ms = scorer._score_latency(100.0)
        score_500ms = scorer._score_latency(500.0)

        assert score_100ms > score_500ms
        assert score_100ms > 0.9
        assert score_500ms > 0.6

    def test_high_latency_low_score(self):
        """High latency results in low score."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        score_2000ms = scorer._score_latency(2000.0)
        score_5000ms = scorer._score_latency(5000.0)

        assert score_2000ms < 0.4
        assert score_5000ms < 0.2
        assert score_2000ms > score_5000ms


class TestBackendScorerThroughputScoring:
    """Test throughput scoring curve."""

    def test_zero_throughput_neutral(self):
        """Zero throughput (no data) returns 0.5 (neutral)."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        assert scorer._score_throughput(0.0) == 0.5

    def test_reference_throughput_capped(self):
        """Reference throughput (100 tok/s) returns ~0.833 on saturation curve."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        # 100 / (100 + 20) = 0.8333...
        assert abs(scorer._score_throughput(100.0) - 0.833) < 0.01

    def test_throughput_linear_scaling(self):
        """Throughput follows saturation curve."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        # 50 / (50 + 20) = 0.714...
        assert abs(scorer._score_throughput(50.0) - 0.714) < 0.01
        # 20 / (20 + 20) = 0.5
        assert scorer._score_throughput(20.0) == 0.5

    def test_throughput_capped_at_one(self):
        """Throughput score approaches 1.0 asymptotically."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        assert scorer._score_throughput(1000.0) > 0.95
        assert scorer._score_throughput(1000.0) < 1.0


class TestBackendScorerScore:
    """Test overall backend scoring."""

    def _create_mock_backend(self, backend_id: str, enabled: bool = True, backend_type: str = "local", priority: int = 1):
        """Create a mock BackendConfig."""
        backend = MagicMock()
        backend.id = backend_id
        backend.enabled = enabled
        backend.type = backend_type
        backend.priority = priority
        return backend

    def test_score_healthy_backend(self):
        """Score a healthy backend with good metrics."""
        from delia.config import BACKEND_METRICS, BackendMetrics, get_backend_health
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        backend = self._create_mock_backend("healthy-backend")

        # Set up metrics
        metrics = BackendMetrics(backend_id="healthy-backend")
        for _ in range(10):
            metrics.record_success(elapsed_ms=100.0, tokens=100)  # Fast, 1000 tok/s
        BACKEND_METRICS["healthy-backend"] = metrics

        scorer = BackendScorer()
        score = scorer.score(backend)

        # Should be high score: low latency + high throughput + 100% success + available
        assert score > 0.9

    def test_score_degraded_backend(self):
        """Score a backend with poor metrics."""
        from delia.config import BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        backend = self._create_mock_backend("degraded-backend")

        # Set up poor metrics
        metrics = BackendMetrics(backend_id="degraded-backend")
        for _ in range(5):
            metrics.record_success(elapsed_ms=2000.0, tokens=20)  # Slow
        for _ in range(5):
            metrics.record_failure(elapsed_ms=1000.0)  # 50% failure rate
        BACKEND_METRICS["degraded-backend"] = metrics

        scorer = BackendScorer()
        score = scorer.score(backend)

        # Should be lower score due to high latency and failures
        assert score < 0.6

    def test_score_new_backend_optimistic(self):
        """New backend with no metrics gets optimistic score + exploration boost."""
        from delia.config import BACKEND_METRICS
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        backend = self._create_mock_backend("new-backend")

        scorer = BackendScorer()
        score = scorer.score(backend)

        # No data = optimistic defaults (latency=1.0, throughput=0.5, reliability=1.0, avail=1.0)
        # Base: 0.35*1.0 + 0.15*0.5 + 0.35*1.0 + 0.15*1.0 = 0.925
        # Exploration Modifier (0 reqs): 1.0 + 0.25 = 1.25
        # Total: 0.925 * 1.25 = 1.156
        assert 1.1 <= score <= 1.2

    def test_score_unavailable_backend(self):
        """Unavailable backend (circuit breaker open) gets lower score."""
        from delia.config import BACKEND_HEALTH, BACKEND_METRICS, BackendHealth
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()
        BACKEND_HEALTH.clear()

        backend = self._create_mock_backend("unavailable-backend")

        # Trip the circuit breaker
        health = BackendHealth(name="unavailable-backend")
        for _ in range(5):  # Exceed consecutive failure threshold
            health.record_failure("test")
        BACKEND_HEALTH["unavailable-backend"] = health

        scorer = BackendScorer()
        score = scorer.score(backend)

        # Availability = 0 reduces score significantly
        assert score < 1.0


class TestBackendScorerSelectBest:
    """Test select_best backend selection."""

    def _create_mock_backend(self, backend_id: str, enabled: bool = True, backend_type: str = "local", priority: int = 1):
        """Create a mock BackendConfig."""
        backend = MagicMock()
        backend.id = backend_id
        backend.enabled = enabled
        backend.type = backend_type
        backend.priority = priority
        return backend

    def test_select_best_single_backend(self):
        """Select from single backend."""
        from delia.config import BACKEND_METRICS
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        backend = self._create_mock_backend("single")
        scorer = BackendScorer()

        result = scorer.select_best([backend])
        assert result is backend

    def test_select_best_multiple_backends(self):
        """Select best from multiple backends based on score."""
        from delia.config import BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        backend_fast = self._create_mock_backend("fast")
        backend_slow = self._create_mock_backend("slow")

        # Fast backend: low latency
        metrics_fast = BackendMetrics(backend_id="fast")
        for _ in range(5):
            metrics_fast.record_success(elapsed_ms=100.0, tokens=100)
        BACKEND_METRICS["fast"] = metrics_fast

        # Slow backend: high latency
        metrics_slow = BackendMetrics(backend_id="slow")
        for _ in range(5):
            metrics_slow.record_success(elapsed_ms=2000.0, tokens=50)
        BACKEND_METRICS["slow"] = metrics_slow

        scorer = BackendScorer()
        result = scorer.select_best([backend_fast, backend_slow])

        assert result is backend_fast

    def test_select_best_by_type_local(self):
        """Filter by backend type (local)."""
        from delia.config import BACKEND_METRICS
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        local_backend = self._create_mock_backend("local-1", backend_type="local")
        remote_backend = self._create_mock_backend("remote-1", backend_type="remote")

        scorer = BackendScorer()
        result = scorer.select_best([local_backend, remote_backend], backend_type="local")

        assert result is local_backend

    def test_select_best_by_type_remote(self):
        """Filter by backend type (remote)."""
        from delia.config import BACKEND_METRICS
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        local_backend = self._create_mock_backend("local-1", backend_type="local")
        remote_backend = self._create_mock_backend("remote-1", backend_type="remote")

        scorer = BackendScorer()
        result = scorer.select_best([local_backend, remote_backend], backend_type="remote")

        assert result is remote_backend

    def test_select_best_no_candidates(self):
        """Return None when no candidates match."""
        from delia.routing import BackendScorer

        # No local backends
        remote = self._create_mock_backend("remote-only", backend_type="remote")

        scorer = BackendScorer()
        result = scorer.select_best([remote], backend_type="local")

        assert result is None

    def test_select_best_disabled_excluded(self):
        """Disabled backends are excluded."""
        from delia.config import BACKEND_METRICS
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        enabled = self._create_mock_backend("enabled", enabled=True)
        disabled = self._create_mock_backend("disabled", enabled=False)

        scorer = BackendScorer()
        result = scorer.select_best([enabled, disabled])

        assert result is enabled

    def test_select_best_all_unavailable_fallback(self):
        """Fall back to priority when all circuit breakers open."""
        from delia.config import BACKEND_HEALTH, BACKEND_METRICS, BackendHealth
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()
        BACKEND_HEALTH.clear()

        high_priority = self._create_mock_backend("high-pri", priority=10)
        low_priority = self._create_mock_backend("low-pri", priority=1)

        # Trip circuit breakers for both
        for backend_id in ["high-pri", "low-pri"]:
            health = BackendHealth(name=backend_id)
            for _ in range(5):
                health.record_failure("test")
            BACKEND_HEALTH[backend_id] = health

        scorer = BackendScorer()
        result = scorer.select_best([high_priority, low_priority])

        # Falls back to highest priority
        assert result is high_priority


class TestBackendScorerCostScoring:
    """Test cost scoring curve."""

    def test_free_provider_score_one(self):
        """Free provider (cost = 0) returns 1.0."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        assert scorer._score_cost("ollama") == 1.0
        assert scorer._score_cost("llamacpp") == 1.0
        assert scorer._score_cost("local") == 1.0

    def test_reference_cost_score_half(self):
        """Reference cost ($0.01/1K) returns 0.5."""
        from delia.config import PROVIDER_COSTS
        from delia.routing import BackendScorer

        # Modify to test exact reference value
        original = PROVIDER_COSTS.get("test-provider")
        try:
            PROVIDER_COSTS["test-provider"] = 0.01  # Reference cost
            scorer = BackendScorer()
            assert scorer._score_cost("test-provider") == 0.5
        finally:
            if original is None:
                PROVIDER_COSTS.pop("test-provider", None)
            else:
                PROVIDER_COSTS["test-provider"] = original

    def test_low_cost_high_score(self):
        """Low cost results in high score."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        score_gemini = scorer._score_cost("gemini")  # $0.0001
        score_openai = scorer._score_cost("openai")  # $0.005

        assert score_gemini > score_openai
        assert score_gemini > 0.9

    def test_high_cost_low_score(self):
        """High cost results in low score."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        score_opus = scorer._score_cost("claude-3-opus")  # $0.015

        assert score_opus < 0.5

    def test_unknown_provider_default_cost(self):
        """Unknown provider uses default cost (0.001)."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        score = scorer._score_cost("unknown-provider")
        # $0.001 should give score > 0.9 (close to free)
        assert 0.9 < score < 1.0


class TestBackendScorerCostAwareRouting:
    """Test cost-aware backend selection."""

    def _create_mock_backend(self, backend_id: str, provider: str, enabled: bool = True):
        """Create a mock BackendConfig."""
        backend = MagicMock()
        backend.id = backend_id
        backend.enabled = enabled
        backend.type = "local" if provider in ("ollama", "llamacpp") else "remote"
        backend.provider = provider
        backend.priority = 1
        return backend

    def test_cost_weight_zero_ignores_cost(self):
        """When cost weight is 0, cost has no effect on score."""
        from delia.config import BACKEND_METRICS
        from delia.routing import BackendScorer, ScoringWeights

        BACKEND_METRICS.clear()

        # Default weights have cost=0.0
        scorer = BackendScorer()
        assert scorer.weights.cost == 0.0

        local_backend = self._create_mock_backend("local", "ollama")
        cloud_backend = self._create_mock_backend("cloud", "claude-3-opus")

        # With cost=0, both should have similar scores (no metrics)
        local_score = scorer.score(local_backend)
        cloud_score = scorer.score(cloud_backend)

        # Scores should be equal since cost is ignored
        assert abs(local_score - cloud_score) < 0.01

    def test_cost_weight_affects_score(self):
        """When cost weight is set, cheaper backends score higher."""
        from delia.config import BACKEND_METRICS
        from delia.routing import BackendScorer, ScoringWeights

        BACKEND_METRICS.clear()

        # Enable cost-aware scoring
        weights = ScoringWeights(
            latency=0.25,
            throughput=0.15,
            reliability=0.25,
            availability=0.15,
            cost=0.2,
        )
        scorer = BackendScorer(weights)

        local_backend = self._create_mock_backend("local", "ollama")
        cloud_backend = self._create_mock_backend("cloud", "claude-3-opus")

        local_score = scorer.score(local_backend)
        cloud_score = scorer.score(cloud_backend)

        # Local (free) should score higher than expensive cloud
        assert local_score > cloud_score

    def test_cost_aware_selection_prefers_cheap(self):
        """select_best prefers cheaper backends when cost-aware."""
        from delia.config import BACKEND_METRICS
        from delia.routing import BackendScorer, ScoringWeights

        BACKEND_METRICS.clear()

        weights = ScoringWeights(
            latency=0.2,
            throughput=0.1,
            reliability=0.2,
            availability=0.1,
            cost=0.4,  # Strong cost preference
        )
        scorer = BackendScorer(weights)

        local = self._create_mock_backend("local", "ollama")
        expensive = self._create_mock_backend("expensive", "claude-3-opus")

        result = scorer.select_best([local, expensive])
        assert result is local


class TestSelectOptimalBackendIntegration:
    """Test select_optimal_backend function uses BackendScorer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear metrics and health state before each test."""
        from delia.config import BACKEND_HEALTH, BACKEND_METRICS

        BACKEND_METRICS.clear()
        BACKEND_HEALTH.clear()
        yield
        BACKEND_METRICS.clear()
        BACKEND_HEALTH.clear()

    @pytest.mark.asyncio
    async def test_selects_best_backend_by_score(self):
        """select_optimal_backend selects based on score when multiple backends available."""
        from delia.backend_manager import BackendConfig, BackendManager
        from delia.config import BACKEND_METRICS, BackendMetrics
        from delia.routing import get_router

        # Create a manager with two backends
        manager = BackendManager()
        manager.backends = {
            "fast": BackendConfig(
                id="fast",
                name="Fast Backend",
                provider="ollama",
                type="local",
                url="http://localhost:11434",
                enabled=True,
                priority=1,
                models={"quick": "qwen2.5:14b"},
            ),
            "slow": BackendConfig(
                id="slow",
                name="Slow Backend",
                provider="ollama",
                type="local",
                url="http://localhost:11435",
                enabled=True,
                priority=2,
                models={"quick": "qwen2.5:14b"},
            ),
        }

        # Set up metrics - fast backend has better latency
        metrics_fast = BackendMetrics(backend_id="fast")
        for _ in range(5):
            metrics_fast.record_success(elapsed_ms=100.0, tokens=100)  # 100ms latency
        BACKEND_METRICS["fast"] = metrics_fast

        metrics_slow = BackendMetrics(backend_id="slow")
        for _ in range(5):
            metrics_slow.record_success(elapsed_ms=2000.0, tokens=50)  # 2000ms latency
        BACKEND_METRICS["slow"] = metrics_slow

        # Use router directly for integration test
        router = get_router()
        original_bm = router.backend_manager
        router.backend_manager = manager

        try:
            _, selected = await router.select_optimal_backend(
                content="test", task_type="quick"
            )

            # Should select fast backend (lower latency = higher score)
            assert selected is not None
            assert selected.id == "fast"
        finally:
            router.backend_manager = original_bm

    @pytest.mark.asyncio
    async def test_respects_backend_type_filter(self):
        """select_optimal_backend respects backend_type filter."""
        from delia.backend_manager import BackendConfig, BackendManager
        from delia.routing import get_router

        # Create a manager with local and remote backends
        manager = BackendManager()
        manager.backends = {
            "local": BackendConfig(
                id="local",
                name="Local Backend",
                provider="ollama",
                type="local",
                url="http://localhost:11434",
                enabled=True,
                priority=1,
                models={"quick": "qwen2.5:14b"},
            ),
            "remote": BackendConfig(
                id="remote",
                name="Remote Backend",
                provider="gemini",
                type="remote",
                url="https://api.example.com",
                enabled=True,
                priority=2,
                models={"quick": "gemini-flash"},
            ),
        }

        router = get_router()
        original_bm = router.backend_manager
        router.backend_manager = manager

        try:
            # Request only remote
            _, selected = await router.select_optimal_backend(
                content="test", task_type="quick", backend_type="remote"
            )
            assert selected is not None
            assert selected.id == "remote"

            # Request only local
            _, selected = await router.select_optimal_backend(
                content="test", task_type="quick", backend_type="local"
            )
            assert selected is not None
            assert selected.id == "local"
        finally:
            router.backend_manager = original_bm


class TestScoringWeightsFromDict:
    """Test ScoringWeights.from_dict classmethod."""

    def test_from_dict_all_values(self):
        """Create weights with all values from dict."""
        from delia.routing import ScoringWeights

        data = {
            "latency": 0.4,
            "throughput": 0.1,
            "reliability": 0.3,
            "availability": 0.1,
            "cost": 0.1,
        }
        weights = ScoringWeights.from_dict(data)

        assert weights.latency == 0.4
        assert weights.throughput == 0.1
        assert weights.reliability == 0.3
        assert weights.availability == 0.1
        assert weights.cost == 0.1

    def test_from_dict_partial_values(self):
        """Missing values use defaults."""
        from delia.routing import ScoringWeights

        data = {"latency": 0.5, "cost": 0.2}
        weights = ScoringWeights.from_dict(data)

        assert weights.latency == 0.5
        assert weights.cost == 0.2
        # Defaults for others
        assert weights.throughput == 0.15
        assert weights.reliability == 0.35
        assert weights.availability == 0.15

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys are ignored."""
        from delia.routing import ScoringWeights

        data = {"latency": 0.5, "unknown_key": 0.9, "foo": "bar"}
        weights = ScoringWeights.from_dict(data)

        assert weights.latency == 0.5
        assert not hasattr(weights, "unknown_key")
        assert not hasattr(weights, "foo")


class TestBackendScorerSelectWeighted:
    """Test weighted random selection for load balancing."""

    def _create_mock_backend(
        self,
        backend_id: str,
        backend_type: str = "local",
        enabled: bool = True,
        priority: int = 0,
        provider: str = "ollama",
    ):
        """Create a mock BackendConfig for testing."""
        from delia.backend_manager import BackendConfig

        return BackendConfig(
            id=backend_id,
            name=f"Test Backend {backend_id}",
            provider=provider,
            type=backend_type,
            url=f"http://localhost:{backend_id}",
            enabled=enabled,
            priority=priority,
            models={"quick": "test-model", "coder": "test-model", "moe": "test-model"},
        )

    def test_select_weighted_returns_backend(self):
        """select_weighted returns a backend from the list."""
        from delia.config import BACKEND_METRICS
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        backends = [
            self._create_mock_backend("backend-1"),
            self._create_mock_backend("backend-2"),
        ]

        scorer = BackendScorer()
        result = scorer.select_weighted(backends)

        assert result is not None
        assert result in backends

    def test_select_weighted_single_backend(self):
        """Single backend is always returned."""
        from delia.routing import BackendScorer

        backend = self._create_mock_backend("only")
        scorer = BackendScorer()

        result = scorer.select_weighted([backend])
        assert result is backend

    def test_select_weighted_empty_list(self):
        """Empty list returns None."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        result = scorer.select_weighted([])
        assert result is None

    def test_select_weighted_respects_type_filter(self):
        """Type filter is respected."""
        from delia.routing import BackendScorer

        local = self._create_mock_backend("local", backend_type="local")
        remote = self._create_mock_backend("remote", backend_type="remote")

        scorer = BackendScorer()

        # Request only local
        result = scorer.select_weighted([local, remote], backend_type="local")
        assert result is local

        # Request only remote
        result = scorer.select_weighted([local, remote], backend_type="remote")
        assert result is remote

    def test_select_weighted_distribution(self):
        """Higher scoring backends are selected more often."""
        from delia.config import BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        fast = self._create_mock_backend("fast")
        slow = self._create_mock_backend("slow")

        # Give fast backend much better metrics
        metrics_fast = BackendMetrics(backend_id="fast")
        for _ in range(10):
            metrics_fast.record_success(elapsed_ms=50.0, tokens=100)
        BACKEND_METRICS["fast"] = metrics_fast

        metrics_slow = BackendMetrics(backend_id="slow")
        for _ in range(10):
            metrics_slow.record_success(elapsed_ms=5000.0, tokens=10)
        BACKEND_METRICS["slow"] = metrics_slow

        scorer = BackendScorer()

        # Sample many times
        selections = {"fast": 0, "slow": 0}
        for _ in range(100):
            result = scorer.select_weighted([fast, slow])
            selections[result.id] += 1

        # Fast should be selected more often (but not always due to randomness)
        assert selections["fast"] > selections["slow"]


# ============================================================
# AFFINITY TRACKER TESTS
# ============================================================


class TestAffinityTracker:
    """Test AffinityTracker class."""

    def test_initial_affinity_neutral(self):
        """Unknown backend+task pairs return neutral 0.5."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker()
        assert tracker.get_affinity("backend1", "review") == 0.5
        assert tracker.get_affinity("unknown", "unknown") == 0.5

    def test_update_success_increases_affinity(self):
        """Successful requests increase affinity above neutral."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)
        tracker.update("backend1", "review", success=True)

        affinity = tracker.get_affinity("backend1", "review")
        assert affinity > 0.5  # Should increase from 0.5

    def test_update_failure_decreases_affinity(self):
        """Failed requests decrease affinity below neutral."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)
        tracker.update("backend1", "review", success=False)

        affinity = tracker.get_affinity("backend1", "review")
        assert affinity < 0.5  # Should decrease from 0.5

    def test_ema_convergence_success(self):
        """Repeated successes converge affinity toward 1.0."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)
        for _ in range(50):
            tracker.update("backend1", "review", success=True)

        affinity = tracker.get_affinity("backend1", "review")
        assert affinity > 0.95  # Should be close to 1.0

    def test_ema_convergence_failure(self):
        """Repeated failures converge affinity toward 0.0."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)
        for _ in range(50):
            tracker.update("backend1", "review", success=False)

        affinity = tracker.get_affinity("backend1", "review")
        assert affinity < 0.05  # Should be close to 0.0

    def test_boost_score_neutral_no_change(self):
        """Neutral affinity (0.5) doesn't change the score."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker()
        base_score = 0.8

        boosted = tracker.boost_score(base_score, "unknown", "unknown")
        assert boosted == base_score  # 0.5 affinity = no change

    def test_boost_score_high_affinity_increases(self):
        """High affinity increases the score."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)
        # Build up high affinity
        for _ in range(50):
            tracker.update("backend1", "review", success=True)

        base_score = 0.8
        boosted = tracker.boost_score(base_score, "backend1", "review")
        assert boosted > base_score  # Should be increased

    def test_boost_score_low_affinity_decreases(self):
        """Low affinity decreases the score."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)
        # Build up low affinity
        for _ in range(50):
            tracker.update("backend1", "review", success=False)

        base_score = 0.8
        boosted = tracker.boost_score(base_score, "backend1", "review")
        assert boosted < base_score  # Should be decreased

    def test_different_task_types_tracked_separately(self):
        """Different task types maintain separate affinities."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)

        # backend1 succeeds at review, fails at generate
        for _ in range(20):
            tracker.update("backend1", "review", success=True)
            tracker.update("backend1", "generate", success=False)

        review_affinity = tracker.get_affinity("backend1", "review")
        generate_affinity = tracker.get_affinity("backend1", "generate")

        assert review_affinity > 0.7  # Good at review
        assert generate_affinity < 0.3  # Bad at generate

    def test_different_backends_tracked_separately(self):
        """Different backends maintain separate affinities."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)

        # backend1 succeeds, backend2 fails at same task
        for _ in range(20):
            tracker.update("backend1", "review", success=True)
            tracker.update("backend2", "review", success=False)

        backend1_affinity = tracker.get_affinity("backend1", "review")
        backend2_affinity = tracker.get_affinity("backend2", "review")

        assert backend1_affinity > 0.7  # backend1 good
        assert backend2_affinity < 0.3  # backend2 bad


class TestAffinityTrackerPersistence:
    """Test AffinityTracker serialization."""

    def test_to_dict_empty(self):
        """Empty tracker serializes correctly."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.2)
        data = tracker.to_dict()

        assert data["alpha"] == 0.2
        assert data["scores"] == {}

    def test_to_dict_with_scores(self):
        """Tracker with scores serializes correctly."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)
        tracker.update("backend1", "review", success=True)
        tracker.update("backend2", "generate", success=False)

        data = tracker.to_dict()

        assert "backend1:review" in data["scores"]
        assert "backend2:generate" in data["scores"]

    def test_from_dict_restores_state(self):
        """Deserialization restores correct state."""
        from delia.config import AffinityTracker

        original = AffinityTracker(alpha=0.15)
        for _ in range(10):
            original.update("backend1", "review", success=True)
        original.update("backend2", "generate", success=False)

        data = original.to_dict()
        restored = AffinityTracker.from_dict(data)

        assert restored.alpha == 0.15
        assert restored.get_affinity("backend1", "review") == original.get_affinity(
            "backend1", "review"
        )
        assert restored.get_affinity("backend2", "generate") == original.get_affinity(
            "backend2", "generate"
        )

    def test_from_dict_handles_empty_data(self):
        """Deserialization handles empty/missing data gracefully."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker.from_dict({})
        assert tracker.alpha == 0.1  # Default
        assert tracker.get_affinity("any", "task") == 0.5  # Neutral

    def test_get_status_returns_info(self):
        """get_status returns useful information."""
        from delia.config import AffinityTracker

        tracker = AffinityTracker()
        tracker.update("backend1", "review", success=True)

        status = tracker.get_status()

        assert "alpha" in status
        assert "tracked_pairs" in status
        assert status["tracked_pairs"] == 1
        assert "scores" in status


class TestBackendScorerAffinityIntegration:
    """Test BackendScorer integration with AffinityTracker."""

    def _create_mock_backend(
        self, backend_id: str, backend_type: str = "local"
    ) -> MagicMock:
        """Create a mock BackendConfig."""
        mock = MagicMock()
        mock.id = backend_id
        mock.type = backend_type
        mock.enabled = True
        mock.priority = 1
        mock.provider = "ollama"
        return mock

    def test_score_with_task_type_uses_affinity(self):
        """score() with task_type applies affinity boost."""
        from delia.config import AFFINITY_TRACKER, BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        backend = self._create_mock_backend("test-backend")

        # Set up metrics so we have a base score
        metrics = BackendMetrics(backend_id="test-backend")
        metrics.record_success(elapsed_ms=100.0, tokens=100)
        BACKEND_METRICS["test-backend"] = metrics

        # Build high affinity for "review" task
        for _ in range(20):
            AFFINITY_TRACKER.update("test-backend", "review", success=True)

        scorer = BackendScorer()

        score_without_task = scorer.score(backend)
        score_with_task = scorer.score(backend, task_type="review")

        # Score with task_type should be boosted
        assert score_with_task > score_without_task

    def test_select_best_with_task_type(self):
        """select_best considers affinity when task_type provided."""
        from delia.config import AFFINITY_TRACKER, BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        # Create two similar backends
        good_at_review = self._create_mock_backend("good-review")
        bad_at_review = self._create_mock_backend("bad-review")

        # Similar base metrics
        for backend_id in ["good-review", "bad-review"]:
            metrics = BackendMetrics(backend_id=backend_id)
            metrics.record_success(elapsed_ms=100.0, tokens=100)
            BACKEND_METRICS[backend_id] = metrics

        # Different affinities for review task
        for _ in range(30):
            AFFINITY_TRACKER.update("good-review", "review", success=True)
            AFFINITY_TRACKER.update("bad-review", "review", success=False)

        scorer = BackendScorer()

        # Without task_type, either could be selected (similar scores)
        # With task_type, good-review should be preferred
        result = scorer.select_best(
            [good_at_review, bad_at_review], task_type="review"
        )
        assert result.id == "good-review"


# ============================================================
# SELECT TOP N TESTS (for hedged requests)
# ============================================================


class TestBackendScorerSelectTopN:
    """Test select_top_n method for hedged request support."""

    def _create_mock_backend(
        self,
        backend_id: str,
        enabled: bool = True,
        backend_type: str = "local",
        priority: int = 0,
    ):
        """Create a mock BackendConfig."""
        from delia.backend_manager import BackendConfig

        return BackendConfig(
            id=backend_id,
            name=f"Test {backend_id}",
            provider="test",
            type=backend_type,
            url=f"http://{backend_id}:8080",
            enabled=enabled,
            priority=priority,
            models={"quick": "test-7b", "coder": "test-14b", "moe": "test-30b"},
        )

    def test_select_top_n_returns_list(self):
        """select_top_n returns a list of backends."""
        from delia.config import BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()
        backend = self._create_mock_backend("test")
        metrics = BackendMetrics(backend_id="test")
        metrics.record_success(elapsed_ms=100.0, tokens=100)
        BACKEND_METRICS["test"] = metrics

        scorer = BackendScorer()
        result = scorer.select_top_n([backend], n=2)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].id == "test"

    def test_select_top_n_returns_up_to_n(self):
        """select_top_n returns at most n backends."""
        from delia.config import BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        backends = [self._create_mock_backend(f"backend-{i}") for i in range(5)]
        for b in backends:
            metrics = BackendMetrics(backend_id=b.id)
            metrics.record_success(elapsed_ms=100.0, tokens=100)
            BACKEND_METRICS[b.id] = metrics

        scorer = BackendScorer()
        result = scorer.select_top_n(backends, n=3)

        assert len(result) == 3

    def test_select_top_n_sorted_by_score(self):
        """select_top_n returns backends sorted by score (best first)."""
        from delia.config import BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        fast = self._create_mock_backend("fast")
        medium = self._create_mock_backend("medium")
        slow = self._create_mock_backend("slow")

        # Set up different latencies (lower is better)
        metrics_fast = BackendMetrics(backend_id="fast")
        for _ in range(10):
            metrics_fast.record_success(elapsed_ms=50.0, tokens=100)
        BACKEND_METRICS["fast"] = metrics_fast

        metrics_medium = BackendMetrics(backend_id="medium")
        for _ in range(10):
            metrics_medium.record_success(elapsed_ms=200.0, tokens=100)
        BACKEND_METRICS["medium"] = metrics_medium

        metrics_slow = BackendMetrics(backend_id="slow")
        for _ in range(10):
            metrics_slow.record_success(elapsed_ms=1000.0, tokens=100)
        BACKEND_METRICS["slow"] = metrics_slow

        scorer = BackendScorer()
        result = scorer.select_top_n([slow, fast, medium], n=3)

        # Should be sorted by score: fast > medium > slow
        assert result[0].id == "fast"
        assert result[1].id == "medium"
        assert result[2].id == "slow"

    def test_select_top_n_respects_type_filter(self):
        """select_top_n filters by backend_type."""
        from delia.config import BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        local1 = self._create_mock_backend("local1", backend_type="local")
        local2 = self._create_mock_backend("local2", backend_type="local")
        remote = self._create_mock_backend("remote", backend_type="remote")

        for b in [local1, local2, remote]:
            metrics = BackendMetrics(backend_id=b.id)
            metrics.record_success(elapsed_ms=100.0, tokens=100)
            BACKEND_METRICS[b.id] = metrics

        scorer = BackendScorer()

        # Only local backends
        local_result = scorer.select_top_n([local1, local2, remote], n=3, backend_type="local")
        assert len(local_result) == 2
        assert all(b.type == "local" for b in local_result)

        # Only remote backends
        remote_result = scorer.select_top_n([local1, local2, remote], n=3, backend_type="remote")
        assert len(remote_result) == 1
        assert remote_result[0].type == "remote"

    def test_select_top_n_excludes_disabled(self):
        """select_top_n excludes disabled backends."""
        from delia.config import BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        enabled = self._create_mock_backend("enabled", enabled=True)
        disabled = self._create_mock_backend("disabled", enabled=False)

        for b in [enabled, disabled]:
            metrics = BackendMetrics(backend_id=b.id)
            metrics.record_success(elapsed_ms=100.0, tokens=100)
            BACKEND_METRICS[b.id] = metrics

        scorer = BackendScorer()
        result = scorer.select_top_n([enabled, disabled], n=2)

        assert len(result) == 1
        assert result[0].id == "enabled"

    def test_select_top_n_empty_list(self):
        """select_top_n returns empty list for no backends."""
        from delia.routing import BackendScorer

        scorer = BackendScorer()
        result = scorer.select_top_n([], n=2)

        assert result == []

    def test_select_top_n_with_task_type(self):
        """select_top_n considers affinity when task_type provided."""
        from delia.config import AFFINITY_TRACKER, BACKEND_METRICS, BackendMetrics
        from delia.routing import BackendScorer

        BACKEND_METRICS.clear()

        good_at_review = self._create_mock_backend("good-review")
        bad_at_review = self._create_mock_backend("bad-review")

        # Similar base metrics
        for backend_id in ["good-review", "bad-review"]:
            metrics = BackendMetrics(backend_id=backend_id)
            metrics.record_success(elapsed_ms=100.0, tokens=100)
            BACKEND_METRICS[backend_id] = metrics

        # Different affinities for review task
        for _ in range(30):
            AFFINITY_TRACKER.update("good-review", "review", success=True)
            AFFINITY_TRACKER.update("bad-review", "review", success=False)

        scorer = BackendScorer()
        result = scorer.select_top_n(
            [good_at_review, bad_at_review], n=2, task_type="review"
        )

        # Good at review should be first
        assert result[0].id == "good-review"
        assert result[1].id == "bad-review"
