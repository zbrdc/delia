# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import math
from hypothesis import given, strategies as st, assume
from delia.routing import BackendScorer, ScoringWeights
from delia.backend_manager import BackendConfig

# Mock metrics object to fuzz
class MockMetrics:
    def __init__(self, latency, tps, success_rate, total_requests=10):
        self.latency_p50 = latency
        self.tokens_per_second = tps
        self.success_rate = success_rate
        self.total_requests = total_requests

class MockHealth:
    def __init__(self, available):
        self._available = available
    def is_available(self):
        return self._available

@given(
    st.floats(allow_nan=True, allow_infinity=True),
    st.floats(allow_nan=True, allow_infinity=True),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans()
)
def test_scorer_robustness(latency, tps, success_rate, available):
    """The scorer should never crash or return NaN/Inf even with crazy metrics."""
    scorer = BackendScorer()
    backend = BackendConfig(id="test", name="test", provider="ollama", type="local", url="")
    
    # We need to mock the global get_backend_metrics and get_backend_health
    # Since they are imported into delia.routing, we patch them there
    import delia.routing
    
    metrics = MockMetrics(latency, tps, success_rate)
    health = MockHealth(available)
    
    # Monkeypatch for this test iteration
    original_metrics = delia.routing.get_backend_metrics
    original_health = delia.routing.get_backend_health
    delia.routing.get_backend_metrics = lambda x: metrics
    delia.routing.get_backend_health = lambda x: health
    
    try:
        score = scorer.score(backend)
        
        # In a robust system, score should be a real number
        assert isinstance(score, (int, float))
        assert not math.isnan(score)
        assert not math.isinf(score)
    except Exception as e:
        pytest.fail(f"Scorer crashed with latency={latency}, tps={tps}, sr={success_rate}: {e}")
    finally:
        delia.routing.get_backend_metrics = original_metrics
        delia.routing.get_backend_health = original_health

@given(st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=5))
def test_select_weighted_robustness(scores):
    """Test weighted selection with various score distributions."""
    scorer = BackendScorer()
    
    # Mock backends
    backends = [
        BackendConfig(id=f"b{i}", name=f"b{i}", provider="ollama", type="local", url="")
        for i in range(len(scores))
    ]
    
    import delia.routing
    import itertools
    original_score = BackendScorer.score
    original_health = delia.routing.get_backend_health
    
    # Cycle scores to ensure we don't hit StopIteration
    score_cycle = itertools.cycle(scores)
    BackendScorer.score = lambda self, b, task=None: next(score_cycle)
    delia.routing.get_backend_health = lambda x: MockHealth(True)
    
    try:
        selected = scorer.select_weighted(backends)
        assert selected in backends
    except Exception as e:
        pytest.fail(f"select_weighted crashed with scores={scores}: {e}")
    finally:
        BackendScorer.score = original_score
        delia.routing.get_backend_health = original_health

def test_select_weighted_all_zero():
    """Test weighted selection when all scores are 0.0."""
    scorer = BackendScorer()
    backends = [
        BackendConfig(id="b1", name="b1", provider="ollama", type="local", url=""),
        BackendConfig(id="b2", name="b2", provider="ollama", type="local", url="")
    ]
    
    import delia.routing
    original_score = BackendScorer.score
    original_health = delia.routing.get_backend_health
    
    BackendScorer.score = lambda self, b, task=None: 0.0
    delia.routing.get_backend_health = lambda x: MockHealth(True)
    
    try:
        # Should not crash even if all scores are 0.0 (uses 0.01 floor)
        selected = scorer.select_weighted(backends)
        assert selected in backends
    finally:
        BackendScorer.score = original_score
        delia.routing.get_backend_health = original_health

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
