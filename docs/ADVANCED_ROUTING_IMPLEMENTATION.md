# Advanced Routing Implementation Plan

## Overview

This document details the implementation plan for Phase 1 of the Advanced Routing features:
1. Backend Metrics Tracking (Score: 12.5)
2. Latency-Aware Backend Scoring (Score: 10.0)
3. Cost-Aware Routing (Score: 10.0)

## Architecture Analysis

### Current State

**BackendConfig** (`backend_manager.py:61-249`)
- Dataclass with: id, name, provider, type, url, enabled, priority, models, etc.
- Runtime state: `_available`, `_client` (not persisted)
- No performance metrics

**BackendHealth** (`config.py:260-374`)
- Circuit breaker pattern implementation
- Tracks: consecutive_failures, last_failure_time, last_error_type
- Context size learning: safe_context_estimate, max_successful_context
- Methods: record_success(), record_failure(), is_available()

**LLMResponse** (`providers/base.py:56-112`)
- Contains: success, response, tokens, elapsed_ms, error, metadata

**Metrics Collection Point** (providers/*.py)
- Providers call `health.record_success(content_size)` after successful calls
- Providers call `health.record_failure(error_type, content_size)` after failures
- `LLMResponse` contains elapsed_ms and tokens

### Gap Analysis
- No latency tracking (elapsed_ms discarded after logging)
- No success rate calculation (only consecutive failures)
- No throughput metrics (tokens per second)
- No cost awareness
- No scoring mechanism for backend selection

---

## Feature 1: Backend Metrics Tracking

### Design

Create `BackendMetrics` dataclass in `config.py` (alongside `BackendHealth`):

```python
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

@dataclass
class BackendMetrics:
    """Rolling performance metrics for a backend."""

    backend_id: str

    # Latency tracking (rolling window)
    latency_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    # Success/failure counts (all-time, for rate calculation)
    total_successes: int = 0
    total_failures: int = 0

    # Token tracking
    total_tokens: int = 0
    total_requests: int = 0

    # Recent window for rate calculation (last N seconds)
    recent_successes: int = 0
    recent_failures: int = 0
    recent_window_start: float = 0.0

    # Timestamps
    last_success_time: float = 0.0
    last_request_time: float = 0.0

    def record_request(self, elapsed_ms: float, tokens: int, success: bool) -> None:
        """Record metrics from a completed request."""
        ...

    @property
    def latency_p50(self) -> float:
        """Median latency in milliseconds."""
        ...

    @property
    def latency_p95(self) -> float:
        """95th percentile latency in milliseconds."""
        ...

    @property
    def success_rate(self) -> float:
        """Success rate as fraction (0.0 to 1.0)."""
        ...

    @property
    def tokens_per_second(self) -> float:
        """Average throughput in tokens per second."""
        ...

    @property
    def requests_per_minute(self) -> float:
        """Request rate based on recent window."""
        ...

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "BackendMetrics":
        """Deserialize from persistence."""
        ...
```

### Integration Points

1. **Storage**: Add `BACKEND_METRICS: dict[str, BackendMetrics]` in `config.py`
2. **Retrieval**: Add `get_backend_metrics(backend_id: str) -> BackendMetrics` function
3. **Persistence**: Save to `~/.cache/delia/backend_metrics.json`
4. **Recording**: Update providers to call `metrics.record_request()` after each call

### Provider Integration

In each provider's call method (e.g., `ollama.py:300-340`), after the existing health recording:

```python
# Existing code
health.record_success(content_size)

# New code - record metrics
from delia.config import get_backend_metrics
metrics = get_backend_metrics(backend_id)
metrics.record_request(
    elapsed_ms=elapsed_ms,
    tokens=tokens,
    success=True
)
```

### Files to Modify

| File | Changes |
|------|---------|
| `config.py` | Add BackendMetrics class, BACKEND_METRICS dict, get_backend_metrics() |
| `providers/base.py` | Add metrics recording to create_success_response/create_error_response helpers |
| `providers/ollama.py` | Pass backend_id to response helpers |
| `providers/llamacpp.py` | Pass backend_id to response helpers |
| `providers/gemini.py` | Pass backend_id to response helpers |
| `paths.py` | Add METRICS_FILE path constant |

### Test Plan

Create `tests/test_backend_metrics.py`:

```python
class TestBackendMetrics:
    def test_record_request_success(self):
        """Verify success recording updates all counters."""

    def test_record_request_failure(self):
        """Verify failure recording updates counters correctly."""

    def test_latency_p50_calculation(self):
        """Verify median calculation with various sample sizes."""

    def test_latency_p95_calculation(self):
        """Verify 95th percentile calculation."""

    def test_success_rate_empty(self):
        """Verify 1.0 rate when no requests recorded."""

    def test_success_rate_calculation(self):
        """Verify rate calculation accuracy."""

    def test_tokens_per_second(self):
        """Verify throughput calculation."""

    def test_rolling_window(self):
        """Verify deque maxlen behavior for latency samples."""

    def test_serialization_roundtrip(self):
        """Verify to_dict/from_dict preserves all data."""

    def test_persistence(self):
        """Verify metrics survive save/load cycle."""
```

---

## Feature 2: Latency-Aware Backend Scoring

### Design

Create `BackendScorer` class in `routing.py`:

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class ScoringWeights:
    """Configurable weights for backend scoring."""
    latency: float = 0.4       # Lower latency = better
    throughput: float = 0.2    # Higher tokens/sec = better
    reliability: float = 0.3   # Higher success rate = better
    availability: float = 0.1  # Circuit breaker state

class BackendScorer:
    """Score backends for optimal routing based on performance metrics."""

    def __init__(self, weights: ScoringWeights | None = None):
        self.weights = weights or ScoringWeights()

    def score(self, backend: BackendConfig) -> float:
        """
        Calculate a 0.0-1.0 score for a backend.

        Higher score = better backend choice.
        """
        metrics = get_backend_metrics(backend.id)
        health = get_backend_health(backend.id)

        # Normalize each component to 0-1 range
        latency_score = self._score_latency(metrics.latency_p50)
        throughput_score = self._score_throughput(metrics.tokens_per_second)
        reliability_score = metrics.success_rate
        availability_score = 1.0 if health.is_available() else 0.0

        return (
            self.weights.latency * latency_score +
            self.weights.throughput * throughput_score +
            self.weights.reliability * reliability_score +
            self.weights.availability * availability_score
        )

    def _score_latency(self, latency_ms: float) -> float:
        """Convert latency to 0-1 score (lower = better)."""
        if latency_ms <= 0:
            return 1.0  # No data = optimistic
        # 500ms = 0.67, 1000ms = 0.5, 2000ms = 0.33
        return 1.0 / (1 + latency_ms / 1000)

    def _score_throughput(self, tps: float) -> float:
        """Convert tokens/sec to 0-1 score (higher = better)."""
        if tps <= 0:
            return 0.5  # No data = neutral
        # 50 tok/s = 0.5, 100 tok/s = 1.0
        return min(tps / 100, 1.0)

    def select_best(
        self,
        backends: list[BackendConfig],
        backend_type: str | None = None
    ) -> BackendConfig | None:
        """
        Select the best backend from a list.

        Args:
            backends: List of backends to choose from
            backend_type: Optional filter ("local" or "remote")

        Returns:
            Best backend or None if none available
        """
        candidates = [
            b for b in backends
            if b.enabled and (backend_type is None or b.type == backend_type)
        ]

        if not candidates:
            return None

        # Filter by circuit breaker availability
        available = [
            b for b in candidates
            if get_backend_health(b.id).is_available()
        ]

        if not available:
            # All circuit breakers open - return highest priority as fallback
            return max(candidates, key=lambda b: b.priority)

        # Score and select best
        return max(available, key=self.score)
```

### Integration with ModelRouter

Update `routing.py` to use scorer in `ModelRouter.select_optimal_backend()`:

```python
class ModelRouter:
    def __init__(self, config: Config, backend_manager: BackendManager):
        self.config = config
        self.backend_manager = backend_manager
        self.scorer = BackendScorer()  # Add scorer
        self._semantic_router = None

    def select_optimal_backend(
        self,
        content: str = "",
        backend_type: str | None = None,
    ) -> BackendConfig | None:
        """Select optimal backend using scoring."""
        backends = self.backend_manager.get_enabled_backends()
        return self.scorer.select_best(backends, backend_type)
```

### Wire into mcp_server.py

Replace `_select_optimal_backend_v2()` with routing module:

```python
# In mcp_server.py, replace:
async def _select_optimal_backend_v2(...):
    # ... simple logic ...

# With:
from .routing import get_router

async def _select_optimal_backend_v2(
    content: str,
    file_path: str | None = None,
    task_type: str = "quick",
    backend_type: str | None = None,
) -> tuple[str | None, BackendConfig | None]:
    """Select optimal backend using scoring-based routing."""
    router = get_router()
    backend = router.select_optimal_backend(content, backend_type)
    return (None, backend)
```

### Files to Modify

| File | Changes |
|------|---------|
| `routing.py` | Add ScoringWeights, BackendScorer classes |
| `routing.py` | Update ModelRouter.select_optimal_backend() to use scorer |
| `mcp_server.py` | Replace _select_optimal_backend_v2 with router call |

### Test Plan

Create `tests/test_backend_scorer.py`:

```python
class TestScoringWeights:
    def test_default_weights_sum_to_one(self):
        """Verify default weights are balanced."""

class TestBackendScorer:
    def test_score_healthy_backend(self):
        """Verify scoring with good metrics."""

    def test_score_degraded_backend(self):
        """Verify lower score with poor metrics."""

    def test_score_unavailable_backend(self):
        """Verify circuit breaker affects score."""

    def test_select_best_single(self):
        """Select from single backend."""

    def test_select_best_multiple(self):
        """Select best from multiple backends."""

    def test_select_best_by_type(self):
        """Filter by backend type."""

    def test_select_best_all_unavailable(self):
        """Fallback when all circuit breakers open."""

    def test_latency_scoring_curve(self):
        """Verify latency score curve shape."""

    def test_throughput_scoring_cap(self):
        """Verify throughput score caps at 1.0."""
```

---

## Feature 3: Cost-Aware Routing

### Design

Extend BackendScorer with cost awareness:

```python
# In config.py or routing.py

# Cost per 1K tokens (input + output average estimate)
PROVIDER_COSTS: dict[str, float] = {
    # Local providers (electricity only)
    "ollama": 0.0,
    "llamacpp": 0.0,
    "lmstudio": 0.0,

    # Google
    "gemini": 0.0001,  # Flash pricing

    # OpenAI
    "openai": 0.002,  # Average across models

    # Anthropic
    "anthropic": 0.008,  # Average across models

    # Default for unknown
    "default": 0.001,
}

@dataclass
class ScoringWeights:
    """Configurable weights for backend scoring."""
    latency: float = 0.35
    throughput: float = 0.15
    reliability: float = 0.25
    availability: float = 0.1
    cost: float = 0.15  # NEW

class BackendScorer:
    def __init__(
        self,
        weights: ScoringWeights | None = None,
        cost_sensitivity: float = 0.5,  # 0=ignore, 1=strongly prefer cheap
    ):
        self.weights = weights or ScoringWeights()
        self.cost_sensitivity = cost_sensitivity

    def score(self, backend: BackendConfig, estimated_tokens: int = 1000) -> float:
        """Calculate score with cost awareness."""
        # ... existing scoring ...

        # Cost scoring
        cost_score = self._score_cost(backend.provider, estimated_tokens)

        return (
            self.weights.latency * latency_score +
            self.weights.throughput * throughput_score +
            self.weights.reliability * reliability_score +
            self.weights.availability * availability_score +
            self.weights.cost * cost_score
        )

    def _score_cost(self, provider: str, estimated_tokens: int) -> float:
        """
        Score cost (higher = cheaper = better).

        Local providers get 1.0, expensive cloud gets lower scores.
        """
        cost_per_1k = PROVIDER_COSTS.get(provider.lower(), PROVIDER_COSTS["default"])

        if cost_per_1k == 0:
            return 1.0  # Free is best

        estimated_cost = cost_per_1k * (estimated_tokens / 1000)

        # Scale: $0.001 = 0.9, $0.01 = 0.5, $0.1 = 0.1
        # Using exponential decay: score = exp(-cost * sensitivity * 100)
        import math
        raw_score = math.exp(-estimated_cost * self.cost_sensitivity * 100)
        return max(0.1, raw_score)  # Floor at 0.1
```

### Configuration

Add to `settings.json` schema:

```json
{
  "routing": {
    "prefer_local": true,
    "fallback_enabled": true,
    "scoring": {
      "enabled": true,
      "weights": {
        "latency": 0.35,
        "throughput": 0.15,
        "reliability": 0.25,
        "availability": 0.1,
        "cost": 0.15
      },
      "cost_sensitivity": 0.5
    }
  }
}
```

### Files to Modify

| File | Changes |
|------|---------|
| `routing.py` | Add PROVIDER_COSTS, update ScoringWeights, add _score_cost() |
| `backend_manager.py` | Load scoring config from settings.json |

### Test Plan

Add to `tests/test_backend_scorer.py`:

```python
class TestCostScoring:
    def test_local_provider_free(self):
        """Local providers get score 1.0."""

    def test_cloud_provider_cost(self):
        """Cloud providers get lower scores."""

    def test_cost_sensitivity_zero(self):
        """Cost ignored when sensitivity is 0."""

    def test_cost_sensitivity_high(self):
        """Cost dominates when sensitivity is 1."""

    def test_unknown_provider_default(self):
        """Unknown providers use default cost."""

    def test_estimated_tokens_impact(self):
        """Higher token estimates increase cost penalty."""
```

---

## Implementation Order

### Phase 1.1: Backend Metrics (3-4 hours)
1. Create `BackendMetrics` dataclass in `config.py`
2. Add persistence to `paths.py` and save/load logic
3. Add `get_backend_metrics()` function
4. Write comprehensive unit tests
5. Verify persistence works

### Phase 1.2: Provider Integration (2-3 hours)
1. Update `providers/base.py` helper functions to accept backend_id
2. Update each provider to pass metrics
3. Write integration tests
4. Verify metrics are recorded correctly

### Phase 1.3: Backend Scorer (2-3 hours)
1. Create `ScoringWeights` dataclass
2. Create `BackendScorer` class with latency/throughput/reliability scoring
3. Write comprehensive unit tests
4. Verify scoring curves are reasonable

### Phase 1.4: Routing Integration (2-3 hours)
1. Update `ModelRouter.select_optimal_backend()` to use scorer
2. Update `mcp_server.py` to use router
3. Write integration tests
4. Verify end-to-end routing works

### Phase 1.5: Cost Awareness (1-2 hours)
1. Add `PROVIDER_COSTS` mapping
2. Extend `ScoringWeights` with cost
3. Add `_score_cost()` method
4. Write cost-specific tests
5. Update settings.json schema

### Phase 1.6: Final Integration (1-2 hours)
1. Add configuration loading for weights
2. Verify all 615+ existing tests pass
3. Run manual testing with real backends
4. Update documentation

---

## Success Criteria

- [ ] All 615+ existing tests pass
- [ ] New tests for BackendMetrics (10+ tests)
- [ ] New tests for BackendScorer (15+ tests)
- [ ] Metrics persist across restarts
- [ ] Scoring correctly prefers faster/more reliable backends
- [ ] Cost awareness correctly prefers local/cheap backends
- [ ] No performance regression in hot paths
- [ ] Clean code following project patterns

---

## Risk Mitigation

1. **Backward compatibility**: All changes are additive; existing behavior preserved
2. **Performance**: Metrics recording is O(1); scoring is O(backends)
3. **Thread safety**: Use dataclass with simple types; deque is thread-safe for append/pop
4. **Cold start**: New backends start with neutral scores (no penalty)
5. **Feature flags**: Scoring can be disabled via config if issues arise
