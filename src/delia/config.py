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
Delia Configuration

All user-configurable values are defined here for easy customization.
Values have been validated with Wolfram Alpha for mathematical accuracy.
"""

import os
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import paths


@dataclass(frozen=True)
class ModelConfig:
    """Model tier configuration."""

    name: str
    default_model: str  # Default model name (provider-agnostic)
    vram_gb: float
    context_tokens: int
    num_ctx: int  # Request context size (conservative for quality)
    max_input_kb: int  # Practical input limit for good output quality

    @property
    def max_input_bytes(self) -> int:
        return self.max_input_kb * 1024


@dataclass(frozen=True)
class VotingConfig:
    """
    K-voting consensus configuration per MDAP paper.

    Implements "first-to-ahead-by-k" voting for mathematical reliability.
    With k=3 and p=0.99 base accuracy: P(correct) = 1/(1+((1-p)/p)^k) = 0.999999

    Formula verified with Wolfram Alpha.
    """

    enabled: bool = True
    default_k: int = 2  # First-to-ahead-by-2 (99.99% with p=0.99)
    max_k: int = 5  # Maximum votes before giving up
    auto_kmin: bool = True  # Auto-calculate k based on task complexity
    max_response_length: int = 700  # Red-flag threshold (tokens), per MDAP paper
    timeout_per_vote: float = 30.0  # Seconds per voting round
    similarity_threshold: float = 0.85  # Semantic similarity for vote matching


@dataclass
class Config:
    """Main configuration for Delia."""

    # ============================================================
    # BACKEND SELECTION (Legacy - used only as fallback)
    # Auto-detected from settings.json, can be overridden via DELIA_BACKEND
    # ============================================================
    backend: str = field(default_factory=lambda: os.getenv("DELIA_BACKEND", ""))

    # ============================================================
    # OLLAMA CONNECTION
    # ============================================================
    ollama_base: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE", "http://localhost:11434"))
    ollama_timeout_seconds: float = 300.0
    ollama_connect_timeout: float = 10.0
    # Backend type: "local" or "remote" - determines how it's referred to
    ollama_type: str = field(default_factory=lambda: os.getenv("OLLAMA_TYPE", "local"))

    # ============================================================
    # LLAMA.CPP CONNECTION (OpenAI-compatible API)
    # ============================================================
    llamacpp_base: str = field(default_factory=lambda: os.getenv("LLAMACPP_BASE", "http://localhost:8080"))
    llamacpp_timeout_seconds: float = 300.0
    llamacpp_connect_timeout: float = 10.0
    # Default model name for llama.cpp (used when no specific model is loaded)
    llamacpp_model: str = field(default_factory=lambda: os.getenv("LLAMACPP_MODEL", "Qwen3-14B-Q4_K_M.gguf"))
    # Context size limit for llama.cpp (in tokens) - used for context-aware routing
    # Set via LLAMACPP_CTX_SIZE env var or adjust based on your llama-server -c setting
    llamacpp_context_tokens: int = field(default_factory=lambda: int(os.getenv("LLAMACPP_CTX_SIZE", "8192")))
    # Backend type: "local" or "remote" - determines how it's referred to
    llamacpp_type: str = field(default_factory=lambda: os.getenv("LLAMACPP_TYPE", "remote"))

    # ============================================================
    # MODEL CONFIGURATION
    # Validated: 40K tokens @ 4 chars/token @ 75% util = 117KB theoretical
    # We use 50KB practical limit for quality output headroom
    # ============================================================
    model_quick: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="quick",
            default_model="qwen3:14b",
            vram_gb=9.0,
            context_tokens=40_000,
            num_ctx=8192,  # 8192 * 4 / 1024 = 32KB request context
            max_input_kb=50,  # 50KB ≈ 12,500 tokens, leaves room for output
        )
    )

    model_coder: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="coder",
            default_model="qwen2.5-coder:14b",
            vram_gb=9.0,
            context_tokens=128_000,
            num_ctx=16384,  # 16384 * 4 / 1024 = 64KB request context
            max_input_kb=100,  # 100KB ≈ 25,000 tokens
        )
    )

    model_moe: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="moe",
            default_model="qwen3:30b-a3b",
            vram_gb=17.0,
            context_tokens=128_000,
            num_ctx=16384,
            max_input_kb=100,
        )
    )

    # Dedicated thinking/reasoning model (uses chain-of-thought)
    model_thinking: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="thinking",
            default_model=os.getenv("THINKING_MODEL", "olmo3:7b-think"),  # or "qwen3-coder:30b" for deeper reasoning
            vram_gb=9.0,  # Adjust based on your chosen model
            context_tokens=128_000,
            num_ctx=16384,
            max_input_kb=100,
        )
    )

    # ============================================================
    # MODEL SELECTION THRESHOLDS
    # Validated: 50KB = 12,500 tokens (well within 40K context of 14B)
    # ============================================================

    # Content size threshold for upgrading to coder/moe (bytes)
    large_content_threshold: int = 50_000  # 50KB → 12,500 tokens

    # Medium content threshold (used when quick is loaded to avoid unnecessary upgrade)
    medium_content_threshold: int = 30_000  # 30KB → 7,500 tokens

    # Tasks that require MoE model (complex multi-step reasoning)
    moe_tasks: frozenset[str] = field(default_factory=lambda: frozenset({"plan", "critique"}))

    # Tasks that benefit from coder model
    coder_tasks: frozenset[str] = field(default_factory=lambda: frozenset({"generate", "review", "analyze"}))

    # Tasks that enable thinking mode
    thinking_tasks: frozenset[str] = field(default_factory=lambda: frozenset({"plan", "analyze", "critique"}))

    # ============================================================
    # GENERATION PARAMETERS
    # ============================================================

    # Temperature for normal generation (lower = more deterministic)
    temperature_normal: float = 0.3

    # Temperature when thinking mode is enabled
    temperature_thinking: float = 0.6

    # ============================================================
    # COST ESTIMATION
    # Validated: GPT-4 averages $0.03 input + $0.06 output = $0.045/1K blended
    # Using conservative $0.03/1K for comparison (input-heavy workloads)
    # ============================================================
    gpt4_cost_per_1k_tokens: float = 0.03

    # ============================================================
    # FILE HANDLING
    # ============================================================
    max_file_size: int = 500_000  # 500KB max file read

    # ============================================================
    # PERSISTENCE
    # ============================================================
    stats_file: Path = field(default_factory=lambda: paths.STATS_FILE)

    # ============================================================
    # AUTHENTICATION & MULTI-USER
    # ============================================================

    # Enable/disable authentication (set via DELIA_AUTH_ENABLED env var)
    # When disabled, all auth endpoints are hidden and tracking uses session IDs
    auth_enabled: bool = field(
        default_factory=lambda: os.getenv("DELIA_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
    )

    # Enable/disable user tracking (works with or without auth)
    tracking_enabled: bool = field(
        default_factory=lambda: os.getenv("DELIA_TRACKING_ENABLED", "true").lower() in ("true", "1", "yes")
    )

    # ============================================================
    # CONCURRENCY CONTROL
    # ============================================================

    # Maximum concurrent LLM requests per backend (prevents GPU memory exhaustion)
    # Set to 0 for unlimited (let Ollama/llama.cpp manage their own queue)
    max_concurrent_requests_per_backend: int = field(
        default_factory=lambda: int(os.getenv("DELIA_MAX_CONCURRENT", "0"))
    )

    # ============================================================
    # BACKEND TYPE HELPERS
    # ============================================================

    def get_backend_type(self, backend: str) -> str:
        """Get the type ('local' or 'remote') for a backend."""
        if backend == "ollama":
            return self.ollama_type
        elif backend == "llamacpp":
            return self.llamacpp_type
        return "local"  # Default fallback

    def get_local_backend(self) -> str | None:
        """Get the name of the backend configured as 'local', or None if none."""
        if self.ollama_type == "local":
            return "ollama"
        elif self.llamacpp_type == "local":
            return "llamacpp"
        return None

    def get_remote_backend(self) -> str | None:
        """Get the name of the backend configured as 'remote', or None if none."""
        if self.ollama_type == "remote":
            return "ollama"
        elif self.llamacpp_type == "remote":
            return "llamacpp"
        return None

    def get_preferred_backend_for_type(self, backend_type: str) -> str | None:
        """Get the preferred backend name for a given type ('local' or 'remote').

        Returns the configured backend or None if not configured.
        """
        if backend_type == "local":
            return self.get_local_backend()
        elif backend_type == "remote":
            return self.get_remote_backend()
        return None

    def is_backend_local(self, backend: str) -> bool:
        """Check if a backend is configured as local."""
        return self.get_backend_type(backend) == "local"

    def is_backend_remote(self, backend: str) -> bool:
        """Check if a backend is configured as remote."""
        return self.get_backend_type(backend) == "remote"


# Global config instance
config = Config()


# ============================================================
# PROVIDER COST ESTIMATES
# Cost per 1K tokens (blended input/output average) for routing decisions
# ============================================================

PROVIDER_COSTS: dict[str, float] = {
    # Local providers (free - just electricity)
    "local": 0.0,
    "ollama": 0.0,
    "llamacpp": 0.0,
    "vllm": 0.0,
    # Google
    "gemini": 0.0001,  # Flash is very cheap, Pro averages out
    "gemini-flash": 0.0001,
    "gemini-pro": 0.001,
    # OpenAI
    "openai": 0.005,  # Averaged across models
    "gpt-4o-mini": 0.0003,
    "gpt-4o": 0.005,
    "gpt-4-turbo": 0.01,
    # Anthropic
    "anthropic": 0.008,  # Averaged across models
    "claude-3-haiku": 0.0003,
    "claude-3-sonnet": 0.003,
    "claude-3-opus": 0.015,
    "claude-3.5-sonnet": 0.003,
    # OpenRouter (varies by model, using moderate default)
    "openrouter": 0.001,
}


def get_provider_cost(provider: str) -> float:
    """Get cost per 1K tokens for a provider.

    Args:
        provider: Provider name (e.g., 'ollama', 'gemini', 'openai')

    Returns:
        Cost per 1K tokens in USD. Returns 0.001 for unknown providers.
    """
    return PROVIDER_COSTS.get(provider.lower(), 0.001)


# ============================================================
# BACKEND HEALTH & CIRCUIT BREAKER
# Tracks failures and implements adaptive routing
# ============================================================


@dataclass
class BackendHealth:
    """
    Tracks backend health and implements circuit breaker pattern.

    Circuit Breaker States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Too many failures, requests are blocked
    - HALF-OPEN: Testing if backend recovered (auto after cooldown)

    Context Size Learning:
    - Tracks successful/failed context sizes
    - Recommends safe context sizes based on history
    """

    name: str

    # Failure tracking
    consecutive_failures: int = 0
    last_failure_time: float = 0.0  # Unix timestamp
    last_error_type: str = ""

    # Context size learning
    last_failed_context_size: int = 0
    safe_context_estimate: int = 100_000  # Start optimistic (100KB)
    max_successful_context: int = 0

    # Circuit breaker state
    circuit_open_until: float = 0.0  # Unix timestamp

    # Configuration
    failure_threshold: int = 3  # Failures before opening circuit
    base_cooldown_seconds: float = 30.0  # Base cooldown time
    max_cooldown_seconds: float = 300.0  # Max 5 min cooldown
    context_reduction_factor: float = 0.7  # Reduce context by 30% after timeout

    def record_failure(self, error_type: str, context_size: int = 0) -> None:
        """Record a backend failure and potentially open circuit."""
        import time

        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.last_error_type = error_type

        # Learn from timeout failures
        if error_type == "timeout" and context_size > 0:
            self.last_failed_context_size = context_size
            # Reduce safe estimate to 70% of failed size
            new_estimate = int(context_size * self.context_reduction_factor)
            self.safe_context_estimate = min(self.safe_context_estimate, new_estimate)

        # Open circuit if threshold exceeded
        if self.consecutive_failures >= self.failure_threshold:
            # Exponential backoff: 30s, 60s, 120s, 240s, 300s (max)
            cooldown = min(
                self.base_cooldown_seconds * (2 ** (self.consecutive_failures - self.failure_threshold)),
                self.max_cooldown_seconds,
            )
            self.circuit_open_until = time.time() + cooldown

    def record_success(self, context_size: int = 0) -> None:
        """Record a successful call and reset circuit."""
        self.consecutive_failures = 0
        self.circuit_open_until = 0.0
        self.last_error_type = ""

        # Learn successful context sizes
        if context_size > 0:
            self.max_successful_context = max(self.max_successful_context, context_size)
            # Slowly increase safe estimate on success (conservative)
            if context_size > self.safe_context_estimate * 0.9:
                self.safe_context_estimate = min(
                    int(context_size * 1.1),  # Allow 10% growth
                    100_000,  # Cap at 100KB
                )

    def is_available(self) -> bool:
        """Check if backend is available (circuit not open)."""
        import time

        return not (self.circuit_open_until > 0 and time.time() < self.circuit_open_until)

    def time_until_available(self) -> float:
        """Seconds until circuit closes (0 if available)."""
        import time

        if not self.is_available():
            return max(0, self.circuit_open_until - time.time())
        return 0.0

    def should_reduce_context(self, proposed_size: int) -> tuple[bool, int]:
        """
        Check if context should be reduced based on failure history.

        Returns:
            (should_reduce, recommended_size)
        """
        # If we have recent failures and proposed size exceeds safe estimate
        if self.consecutive_failures > 0 and proposed_size > self.safe_context_estimate:
            recommended = int(self.safe_context_estimate * 0.9)  # 10% safety margin
            return True, recommended
        return False, proposed_size

    def get_status(self) -> dict[str, object]:
        """Get current health status as dict."""
        import time

        return {
            "available": self.is_available(),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error_type,
            "circuit_open": self.circuit_open_until > time.time(),
            "seconds_until_available": round(self.time_until_available(), 1),
            "safe_context_kb": self.safe_context_estimate // 1024,
        }


# Maximum number of latency samples to keep (rolling window)
_METRICS_WINDOW_SIZE: int = 100


@dataclass
class BackendMetrics:
    """
    Rolling performance metrics for a backend.

    Tracks latency, success rate, and throughput for routing decisions.
    Uses a rolling window for latency samples to avoid unbounded memory growth.

    Thread Safety:
        This class is NOT thread-safe. The caller (typically config module)
        should ensure thread-safe access if needed.
    """

    backend_id: str

    # Latency tracking (rolling window of milliseconds)
    _latency_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=_METRICS_WINDOW_SIZE)
    )

    # Request counts (all-time)
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0

    # Token tracking (all-time)
    total_tokens: int = 0

    # Timestamps
    last_request_time: float = 0.0

    def record_success(self, elapsed_ms: float, tokens: int = 0) -> None:
        """
        Record a successful request.

        Args:
            elapsed_ms: Request latency in milliseconds
            tokens: Number of tokens processed (0 if unknown)
        """
        self._latency_samples.append(elapsed_ms)
        self.total_requests += 1
        self.total_successes += 1
        self.total_tokens += tokens
        self.last_request_time = time.time()

    def record_failure(self, elapsed_ms: float = 0) -> None:
        """
        Record a failed request.

        Args:
            elapsed_ms: Request latency in milliseconds (0 if timed out)
        """
        if elapsed_ms > 0:
            self._latency_samples.append(elapsed_ms)
        self.total_requests += 1
        self.total_failures += 1
        self.last_request_time = time.time()

    @property
    def success_rate(self) -> float:
        """
        Success rate as a fraction (0.0 to 1.0).

        Returns 1.0 if no requests have been made (optimistic default).
        """
        if self.total_requests == 0:
            return 1.0
        return self.total_successes / self.total_requests

    @property
    def latency_p50(self) -> float:
        """
        Median latency in milliseconds.

        Returns 0.0 if no samples available.
        """
        if not self._latency_samples:
            return 0.0
        sorted_samples = sorted(self._latency_samples)
        mid = len(sorted_samples) // 2
        return sorted_samples[mid]

    @property
    def latency_p95(self) -> float:
        """
        95th percentile latency in milliseconds.

        Returns 0.0 if no samples available.
        """
        if not self._latency_samples:
            return 0.0
        sorted_samples = sorted(self._latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        # Clamp to valid range
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def tokens_per_second(self) -> float:
        """
        Average throughput in tokens per second.

        Calculated from total tokens and average latency.
        Returns 0.0 if no successful requests with tokens.
        """
        if self.total_successes == 0 or self.total_tokens == 0:
            return 0.0
        if not self._latency_samples:
            return 0.0

        avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
        if avg_latency_ms <= 0:
            return 0.0

        avg_tokens_per_request = self.total_tokens / self.total_successes
        # Convert ms to seconds: tokens / (ms / 1000) = tokens * 1000 / ms
        return avg_tokens_per_request * 1000 / avg_latency_ms

    @property
    def sample_count(self) -> int:
        """Number of latency samples currently stored."""
        return len(self._latency_samples)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics for persistence."""
        return {
            "backend_id": self.backend_id,
            "latency_samples": list(self._latency_samples),
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_tokens": self.total_tokens,
            "last_request_time": self.last_request_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackendMetrics":
        """Deserialize metrics from persistence."""
        metrics = cls(backend_id=data.get("backend_id", "unknown"))
        metrics._latency_samples = deque(
            data.get("latency_samples", []),
            maxlen=_METRICS_WINDOW_SIZE,
        )
        metrics.total_requests = data.get("total_requests", 0)
        metrics.total_successes = data.get("total_successes", 0)
        metrics.total_failures = data.get("total_failures", 0)
        metrics.total_tokens = data.get("total_tokens", 0)
        metrics.last_request_time = data.get("last_request_time", 0.0)
        return metrics

    def get_status(self) -> dict[str, Any]:
        """Get current metrics status as dict (for health endpoint)."""
        return {
            "backend_id": self.backend_id,
            "success_rate": round(self.success_rate, 3),
            "latency_p50_ms": round(self.latency_p50, 1),
            "latency_p95_ms": round(self.latency_p95, 1),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "total_requests": self.total_requests,
            "sample_count": self.sample_count,
        }


# Global health trackers for each backend (dynamic)
BACKEND_HEALTH: dict[str, BackendHealth] = {}


def get_backend_health(backend: str) -> BackendHealth:
    """Get or create health tracker for a backend."""
    if backend not in BACKEND_HEALTH:
        BACKEND_HEALTH[backend] = BackendHealth(name=backend)
    return BACKEND_HEALTH[backend]


# Global metrics trackers for each backend (dynamic)
BACKEND_METRICS: dict[str, BackendMetrics] = {}


def get_backend_metrics(backend_id: str) -> BackendMetrics:
    """
    Get or create metrics tracker for a backend.

    Args:
        backend_id: Unique identifier for the backend

    Returns:
        BackendMetrics instance for the backend
    """
    if backend_id not in BACKEND_METRICS:
        BACKEND_METRICS[backend_id] = BackendMetrics(backend_id=backend_id)
    return BACKEND_METRICS[backend_id]


def load_backend_metrics() -> None:
    """
    Load backend metrics from disk.

    Called at startup to restore metrics from previous session.
    Silently handles missing or corrupt files.
    """
    import json
    import structlog

    log = structlog.get_logger()
    metrics_file = paths.BACKEND_METRICS_FILE

    if not metrics_file.exists():
        return

    try:
        data = json.loads(metrics_file.read_text())
        for backend_id, metrics_data in data.items():
            BACKEND_METRICS[backend_id] = BackendMetrics.from_dict(metrics_data)
        log.debug("backend_metrics_loaded", count=len(BACKEND_METRICS))
    except json.JSONDecodeError as e:
        log.warning("backend_metrics_load_failed", error=str(e), reason="invalid_json")
    except Exception as e:
        log.warning("backend_metrics_load_failed", error=str(e))


def save_backend_metrics() -> None:
    """
    Save backend metrics to disk.

    Uses atomic write (temp file + rename) to prevent corruption.
    """
    import json
    import structlog

    log = structlog.get_logger()
    metrics_file = paths.BACKEND_METRICS_FILE

    try:
        paths.ensure_directories()
        data = {
            backend_id: metrics.to_dict()
            for backend_id, metrics in BACKEND_METRICS.items()
        }
        temp_file = metrics_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(metrics_file)  # Atomic on POSIX
    except Exception as e:
        log.warning("backend_metrics_save_failed", error=str(e))


# ============================================================
# TASK-BACKEND AFFINITY TRACKING
# ============================================================


@dataclass
class AffinityTracker:
    """
    Track backend performance per task type using Exponential Moving Average.

    Learns which backends perform better for specific task types (review, generate,
    analyze, etc.) based on success/failure outcomes. Used by BackendScorer to
    boost scores for backends with proven task-specific performance.

    EMA Formula: new_score = old_score * (1 - alpha) + quality * alpha
    Where quality is 1.0 for success, 0.0 for failure.
    """

    alpha: float = 0.1  # EMA decay factor (higher = faster adaptation)

    # (backend_id, task_type) -> EMA score [0.0-1.0]
    _scores: dict[tuple[str, str], float] = field(default_factory=dict)

    def update(
        self,
        backend_id: str,
        task_type: str,
        success: bool | None = None,
        quality: float | None = None,
    ) -> None:
        """
        Update affinity score with new observation.

        Accepts either a boolean success flag OR a float quality score (0.0-1.0).
        The quality score allows more nuanced learning from response quality
        rather than just binary success/failure.

        Args:
            backend_id: Backend identifier
            task_type: Task type (review, generate, analyze, etc.)
            success: Whether the request succeeded (legacy, maps to 1.0/0.0)
            quality: Quality score from 0.0 to 1.0 (preferred)

        Raises:
            ValueError: If neither success nor quality is provided
        """
        # Determine quality score from inputs
        if quality is not None:
            # Use explicit quality score, clamp to [0.0, 1.0]
            q = max(0.0, min(1.0, quality))
        elif success is not None:
            # Legacy boolean: map to 1.0/0.0
            q = 1.0 if success else 0.0
        else:
            raise ValueError("Either 'success' or 'quality' must be provided")

        key = (backend_id, task_type)
        old = self._scores.get(key, 0.5)  # Start neutral
        self._scores[key] = old * (1 - self.alpha) + q * self.alpha

    def get_affinity(self, backend_id: str, task_type: str) -> float:
        """
        Get affinity score for backend+task combo.

        Returns:
            Score from 0.0 to 1.0 (0.5 = neutral/unknown)
        """
        return self._scores.get((backend_id, task_type), 0.5)

    def update_with_outcome(
        self,
        backend_id: str,
        task_type: str,
        succeeded: bool,
        efficiency: float = 1.0,
    ) -> None:
        """
        Update affinity based on outcome + efficiency (ToolOrchestra-style).

        Implements a three-reward system inspired by ToolOrchestra paper:
        - Outcome reward: Binary success/failure (50% weight)
        - Efficiency reward: Token/latency efficiency (30% weight)
        - Preference: Historical preference score (20% weight)

        Formula: combined = 0.5*outcome + 0.3*efficiency + 0.2*preference

        Args:
            backend_id: Backend identifier
            task_type: Task type (review, generate, analyze, etc.)
            succeeded: Whether the task completed successfully
            efficiency: Token efficiency score 0.0-1.0 (1.0 = optimal)
        """
        # Outcome reward: 1.0 for success, 0.0 for failure
        outcome_reward = 1.0 if succeeded else 0.0

        # Efficiency reward: already 0.0-1.0
        efficiency_reward = max(0.0, min(1.0, efficiency))

        # Preference: use current affinity as proxy for learned preference
        preference = self.get_affinity(backend_id, task_type)

        # Combined score (ToolOrchestra weights)
        combined = 0.5 * outcome_reward + 0.3 * efficiency_reward + 0.2 * preference

        # EMA update with combined score
        self.update(backend_id, task_type, quality=combined)

    def boost_score(
        self, base_score: float, backend_id: str, task_type: str
    ) -> float:
        """
        Adjust a backend score based on task affinity.

        Args:
            base_score: Original backend score
            backend_id: Backend identifier
            task_type: Task type

        Returns:
            Adjusted score (0.5 affinity = no change,
            1.0 = +20% boost, 0.0 = -20% penalty)
        """
        affinity = self.get_affinity(backend_id, task_type)
        # Linear modifier: 0.5 -> 1.0, 1.0 -> 1.2, 0.0 -> 0.8
        modifier = 1.0 + (affinity - 0.5) * 0.4
        return base_score * modifier

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        # Convert tuple keys to strings for JSON
        return {
            "alpha": self.alpha,
            "scores": {f"{k[0]}:{k[1]}": v for k, v in self._scores.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AffinityTracker":
        """Deserialize from persistence."""
        tracker = cls(alpha=data.get("alpha", 0.1))
        scores_data = data.get("scores", {})
        for key_str, score in scores_data.items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                tracker._scores[(parts[0], parts[1])] = score
        return tracker

    def get_status(self) -> dict[str, Any]:
        """Get current status for health endpoint."""
        return {
            "alpha": self.alpha,
            "tracked_pairs": len(self._scores),
            "scores": {f"{k[0]}:{k[1]}": round(v, 3) for k, v in self._scores.items()},
        }


# Global affinity tracker instance
AFFINITY_TRACKER = AffinityTracker()


def get_affinity_tracker() -> AffinityTracker:
    """Get the global affinity tracker instance."""
    return AFFINITY_TRACKER


def load_affinity() -> None:
    """
    Load affinity data from disk.

    Called at startup to restore learned affinities from previous session.
    """
    import json
    import structlog

    log = structlog.get_logger()
    affinity_file = paths.AFFINITY_FILE

    if not affinity_file.exists():
        return

    try:
        data = json.loads(affinity_file.read_text())
        global AFFINITY_TRACKER
        AFFINITY_TRACKER = AffinityTracker.from_dict(data)
        log.debug("affinity_loaded", tracked_pairs=len(AFFINITY_TRACKER._scores))
    except json.JSONDecodeError as e:
        log.warning("affinity_load_failed", error=str(e), reason="invalid_json")
    except Exception as e:
        log.warning("affinity_load_failed", error=str(e))


def save_affinity() -> None:
    """
    Save affinity data to disk.

    Uses atomic write (temp file + rename) to prevent corruption.
    """
    import json
    import structlog

    log = structlog.get_logger()
    affinity_file = paths.AFFINITY_FILE

    try:
        paths.ensure_directories()
        data = AFFINITY_TRACKER.to_dict()
        temp_file = affinity_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(affinity_file)  # Atomic on POSIX
    except Exception as e:
        log.warning("affinity_save_failed", error=str(e))


# ============================================================
# PRE-WARMING TRACKER
# Predicts model needs based on hourly usage patterns
# ============================================================


@dataclass
class PrewarmTracker:
    """
    Track hourly model tier usage patterns using Exponential Moving Average.

    Learns which model tiers are used at different hours of the day to enable
    predictive pre-warming. For each hour (0-23), tracks EMA scores per tier.

    EMA Formula: new_score = old_score * (1 - alpha) + 1.0 * alpha
    When a tier is used, its score for the current hour increases.
    """

    alpha: float = 0.15  # EMA decay factor (higher = faster adaptation)
    threshold: float = 0.3  # Minimum score to recommend pre-warming

    # (hour, tier) -> EMA score [0.0-1.0]
    _scores: dict[tuple[int, str], float] = field(default_factory=dict)

    def update(self, tier: str) -> None:
        """
        Update usage score for current hour and tier.

        Args:
            tier: Model tier that was used (quick, coder, moe, thinking)
        """
        from datetime import datetime

        hour = datetime.now().hour
        key = (hour, tier)
        old = self._scores.get(key, 0.0)
        self._scores[key] = old * (1 - self.alpha) + self.alpha

    def get_predicted_tiers(self, hour: int | None = None) -> list[str]:
        """
        Get tiers likely to be needed at the given hour.

        Args:
            hour: Hour of day (0-23). If None, uses current hour.

        Returns:
            List of tier names with scores above threshold, sorted by score descending.
        """
        from datetime import datetime

        if hour is None:
            hour = datetime.now().hour

        # Find all tiers for this hour with score above threshold
        candidates = [
            (tier, score)
            for (h, tier), score in self._scores.items()
            if h == hour and score >= self.threshold
        ]
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [tier for tier, _ in candidates]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "scores": {f"{k[0]}:{k[1]}": v for k, v in self._scores.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrewarmTracker":
        """Deserialize from persistence."""
        tracker = cls(
            alpha=data.get("alpha", 0.15),
            threshold=data.get("threshold", 0.3),
        )
        scores_data = data.get("scores", {})
        for key_str, score in scores_data.items():
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                try:
                    hour = int(parts[0])
                    tier = parts[1]
                    tracker._scores[(hour, tier)] = score
                except ValueError:
                    pass  # Skip malformed entries
        return tracker

    def get_status(self) -> dict[str, Any]:
        """Get current status for health endpoint."""
        from datetime import datetime

        current_hour = datetime.now().hour
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "tracked_entries": len(self._scores),
            "current_hour": current_hour,
            "predicted_tiers": self.get_predicted_tiers(current_hour),
            "top_scores": {
                f"{h}:{t}": round(s, 3)
                for (h, t), s in sorted(
                    self._scores.items(), key=lambda x: x[1], reverse=True
                )[:10]
            },
        }


# Global prewarm tracker instance
PREWARM_TRACKER = PrewarmTracker()


def get_prewarm_tracker() -> PrewarmTracker:
    """Get the global prewarm tracker instance."""
    return PREWARM_TRACKER


def load_prewarm() -> None:
    """
    Load prewarm data from disk.

    Called at startup to restore learned patterns from previous session.
    """
    import json
    import structlog

    log = structlog.get_logger()
    prewarm_file = paths.PREWARM_FILE

    if not prewarm_file.exists():
        return

    try:
        data = json.loads(prewarm_file.read_text())
        global PREWARM_TRACKER
        PREWARM_TRACKER = PrewarmTracker.from_dict(data)
        log.debug("prewarm_loaded", tracked_entries=len(PREWARM_TRACKER._scores))
    except json.JSONDecodeError as e:
        log.warning("prewarm_load_failed", error=str(e), reason="invalid_json")
    except Exception as e:
        log.warning("prewarm_load_failed", error=str(e))


def save_prewarm() -> None:
    """
    Save prewarm data to disk.

    Uses atomic write (temp file + rename) to prevent corruption.
    """
    import json
    import structlog

    log = structlog.get_logger()
    prewarm_file = paths.PREWARM_FILE

    try:
        paths.ensure_directories()
        data = PREWARM_TRACKER.to_dict()
        temp_file = prewarm_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(prewarm_file)  # Atomic on POSIX
    except Exception as e:
        log.warning("prewarm_save_failed", error=str(e))


# ============================================================
# DERIVED VALUES (for backwards compatibility)
# ============================================================

# Load models from backends.json if available, otherwise use config defaults
# DEPRECATED: Now handled by backend_manager.py and settings.json

STATS_FILE = config.stats_file


# ============================================================
# DYNAMIC MODEL DETECTION
# HuggingFace-informed model parsing for natural language model names
# Supports: Qwen, Llama, Mistral, DeepSeek, Phi, Gemma, Falcon, Yi, GPT variants
# ============================================================

import re
from typing import NamedTuple

# ============================================================
# REGEX PATTERNS FOR MODEL NAME PARSING
# ============================================================

# Extract parameter count: "14B", "7b", "72B", "0.5B", "1.5b"
_PARAM_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:illion)?(?![a-zA-Z])", re.IGNORECASE)

# Extract version numbers: "v0.1", "2.5", "3.1", "-v3"
_VERSION_REGEX = re.compile(r"(?:v?(\d+(?:\.\d+)*))", re.IGNORECASE)

# Detect MoE (Mixture of Experts) patterns: "-a3b", "MoE", "mixtral"
_MOE_REGEX = re.compile(r"(-a\d+b|moe|mixtral)", re.IGNORECASE)

# ============================================================
# MODEL FAMILY DETECTION (HuggingFace top models)
# ============================================================

MODEL_FAMILIES = {
    # Family name -> (aliases, is_primarily_coder)
    "qwen": (["qwen", "qwen2", "qwen2.5", "qwen3"], False),
    "llama": (["llama", "llama2", "llama3", "llama-3", "meta-llama", "codellama"], False),
    "mistral": (["mistral", "mixtral", "mistral-nemo"], False),
    "deepseek": (["deepseek", "deepseek-v2", "deepseek-v3", "deepseek-coder"], False),
    "phi": (["phi", "phi-2", "phi-3", "phi-4", "phi2", "phi3", "phi4"], False),
    "gemma": (["gemma", "gemma2", "gemma-2", "codegemma"], False),
    "falcon": (["falcon", "falcon2"], False),
    "yi": (["yi", "yi-coder", "yi-1.5"], False),
    "gpt": (["gpt2", "gpt-j", "gpt-neox", "gpt-neo"], False),
    "starcoder": (["starcoder", "starcoder2", "starcoderbase"], True),
    "codellama": (["codellama", "code-llama"], True),
    "codegemma": (["codegemma", "code-gemma"], True),
    "nemotron": (["nemotron"], False),
    "hermes": (["hermes", "nous-hermes"], False),
    "orca": (["orca", "orca2"], False),
    "wizardcoder": (["wizardcoder", "wizard-coder"], True),
    "magicoder": (["magicoder"], True),
}

# ============================================================
# VARIANT DETECTION
# ============================================================

# Coder-specialized keywords
CODER_KEYWORDS = frozenset(
    [
        "coder",
        "code",
        "codellama",
        "starcoder",
        "codegemma",
        "wizardcoder",
        "magicoder",
        "deepseek-coder",
        "yi-coder",
        "codestral",
        "granite-code",
        "stable-code",
    ]
)

# Instruction-tuned keywords
INSTRUCT_KEYWORDS = frozenset(["instruct", "chat", "-it", "assistant", "rlhf"])

# Base model keywords
BASE_KEYWORDS = frozenset(["base", "foundation", "pretrain", "raw"])


class ModelInfo(NamedTuple):
    """Parsed model information."""

    params_b: float  # Parameter count in billions (0 if unknown)
    family: str | None  # Model family (qwen, llama, etc.)
    is_coder: bool  # Is this a code-specialized model?
    is_moe: bool  # Is this a Mixture of Experts model?
    is_instruct: bool  # Is this instruction-tuned?
    raw_name: str  # Original model name


@lru_cache(maxsize=256)
def parse_model_name(name: str) -> ModelInfo:
    """
    Parse a model name to extract characteristics.

    Cached to avoid repeated regex parsing for the same model names.

    Examples:
        "qwen2.5-coder:14b" → (14.0, "qwen", True, False, False, ...)
        "llama-3.1-70b-instruct" → (70.0, "llama", False, False, True, ...)
        "mixtral-8x7b" → (56.0, "mistral", False, True, False, ...)
        "deepseek-v3" → (0.0, "deepseek", False, False, False, ...)
    """
    name_lower = name.lower()

    # Extract parameter count
    param_match = _PARAM_REGEX.search(name)
    params = float(param_match.group(1)) if param_match else 0.0

    # Handle MoE notation like "8x7b" → 56B total
    moe_mult_match = re.search(r"(\d+)x(\d+)b", name_lower)
    if moe_mult_match:
        experts = int(moe_mult_match.group(1))
        per_expert = int(moe_mult_match.group(2))
        params = float(experts * per_expert)

    # Detect model family
    family = None
    family_is_coder = False
    for fam_name, (aliases, is_coder_family) in MODEL_FAMILIES.items():
        if any(alias in name_lower for alias in aliases):
            family = fam_name
            family_is_coder = is_coder_family
            break

    # Detect coder variant
    is_coder = family_is_coder or any(kw in name_lower for kw in CODER_KEYWORDS)

    # Detect MoE variant
    is_moe = bool(_MOE_REGEX.search(name))

    # Detect instruction-tuning
    is_instruct = any(kw in name_lower for kw in INSTRUCT_KEYWORDS)

    return ModelInfo(
        params_b=params, family=family, is_coder=is_coder, is_moe=is_moe, is_instruct=is_instruct, raw_name=name
    )


@lru_cache(maxsize=256)
def _detect_model_tier_cached(model_name: str) -> str:
    """
    Cached tier detection based purely on model name parsing.

    This is the core logic - separated to enable caching.
    """
    info = parse_model_name(model_name)

    # MoE models get moe tier (complex reasoning capability)
    if info.is_moe:
        return "moe"

    # Large models (≥30B) get moe tier
    if info.params_b >= 30:
        return "moe"

    # Coder-specialized models get coder tier
    if info.is_coder:
        return "coder"

    # Medium models (15-30B) can handle complex tasks
    if 15 <= info.params_b < 30:
        return "coder"

    # Small models or unknown → quick tier
    return "quick"


def detect_model_tier(model_name: str, known_models: dict[str, str] | None = None) -> str:
    """
    Detect which tier a model belongs to based on parsed characteristics.

    Tier Assignment Strategy:
    1. Exact match against configured tier models (if provided)
    2. MoE or large params (≥30B) → moe tier
    3. Coder-specialized → coder tier
    4. Medium params (15-30B) → coder tier (capable enough)
    5. Small params (<15B) or unknown → quick tier

    Returns 'quick', 'coder', or 'moe'.
    """
    # Priority 1: Exact match against configured models (not cached - dict not hashable)
    if known_models:
        model_lower = model_name.lower()
        for tier, tier_model in known_models.items():
            tier_normalized = tier_model.lower().replace(":latest", "")
            model_normalized = model_lower.replace(":latest", "")
            if tier_normalized == model_normalized or tier_normalized in model_normalized:
                return tier

    # Priority 2: Use cached tier detection based on model parsing
    return _detect_model_tier_cached(model_name)
