"""Voting statistics tracker for MDAP k-voting consensus.

Tracks:
- Consensus metrics (votes to reach, distribution, success rate)
- Rejection metrics (reasons, by backend, by tier)
- Per-tier quality statistics

Used for tuning kmin, quality thresholds, and backend affinity.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from .paths import DATA_DIR

log = structlog.get_logger()

# Singleton instance
_voting_stats_tracker: VotingStatsTracker | None = None
_tracker_lock = threading.Lock()


@dataclass
class ConsensusStats:
    """Statistics about voting consensus outcomes."""

    total_attempts: int = 0
    successful: int = 0
    failed: int = 0
    # Histogram: votes_needed -> count
    votes_histogram: dict[int, int] = field(default_factory=dict)
    # Recent samples for percentile calculation (rolling window)
    votes_samples: list[int] = field(default_factory=list)
    # Max samples to keep for percentile calculation
    max_samples: int = 1000

    def record(self, votes_needed: int, success: bool) -> None:
        """Record a consensus attempt."""
        self.total_attempts += 1
        if success:
            self.successful += 1
            # Track histogram
            key = str(votes_needed)  # JSON keys must be strings
            self.votes_histogram[key] = self.votes_histogram.get(key, 0) + 1
            # Track samples for percentile
            self.votes_samples.append(votes_needed)
            if len(self.votes_samples) > self.max_samples:
                self.votes_samples.pop(0)
        else:
            self.failed += 1

    @property
    def consensus_rate(self) -> float:
        """Percentage of attempts that reached consensus."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful / self.total_attempts

    @property
    def avg_votes(self) -> float:
        """Average votes needed to reach consensus."""
        if not self.votes_samples:
            return 0.0
        return sum(self.votes_samples) / len(self.votes_samples)

    def percentile(self, p: float) -> int:
        """Calculate percentile of votes needed (0.0-1.0)."""
        if not self.votes_samples:
            return 0
        sorted_samples = sorted(self.votes_samples)
        # Verified with Wolfram: ceiling(p * n) - 1 for 0-indexed
        idx = min(math.ceil(p * len(sorted_samples)) - 1, len(sorted_samples) - 1)
        idx = max(0, idx)
        return sorted_samples[idx]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence."""
        return {
            "total_attempts": self.total_attempts,
            "successful": self.successful,
            "failed": self.failed,
            "votes_histogram": self.votes_histogram,
            "votes_samples": self.votes_samples[-self.max_samples :],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsensusStats:
        """Deserialize from JSON."""
        stats = cls()
        stats.total_attempts = data.get("total_attempts", 0)
        stats.successful = data.get("successful", 0)
        stats.failed = data.get("failed", 0)
        stats.votes_histogram = data.get("votes_histogram", {})
        stats.votes_samples = data.get("votes_samples", [])
        return stats


@dataclass
class RejectionRecord:
    """A single rejection event."""

    timestamp: float
    reason: str
    backend_id: str
    tier: str
    response_preview: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON."""
        return {
            "timestamp": self.timestamp,
            "reason": self.reason,
            "backend_id": self.backend_id,
            "tier": self.tier,
            "response_preview": self.response_preview,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RejectionRecord:
        """Deserialize from JSON."""
        return cls(
            timestamp=data.get("timestamp", 0),
            reason=data.get("reason", "unknown"),
            backend_id=data.get("backend_id", "unknown"),
            tier=data.get("tier", "unknown"),
            response_preview=data.get("response_preview", ""),
        )


@dataclass
class RejectionStats:
    """Statistics about quality rejections."""

    total: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)
    by_backend: dict[str, int] = field(default_factory=dict)
    by_tier: dict[str, int] = field(default_factory=dict)
    # Recent rejections for debugging (last 100)
    recent: deque[RejectionRecord] = field(default_factory=lambda: deque(maxlen=100))

    def record(
        self, reason: str, backend_id: str, tier: str, response_preview: str
    ) -> None:
        """Record a rejection event."""
        self.total += 1
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1
        self.by_backend[backend_id] = self.by_backend.get(backend_id, 0) + 1
        self.by_tier[tier] = self.by_tier.get(tier, 0) + 1

        self.recent.append(
            RejectionRecord(
                timestamp=time.time(),
                reason=reason,
                backend_id=backend_id,
                tier=tier,
                response_preview=response_preview[:100],
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence."""
        return {
            "total": self.total,
            "by_reason": self.by_reason,
            "by_backend": self.by_backend,
            "by_tier": self.by_tier,
            "recent": [r.to_dict() for r in self.recent],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RejectionStats:
        """Deserialize from JSON."""
        stats = cls()
        stats.total = data.get("total", 0)
        stats.by_reason = data.get("by_reason", {})
        stats.by_backend = data.get("by_backend", {})
        stats.by_tier = data.get("by_tier", {})
        recent_data = data.get("recent", [])
        stats.recent = deque(
            [RejectionRecord.from_dict(r) for r in recent_data], maxlen=100
        )
        return stats


@dataclass
class TierStats:
    """Statistics for a single model tier."""

    calls: int = 0
    quality_sum: float = 0.0
    rejections: int = 0
    consensus_attempts: int = 0
    consensus_successes: int = 0
    # EMA for quality (alpha=0.2, verified with Wolfram)
    quality_ema: float = 0.5
    ema_alpha: float = 0.2

    def record_call(self, quality_score: float) -> None:
        """Record a call with its quality score."""
        self.calls += 1
        self.quality_sum += quality_score
        # Update EMA: new = alpha * value + (1-alpha) * old
        self.quality_ema = self.ema_alpha * quality_score + (1 - self.ema_alpha) * self.quality_ema

    def record_rejection(self) -> None:
        """Record a quality rejection."""
        self.rejections += 1

    def record_consensus(self, success: bool) -> None:
        """Record a consensus attempt."""
        self.consensus_attempts += 1
        if success:
            self.consensus_successes += 1

    @property
    def avg_quality(self) -> float:
        """Average quality score."""
        if self.calls == 0:
            return 0.0
        return self.quality_sum / self.calls

    @property
    def rejection_rate(self) -> float:
        """Percentage of calls that were rejected."""
        if self.calls == 0:
            return 0.0
        return self.rejections / self.calls

    @property
    def consensus_rate(self) -> float:
        """Percentage of consensus attempts that succeeded."""
        if self.consensus_attempts == 0:
            return 0.0
        return self.consensus_successes / self.consensus_attempts

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON."""
        return {
            "calls": self.calls,
            "quality_sum": self.quality_sum,
            "rejections": self.rejections,
            "consensus_attempts": self.consensus_attempts,
            "consensus_successes": self.consensus_successes,
            "quality_ema": self.quality_ema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TierStats:
        """Deserialize from JSON."""
        stats = cls()
        stats.calls = data.get("calls", 0)
        stats.quality_sum = data.get("quality_sum", 0.0)
        stats.rejections = data.get("rejections", 0)
        stats.consensus_attempts = data.get("consensus_attempts", 0)
        stats.consensus_successes = data.get("consensus_successes", 0)
        stats.quality_ema = data.get("quality_ema", 0.5)
        return stats


class VotingStatsTracker:
    """Tracks voting consensus and quality metrics.

    Thread-safe singleton that persists to voting_stats.json.

    Usage:
        tracker = get_voting_stats_tracker()
        tracker.record_consensus(votes_cast=2, k=2, tier="quick", backend_id="ollama")
        tracker.record_rejection(reason="too_long", backend_id="ollama", tier="coder", preview="...")
        stats = tracker.get_stats()
    """

    def __init__(self, stats_file: Path | None = None):
        """Initialize tracker with optional custom stats file path."""
        self._stats_file = stats_file or (DATA_DIR / "voting_stats.json")
        self._lock = threading.Lock()

        # Initialize stats
        self.consensus = ConsensusStats()
        self.rejections = RejectionStats()
        self.tiers: dict[str, TierStats] = {}

        # Load existing stats
        self._load()

    def _load(self) -> None:
        """Load stats from file."""
        if not self._stats_file.exists():
            return

        try:
            with open(self._stats_file) as f:
                data = json.load(f)

            self.consensus = ConsensusStats.from_dict(data.get("consensus", {}))
            self.rejections = RejectionStats.from_dict(data.get("rejections", {}))

            for tier_name, tier_data in data.get("tiers", {}).items():
                self.tiers[tier_name] = TierStats.from_dict(tier_data)

            log.debug(
                "voting_stats_loaded",
                consensus_attempts=self.consensus.total_attempts,
                rejections=self.rejections.total,
                tiers=list(self.tiers.keys()),
            )
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("voting_stats_load_failed", error=str(e))

    def _save(self) -> None:
        """Save stats to file atomically."""
        data = {
            "consensus": self.consensus.to_dict(),
            "rejections": self.rejections.to_dict(),
            "tiers": {name: tier.to_dict() for name, tier in self.tiers.items()},
            "last_updated": time.time(),
        }

        # Atomic write: write to temp file, then rename
        temp_file = self._stats_file.with_suffix(".tmp")
        try:
            self._stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self._stats_file)
        except OSError as e:
            log.error("voting_stats_save_failed", error=str(e))
            if temp_file.exists():
                temp_file.unlink()

    def _get_tier(self, tier: str) -> TierStats:
        """Get or create tier stats."""
        if tier not in self.tiers:
            self.tiers[tier] = TierStats()
        return self.tiers[tier]

    def record_consensus(
        self, votes_cast: int, k: int, tier: str, backend_id: str, success: bool = True
    ) -> None:
        """Record a consensus attempt.

        Args:
            votes_cast: Number of votes cast before consensus/failure
            k: The k value used (votes needed)
            tier: Model tier (quick, coder, moe, thinking)
            backend_id: Winning backend ID
            success: Whether consensus was reached
        """
        with self._lock:
            self.consensus.record(votes_cast, success)
            self._get_tier(tier).record_consensus(success)
            self._save()

        log.debug(
            "voting_consensus_recorded",
            votes=votes_cast,
            k=k,
            tier=tier,
            backend=backend_id,
            success=success,
        )

    def record_rejection(
        self, reason: str, backend_id: str, tier: str, response_preview: str
    ) -> None:
        """Record a quality rejection (red-flag).

        Args:
            reason: Rejection reason (too_long, repetitive, etc.)
            backend_id: Backend that produced the rejected response
            tier: Model tier
            response_preview: First 100 chars of rejected response
        """
        with self._lock:
            self.rejections.record(reason, backend_id, tier, response_preview)
            self._get_tier(tier).record_rejection()
            self._save()

        log.debug(
            "voting_rejection_recorded",
            reason=reason,
            backend=backend_id,
            tier=tier,
        )

    def record_quality(self, tier: str, quality_score: float) -> None:
        """Record a quality score for a response.

        Args:
            tier: Model tier
            quality_score: Quality score 0.0-1.0
        """
        with self._lock:
            self._get_tier(tier).record_call(quality_score)
            # Don't save on every quality record (too frequent)
            # Will be saved on next consensus/rejection

    def get_stats(self) -> dict[str, Any]:
        """Get all statistics as a dictionary.

        Returns formatted stats suitable for dashboard display.
        """
        with self._lock:
            # Consensus summary
            consensus_summary = {
                "total_attempts": self.consensus.total_attempts,
                "successful": self.consensus.successful,
                "failed": self.consensus.failed,
                "consensus_rate": round(self.consensus.consensus_rate * 100, 1),
                "avg_votes": round(self.consensus.avg_votes, 2),
                "p50_votes": self.consensus.percentile(0.5),
                "p95_votes": self.consensus.percentile(0.95),
                "distribution": self._format_distribution(),
            }

            # Rejection summary
            rejection_summary = {
                "total": self.rejections.total,
                "by_reason": self.rejections.by_reason,
                "by_backend": self.rejections.by_backend,
                "by_tier": self.rejections.by_tier,
                "recent": [r.to_dict() for r in list(self.rejections.recent)[-10:]],
            }

            # Per-tier summary
            tier_summary = {}
            for name, tier in self.tiers.items():
                tier_summary[name] = {
                    "calls": tier.calls,
                    "avg_quality": round(tier.avg_quality, 3),
                    "quality_ema": round(tier.quality_ema, 3),
                    "rejection_rate": round(tier.rejection_rate * 100, 1),
                    "consensus_rate": round(tier.consensus_rate * 100, 1),
                }

            return {
                "consensus": consensus_summary,
                "rejections": rejection_summary,
                "tiers": tier_summary,
            }

    def _format_distribution(self) -> dict[str, float]:
        """Format votes histogram as percentages."""
        total = sum(self.consensus.votes_histogram.values())
        if total == 0:
            return {}

        result = {}
        for k, count in sorted(self.consensus.votes_histogram.items(), key=lambda x: int(x[0])):
            pct = round(count / total * 100, 1)
            result[f"k={k}"] = pct
        return result

    def reset(self) -> None:
        """Reset all statistics (for testing)."""
        with self._lock:
            self.consensus = ConsensusStats()
            self.rejections = RejectionStats()
            self.tiers = {}
            if self._stats_file.exists():
                self._stats_file.unlink()


def get_voting_stats_tracker() -> VotingStatsTracker:
    """Get the singleton VotingStatsTracker instance."""
    global _voting_stats_tracker
    with _tracker_lock:
        if _voting_stats_tracker is None:
            _voting_stats_tracker = VotingStatsTracker()
        return _voting_stats_tracker


def reset_voting_stats_tracker() -> None:
    """Reset the singleton (for testing)."""
    global _voting_stats_tracker
    with _tracker_lock:
        if _voting_stats_tracker is not None:
            _voting_stats_tracker.reset()
        _voting_stats_tracker = None
