# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Economic Tracking System for Delia.

Tracks REAL metrics that matter for routing decisions:
- Cost per token (actual $ spent)
- Quality scores (empirical performance)
- Latency (time is money)
- Success rates (reliability)

This replaces the melon system for routing decisions.
Melons are repurposed as a savings display ($ saved vs cloud baseline).

Usage:
    from delia.economics import get_economic_tracker, record_call

    # After each LLM call
    record_call(
        model_id="qwen2.5:14b",
        task_type="coder",
        tokens=1500,
        cost_usd=0.0,  # Local = free
        latency_ms=2300,
        quality_score=0.92,
        success=True,
    )

    # Get routing score
    tracker = get_economic_tracker()
    score = tracker.get_routing_score("qwen2.5:14b", "coder")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import structlog

from . import paths

log = structlog.get_logger()


# Baseline costs for cloud APIs (per 1K tokens, input)
# Used to calculate savings when using local models
CLOUD_BASELINE_COSTS = {
    "gpt-4": 0.03,           # $0.03 per 1K input tokens
    "gpt-4-turbo": 0.01,     # $0.01 per 1K input tokens
    "gpt-3.5-turbo": 0.0005, # $0.0005 per 1K input tokens
    "claude-3-opus": 0.015,  # $0.015 per 1K input tokens
    "claude-3-sonnet": 0.003, # $0.003 per 1K input tokens
    "claude-3-haiku": 0.00025, # $0.00025 per 1K input tokens
}

# Default baseline for savings calculation
DEFAULT_BASELINE = "gpt-4-turbo"


@dataclass
class ModelEconomics:
    """
    Economic metrics for a model+task combination.

    These are REAL metrics that inform routing decisions.
    """
    model_id: str
    task_type: str

    # Call counts
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Cost tracking (in USD)
    total_cost_usd: float = 0.0

    # Latency tracking (in milliseconds)
    total_latency_ms: int = 0

    # Quality tracking (sum of quality scores for averaging)
    quality_sum: float = 0.0
    quality_count: int = 0

    # Savings vs cloud baseline
    total_savings_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def success_rate(self) -> float:
        """Success rate as fraction (0-1)."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_quality(self) -> float:
        """Average quality score (0-1)."""
        if self.quality_count == 0:
            return 0.0
        return self.quality_sum / self.quality_count

    @property
    def cost_per_1k_tokens(self) -> float:
        """Cost per 1000 tokens in USD."""
        if self.total_tokens == 0:
            return 0.0
        return (self.total_cost_usd / self.total_tokens) * 1000

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def quality_per_dollar(self) -> float:
        """
        Quality per dollar spent - the key ROI metric.

        Higher = better value. For free local models, this is very high.
        """
        if self.total_cost_usd <= 0:
            # Free models get high but not infinite score
            return self.avg_quality * 1000
        return (self.avg_quality * self.successful_calls) / self.total_cost_usd

    @property
    def tokens_per_second(self) -> float:
        """Throughput in tokens per second."""
        if self.total_latency_ms == 0:
            return 0.0
        return (self.total_tokens / self.total_latency_ms) * 1000

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: int,
        quality_score: float | None,
        success: bool,
        baseline: str = DEFAULT_BASELINE,
    ) -> float:
        """
        Record a completed LLM call.

        Returns:
            Savings in USD vs cloud baseline
        """
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd
        self.total_latency_ms += latency_ms

        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1

        if quality_score is not None and success:
            self.quality_sum += quality_score
            self.quality_count += 1

        # Calculate savings vs cloud baseline
        baseline_cost = CLOUD_BASELINE_COSTS.get(baseline, 0.01)
        total_tokens = input_tokens + output_tokens
        cloud_cost = (total_tokens / 1000) * baseline_cost
        savings = max(0, cloud_cost - cost_usd)
        self.total_savings_usd += savings

        return savings

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEconomics:
        """Deserialize from dictionary."""
        return cls(**data)


class EconomicTracker:
    """
    Tracks economic metrics for all models and task types.

    Provides REAL signals for routing decisions:
    - Cost efficiency (quality per dollar)
    - Reliability (success rate)
    - Performance (latency, throughput)
    """

    def __init__(self, stats_file: Path | None = None):
        self._stats_file = stats_file or (paths.DATA_DIR / "economics.json")
        self._stats: dict[str, ModelEconomics] = {}
        self._load()

    def _key(self, model_id: str, task_type: str) -> str:
        """Generate unique key for model+task combination."""
        return f"{model_id}:{task_type}"

    def _get_or_create(self, model_id: str, task_type: str) -> ModelEconomics:
        """Get existing stats or create new entry."""
        key = self._key(model_id, task_type)
        if key not in self._stats:
            self._stats[key] = ModelEconomics(model_id=model_id, task_type=task_type)
        return self._stats[key]

    def record_call(
        self,
        model_id: str,
        task_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        quality_score: float | None = None,
        success: bool = True,
        baseline: str = DEFAULT_BASELINE,
    ) -> float:
        """
        Record a completed LLM call.

        Args:
            model_id: The model that was used
            task_type: Type of task (quick, coder, moe, etc.)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Actual cost in USD (0 for local models)
            latency_ms: Time taken in milliseconds
            quality_score: Quality score if available (0-1)
            success: Whether the call succeeded
            baseline: Cloud baseline for savings calculation

        Returns:
            Savings in USD vs cloud baseline
        """
        stats = self._get_or_create(model_id, task_type)
        savings = stats.record_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            quality_score=quality_score,
            success=success,
            baseline=baseline,
        )

        log.debug(
            "economic_call_recorded",
            model=model_id,
            task=task_type,
            tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            savings_usd=savings,
            quality=quality_score,
        )

        self._save()
        return savings

    def get_economics(self, model_id: str, task_type: str) -> ModelEconomics:
        """Get economic metrics for a model+task combination."""
        return self._get_or_create(model_id, task_type)

    def get_routing_score(
        self,
        model_id: str,
        task_type: str,
        priority: int = 100,
    ) -> float:
        """
        Calculate routing score based on economic metrics.

        This is the PRIMARY routing signal, replacing melon boost.

        Formula:
            score = priority × success_rate × quality × cost_efficiency × speed_factor

        Where:
            - success_rate: Reliability (0-1)
            - quality: Average quality score (0-1)
            - cost_efficiency: 1 / (1 + cost_per_1k_tokens)
            - speed_factor: Normalized speed bonus

        Returns:
            Routing score (higher = more likely to be selected)
        """
        stats = self._get_or_create(model_id, task_type)

        # Base priority
        base = float(priority)

        # Success rate factor (default 0.8 for new models)
        success_factor = stats.success_rate if stats.total_calls >= 5 else 0.8

        # Quality factor (default 0.7 for new models)
        quality_factor = stats.avg_quality if stats.quality_count >= 3 else 0.7

        # Cost efficiency: free models get 1.0, expensive models get penalized
        # Uses sigmoid-like curve: 1 / (1 + cost)
        cost_factor = 1.0 / (1.0 + stats.cost_per_1k_tokens)

        # Speed factor: faster models get slight bonus (capped at 1.2)
        # Baseline: 50 tokens/second
        if stats.tokens_per_second > 0:
            speed_factor = min(1.2, 0.8 + (stats.tokens_per_second / 250))
        else:
            speed_factor = 1.0

        score = base * success_factor * quality_factor * cost_factor * speed_factor

        log.debug(
            "economic_routing_score",
            model=model_id,
            task=task_type,
            base=base,
            success=success_factor,
            quality=quality_factor,
            cost=cost_factor,
            speed=speed_factor,
            final_score=score,
        )

        return score

    def get_total_savings(self) -> float:
        """Get total savings across all models and tasks."""
        return sum(s.total_savings_usd for s in self._stats.values())

    def get_total_cost(self) -> float:
        """Get total cost across all models and tasks."""
        return sum(s.total_cost_usd for s in self._stats.values())

    def get_total_calls(self) -> int:
        """Get total call count across all models and tasks."""
        return sum(s.total_calls for s in self._stats.values())

    def get_leaderboard(self, task_type: str | None = None) -> list[ModelEconomics]:
        """
        Get leaderboard sorted by savings.

        Args:
            task_type: Filter by task type (None = all)

        Returns:
            List of ModelEconomics sorted by total savings
        """
        stats_list = list(self._stats.values())

        if task_type:
            stats_list = [s for s in stats_list if s.task_type == task_type]

        return sorted(
            stats_list,
            key=lambda s: (s.total_savings_usd, s.avg_quality, s.success_rate),
            reverse=True,
        )

    def get_savings_display(self) -> str:
        """
        Get formatted savings display for users.

        Shows tangible value: actual dollars saved by using local models.
        """
        total_savings = self.get_total_savings()
        total_cost = self.get_total_cost()
        total_calls = self.get_total_calls()

        # Aggregate by model (across task types)
        model_totals: dict[str, dict[str, float]] = {}
        for stats in self._stats.values():
            if stats.model_id not in model_totals:
                model_totals[stats.model_id] = {
                    "savings": 0.0,
                    "quality": 0.0,
                    "quality_count": 0,
                    "calls": 0,
                }
            model_totals[stats.model_id]["savings"] += stats.total_savings_usd
            model_totals[stats.model_id]["quality"] += stats.quality_sum
            model_totals[stats.model_id]["quality_count"] += stats.quality_count
            model_totals[stats.model_id]["calls"] += stats.total_calls

        # Sort by savings
        sorted_models = sorted(
            model_totals.items(),
            key=lambda x: x[1]["savings"],
            reverse=True,
        )

        # Build display
        lines = []
        medals = ["🥇", "🥈", "🥉"]

        for i, (model_id, totals) in enumerate(sorted_models[:10]):
            medal = medals[i] if i < 3 else f"{i+1:>2}."
            savings = totals["savings"]
            avg_qual = totals["quality"] / max(1, totals["quality_count"])
            calls = totals["calls"]

            lines.append(
                f"{medal} {model_id:<28} ${savings:>7.2f} saved  "
                f"{avg_qual:.0%} quality  {calls:>5} calls"
            )

        if len(sorted_models) > 10:
            lines.append(f"    ...and {len(sorted_models) - 10} more models")

        return f"""💰 SAVINGS DASHBOARD
══════════════════════════════════════════════════
Total Saved: ${total_savings:.2f}  |  Total Spent: ${total_cost:.2f}  |  Calls: {total_calls:,}

{chr(10).join(lines) if lines else "No data yet - make some LLM calls!"}

💡 Savings calculated vs {DEFAULT_BASELINE} (${CLOUD_BASELINE_COSTS[DEFAULT_BASELINE]}/1K tokens)
"""

    def _load(self) -> None:
        """Load economic stats from disk."""
        if not self._stats_file.exists():
            return

        try:
            data = json.loads(self._stats_file.read_text())
            for key, stats_dict in data.get("stats", {}).items():
                self._stats[key] = ModelEconomics.from_dict(stats_dict)
            log.debug("economics_loaded", count=len(self._stats))
        except Exception as e:
            log.warning("economics_load_error", error=str(e))

    def _save(self) -> None:
        """Save economic stats to disk."""
        try:
            self._stats_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "baseline": DEFAULT_BASELINE,
                "stats": {k: v.to_dict() for k, v in self._stats.items()},
            }
            self._stats_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.warning("economics_save_error", error=str(e))


# Module-level singleton
_tracker: EconomicTracker | None = None


def get_economic_tracker() -> EconomicTracker:
    """Get the global economic tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = EconomicTracker()
    return _tracker


def reset_economic_tracker() -> None:
    """Reset the global tracker (for testing)."""
    global _tracker
    _tracker = None


def record_call(
    model_id: str,
    task_type: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
    quality_score: float | None = None,
    success: bool = True,
) -> float:
    """
    Convenience function to record a call.

    Returns savings in USD.
    """
    return get_economic_tracker().record_call(
        model_id=model_id,
        task_type=task_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        quality_score=quality_score,
        success=success,
    )
