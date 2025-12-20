# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Melon Savings Display System 🍈

Melons represent REAL SAVINGS - money you've saved by using local models
instead of expensive cloud APIs.

1 melon = $0.001 saved
500 melons = 1 golden melon = $0.50 saved

NOTE: Melons are now DISPLAY ONLY - they do NOT affect routing decisions.
Routing is handled by economics.py using real metrics (quality, cost, latency).
Melons are kept as a fun way to visualize your savings.

Usage:
    from delia.melons import get_melon_tracker, award_savings

    # Record savings after each call
    tracker = get_melon_tracker()
    tracker.record_savings("model-id", "coder", savings_usd=0.025)  # 25 melons!

    # Get fun leaderboard
    print(tracker.get_leaderboard_text())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import structlog

from . import paths

log = structlog.get_logger()


# Conversion rate: 1 melon = $0.001 saved
MELON_TO_USD = 0.001
USD_TO_MELON = 1000  # $1 = 1000 melons


@dataclass
class MelonStats:
    """
    Melon statistics for a model+task combination.

    Now represents SAVINGS, not arbitrary rewards.
    1 melon = $0.001 saved vs cloud baseline.
    """

    model_id: str
    task_type: str
    melons: int = 0
    golden_melons: int = 0
    total_calls: int = 0
    total_savings_usd: float = 0.0

    @property
    def total_melon_value(self) -> int:
        """Total value in melons (golden = 500 each)."""
        return self.melons + (self.golden_melons * 500)

    @property
    def savings_usd(self) -> float:
        """Total savings in USD."""
        return self.total_melon_value * MELON_TO_USD

    def record_savings(self, savings_usd: float) -> bool:
        """
        Record savings from a call. Returns True if a golden melon was earned.

        Args:
            savings_usd: Amount saved vs cloud baseline

        Returns:
            True if this pushed us over a golden melon threshold
        """
        self.total_calls += 1
        self.total_savings_usd += savings_usd

        # Convert savings to melons
        new_melons = int(savings_usd * USD_TO_MELON)
        if new_melons <= 0:
            return False

        self.melons += new_melons
        earned_golden = False

        # Check for golden melon promotion (500 melons = $0.50 saved)
        while self.melons >= 500:
            self.melons -= 500
            self.golden_melons += 1
            earned_golden = True
            log.info(
                "🏆 GOLDEN MELON EARNED!",
                model=self.model_id,
                task=self.task_type,
                total_golden=self.golden_melons,
                total_saved=f"${self.total_savings_usd:.2f}",
            )

        return earned_golden

    # Keep legacy methods for backward compatibility during transition
    def award(self, count: int = 1) -> bool:
        """Legacy: Award melons directly."""
        if count <= 0:
            return False
        self.melons += count
        earned_golden = False
        while self.melons >= 500:
            self.melons -= 500
            self.golden_melons += 1
            earned_golden = True
        return earned_golden

    def penalize(self, count: int = 1) -> None:
        """Legacy: Remove melons. Deprecated - savings can't be negative."""
        self.melons = max(0, self.melons - count)

    def record_response(self, success: bool) -> None:
        """Legacy: Record response. Use record_savings instead."""
        self.total_calls += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MelonStats:
        # Handle migration from old format
        if "total_responses" in data:
            data["total_calls"] = data.pop("total_responses", 0)
        if "successful_responses" in data:
            data.pop("successful_responses", None)
        if "total_savings_usd" not in data:
            # Estimate savings from existing melons
            total_melons = data.get("melons", 0) + data.get("golden_melons", 0) * 500
            data["total_savings_usd"] = total_melons * MELON_TO_USD
        return cls(**data)


class MelonTracker:
    """
    Tracks savings as melons across all models and task types.

    Melons are a FUN way to visualize your savings from using local models.
    1 melon = $0.001 saved vs cloud baseline.
    500 melons = 1 golden melon = $0.50 saved.

    NOTE: Melons are DISPLAY ONLY - they do NOT affect routing decisions.
    See economics.py for actual routing logic.
    """

    def __init__(self, stats_file: Path | None = None):
        self._stats_file = stats_file or (paths.DATA_DIR / "melons.json")
        self._stats: dict[str, MelonStats] = {}
        self._load()

    def _key(self, model_id: str, task_type: str) -> str:
        return f"{model_id}:{task_type}"

    def _get_or_create(self, model_id: str, task_type: str) -> MelonStats:
        key = self._key(model_id, task_type)
        if key not in self._stats:
            self._stats[key] = MelonStats(model_id=model_id, task_type=task_type)
        return self._stats[key]

    def record_savings(
        self,
        model_id: str,
        task_type: str,
        savings_usd: float,
    ) -> bool:
        """
        Record savings from a call as melons.

        This is the PRIMARY method for adding melons.
        1 melon = $0.001 saved.

        Args:
            model_id: The model that was used
            task_type: Type of task (quick, coder, moe, etc.)
            savings_usd: Amount saved vs cloud baseline

        Returns:
            True if a golden melon was earned
        """
        stats = self._get_or_create(model_id, task_type)
        earned_golden = stats.record_savings(savings_usd)

        log.debug(
            "🍈 savings_recorded",
            model=model_id,
            task=task_type,
            saved_usd=f"${savings_usd:.4f}",
            melons_earned=int(savings_usd * USD_TO_MELON),
            total_melons=stats.melons,
            golden=stats.golden_melons,
        )

        self._save()
        return earned_golden

    # Legacy methods for backward compatibility
    def award(
        self,
        model_id: str,
        task_type: str,
        melons: int = 1,
        success: bool = True,
    ) -> bool:
        """Legacy: Award melons directly. Use record_savings instead."""
        stats = self._get_or_create(model_id, task_type)
        stats.record_response(success)
        earned_golden = stats.award(melons)
        self._save()
        return earned_golden

    def penalize(
        self,
        model_id: str,
        task_type: str,
        melons: int = 1,
    ) -> None:
        """Legacy: Remove melons. Deprecated - savings can't be negative."""
        stats = self._get_or_create(model_id, task_type)
        stats.penalize(melons)
        self._save()

    def get_stats(self, model_id: str, task_type: str) -> MelonStats:
        """Get melon stats for a specific model+task."""
        return self._get_or_create(model_id, task_type)

    def get_total_savings(self) -> float:
        """Get total savings in USD across all models."""
        return sum(s.total_savings_usd for s in self._stats.values())

    def get_leaderboard(self, task_type: str | None = None) -> list[MelonStats]:
        """
        Get leaderboard sorted by savings.

        Args:
            task_type: Filter by task type (None = all)

        Returns:
            List of MelonStats sorted by total savings
        """
        stats_list = list(self._stats.values())

        if task_type:
            stats_list = [s for s in stats_list if s.task_type == task_type]

        return sorted(
            stats_list,
            key=lambda s: (s.golden_melons, s.melons, s.total_savings_usd),
            reverse=True,
        )

    def get_leaderboard_text(self) -> str:
        """Get formatted leaderboard for display - now shows REAL savings."""
        # Calculate totals
        total_melons = sum(s.melons for s in self._stats.values())
        total_golden = sum(s.golden_melons for s in self._stats.values())
        total_savings = sum(s.total_savings_usd for s in self._stats.values())
        total_calls = sum(s.total_calls for s in self._stats.values())

        # Aggregate by model (across task types)
        model_totals: dict[str, dict[str, Any]] = {}
        for stats in self._stats.values():
            if stats.model_id not in model_totals:
                model_totals[stats.model_id] = {
                    "melons": 0,
                    "golden": 0,
                    "savings": 0.0,
                    "calls": 0,
                }
            model_totals[stats.model_id]["melons"] += stats.melons
            model_totals[stats.model_id]["golden"] += stats.golden_melons
            model_totals[stats.model_id]["savings"] += stats.total_savings_usd
            model_totals[stats.model_id]["calls"] += stats.total_calls

        # Sort by savings
        sorted_models = sorted(
            model_totals.items(),
            key=lambda x: x[1]["savings"],
            reverse=True,
        )

        medals = ["🥇", "🥈", "🥉"]

        # Build leaderboard lines
        board_lines = []
        for i, (model_id, totals) in enumerate(sorted_models[:10]):
            medal = medals[i] if i < 3 else f"{i+1:>2}."
            golden = f"🏆×{totals['golden']}" if totals["golden"] else "     "
            savings = totals["savings"]
            calls = totals["calls"]
            board_lines.append(
                f"{medal} {model_id:<24} ${savings:>7.2f} saved  "
                f"{golden}  {calls:>5} calls"
            )

        if len(sorted_models) > 10:
            board_lines.append(f"    ...and {len(sorted_models) - 10} more models")

        board_text = "\n".join(board_lines) if board_lines else "No savings yet - make some LLM calls!"

        return f"""🍈 SAVINGS LEADERBOARD
══════════════════════════════════════════════════════
Total Saved: ${total_savings:.2f}  ({total_melons:,} melons + {total_golden} golden)
Total Calls: {total_calls:,}

{board_text}

💡 1 melon = $0.001 saved vs cloud API pricing
🏆 Golden melon = $0.50 milestone"""
    
    def _load(self) -> None:
        """Load melon stats from disk."""
        if not self._stats_file.exists():
            return
        
        try:
            data = json.loads(self._stats_file.read_text())
            for key, stats_dict in data.get("stats", {}).items():
                self._stats[key] = MelonStats.from_dict(stats_dict)
            log.debug("melons_loaded", count=len(self._stats))
        except Exception as e:
            log.warning("melons_load_error", error=str(e))
    
    def _save(self) -> None:
        """Save melon stats to disk."""
        try:
            self._stats_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "stats": {k: v.to_dict() for k, v in self._stats.items()},
            }
            self._stats_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.warning("melons_save_error", error=str(e))

    def save(self) -> None:
        """Public alias for _save to match OrchestrationService expectations."""
        self._save()
    
    def reset(self) -> None:
        """Reset all stats (for testing)."""
        self._stats.clear()


# Singleton instance
_melon_tracker: MelonTracker | None = None


def get_melon_tracker() -> MelonTracker:
    """Get the global melon tracker instance."""
    global _melon_tracker
    if _melon_tracker is None:
        _melon_tracker = MelonTracker()
    return _melon_tracker


def reset_melon_tracker() -> None:
    """Reset the global melon tracker (for testing)."""
    global _melon_tracker
    _melon_tracker = None


class RewardCollector:
    """
    Automatically collects high-quality training pairs for fine-tuning.
    
    Objective: Save prompt/response pairs that receive a "Golden Melon"
    or exceptional quality score (>= 0.95).
    """

    def __init__(self, output_file: Path | None = None):
        self.output_file = output_file or (paths.DATA_DIR / "training_feedback.jsonl")

    def record_winning_pair(
        self,
        prompt: str,
        response: str,
        model_id: str,
        task_type: str,
        quality_score: float,
    ) -> None:
        """
        Record a high-quality interaction for the fine-tuning dataset.
        """
        if quality_score < 0.95:
            return

        import datetime
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model_id,
            "task": task_type,
            "quality": quality_score,
            "prompt": prompt,
            "response": response,
        }

        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "a") as f:
                f.write(json.dumps(data) + "\n")
            log.info("🏆 interaction_recorded_for_training", model=model_id, quality=quality_score)
        except Exception as e:
            log.warning("failed_to_record_reward", error=str(e))


# Singleton instances
_reward_collector: RewardCollector | None = None


def get_reward_collector() -> RewardCollector:
    """Get the global reward collector instance."""
    global _reward_collector
    if _reward_collector is None:
        _reward_collector = RewardCollector()
    return _reward_collector


# Helper functions for common operations
def record_call_economics(
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
    Record economic metrics for a call AND convert savings to melons.

    This is the main entry point for recording LLM call outcomes.
    It updates both the EconomicTracker (for routing) and MelonTracker (for display).

    Args:
        model_id: Model that was used
        task_type: Task type (quick, coder, moe, etc.)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_usd: Actual cost in USD (0 for local models)
        latency_ms: Latency in milliseconds
        quality_score: Quality score 0.0-1.0 (optional)
        success: Whether the call succeeded

    Returns:
        Savings in USD
    """
    from .economics import get_economic_tracker

    # Record in economic tracker (affects routing)
    economics = get_economic_tracker()
    savings = economics.record_call(
        model_id=model_id,
        task_type=task_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        quality_score=quality_score,
        success=success,
    )

    # Record savings as melons (for display)
    if savings > 0:
        tracker = get_melon_tracker()
        tracker.record_savings(model_id, task_type, savings)

    return savings


def award_melons_for_quality(
    model_id: str,
    task_type: str,
    quality_score: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
) -> int:
    """
    Record a call and award melons based on response quality.

    This function now:
    1. Records economics (quality, tokens, cost) for routing decisions
    2. Calculates savings and awards melons for display

    Melons are now based on SAVINGS, not arbitrary quality thresholds.
    The quality affects the economic routing score, not melon count.

    Args:
        model_id: Model that responded
        task_type: Task type
        quality_score: Quality score 0.0-1.0
        input_tokens: Number of input tokens (optional, for economics)
        output_tokens: Number of output tokens (optional, for economics)
        cost_usd: Actual cost in USD (optional, 0 for local)
        latency_ms: Latency in ms (optional)

    Returns:
        Estimated melons from savings (for backward compatibility)
    """
    # Record economics (this also awards savings-based melons)
    success = quality_score >= 0.3
    savings = record_call_economics(
        model_id=model_id,
        task_type=task_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        quality_score=quality_score if success else None,
        success=success,
    )

    # Return estimated melons for backward compatibility
    return int(savings * USD_TO_MELON)
