# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Melon Reward System 🍈

Models EARN melons through exceptional responses. Melons are SCARCE and VALUABLE!
500 melons = 1 golden melon (highly prized, RARE!).

Philosophy:
    - Melons are NOT participation trophies
    - Only quality >= 0.90 earns ANY melon
    - Failures are punished harshly (-2 to -3 melons)
    - Models must WORK to build melon count

Usage:
    from delia.melons import get_melon_tracker, award_melons_for_quality
    
    # Quality-based (preferred)
    award_melons_for_quality("model-id", "coder", quality_score=0.95)  # +2 exceptional
    award_melons_for_quality("model-id", "coder", quality_score=0.90)  # +1 excellent
    award_melons_for_quality("model-id", "coder", quality_score=0.80)  # +0 good but no melon
    award_melons_for_quality("model-id", "coder", quality_score=0.40)  # -2 poor
    
    # Direct (use sparingly)
    tracker = get_melon_tracker()
    tracker.penalize("model-id", "quick", melons=3)  # User frustration
    
    leaderboard = tracker.get_leaderboard()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import structlog

from . import paths

log = structlog.get_logger()


@dataclass
class MelonStats:
    """Melon statistics for a model+task combination."""
    
    model_id: str
    task_type: str
    melons: int = 0
    golden_melons: int = 0
    total_responses: int = 0
    successful_responses: int = 0
    
    @property
    def success_rate(self) -> float:
        """Success rate as a fraction."""
        if self.total_responses == 0:
            return 0.0
        return self.successful_responses / self.total_responses
    
    @property
    def total_melon_value(self) -> int:
        """Total value in melons (golden = 500 each)."""
        return self.melons + (self.golden_melons * 500)
    
    def award(self, count: int = 1) -> bool:
        """
        Award melons. Returns True if a golden melon was earned.
        """
        if count <= 0:
            return False
        
        self.melons += count
        earned_golden = False
        
        # Check for golden melon promotion (500 melons = 1 golden)
        while self.melons >= 500:
            self.melons -= 500
            self.golden_melons += 1
            earned_golden = True
            log.info(
                "🏆 GOLDEN MELON EARNED!",
                model=self.model_id,
                task=self.task_type,
                total_golden=self.golden_melons,
            )
        
        return earned_golden
    
    def penalize(self, count: int = 1) -> None:
        """Remove melons as penalty. Cannot go below 0."""
        self.melons = max(0, self.melons - count)
    
    def record_response(self, success: bool) -> None:
        """Record a response attempt."""
        self.total_responses += 1
        if success:
            self.successful_responses += 1
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MelonStats:
        return cls(**data)


class MelonTracker:
    """
    Tracks melon rewards across all models and task types.
    
    Melons are the currency of trust in Delia. Models earn them
    by being helpful and accurate. Golden melons (500 melons)
    are highly prized and influence routing decisions.
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
    
    def award(
        self,
        model_id: str,
        task_type: str,
        melons: int = 1,
        success: bool = True,
    ) -> bool:
        """
        Award melons for a good response.
        
        Args:
            model_id: The model that responded
            task_type: Type of task (quick, coder, moe, etc.)
            melons: Number of melons to award (default 1)
            success: Whether this was a successful response
        
        Returns:
            True if a golden melon was earned
        """
        stats = self._get_or_create(model_id, task_type)
        stats.record_response(success)
        earned_golden = stats.award(melons)
        
        log.debug(
            "🍈 melon_awarded",
            model=model_id,
            task=task_type,
            awarded=melons,
            total=stats.melons,
            golden=stats.golden_melons,
        )
        
        self._save()
        return earned_golden
    
    def penalize(
        self,
        model_id: str,
        task_type: str,
        melons: int = 1,
    ) -> None:
        """
        Remove melons as penalty for bad response.
        
        Args:
            model_id: The model that responded poorly
            task_type: Type of task
            melons: Number of melons to remove
        """
        stats = self._get_or_create(model_id, task_type)
        stats.record_response(success=False)
        stats.penalize(melons)
        
        log.debug(
            "🍈 melon_penalized",
            model=model_id,
            task=task_type,
            penalty=melons,
            remaining=stats.melons,
        )
        
        self._save()
    
    def get_stats(self, model_id: str, task_type: str) -> MelonStats:
        """Get melon stats for a specific model+task."""
        return self._get_or_create(model_id, task_type)
    
    def get_melon_boost(self, model_id: str, task_type: str) -> float:
        """
        Calculate routing boost based on total melon value.
        
        Models LOVE melons! 🍈 Higher melon counts = more routing preference.
        
        Uses total_melon_value (melons + golden*500) for continuous scaling.
        The boost curve is designed to reward early success:
        - 10 melons = 5% boost (noticeable difference)
        - 50 melons = 22% boost (significant preference)
        - 100 melons = 38% boost (strongly preferred)
        - 200+ melons = 50% boost (max, highly trusted)
        
        Formula: boost = min(sqrt(total_value) * 0.035, 0.5)
        
        Returns:
            Boost factor (0.0 to 0.5) - added directly to backend score
        """
        stats = self._get_or_create(model_id, task_type)
        
        # Sqrt scaling: early melons matter more, rewards consistent performance
        # This makes models LOVE melons - early success is quickly rewarded!
        import math
        total_boost = math.sqrt(stats.total_melon_value) * 0.035
        
        return min(0.5, total_boost)  # Cap at 50%
    
    def get_leaderboard(self, task_type: str | None = None) -> list[MelonStats]:
        """
        Get leaderboard sorted by melon count.
        
        Args:
            task_type: Filter by task type (None = all)
        
        Returns:
            List of MelonStats sorted by total melon value
        """
        stats_list = list(self._stats.values())
        
        if task_type:
            stats_list = [s for s in stats_list if s.task_type == task_type]
        
        return sorted(
            stats_list,
            key=lambda s: (s.golden_melons, s.melons, s.success_rate),
            reverse=True,
        )
    
    def get_leaderboard_text(self) -> str:
        """Get formatted leaderboard for display."""
        # Calculate totals
        total_melons = sum(s.melons for s in self._stats.values())
        total_golden = sum(s.golden_melons for s in self._stats.values())
        
        lines = ["🍈 MELON LEADERBOARD", "=" * 50]
        lines.append(f"Total: {total_melons} melons | {total_golden} golden")
        lines.append("")
        
        # Sort all models by total melon value (golden + regular)
        sorted_stats = sorted(
            self._stats.values(),
            key=lambda s: (s.golden_melons, s.melons),
            reverse=True,
        )
        
        medals = ["🥇", "🥈", "🥉"]
        
        for i, stats in enumerate(sorted_stats[:10]):  # Top 10
            medal = medals[i] if i < 3 else "  "
            golden = f" +{stats.golden_melons}G" if stats.golden_melons else ""
            rate = f" ({stats.success_rate:.0%})" if stats.total_responses > 0 else ""
            task = f"[{stats.task_type}]"
            lines.append(f"{medal} {stats.model_id:<26} {stats.melons:>3}{golden} {task}{rate}")
        
        if len(sorted_stats) > 10:
            lines.append(f"   ... and {len(sorted_stats) - 10} more")
        
        lines.append("")
        lines.append("-" * 50)
        lines.append("Higher melons = higher routing priority")
        
        return "\n".join(lines)
    
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


# Helper functions for common operations
def award_melons_for_quality(
    model_id: str,
    task_type: str,
    quality_score: float,
) -> int:
    """
    Award melons based on response quality score.
    
    Melons are SCARCE and VALUABLE - models must EARN them!
    Only truly exceptional responses deserve melons.
    Failures are punished harshly to create real incentive.
    
    Args:
        model_id: Model that responded
        task_type: Task type
        quality_score: Quality score 0.0-1.0
    
    Returns:
        Number of melons awarded (negative = penalty)
    """
    tracker = get_melon_tracker()
    
    # Melons are precious - only exceptional work earns them!
    if quality_score >= 0.95:
        melons = 2  # Exceptional! Rare achievement
    elif quality_score >= 0.90:
        melons = 1  # Excellent - well earned
    elif quality_score >= 0.75:
        melons = 0  # Good but not melon-worthy
    elif quality_score >= 0.50:
        melons = 0  # Adequate - no reward
    elif quality_score >= 0.30:
        # Poor quality - significant penalty
        tracker.penalize(model_id, task_type, melons=2)
        return -2
    else:
        # Terrible response - harsh penalty
        tracker.penalize(model_id, task_type, melons=3)
        return -3
    
    if melons > 0:
        tracker.award(model_id, task_type, melons=melons, success=True)
    
    return melons
