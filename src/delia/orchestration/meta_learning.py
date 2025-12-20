# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Orchestration Meta-Learning System.

This module implements the meta-learner for Delia's ToT (Tree of Thoughts)
meta-orchestration system. It learns which orchestration modes work best
for which task patterns, using:

- Bayesian updating for confidence (Beta distribution)
- UCB1 for exploration-exploitation trade-off
- Exponential decay for ToT trigger probability

Mathematical foundations (Wolfram-validated):
- UCB1: avg + sqrt(2 * ln(n) / n_i) with exploration constant sqrt(2)
- Beta posterior: confidence = α / (α + β) with Laplace smoothing
- Decay: P(ToT) = 0.8 * exp(-t/τ) where τ=100 is learning time constant
"""

from __future__ import annotations

import json
import math
import random
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import structlog

from .. import paths
from ..prompts import OrchestrationMode

if TYPE_CHECKING:
    from .result import DetectedIntent

log = structlog.get_logger()

# High-stakes keywords that boost ToT probability
HIGH_STAKES_KEYWORDS = frozenset([
    "security", "secure", "vulnerability", "exploit", "injection",
    "crypto", "cryptographic", "encryption", "decrypt",
    "auth", "authentication", "authorization", "oauth", "jwt",
    "payment", "billing", "transaction", "financial",
    "medical", "health", "patient", "hipaa",
    "safety", "critical", "production", "deploy",
    "password", "secret", "credential", "api key",
])


@dataclass
class OrchestrationPattern:
    """
    A learned orchestration strategy.

    Stores Bayesian confidence (α, β) and UCB exploration stats
    for a specific task pattern.
    """
    pattern_id: str
    task_pattern: str  # Computed hash of task features
    keywords: list[str] = field(default_factory=list)  # Trigger keywords
    best_mode: str = "voting"  # Winning mode (stored as string for JSON)

    # Bayesian confidence tracking
    alpha: int = 1  # Success count (prior: 1 for Laplace smoothing)
    beta: int = 1   # Failure count (prior: 1 for Laplace smoothing)

    # UCB1 exploration tracking
    ucb_pulls: int = 0  # Total times this pattern was in ToT
    ucb_rewards: float = 0.0  # Sum of quality scores

    # Mode-specific UCB stats
    mode_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def confidence(self) -> float:
        """
        Beta distribution mean (Bayesian confidence).

        Returns value in [0, 1] representing how confident we are
        that best_mode is the correct choice for this pattern.
        """
        return self.alpha / (self.alpha + self.beta)

    def ucb_score(self, total_pulls: int) -> float:
        """
        UCB1 exploration bonus for this pattern.

        Returns avg_quality + exploration_bonus.
        """
        if self.ucb_pulls == 0:
            return float('inf')  # Infinite bonus for unexplored

        avg_quality = self.ucb_rewards / self.ucb_pulls
        exploration_bonus = math.sqrt(2 * math.log(max(1, total_pulls)) / self.ucb_pulls)
        return avg_quality + exploration_bonus

    def get_mode_ucb(self, mode: str, total_pulls: int) -> float:
        """Get UCB score for a specific mode within this pattern."""
        if mode not in self.mode_stats:
            return float('inf')  # Unexplored mode

        stats = self.mode_stats[mode]
        pulls = stats.get("pulls", 0)
        if pulls == 0:
            return float('inf')

        avg = stats.get("rewards", 0) / pulls
        bonus = math.sqrt(2 * math.log(max(1, total_pulls)) / pulls)
        return avg + bonus

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestrationPattern":
        """Create from dict (JSON deserialization)."""
        return cls(**data)


class OrchestrationLearner:
    """
    Meta-learner for orchestration strategies.

    Learns which orchestration modes work best for which task patterns.
    Implements explore-exploit trade-off using UCB1 and Bayesian updating.

    Key methods:
    - should_use_tot(): Decide whether to trigger ToT meta-orchestration
    - get_best_mode(): Get learned mode for a task pattern
    - select_exploration_modes(): UCB1-based mode selection for ToT
    - learn_from_tot(): Update patterns from ToT outcome
    """

    # Learning time constant (τ in exponential decay)
    LEARNING_TAU = 100

    # Base ToT trigger rate (decays with experience)
    BASE_TOT_RATE = 0.8

    # Confidence threshold for using learned pattern directly
    CONFIDENCE_THRESHOLD = 0.7

    # Modes available for ToT exploration
    EXPLORABLE_MODES = [
        OrchestrationMode.VOTING,
        OrchestrationMode.AGENTIC,
        OrchestrationMode.DEEP_THINKING,
        OrchestrationMode.COMPARISON,
    ]

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or paths.DATA_DIR
        self.patterns: dict[str, OrchestrationPattern] = {}
        self.total_tasks: int = 0
        self._load()

    def _get_path(self) -> Path:
        """Get path to the learner's persistent storage."""
        return self.data_dir / "orchestration_patterns.json"

    def _load(self) -> None:
        """Load patterns from disk."""
        path = self._get_path()
        if not path.exists():
            log.debug("orchestration_learner_no_existing_data")
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
                self.total_tasks = data.get("total_tasks", 0)
                for p_data in data.get("patterns", []):
                    pattern = OrchestrationPattern.from_dict(p_data)
                    self.patterns[pattern.task_pattern] = pattern
                log.info("orchestration_learner_loaded", patterns=len(self.patterns))
        except Exception as e:
            log.error("orchestration_learner_load_failed", error=str(e))

    def _save(self) -> None:
        """Persist patterns to disk."""
        path = self._get_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump({
                    "total_tasks": self.total_tasks,
                    "patterns": [p.to_dict() for p in self.patterns.values()],
                }, f, indent=2)
        except Exception as e:
            log.error("orchestration_learner_save_failed", error=str(e))

    def _extract_features(self, message: str) -> dict[str, Any]:
        """
        Extract task features from message for pattern matching.

        Returns dict with:
        - keywords: List of relevant keywords found
        - has_code: Whether message contains code
        - task_type_hints: Detected task type hints
        - stakes_keywords: High-stakes keywords found
        """
        msg_lower = message.lower()

        # Extract keywords (simplified - could use TF-IDF or embeddings)
        words = set(re.findall(r'\b[a-z]{3,}\b', msg_lower))

        # Detect code
        has_code = bool(re.search(r'```|\bdef\s|\bclass\s|\bfunction\s|=>\s*\{', message))

        # Task type hints
        task_hints = []
        if any(w in msg_lower for w in ["review", "check", "audit", "verify"]):
            task_hints.append("review")
        if any(w in msg_lower for w in ["write", "create", "implement", "build"]):
            task_hints.append("generate")
        if any(w in msg_lower for w in ["refactor", "optimize", "improve", "clean"]):
            task_hints.append("refactor")
        if any(w in msg_lower for w in ["explain", "what", "how", "why"]):
            task_hints.append("explain")
        if any(w in msg_lower for w in ["debug", "fix", "error", "bug"]):
            task_hints.append("debug")

        # High-stakes detection
        stakes_keywords = [w for w in HIGH_STAKES_KEYWORDS if w in msg_lower]

        return {
            "keywords": list(words)[:20],  # Limit to avoid huge patterns
            "has_code": has_code,
            "task_hints": task_hints,
            "stakes_keywords": stakes_keywords,
        }

    def _compute_pattern_key(self, features: dict[str, Any]) -> str:
        """
        Compute a pattern key from features.

        Uses task hints + stakes keywords for coarse-grained matching.
        """
        # Sort for deterministic key
        hints = sorted(features.get("task_hints", []))
        stakes = sorted(features.get("stakes_keywords", []))
        has_code = features.get("has_code", False)

        key_parts = []
        if hints:
            key_parts.append("_".join(hints))
        if stakes:
            key_parts.append("stakes:" + "_".join(stakes[:3]))
        if has_code:
            key_parts.append("code")

        return "|".join(key_parts) if key_parts else "general"

    def _find_matching_pattern(self, features: dict[str, Any]) -> OrchestrationPattern | None:
        """Find the best matching pattern for given features."""
        key = self._compute_pattern_key(features)
        return self.patterns.get(key)

    def _calculate_stakes(self, message: str) -> float:
        """
        Calculate stakes multiplier for ToT trigger probability.

        Returns 1.0 for normal tasks, higher for high-stakes tasks.
        """
        msg_lower = message.lower()
        stakes_count = sum(1 for kw in HIGH_STAKES_KEYWORDS if kw in msg_lower)

        if stakes_count == 0:
            return 1.0
        elif stakes_count == 1:
            return 1.3
        elif stakes_count == 2:
            return 1.6
        else:
            return 2.0  # Cap at 2x

    def should_use_tot(self, message: str, current_intent: "DetectedIntent") -> tuple[bool, str]:
        """
        Decide whether to trigger ToT meta-orchestration.

        Uses the formula:
        P(ToT) = base_rate(t) * novelty * stakes * (1 - confidence)

        Where:
        - base_rate(t) = 0.8 * exp(-t/τ) decays with experience
        - novelty = 1.0 if pattern unknown, 0.3 if known
        - stakes = 1.0-2.0 based on high-risk keywords
        - confidence = pattern confidence from Beta distribution

        Returns: (use_tot, reasoning)
        """
        features = self._extract_features(message)
        pattern = self._find_matching_pattern(features)

        # Calculate each factor
        base_rate = self.BASE_TOT_RATE * math.exp(-self.total_tasks / self.LEARNING_TAU)
        novelty = 1.0 if pattern is None else 0.3
        stakes = self._calculate_stakes(message)
        confidence = pattern.confidence if pattern else 0.0

        # Combined probability
        p_tot = base_rate * novelty * stakes * (1 - confidence)

        # Clamp to [0, 1]
        p_tot = max(0.0, min(1.0, p_tot))

        # Decide
        use_tot = random.random() < p_tot

        # Force ToT for high-stakes + novel
        if stakes > 1.5 and novelty > 0.5:
            use_tot = True

        reasoning = (
            f"P(ToT)={p_tot:.3f} "
            f"(base={base_rate:.3f}, novelty={novelty:.1f}, "
            f"stakes={stakes:.2f}, conf={confidence:.3f}) "
            f"→ {'EXPLORE' if use_tot else 'EXPLOIT'}"
        )

        log.debug(
            "tot_trigger_decision",
            p_tot=p_tot,
            use_tot=use_tot,
            base_rate=base_rate,
            novelty=novelty,
            stakes=stakes,
            confidence=confidence,
        )

        return use_tot, reasoning

    def get_best_mode(self, message: str) -> tuple[OrchestrationMode | None, float]:
        """
        Get the best known orchestration mode for this task pattern.

        Returns: (mode, confidence) or (None, 0.0) if no pattern found.
        """
        features = self._extract_features(message)
        pattern = self._find_matching_pattern(features)

        if pattern is None:
            return None, 0.0

        try:
            mode = OrchestrationMode(pattern.best_mode)
            return mode, pattern.confidence
        except ValueError:
            log.warning("invalid_mode_in_pattern", mode=pattern.best_mode)
            return None, 0.0

    def select_exploration_modes(self, message: str | None = None) -> list[OrchestrationMode]:
        """
        Select modes to try in ToT using UCB1.

        Returns list of 3 modes ordered by UCB score (exploration vs exploitation).
        """
        # Calculate total pulls for UCB formula
        total_pulls = sum(p.ucb_pulls for p in self.patterns.values()) + 1

        # Score each explorable mode
        mode_scores: list[tuple[OrchestrationMode, float]] = []

        for mode in self.EXPLORABLE_MODES:
            # Aggregate stats across all patterns for this mode
            mode_pulls = 0
            mode_rewards = 0.0

            for pattern in self.patterns.values():
                if mode.value in pattern.mode_stats:
                    stats = pattern.mode_stats[mode.value]
                    mode_pulls += stats.get("pulls", 0)
                    mode_rewards += stats.get("rewards", 0.0)

            if mode_pulls == 0:
                # Unexplored mode gets infinite UCB (prioritize exploration)
                ucb = float('inf')
            else:
                avg_quality = mode_rewards / mode_pulls
                exploration_bonus = math.sqrt(2 * math.log(max(1, total_pulls)) / mode_pulls)
                ucb = avg_quality + exploration_bonus

            mode_scores.append((mode, ucb))

        # Sort by UCB score descending
        mode_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top 3 modes
        selected = [mode for mode, _ in mode_scores[:3]]

        log.debug(
            "tot_mode_selection",
            selected=[m.value for m in selected],
            scores=[(m.value, f"{s:.3f}") for m, s in mode_scores],
        )

        return selected

    def learn_from_tot(
        self,
        message: str,
        branch_results: list[tuple[OrchestrationMode, Any]],  # (mode, OrchestrationResult)
        winner_mode: OrchestrationMode,
        critic_reasoning: str,
    ) -> None:
        """
        Update patterns based on ToT outcome.

        Uses:
        - Bayesian updating for pattern confidence
        - UCB stats for exploration tracking
        - Mode-specific quality tracking
        """
        features = self._extract_features(message)
        pattern_key = self._compute_pattern_key(features)

        # Get or create pattern
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = OrchestrationPattern(
                pattern_id=f"orch-{uuid.uuid4().hex[:8]}",
                task_pattern=pattern_key,
                keywords=features.get("keywords", [])[:10],
                best_mode=winner_mode.value,
                alpha=2,  # Start with slight confidence in winner
                beta=1,
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
            )
            log.info("orchestration_pattern_created", pattern=pattern_key, mode=winner_mode.value)

        pattern = self.patterns[pattern_key]

        # Bayesian update
        if winner_mode.value == pattern.best_mode:
            pattern.alpha += 1  # Reinforce current best mode
        else:
            pattern.beta += 1  # Weaken confidence in current best

            # Switch best_mode if confidence drops too low
            if pattern.confidence < 0.4:
                log.info(
                    "orchestration_pattern_mode_switch",
                    pattern=pattern_key,
                    old_mode=pattern.best_mode,
                    new_mode=winner_mode.value,
                    old_confidence=pattern.confidence,
                )
                pattern.best_mode = winner_mode.value
                pattern.alpha = 2
                pattern.beta = 1

        # Update UCB stats for pattern
        pattern.ucb_pulls += 1

        # Update mode-specific stats
        for mode, result in branch_results:
            mode_key = mode.value
            if mode_key not in pattern.mode_stats:
                pattern.mode_stats[mode_key] = {"pulls": 0, "rewards": 0.0}

            pattern.mode_stats[mode_key]["pulls"] += 1

            # Use quality_score as reward (default 0.5 if not available)
            quality = getattr(result, 'quality_score', None) or 0.5
            pattern.mode_stats[mode_key]["rewards"] += quality

            # Extra reward for winner
            if mode == winner_mode:
                pattern.mode_stats[mode_key]["rewards"] += 0.2  # Bonus for winning
                pattern.ucb_rewards += quality + 0.2

        pattern.last_updated = datetime.now().isoformat()
        self.total_tasks += 1
        self._save()

        log.info(
            "orchestration_pattern_updated",
            pattern=pattern_key,
            winner=winner_mode.value,
            confidence=pattern.confidence,
            total_tasks=self.total_tasks,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics for dashboard/monitoring."""
        if not self.patterns:
            return {
                "total_tasks": self.total_tasks,
                "patterns_count": 0,
                "avg_confidence": 0.0,
                "tot_trigger_rate": self.BASE_TOT_RATE,
            }

        confidences = [p.confidence for p in self.patterns.values()]
        current_tot_rate = self.BASE_TOT_RATE * math.exp(-self.total_tasks / self.LEARNING_TAU)

        return {
            "total_tasks": self.total_tasks,
            "patterns_count": len(self.patterns),
            "avg_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
            "tot_trigger_rate": current_tot_rate,
            "mode_distribution": self._get_mode_distribution(),
        }

    def _get_mode_distribution(self) -> dict[str, int]:
        """Get distribution of best modes across patterns."""
        dist: dict[str, int] = {}
        for pattern in self.patterns.values():
            dist[pattern.best_mode] = dist.get(pattern.best_mode, 0) + 1
        return dist


# Singleton instance
_learner: OrchestrationLearner | None = None


def get_orchestration_learner() -> OrchestrationLearner:
    """Get the global OrchestrationLearner instance."""
    global _learner
    if _learner is None:
        _learner = OrchestrationLearner()
    return _learner


def reset_orchestration_learner() -> None:
    """Reset the global learner (for testing)."""
    global _learner
    _learner = None
