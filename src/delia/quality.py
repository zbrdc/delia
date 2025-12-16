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
Response Quality Validation

Validates LLM response quality and provides continuous quality scores (0.0-1.0)
for affinity learning. Detects issues like:
- Repetition/looping (model stuck)
- Empty/truncated output
- Gibberish/nonsense text
- Low vocabulary diversity
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityConfig:
    """Configuration for quality validation."""

    enabled: bool = True

    # Repetition detection
    repetition_ngram_size: int = 5  # Word-level n-grams
    repetition_threshold: int = 3  # Max allowed repeats before penalty
    sentence_repeat_threshold: int = 2  # Max sentence repeats

    # Length validation
    min_response_length: int = 10  # Minimum characters
    max_response_length_tokens: int = 700  # MDAP red-flag threshold (tokens)
    max_empty_ratio: float = 0.8  # Max whitespace ratio

    # Task-specific minimum word counts
    min_words_per_task: dict[str, int] = field(
        default_factory=lambda: {
            "review": 20,
            "generate": 10,
            "analyze": 30,
            "think": 20,
            "delegate": 10,
        }
    )

    # Coherence checks
    max_non_ascii_ratio: float = 0.3  # Detect encoding issues
    min_vocabulary_diversity: float = 0.15  # Unique words / total words
    max_single_char_ratio: float = 0.3  # Detect character spam

    # Score weights for final calculation
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "repetition": 0.35,
            "length": 0.25,
            "coherence": 0.40,
        }
    )


@dataclass
class QualityScore:
    """Quality assessment result."""

    overall: float  # 0.0-1.0 final score
    repetition_score: float  # 1.0 = no repetition
    length_score: float  # 1.0 = appropriate length
    coherence_score: float  # 1.0 = coherent text
    issues: list[str] = field(default_factory=list)  # Human-readable issues

    @property
    def is_valid(self) -> bool:
        """Check if response passes quality threshold for k-voting."""
        return self.overall >= 0.5 and "response_too_long" not in str(self.issues)

    @property
    def reason(self) -> str | None:
        """Get primary failure reason for red-flagging."""
        if self.is_valid:
            return None
        if self.issues:
            return self.issues[0]
        return "low_quality"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/persistence."""
        return {
            "overall": round(self.overall, 3),
            "repetition": round(self.repetition_score, 3),
            "length": round(self.length_score, 3),
            "coherence": round(self.coherence_score, 3),
            "issues": self.issues,
        }


# Default config singleton
_DEFAULT_CONFIG = QualityConfig()


class ResponseQualityValidator:
    """
    Validate LLM response quality for affinity learning.

    Provides continuous quality scores (0.0-1.0) instead of binary success/failure,
    enabling the routing system to learn from response quality patterns.
    """

    def __init__(self, config: QualityConfig | None = None):
        self.config = config or _DEFAULT_CONFIG

    def validate(self, response: str, task_type: str | None = None) -> QualityScore:
        """
        Assess response quality, return score 0.0-1.0.

        Args:
            response: The LLM response text to validate
            task_type: Optional task type for length requirements

        Returns:
            QualityScore with overall score and component scores
        """
        if not response:
            return QualityScore(
                overall=0.0,
                repetition_score=0.0,
                length_score=0.0,
                coherence_score=0.0,
                issues=["empty_response"],
            )

        issues: list[str] = []

        # Calculate component scores
        repetition_score = self._check_repetition(response, issues)
        length_score = self._check_length(response, task_type, issues)
        coherence_score = self._check_coherence(response, issues)

        # Weighted combination
        weights = self.config.weights
        overall = (
            weights["repetition"] * repetition_score
            + weights["length"] * length_score
            + weights["coherence"] * coherence_score
        )

        # Clamp to [0.0, 1.0]
        overall = max(0.0, min(1.0, overall))

        return QualityScore(
            overall=overall,
            repetition_score=repetition_score,
            length_score=length_score,
            coherence_score=coherence_score,
            issues=issues,
        )

    def _check_repetition(self, response: str, issues: list[str]) -> float:
        """
        Check for repetitive content (looping models).

        Returns:
            Score from 0.0 (severe repetition) to 1.0 (no repetition)
        """
        words = response.lower().split()
        if len(words) < self.config.repetition_ngram_size:
            return 1.0  # Too short to detect patterns

        # Check n-gram repetition
        ngram_size = self.config.repetition_ngram_size
        ngrams = [
            tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)
        ]
        ngram_counts = Counter(ngrams)

        # Find most repeated n-gram
        if ngram_counts:
            max_repeats = max(ngram_counts.values())
            if max_repeats >= self.config.repetition_threshold:
                severity = min(max_repeats / 10, 1.0)  # Cap at 10 repeats
                issues.append(f"ngram_repetition:{max_repeats}")
                return max(0.0, 1.0 - severity)

        # Check sentence-level repetition
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip().lower() for s in sentences if s.strip()]
        if sentences:
            sentence_counts = Counter(sentences)
            max_sentence_repeats = max(sentence_counts.values())
            if max_sentence_repeats >= self.config.sentence_repeat_threshold:
                severity = min(max_sentence_repeats / 5, 1.0)
                issues.append(f"sentence_repetition:{max_sentence_repeats}")
                return max(0.0, 1.0 - severity * 0.5)

        # Check for token-level loops (e.g., "the the the")
        if len(words) >= 4:
            consecutive_repeats = 1
            max_consecutive = 1
            for i in range(1, len(words)):
                if words[i] == words[i - 1]:
                    consecutive_repeats += 1
                    max_consecutive = max(max_consecutive, consecutive_repeats)
                else:
                    consecutive_repeats = 1

            if max_consecutive >= 4:
                issues.append(f"token_loop:{max_consecutive}")
                return max(0.0, 1.0 - min(max_consecutive / 8, 0.8))

        return 1.0

    def _check_length(
        self, response: str, task_type: str | None, issues: list[str]
    ) -> float:
        """
        Check if response length is appropriate.

        Returns:
            Score from 0.0 (too short/empty) to 1.0 (appropriate length)
        """
        # Check minimum length
        if len(response) < self.config.min_response_length:
            issues.append(f"too_short:{len(response)}")
            return len(response) / self.config.min_response_length

        # Check whitespace ratio
        non_whitespace = len(response.replace(" ", "").replace("\n", "").replace("\t", ""))
        if len(response) > 0:
            whitespace_ratio = 1 - (non_whitespace / len(response))
            if whitespace_ratio > self.config.max_empty_ratio:
                issues.append(f"mostly_whitespace:{whitespace_ratio:.2f}")
                return max(0.0, 1.0 - whitespace_ratio)

        # Check task-specific minimum words
        words = response.split()
        word_count = len(words)

        if task_type:
            min_words = self.config.min_words_per_task.get(task_type, 10)
            if word_count < min_words:
                issues.append(f"insufficient_words:{word_count}/{min_words}")
                return min(1.0, word_count / min_words)

        # MDAP red-flag: Check maximum response length (tokens)
        # Per MDAP paper, responses >700 tokens are often low quality
        estimated_tokens = len(response) // 4  # ~4 chars per token
        if estimated_tokens > self.config.max_response_length_tokens:
            issues.append(
                f"response_too_long:{estimated_tokens}/{self.config.max_response_length_tokens}"
            )
            # Gradual penalty: 1.0 at threshold, 0.0 at 2x threshold
            ratio = estimated_tokens / self.config.max_response_length_tokens
            return max(0.0, 2.0 - ratio)

        return 1.0

    def _check_coherence(self, response: str, issues: list[str]) -> float:
        """
        Check for gibberish, encoding issues, and low vocabulary diversity.

        Returns:
            Score from 0.0 (incoherent) to 1.0 (coherent)
        """
        scores = []

        # Check non-ASCII ratio (encoding issues)
        ascii_count = sum(1 for c in response if ord(c) < 128)
        if len(response) > 0:
            non_ascii_ratio = 1 - (ascii_count / len(response))
            if non_ascii_ratio > self.config.max_non_ascii_ratio:
                issues.append(f"encoding_issues:{non_ascii_ratio:.2f}")
                scores.append(max(0.0, 1.0 - non_ascii_ratio))
            else:
                scores.append(1.0)

        # Check vocabulary diversity
        words = response.lower().split()
        if len(words) >= 5:  # Need enough words to measure
            unique_words = len(set(words))
            diversity = unique_words / len(words)
            if diversity < self.config.min_vocabulary_diversity:
                issues.append(f"low_diversity:{diversity:.2f}")
                scores.append(diversity / self.config.min_vocabulary_diversity)
            else:
                scores.append(1.0)
        else:
            scores.append(1.0)

        # Check for excessive single characters (gibberish)
        single_chars = sum(1 for c in response if c.isalpha())
        if len(response) > 10:
            # Count sequences of single characters separated by spaces
            char_pattern = re.findall(r"\b[a-zA-Z]\b", response)
            single_char_ratio = len(char_pattern) / max(len(words), 1)
            if single_char_ratio > self.config.max_single_char_ratio:
                issues.append(f"char_spam:{single_char_ratio:.2f}")
                scores.append(max(0.0, 1.0 - single_char_ratio))
            else:
                scores.append(1.0)

        # Check for incomplete sentences (truncation)
        stripped = response.rstrip()
        if stripped and len(stripped) > 20:
            # Check if ends with reasonable punctuation or word
            if not re.search(r"[.!?:;)\]\"'`]$|\w$", stripped):
                issues.append("truncated")
                scores.append(0.8)  # Minor penalty
            else:
                scores.append(1.0)
        else:
            scores.append(1.0)

        return sum(scores) / len(scores) if scores else 1.0


# Module-level singleton
_VALIDATOR: ResponseQualityValidator | None = None



def _load_quality_config() -> QualityConfig:
    """Load quality config from settings if available."""
    try:
        from .backend_manager import backend_manager

        settings = backend_manager.settings
        if not settings:
            return QualityConfig()

        routing = settings.get("routing", {})
        quality_cfg = routing.get("quality", {})

        if not quality_cfg.get("enabled", True):
            # Quality validation disabled
            return QualityConfig(enabled=False)

        return QualityConfig(
            enabled=quality_cfg.get("enabled", True),
            repetition_ngram_size=quality_cfg.get("repetition_ngram_size", 5),
            repetition_threshold=quality_cfg.get("repetition_threshold", 3),
            min_response_length=quality_cfg.get("min_response_length", 10),
            min_vocabulary_diversity=quality_cfg.get("min_vocabulary_diversity", 0.15),
        )
    except Exception:
        # Fallback to defaults if settings unavailable
        return QualityConfig()

def get_quality_validator() -> ResponseQualityValidator:
    """Get the singleton quality validator instance, configured from settings."""
    global _VALIDATOR
    if _VALIDATOR is None:
        config = _load_quality_config()
        _VALIDATOR = ResponseQualityValidator(config)
    return _VALIDATOR


def validate_response(response: str, task_type: str | None = None) -> QualityScore:
    """
    Convenience function to validate response quality.

    Args:
        response: The LLM response text to validate
        task_type: Optional task type for context-aware validation

    Returns:
        QualityScore with overall score and component scores.
        If quality validation is disabled, returns perfect scores.
    """
    validator = get_quality_validator()
    if not validator.config.enabled:
        # Quality validation disabled - return perfect score
        # (effectively maintaining binary success/failure behavior)
        return QualityScore(
            overall=1.0,
            repetition_score=1.0,
            length_score=1.0,
            coherence_score=1.0,
            issues=[],
        )
    return validator.validate(response, task_type)
