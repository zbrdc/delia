"""
K-Voting Consensus Implementation.

Implements the "first-to-ahead-by-k" voting mechanism from the MDAP paper
"Smashing Intelligence into a Million Pieces" for mathematically-guaranteed
reliability in agentic systems.

Key insight: With base model accuracy p=0.99 and k=3 votes:
    P(correct) = 1 / (1 + ((1-p)/p)^k) = 0.999999

Formula verified with Wolfram Alpha.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from delia.quality import ResponseQualityValidator, ValidationResult


@dataclass
class VoteResult:
    """Result of adding a vote to consensus."""

    consensus_reached: bool
    winning_response: str | None = None
    votes_for_winner: int = 0
    total_votes: int = 0
    red_flagged: bool = False
    red_flag_reason: str | None = None


@dataclass
class ConsensusMetadata:
    """Metadata about the consensus process."""

    total_votes: int = 0
    red_flagged_count: int = 0
    unique_responses: int = 0
    winning_votes: int = 0
    k_used: int = 2


class VotingConsensus:
    """
    K-voting consensus per MDAP paper.

    Implements "first-to-ahead-by-k" voting where the first response to
    receive k matching votes wins. This provides mathematical guarantees
    on correctness probability.

    Responses are compared using normalized text similarity. Red-flagged
    responses (too long, repetitive, incoherent) are excluded from voting.
    """

    def __init__(
        self,
        k: int,
        quality_validator: ResponseQualityValidator | None = None,
        similarity_threshold: float = 0.9,
        max_response_length: int = 700,
    ):
        """
        Initialize voting consensus.

        Args:
            k: Number of matching votes needed (first-to-ahead-by-k)
            quality_validator: Optional validator for red-flagging
            similarity_threshold: Min similarity for responses to match (0.0-1.0)
            max_response_length: Max tokens before red-flagging
        """
        self.k = k
        self.validator = quality_validator
        self.similarity_threshold = similarity_threshold
        self.max_response_length = max_response_length

        # response_hash -> (count, original_response)
        self._votes: dict[str, tuple[int, str]] = {}
        self._red_flagged: int = 0
        self._total_votes: int = 0

    def add_vote(self, response: str) -> VoteResult:
        """
        Add a vote for a response.

        Returns:
            VoteResult indicating if consensus was reached
        """
        self._total_votes += 1

        # Red-flag check first
        if self.validator:
            validation = self.validator.validate(response)
            if not validation.is_valid:
                self._red_flagged += 1
                return VoteResult(
                    consensus_reached=False,
                    total_votes=self._total_votes,
                    red_flagged=True,
                    red_flag_reason=validation.reason,
                )

        # Simple length check if no validator
        elif self._estimate_tokens(response) > self.max_response_length:
            self._red_flagged += 1
            return VoteResult(
                consensus_reached=False,
                total_votes=self._total_votes,
                red_flagged=True,
                red_flag_reason="response_too_long",
            )

        # Normalize and hash for comparison
        normalized = self._normalize(response)
        response_hash = self._semantic_hash(normalized)

        # Check if similar to existing responses
        matched_hash = self._find_similar(normalized, response_hash)

        if matched_hash:
            # Increment existing vote
            count, original = self._votes[matched_hash]
            self._votes[matched_hash] = (count + 1, original)
            current_count = count + 1
        else:
            # New unique response
            self._votes[response_hash] = (1, response)
            current_count = 1

        # Check if any response is ahead by k
        if current_count >= self.k:
            winning_response = (
                self._votes[matched_hash][1] if matched_hash else response
            )
            return VoteResult(
                consensus_reached=True,
                winning_response=winning_response,
                votes_for_winner=current_count,
                total_votes=self._total_votes,
            )

        return VoteResult(
            consensus_reached=False,
            total_votes=self._total_votes,
        )

    def get_best_response(self) -> tuple[str | None, ConsensusMetadata]:
        """
        Get the best response even without full consensus.

        Returns:
            (best_response, metadata) - response with most votes
        """
        if not self._votes:
            return None, ConsensusMetadata(
                total_votes=self._total_votes,
                red_flagged_count=self._red_flagged,
            )

        # Find response with most votes
        best_hash = max(self._votes.keys(), key=lambda h: self._votes[h][0])
        count, response = self._votes[best_hash]

        return response, ConsensusMetadata(
            total_votes=self._total_votes,
            red_flagged_count=self._red_flagged,
            unique_responses=len(self._votes),
            winning_votes=count,
            k_used=self.k,
        )

    def reset(self) -> None:
        """Reset voting state for reuse."""
        self._votes.clear()
        self._red_flagged = 0
        self._total_votes = 0

    @staticmethod
    def calculate_kmin(
        total_steps: int,
        target_accuracy: float = 0.9999,
        base_accuracy: float | None = None,
    ) -> int:
        """
        Calculate minimum k for target overall accuracy.

        Implements kmin = Θ(ln s) from MDAP paper.

        Args:
            total_steps: Number of steps in the task
            target_accuracy: Target overall accuracy (e.g., 0.9999)
            base_accuracy: Per-step model accuracy (0.0-1.0). 
                           If None, defaults to 0.99.

        Returns:
            Minimum k value needed (clamped to [2, 5])
        """
        if total_steps <= 0:
            return 2

        # Use provided accuracy or optimistic default
        p = base_accuracy if base_accuracy is not None else 0.99

        # For each step, need per-step accuracy >= target^(1/s)
        # P(correct|k) = 1 / (1 + ((1-p)/p)^k) >= step_target
        # Solve for k: k >= ln((1-step_target)/step_target) / ln((1-p)/p)

        step_target = target_accuracy ** (1 / total_steps)

        # Avoid division by zero or invalid log arguments
        if step_target >= 1.0:
            return 2  # Already at target
        
        # If model is no better than random guessing, use max redundancy
        if p <= 0.5:
            return 5

        try:
            ratio = (1 - step_target) / step_target
            log_base = (1 - p) / p
            if ratio <= 0 or log_base <= 0:
                return 2
            k = math.ceil(math.log(ratio) / math.log(log_base))
        except (ValueError, ZeroDivisionError):
            k = 2

        # Clamp to reasonable range (MDAP paper suggests k=3-5 for high stakes)
        return max(2, min(k, 5))

    @staticmethod
    def voting_probability(k: int, base_accuracy: float = 0.99) -> float:
        """
        Calculate probability of correct outcome with k votes.

        Formula: P(correct) = 1 / (1 + ((1-p)/p)^k)

        Args:
            k: Number of matching votes required
            base_accuracy: Per-model accuracy (0.0-1.0)

        Returns:
            Probability of correct consensus
        """
        p = base_accuracy
        if p >= 1.0:
            return 1.0
        if p <= 0.0:
            return 0.0

        return 1.0 / (1.0 + ((1 - p) / p) ** k)

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove markdown formatting
        text = re.sub(r"[*_`#\[\]()]", "", text)
        # Remove punctuation for semantic grouping
        text = re.sub(r"[.,!?;:]", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _semantic_hash(self, normalized_text: str) -> str:
        """Create a hash for normalized text."""
        return hashlib.sha256(normalized_text.encode()).hexdigest()[:16]

    def _find_similar(
        self, normalized: str, new_hash: str
    ) -> str | None:
        """
        Find if normalized text is similar to any existing vote.

        Returns:
            Hash of matching response, or None if no match
        """
        # First check exact match
        if new_hash in self._votes:
            return new_hash

        # Check semantic similarity with existing responses
        for existing_hash, (_, existing_response) in self._votes.items():
            existing_normalized = self._normalize(existing_response)
            similarity = self._text_similarity(normalized, existing_normalized)
            if similarity >= self.similarity_threshold:
                return existing_hash

        return None

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using difflib.SequenceMatcher.

        Returns:
            Similarity score 0.0-1.0 (Gestalt Pattern Matching)
        """
        if not text1 or not text2:
            return 0.0

        # difflib is much better than Jaccard for sentence structure
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(text) // 4


def estimate_task_complexity(prompt: str) -> int:
    """
    Estimate number of steps in a task based on prompt analysis.

    Used to auto-calculate kmin for the voting system.

    Args:
        prompt: The task prompt

    Returns:
        Estimated number of reasoning steps (1-100)
    """
    # Simple heuristics for step estimation
    steps = 1

    # Count explicit step indicators
    step_patterns = [
        r"\b(first|second|third|then|next|after|finally)\b",
        r"\b(step \d+|1\)|2\)|3\))",
        r"\b(and then|followed by|subsequently)\b",
    ]

    for pattern in step_patterns:
        matches = re.findall(pattern, prompt.lower())
        steps += len(matches)

    # Estimate based on length (longer = more complex)
    word_count = len(prompt.split())
    if word_count > 500:
        steps += 5
    elif word_count > 200:
        steps += 3
    elif word_count > 100:
        steps += 1

    # Cap at reasonable max
    return min(steps, 100)
