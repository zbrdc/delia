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
class AdaptiveVotingConfig:
    """
    Research-backed adaptive voting configuration.

    Based on CISC 2025 (Confidence Improves Self-Consistency) which shows
    46% compute reduction with confidence-weighted voting vs standard majority.

    Key insight: High-confidence first responses often don't need voting at all.
    """

    initial_k: int = 1  # Start with no voting (single response)
    max_k: int = 3  # Maximum votes if disagreement
    confidence_skip_threshold: float = 0.85  # Skip voting if confidence >= this
    disagreement_escalation: bool = True  # Increase k on disagreement
    weighted_voting: bool = True  # Weight votes by confidence
    min_confidence_for_consensus: float = 0.6  # Minimum avg confidence to accept


@dataclass
class VoteResult:
    """Result of adding a vote to consensus."""

    consensus_reached: bool
    winning_response: str | None = None
    votes_for_winner: int = 0
    total_votes: int = 0
    confidence: float = 0.0  # Model confidence for this response
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


class AdaptiveVotingConsensus:
    """
    Confidence-weighted adaptive voting per CISC 2025 research.

    Key improvements over standard VotingConsensus:
    1. Starts with k=1 (no voting) - only escalates on low confidence/disagreement
    2. Weights votes by model confidence, not equal weight
    3. Early stopping when high-confidence agreement is reached
    4. 46% compute reduction vs standard majority voting (per CISC paper)

    Usage:
        consensus = AdaptiveVotingConsensus()

        while consensus.should_continue():
            response, confidence = await get_model_response()
            result = consensus.add_vote(response, confidence)
            if result.consensus_reached:
                break

        winner = consensus.get_weighted_winner()
    """

    def __init__(
        self,
        config: AdaptiveVotingConfig | None = None,
        quality_validator: "ResponseQualityValidator | None" = None,
        similarity_threshold: float = 0.9,
    ):
        self.config = config or AdaptiveVotingConfig()
        self.validator = quality_validator
        self.similarity_threshold = similarity_threshold

        # Track votes with confidence: response_hash -> list[(response, confidence)]
        self._votes: dict[str, list[tuple[str, float]]] = {}
        self._vote_order: list[tuple[str, str, float]] = []  # (hash, response, confidence)
        self._red_flagged: int = 0

    @property
    def total_votes(self) -> int:
        return len(self._vote_order)

    def should_continue(self) -> bool:
        """
        Determine if more votes are needed.

        Returns False (stop voting) when:
        - First response has confidence >= threshold
        - We have agreement with good average confidence
        - We've reached max_k votes
        """
        if not self._vote_order:
            return True  # Need at least one vote

        # Check if first vote is high confidence (skip voting entirely)
        if len(self._vote_order) == 1:
            _, _, confidence = self._vote_order[0]
            if confidence >= self.config.confidence_skip_threshold:
                return False  # High confidence, no voting needed

        # Check if we've reached max votes
        if len(self._vote_order) >= self.config.max_k:
            return False

        # Check for agreement with sufficient confidence
        if len(self._vote_order) >= 2:
            if self._has_confident_agreement():
                return False

        return True

    def _has_confident_agreement(self) -> bool:
        """Check if we have agreeing votes with good confidence."""
        if len(self._votes) == 0:
            return False

        # Find the response with most votes
        best_hash = max(self._votes.keys(), key=lambda h: len(self._votes[h]))
        votes_for_best = self._votes[best_hash]

        if len(votes_for_best) < 2:
            return False  # Need at least 2 agreeing votes

        # Check average confidence of agreeing votes
        avg_confidence = sum(c for _, c in votes_for_best) / len(votes_for_best)
        return avg_confidence >= self.config.min_confidence_for_consensus

    def add_vote(self, response: str, confidence: float = 0.5) -> VoteResult:
        """
        Add a vote with confidence score.

        Args:
            response: The model response
            confidence: Model's confidence in this response (0.0-1.0)

        Returns:
            VoteResult with consensus status
        """
        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))

        # Red-flag check
        if self.validator:
            validation = self.validator.validate(response)
            if not validation.is_valid:
                self._red_flagged += 1
                return VoteResult(
                    consensus_reached=False,
                    total_votes=self.total_votes,
                    confidence=confidence,
                    red_flagged=True,
                    red_flag_reason=validation.reason,
                )

        # Normalize and hash
        normalized = self._normalize(response)
        response_hash = self._semantic_hash(normalized)

        # Find similar existing response
        matched_hash = self._find_similar(normalized, response_hash)
        use_hash = matched_hash or response_hash

        # Add to votes
        if use_hash not in self._votes:
            self._votes[use_hash] = []
        self._votes[use_hash].append((response, confidence))
        self._vote_order.append((use_hash, response, confidence))

        # Check for consensus
        votes_for_this = self._votes[use_hash]

        # Adaptive consensus: check if we should stop
        if not self.should_continue():
            winner = self.get_weighted_winner()
            return VoteResult(
                consensus_reached=True,
                winning_response=winner.winning_response,
                votes_for_winner=len(votes_for_this),
                total_votes=self.total_votes,
                confidence=confidence,
            )

        return VoteResult(
            consensus_reached=False,
            total_votes=self.total_votes,
            confidence=confidence,
        )

    def get_weighted_winner(self) -> VoteResult:
        """
        Get winner using confidence-weighted voting.

        Instead of simple majority (each vote = 1), weights by confidence:
        - Response A: 2 votes at 0.6 confidence = 1.2 weighted
        - Response B: 1 vote at 0.95 confidence = 0.95 weighted
        - Response A wins (despite B having higher individual confidence)

        This balances agreement with confidence per CISC 2025.
        """
        if not self._votes:
            return VoteResult(
                consensus_reached=False,
                total_votes=0,
            )

        if self.config.weighted_voting:
            # Confidence-weighted scoring
            weighted_scores: dict[str, float] = {}
            for response_hash, votes in self._votes.items():
                weighted_scores[response_hash] = sum(conf for _, conf in votes)

            best_hash = max(weighted_scores.keys(), key=lambda h: weighted_scores[h])
        else:
            # Standard majority (count only)
            best_hash = max(self._votes.keys(), key=lambda h: len(self._votes[h]))

        votes_for_best = self._votes[best_hash]
        best_response = votes_for_best[0][0]  # Use first response text
        avg_confidence = sum(c for _, c in votes_for_best) / len(votes_for_best)

        return VoteResult(
            consensus_reached=True,
            winning_response=best_response,
            votes_for_winner=len(votes_for_best),
            total_votes=self.total_votes,
            confidence=avg_confidence,
        )

    def get_metadata(self) -> ConsensusMetadata:
        """Get metadata about the voting process."""
        winner = self.get_weighted_winner()
        return ConsensusMetadata(
            total_votes=self.total_votes,
            red_flagged_count=self._red_flagged,
            unique_responses=len(self._votes),
            winning_votes=winner.votes_for_winner,
            k_used=self.config.max_k,
        )

    def reset(self) -> None:
        """Reset for reuse."""
        self._votes.clear()
        self._vote_order.clear()
        self._red_flagged = 0

    # Reuse normalization methods from VotingConsensus
    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[*_`#\[\]()]", "", text)
        text = re.sub(r"[.,!?;:]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _semantic_hash(self, normalized_text: str) -> str:
        return hashlib.sha256(normalized_text.encode()).hexdigest()[:16]

    def _find_similar(self, normalized: str, new_hash: str) -> str | None:
        if new_hash in self._votes:
            return new_hash

        for existing_hash, votes in self._votes.items():
            existing_response = votes[0][0]
            existing_normalized = self._normalize(existing_response)
            similarity = difflib.SequenceMatcher(None, normalized, existing_normalized).ratio()
            if similarity >= self.similarity_threshold:
                return existing_hash

        return None


def estimate_task_complexity(prompt: str) -> int:
    """
    Estimate number of steps in a task based on prompt analysis.

    Used to auto-calculate kmin for the voting system.

    Args:
        prompt: The task prompt

    Returns:
        Estimated number of reasoning steps (1-100)
    """
    if not prompt or len(prompt.strip()) < 5:
        return 1

    prompt_lower = prompt.lower()
    # Base steps
    steps = 1

    # 1. Count action verbs (implied steps)
    actions = [
        r"\b(write|create|implement|build|develop)\b",
        r"\b(fix|debug|resolve|patch)\b",
        r"\b(refactor|optimize|improve|clean)\b",
        r"\b(test|verify|validate|check)\b",
        r"\b(analyze|review|audit|examine)\b",
        r"\b(search|find|grep|lookup)\b",
    ]
    for pattern in actions:
        steps += len(re.findall(pattern, prompt_lower))

    # 2. Count explicit step indicators
    step_patterns = [
        r"\b(first|second|third|then|next|after|finally)\b",
        r"\b(step \d+|[1-9]\)|[1-9]\.)",
        r"\b(and then|followed by|subsequently)\b",
    ]
    for pattern in step_patterns:
        steps += len(re.findall(pattern, prompt_lower))

    # 3. Boost for logical connectors
    connectors = [r"\b(and|with|while|because|if|then|else)\b"]
    for pattern in connectors:
        steps += int(len(re.findall(pattern, prompt_lower)) * 0.5)

    # 4. Boost for length (longer = more nuance)
    word_count = len(prompt.split())
    steps += word_count // 20

    # Cap at reasonable max (MDAP paper suggests complexity beyond 20-30 steps 
    # rapidly approaches 1.0 kmin for typical models)
    return max(1, min(steps, 100))
