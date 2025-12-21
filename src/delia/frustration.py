# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
DEPRECATED: This module has been consolidated into orchestration/intrinsics.py

The frustration detection logic is now part of IntrinsicsEngine.check_user_state(),
which provides a unified pre-execution sanity check combining:
- Answerability (context sufficiency)
- User state (frustration detection)
- Groundedness (hallucination detection)

Use:
    from delia.orchestration.intrinsics import get_intrinsics_engine, FrustrationLevel
    intrinsics = get_intrinsics_engine()
    user_state = intrinsics.check_user_state(message, is_repeat=..., repeat_count=...)

This file is kept for backwards compatibility but will be removed in a future release.
---

Frustration Detection for Quality Feedback (LEGACY).

When users repeat a question, it's a strong signal that the previous
answer was wrong or unhelpful. This module detects repeated questions
and provides feedback for the melon reward system.

Key insight: Repeated questions = implicit negative feedback.

Usage:
    from delia.frustration import get_frustration_tracker
    
    tracker = get_frustration_tracker()
    
    # Check for repeat before answering
    repeat_info = tracker.check_repeat(session_id, message, model_used)
    
    if repeat_info.is_repeat:
        # Previous model answered poorly - penalize
        melon_tracker.penalize(repeat_info.previous_model, task_type, melons=3)
        # Consider voting mode for more reliable answer
        
    # After answering, record the response
    tracker.record_response(session_id, message, model_used)
"""

from __future__ import annotations

import hashlib
import time
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger()

# Similarity threshold for considering questions as "repeats"
SIMILARITY_THRESHOLD = 0.85

# Time window for considering repeats (5 minutes)
REPEAT_WINDOW_SECONDS = 300

# Angry/Frustrated Keywords (Sentiment Analysis)
ANGRY_KEYWORDS = {
    "stupid", "dumb", "idiot", "useless", "broken", "fail", "wrong", "bad",
    "terrible", "horrible", "awful", "trash", "garbage", "waste", "clown",
    "stop", "quit", "exit", "kill", "die", "hate", "shut up"
}

# Negative Feedback Patterns
NEGATIVE_PATTERNS = [
    r"\b(that'?s|is)\s+(wrong|incorrect|false|not right)\b",
    r"\b(no|nope|nah)\b.{0,20}\b(wrong|incorrect)\b",
    r"\b(you|it)\s+(missed|failed|didn'?t)\b",
    r"\b(not|isn'?t)\s+(what|the answer)\b",
    r"\b(stop|don'?t)\s+(doing|saying)\b",
    r"\b(again|repeat)\b",
]

class FrustrationLevel(Enum):
    """Level of user frustration."""
    NONE = "none"
    LOW = "low"       # Mild annoyance, single repeat
    MEDIUM = "medium" # Explicit negative feedback, 2+ repeats
    HIGH = "high"     # Angry keywords, 3+ repeats, rapid-fire


@dataclass
class RepeatInfo:
    """Information about a repeated question."""
    is_repeat: bool = False
    repeat_count: int = 0
    previous_model: str | None = None
    previous_response: str | None = None
    time_since_last: float = 0.0  # seconds
    similarity: float = 0.0
    
    # Enhanced metrics
    level: FrustrationLevel = FrustrationLevel.NONE
    has_angry_keywords: bool = False
    has_negative_feedback: bool = False
    is_rapid_fire: bool = False  # < 5s since last message


@dataclass
class QuestionRecord:
    """Record of a question and its response."""
    message: str
    message_hash: str
    model_used: str
    response: str
    timestamp: float
    session_id: str


class FrustrationTracker:
    """
    Tracks user frustration through multiple signals:
    1. Repeated questions (the classic signal)
    2. Sentiment analysis (angry keywords)
    3. Negative feedback patterns ("that's wrong")
    4. Rapid-fire interaction (mash frustration)
    """
    
    def __init__(self, max_records_per_session: int = 50):
        self._records: dict[str, list[QuestionRecord]] = {}  # session_id -> records
        self._max_records = max_records_per_session
        self._negative_regexes = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]
        
    def _hash_message(self, message: str) -> str:
        """Create normalized hash for message comparison."""
        # Normalize: lowercase, strip, remove extra whitespace
        normalized = " ".join(message.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _check_sentiment(self, message: str) -> bool:
        """Check for angry keywords."""
        words = set(message.lower().split())
        return bool(words & ANGRY_KEYWORDS)

    def _check_negative_feedback(self, message: str) -> bool:
        """Check for explicit negative feedback patterns."""
        return any(p.search(message) for p in self._negative_regexes)

    def _calculate_similarity(self, msg1: str, msg2: str) -> float:
        """
        Calculate similarity between two messages.
        
        Uses hash for exact matches, then tries semantic similarity,
        falls back to word overlap.
        """
        # Exact match (after normalization)
        if self._hash_message(msg1) == self._hash_message(msg2):
            return 1.0
        
        # Try semantic similarity using sentence-transformers (if available)
        try:
            from .orchestration.semantic import get_semantic_matcher
            matcher = get_semantic_matcher()
            if matcher.is_initialized:
                # Use the model directly for pairwise similarity
                emb1 = matcher._model.encode(msg1, normalize_embeddings=True)
                emb2 = matcher._model.encode(msg2, normalize_embeddings=True)
                import numpy as np
                semantic_sim = float(np.dot(emb1, emb2))
                if semantic_sim >= 0.7:
                    return semantic_sim
        except Exception:
            pass  # Fall back to word overlap
        
        # Fall back: Word overlap with stop word filtering
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "what", "how", "why", "when", "where", "which", "who",
            "to", "for", "of", "in", "on", "at", "by", "with",
            "can", "could", "would", "should", "please", "me", "i", "you",
        }
        
        words1 = {w for w in msg1.lower().split() if w not in stop_words and len(w) > 2}
        words2 = {w for w in msg2.lower().split() if w not in stop_words and len(w) > 2}
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity on meaningful words
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard = len(intersection) / len(union)
        
        # Boost if key content words are shared
        if len(intersection) >= 2:
            jaccard = min(1.0, jaccard + 0.2)
        
        return jaccard
    
    def check_repeat(
        self,
        session_id: str,
        message: str,
        current_model: str | None = None,
    ) -> RepeatInfo:
        """
        Check if this message indicates frustration via repeats or sentiment.
        
        Args:
            session_id: Current session
            message: User's message
            current_model: Model about to respond (optional)
            
        Returns:
            RepeatInfo with detailed frustration analysis
        """
        if session_id not in self._records:
            self._records[session_id] = []
        
        records = self._records[session_id]
        current_time = time.time()
        
        # 1. Check for rapid-fire (mash frustration)
        is_rapid_fire = False
        if records:
            last_record = records[-1]
            if (current_time - last_record.timestamp) < 5.0:
                is_rapid_fire = True

        # 2. Check for angry keywords
        has_angry_keywords = self._check_sentiment(message)

        # 3. Check for negative feedback
        has_negative_feedback = self._check_negative_feedback(message)
        
        # 4. Check for repeats
        repeat_count = 0
        most_recent_match: QuestionRecord | None = None
        highest_similarity = 0.0
        
        if records:
            for record in reversed(records):  # Most recent first
                # Skip if too old
                age = current_time - record.timestamp
                if age > REPEAT_WINDOW_SECONDS:
                    break
                
                # Check similarity
                similarity = self._calculate_similarity(message, record.message)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    repeat_count += 1
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_recent_match = record

        # Determine Frustration Level
        level = FrustrationLevel.NONE
        
        # Scoring system - refined thresholds
        score = 0
        if repeat_count > 0: score += repeat_count * 1.5  # Repeats are strong signal
        if has_angry_keywords: score += 3.5              # Anger is very strong
        if has_negative_feedback: score += 2.5           # Explicit feedback is strong
        if is_rapid_fire: score += 0.5                   # Mash is minor boost
        
        if score >= 5.0:
            level = FrustrationLevel.HIGH
        elif score >= 3.0:
            level = FrustrationLevel.MEDIUM
        elif score >= 1.5:
            level = FrustrationLevel.LOW
            
        info = RepeatInfo(
            is_repeat=(repeat_count > 0),
            repeat_count=repeat_count,
            previous_model=most_recent_match.model_used if most_recent_match else None,
            previous_response=most_recent_match.response[:200] if most_recent_match else None,
            time_since_last=(current_time - most_recent_match.timestamp) if most_recent_match else 0.0,
            similarity=highest_similarity,
            level=level,
            has_angry_keywords=has_angry_keywords,
            has_negative_feedback=has_negative_feedback,
            is_rapid_fire=is_rapid_fire,
        )
        
        if level != FrustrationLevel.NONE:
            log.warning(
                "frustration_detected",
                session=session_id[:8],
                level=level.value,
                score=score,
                repeat=repeat_count,
                angry=has_angry_keywords,
                negative=has_negative_feedback,
                rapid=is_rapid_fire,
            )
            
        return info
    
    def record_response(
        self,
        session_id: str,
        message: str,
        model_used: str,
        response: str,
    ) -> None:
        """
        Record a question and its response for future repeat detection.
        
        Args:
            session_id: Session ID
            message: User's message
            model_used: Model that responded
            response: The model's response
        """
        if session_id not in self._records:
            self._records[session_id] = []
        
        records = self._records[session_id]
        
        # Add new record
        record = QuestionRecord(
            message=message,
            message_hash=self._hash_message(message),
            model_used=model_used,
            response=response,
            timestamp=time.time(),
            session_id=session_id,
        )
        records.append(record)
        
        # Trim old records
        if len(records) > self._max_records:
            self._records[session_id] = records[-self._max_records:]
    
    def clear_session(self, session_id: str) -> None:
        """Clear records for a session."""
        self._records.pop(session_id, None)
    
    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        total_records = sum(len(r) for r in self._records.values())
        return {
            "sessions_tracked": len(self._records),
            "total_records": total_records,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_frustration_tracker: FrustrationTracker | None = None


def get_frustration_tracker() -> FrustrationTracker:
    """Get the global frustration tracker."""
    global _frustration_tracker
    if _frustration_tracker is None:
        _frustration_tracker = FrustrationTracker()
    return _frustration_tracker


def reset_frustration_tracker() -> None:
    """Reset the global tracker (for testing)."""
    global _frustration_tracker
    _frustration_tracker = None


__all__ = [
    "FrustrationTracker",
    "RepeatInfo",
    "get_frustration_tracker",
    "reset_frustration_tracker",
]

