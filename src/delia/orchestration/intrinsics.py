# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
LLM Intrinsics Engine - Inspired by IBM Granite RAG.

Provides meta-cognitive sanity checks for RAG and Agentic workflows:
- Answerability: Can the task be completed with current context?
- Uncertainty: How confident is the model in its response?
- Groundedness: Is the response supported by the provided facts?

Uses the configured quick-tier model for fast (~100ms) intrinsic checks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger()


class IntrinsicAction(str, Enum):
    """Recommended action based on intrinsic check."""
    PROCEED = "proceed"           # Context sufficient, go ahead
    FETCH_MORE = "fetch_more"     # Need more context (trigger RAG)
    ESCALATE = "escalate"         # Use stronger model or ToT
    ESCALATE_VOTING = "escalate_voting"  # Use voting for consensus
    ESCALATE_DEEP = "escalate_deep"      # Use deep thinking
    CLARIFY = "clarify"           # Ask user for clarification


class FrustrationLevel(str, Enum):
    """Level of user frustration (absorbed from frustration.py)."""
    NONE = "none"
    LOW = "low"       # Mild annoyance
    MEDIUM = "medium" # Explicit negative feedback
    HIGH = "high"     # Angry keywords, multiple signals


# Angry/Frustrated Keywords
_ANGRY_KEYWORDS = {
    "stupid", "dumb", "idiot", "useless", "broken", "fail", "wrong", "bad",
    "terrible", "horrible", "awful", "trash", "garbage", "waste", "clown",
    "stop", "quit", "exit", "kill", "die", "hate", "shut up"
}

# Negative Feedback Patterns
_NEGATIVE_PATTERNS = [
    re.compile(r"\b(that'?s|is)\s+(wrong|incorrect|false|not right)\b", re.IGNORECASE),
    re.compile(r"\b(no|nope|nah)\b.{0,20}\b(wrong|incorrect)\b", re.IGNORECASE),
    re.compile(r"\b(you|it)\s+(missed|failed|didn'?t)\b", re.IGNORECASE),
    re.compile(r"\b(not|isn'?t)\s+(what|the answer)\b", re.IGNORECASE),
    re.compile(r"\b(stop|don'?t)\s+(doing|saying)\b", re.IGNORECASE),
    re.compile(r"\b(again|repeat)\b", re.IGNORECASE),
]


@dataclass
class IntrinsicResult:
    """Result of an intrinsic check."""
    score: float          # 0.0 to 1.0 (higher = more confident)
    reasoning: str        # Why this score
    passed: bool          # Whether threshold was met
    action: IntrinsicAction = IntrinsicAction.PROCEED


@dataclass
class AnswerabilityResult(IntrinsicResult):
    """Extended result for answerability checks."""
    missing_info: list[str] | None = None  # What's missing from context


@dataclass
class UserStateResult:
    """Result of user state/frustration analysis (no LLM needed)."""
    level: FrustrationLevel
    has_angry_keywords: bool
    has_negative_feedback: bool
    is_repeat: bool  # Set by caller based on session history
    repeat_count: int  # Set by caller
    action: IntrinsicAction
    reasoning: str


class IntrinsicsEngine:
    """
    Manages LLM meta-cognitive checks using the quick-tier model.

    These are fast (~100ms) sanity checks that gate expensive operations:
    - Before RAG: Is the context sufficient?
    - After generation: Is the response grounded?
    - On uncertainty: Should we escalate to a stronger model?
    """

    # Thresholds for actions
    PROCEED_THRESHOLD = 0.7      # Above this: proceed normally
    FETCH_THRESHOLD = 0.4       # Below proceed, above this: fetch more context
    ESCALATE_THRESHOLD = 0.4    # Below this: escalate to stronger model

    def __init__(self, call_llm_fn: Any):
        self.call_llm = call_llm_fn
        self._model: str | None = None

    async def _get_model(self) -> str:
        """Get the quick-tier model for intrinsic checks."""
        if self._model is None:
            from ..routing import select_model
            self._model = await select_model("quick")
        return self._model

    async def check_answerability(
        self,
        task: str,
        context: str,
        threshold: float = 0.6,
    ) -> AnswerabilityResult:
        """
        Check if the provided context is sufficient to complete the task.

        Args:
            task: The user's request/question
            context: Available context (files, docs, etc.)
            threshold: Minimum score to pass (default 0.6)

        Returns:
            AnswerabilityResult with score, reasoning, and recommended action
        """
        model = await self._get_model()

        # Truncate context to avoid overwhelming small models
        context_preview = context[:3000] if len(context) > 3000 else context

        system = """You evaluate if context is sufficient to answer a question.
Output JSON only: {"score": 0.0-1.0, "reasoning": "why", "missing": ["item1", "item2"]}
- score 1.0 = context fully answers the question
- score 0.0 = context is completely irrelevant
- missing = list of information that would help (empty if sufficient)"""

        prompt = f"""TASK: {task}

CONTEXT:
{context_preview}

Is this context sufficient to complete the task?"""

        try:
            res = await self.call_llm(
                model=model,
                prompt=prompt,
                system=system,
                task_type="quick",
                temperature=0.1,
                max_tokens=200,
            )

            response_text = res.get("response", "{}")

            # Extract JSON from response
            match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if match:
                # Clean markdown artifacts
                json_str = match.group(0)
                json_str = re.sub(r'\*\*([^*]+)\*\*', r'\1', json_str)
                data = json.loads(json_str)
            else:
                data = {}

            score = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            missing = data.get("missing", [])

            # Determine action based on score
            if score >= self.PROCEED_THRESHOLD:
                action = IntrinsicAction.PROCEED
            elif score >= self.FETCH_THRESHOLD:
                action = IntrinsicAction.FETCH_MORE
            else:
                action = IntrinsicAction.ESCALATE

            return AnswerabilityResult(
                score=score,
                reasoning=reasoning,
                passed=score >= threshold,
                action=action,
                missing_info=missing if missing else None,
            )

        except Exception as e:
            log.warning("answerability_check_failed", error=str(e))
            # On failure, be permissive but flag it
            return AnswerabilityResult(
                score=0.5,
                reasoning=f"Check failed: {e}",
                passed=True,  # Don't block on intrinsic failure
                action=IntrinsicAction.PROCEED,
                missing_info=None,
            )

    async def check_confidence(
        self,
        task: str,
        response: str,
        threshold: float = 0.6,
    ) -> IntrinsicResult:
        """
        Quantify confidence in a generated response.

        Args:
            task: The original request
            response: The generated response to evaluate
            threshold: Minimum score to pass

        Returns:
            IntrinsicResult with confidence score and action
        """
        model = await self._get_model()

        # Truncate for speed
        response_preview = response[:2000] if len(response) > 2000 else response

        system = """Rate how confident and accurate this response is.
Output JSON only: {"score": 0.0-1.0, "reasoning": "why"}
- score 1.0 = completely confident and accurate
- score 0.5 = uncertain, may have errors
- score 0.0 = likely wrong or hallucinated"""

        prompt = f"""TASK: {task}

RESPONSE:
{response_preview}

How confident and accurate is this response?"""

        try:
            res = await self.call_llm(
                model=model,
                prompt=prompt,
                system=system,
                task_type="quick",
                temperature=0.1,
                max_tokens=150,
            )

            response_text = res.get("response", "{}")
            match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if match:
                json_str = re.sub(r'\*\*([^*]+)\*\*', r'\1', match.group(0))
                data = json.loads(json_str)
            else:
                data = {}

            score = float(data.get("score", 0.7))
            reasoning = data.get("reasoning", "No reasoning provided")

            # Determine action
            if score >= self.PROCEED_THRESHOLD:
                action = IntrinsicAction.PROCEED
            elif score >= self.ESCALATE_THRESHOLD:
                action = IntrinsicAction.CLARIFY
            else:
                action = IntrinsicAction.ESCALATE

            return IntrinsicResult(
                score=score,
                reasoning=reasoning,
                passed=score >= threshold,
                action=action,
            )

        except Exception as e:
            log.warning("confidence_check_failed", error=str(e))
            return IntrinsicResult(
                score=0.7,
                reasoning=f"Check failed: {e}",
                passed=True,
                action=IntrinsicAction.PROCEED,
            )

    async def check_groundedness(
        self,
        response: str,
        sources: str,
        threshold: float = 0.6,
    ) -> IntrinsicResult:
        """
        Check if a response is grounded in the provided sources.

        Detects hallucination by verifying claims against source material.

        Args:
            response: The generated response
            sources: The source material it should be grounded in
            threshold: Minimum score to pass

        Returns:
            IntrinsicResult with groundedness score
        """
        model = await self._get_model()

        sources_preview = sources[:3000] if len(sources) > 3000 else sources
        response_preview = response[:1500] if len(response) > 1500 else response

        system = """Check if the response is grounded in the sources (not hallucinated).
Output JSON only: {"score": 0.0-1.0, "reasoning": "why"}
- score 1.0 = every claim is supported by sources
- score 0.5 = some claims lack support
- score 0.0 = contains fabricated information"""

        prompt = f"""SOURCES:
{sources_preview}

RESPONSE:
{response_preview}

Is this response grounded in the sources?"""

        try:
            res = await self.call_llm(
                model=model,
                prompt=prompt,
                system=system,
                task_type="quick",
                temperature=0.1,
                max_tokens=150,
            )

            response_text = res.get("response", "{}")
            match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if match:
                json_str = re.sub(r'\*\*([^*]+)\*\*', r'\1', match.group(0))
                data = json.loads(json_str)
            else:
                data = {}

            score = float(data.get("score", 0.7))
            reasoning = data.get("reasoning", "No reasoning provided")

            action = IntrinsicAction.PROCEED if score >= threshold else IntrinsicAction.ESCALATE

            return IntrinsicResult(
                score=score,
                reasoning=reasoning,
                passed=score >= threshold,
                action=action,
            )

        except Exception as e:
            log.warning("groundedness_check_failed", error=str(e))
            return IntrinsicResult(
                score=0.7,
                reasoning=f"Check failed: {e}",
                passed=True,
                action=IntrinsicAction.PROCEED,
            )


    def check_user_state(
        self,
        message: str,
        is_repeat: bool = False,
        repeat_count: int = 0,
    ) -> UserStateResult:
        """
        Analyze user message for frustration signals (no LLM needed).

        This consolidates frustration.py logic into intrinsics for a unified
        pre-execution check. Fast (~0ms) since it's pure regex/keyword matching.

        Args:
            message: User's current message
            is_repeat: Whether this is a repeated question (caller determines)
            repeat_count: How many times repeated (caller determines)

        Returns:
            UserStateResult with frustration level and recommended action
        """
        # Check for angry keywords
        words = set(message.lower().split())
        has_angry_keywords = bool(words & _ANGRY_KEYWORDS)

        # Check for negative feedback patterns
        has_negative_feedback = any(p.search(message) for p in _NEGATIVE_PATTERNS)

        # Calculate frustration score
        score = 0.0
        if repeat_count > 0:
            score += repeat_count * 1.5  # Repeats are strong signal
        if has_angry_keywords:
            score += 3.5  # Anger is very strong
        if has_negative_feedback:
            score += 2.5  # Explicit feedback is strong

        # Determine level
        if score >= 5.0:
            level = FrustrationLevel.HIGH
        elif score >= 3.0:
            level = FrustrationLevel.MEDIUM
        elif score >= 1.5:
            level = FrustrationLevel.LOW
        else:
            level = FrustrationLevel.NONE

        # Determine action based on level
        if level == FrustrationLevel.HIGH:
            action = IntrinsicAction.ESCALATE_DEEP
            reasoning = f"High frustration (score={score:.1f}): escalate to deep thinking"
        elif level == FrustrationLevel.MEDIUM:
            action = IntrinsicAction.ESCALATE_VOTING
            reasoning = f"Medium frustration (score={score:.1f}): escalate to voting"
        elif level == FrustrationLevel.LOW:
            action = IntrinsicAction.ESCALATE_VOTING
            reasoning = f"Low frustration (score={score:.1f}): consider voting"
        else:
            action = IntrinsicAction.PROCEED
            reasoning = "No frustration signals detected"

        if level != FrustrationLevel.NONE:
            log.warning(
                "user_frustration_detected",
                level=level.value,
                score=score,
                angry=has_angry_keywords,
                negative=has_negative_feedback,
                repeats=repeat_count,
            )

        return UserStateResult(
            level=level,
            has_angry_keywords=has_angry_keywords,
            has_negative_feedback=has_negative_feedback,
            is_repeat=is_repeat,
            repeat_count=repeat_count,
            action=action,
            reasoning=reasoning,
        )


# Singleton instance
_engine: IntrinsicsEngine | None = None


def get_intrinsics_engine(call_llm_fn: Any = None) -> IntrinsicsEngine:
    """Get the global IntrinsicsEngine instance."""
    global _engine
    if _engine is None:
        if call_llm_fn is None:
            from ..llm import call_llm
            call_llm_fn = call_llm
        _engine = IntrinsicsEngine(call_llm_fn)
    return _engine
