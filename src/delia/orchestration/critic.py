# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Senior QA Critic Implementation.

Uses a high-fidelity model to verify the output of the Executor
before it reaches the user.

Enhanced with multi-branch evaluation for ToT meta-orchestration:
- evaluate_branches(): Compare multiple orchestration results
- Weighted scoring: 0.35*correctness + 0.25*completeness + 0.25*quality + 0.15*confidence
- Returns structured BranchEvaluation with reasoning for Delia learning
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from .result import DetectedIntent, OrchestrationResult
from ..prompts import ModelRole, OrchestrationMode, build_system_prompt

if TYPE_CHECKING:
    from ..backend_manager import BackendConfig

log = structlog.get_logger()


# Scoring weights for branch evaluation (sum to 1.0)
WEIGHT_CORRECTNESS = 0.35  # Does it solve the problem?
WEIGHT_COMPLETENESS = 0.25  # All requirements met?
WEIGHT_QUALITY = 0.25  # Code quality, reasoning depth
WEIGHT_CONFIDENCE = 0.15  # Critic's certainty


@dataclass
class BranchScore:
    """
    Detailed scoring for a single ToT branch.

    Each dimension is scored 0-1 (normalized from 0-10 LLM output).
    """
    mode: OrchestrationMode
    correctness: float  # 0-1: Does it solve the problem?
    completeness: float  # 0-1: All requirements met?
    quality: float  # 0-1: Code quality, reasoning depth
    confidence: float  # 0-1: How certain is the critic?

    @property
    def weighted_score(self) -> float:
        """
        Compute weighted composite score.

        Uses the mathematically validated weights:
        0.35*correctness + 0.25*completeness + 0.25*quality + 0.15*confidence
        """
        return (
            WEIGHT_CORRECTNESS * self.correctness +
            WEIGHT_COMPLETENESS * self.completeness +
            WEIGHT_QUALITY * self.quality +
            WEIGHT_CONFIDENCE * self.confidence
        )


@dataclass
class BranchEvaluation:
    """
    Result of evaluating multiple ToT branches.

    Contains the winner selection plus detailed reasoning for Delia meta-learning.
    """
    winner_index: int  # Index into the branches list
    winner_mode: OrchestrationMode
    scores: list[BranchScore] = field(default_factory=list)
    reasoning: str = ""  # Why this branch won
    insights: str = ""  # Meta-insights for orchestration learning
    raw_response: str = ""  # Raw critic response for debugging

class ResponseCritic:
    """
    Acts as the final verification step in the orchestration loop.
    
    The Critic reviews the work of the Executor and either 'Approves'
    it or provides feedback for correction.
    """

    def __init__(self, call_llm_fn: Any):
        self.call_llm = call_llm_fn

    async def verify(
        self,
        original_prompt: str,
        response_to_verify: str,
        backend_obj: BackendConfig | None = None,
    ) -> tuple[bool, str]:
        """
        Verifies the response against the original prompt.
        
        Returns:
            (is_approved, feedback_or_original_text)
        """
        from ..config import config
        
        # Use 'quick' tier for speed, or a dedicated critic if configured
        critic_model = getattr(config, 'model_critic', config.model_quick).default_model
        
        system_prompt = build_system_prompt(ModelRole.CRITIC)
        
        verification_prompt = f"""### Original Request:
{original_prompt}

### Assistant Response to Review:
{response_to_verify}

---
Does this response accurately and fully address the request? 
Respond with 'APPROVED' or provide feedback."""

        log.info("critic_verification_start", model=critic_model)

        result = await self.call_llm(
            model=critic_model,
            prompt=verification_prompt,
            system=system_prompt,
            task_type="critique",
            backend_obj=backend_obj,
            temperature=0.1, # Highly deterministic
        )

        if not result.get("success"):
            log.warning("critic_failed_skipping_verification")
            return True, response_to_verify # Fail open to avoid blocking user

        critic_response = result.get("response", "").strip()
        is_approved = "APPROVED" in critic_response or "approved" in critic_response.lower()

        if is_approved:
            log.info("critic_approved_response")
            return True, response_to_verify
        else:
            log.warning("critic_rejected_response", feedback_len=len(critic_response))
            return False, critic_response

    async def evaluate_branches(
        self,
        original_prompt: str,
        branches: list[tuple[OrchestrationMode, OrchestrationResult]],
        backend_obj: "BackendConfig | None" = None,
    ) -> BranchEvaluation:
        """
        Compare multiple ToT branches and select the best one.

        Uses structured scoring with reasoning for Delia meta-learning.
        Each branch is scored on:
        - Correctness (0-10): Does it solve the problem?
        - Completeness (0-10): Are all requirements addressed?
        - Quality (0-10): Code quality, reasoning depth, clarity
        - Confidence (0-10): Critic's certainty in the assessment

        Args:
            original_prompt: The original user request
            branches: List of (mode, result) tuples to compare
            backend_obj: Optional backend to use for critic LLM

        Returns:
            BranchEvaluation with winner, scores, and reasoning
        """
        from ..config import config
        from ..text_utils import strip_thinking_tags

        if not branches:
            raise ValueError("No branches to evaluate")

        if len(branches) == 1:
            # Single branch - trivial winner
            mode, result = branches[0]
            return BranchEvaluation(
                winner_index=0,
                winner_mode=mode,
                scores=[BranchScore(
                    mode=mode,
                    correctness=0.7,
                    completeness=0.7,
                    quality=result.quality_score or 0.5,
                    confidence=0.7,
                )],
                reasoning="Single branch - automatic winner",
                insights="",
            )

        # Use MoE tier for complex multi-branch comparison
        critic_model = config.model_moe.default_model

        # Build comparison prompt with all branches
        branches_text = self._format_branches_for_comparison(branches)

        evaluation_prompt = f"""### Original Task:
{original_prompt}

### Branch Results to Compare:
{branches_text}

---
As the Senior QA Critic, evaluate ALL branches and select the BEST response.

Score each branch on these dimensions (0-10 scale):
1. Correctness: Does it correctly and accurately solve the problem?
2. Completeness: Are all requirements and edge cases addressed?
3. Quality: Code quality, reasoning depth, clarity, best practices
4. Confidence: How certain are you in this assessment?

After scoring, select the winner and explain:
- WHY this branch won over the others
- What PROPERTIES of the task made this approach optimal
- INSIGHTS for future orchestration decisions

OUTPUT FORMAT (strict JSON):
{{
  "winner_index": 0,
  "reasoning": "Detailed explanation of why this branch won",
  "insights": "Meta-insight about what task properties favor this orchestration mode",
  "scores": [
    {{"mode": "voting", "correctness": 9, "completeness": 8, "quality": 9, "confidence": 8}},
    {{"mode": "agentic", "correctness": 7, "completeness": 7, "quality": 7, "confidence": 7}},
    {{"mode": "deep_thinking", "correctness": 8, "completeness": 6, "quality": 8, "confidence": 7}}
  ]
}}"""

        log.info("critic_branch_evaluation_start", branch_count=len(branches), model=critic_model)

        result = await self.call_llm(
            model=critic_model,
            prompt=evaluation_prompt,
            system=build_system_prompt(ModelRole.CRITIC),
            task_type="critique",
            backend_obj=backend_obj,
            temperature=0.1,  # Highly deterministic
            enable_thinking=True,  # Allow extended reasoning
        )

        if not result.get("success"):
            log.warning("critic_branch_evaluation_failed", error=result.get("error"))
            return self._fallback_evaluation(branches)

        # Parse structured output
        response_text = strip_thinking_tags(result.get("response", ""))

        try:
            evaluation = self._parse_evaluation_response(response_text, branches)
            log.info(
                "critic_branch_evaluation_complete",
                winner_mode=evaluation.winner_mode.value,
                winner_score=evaluation.scores[evaluation.winner_index].weighted_score
                if evaluation.scores else 0,
            )
            return evaluation
        except Exception as e:
            log.error("critic_parse_failed", error=str(e))
            return self._fallback_evaluation(branches)

    def _format_branches_for_comparison(
        self,
        branches: list[tuple[OrchestrationMode, OrchestrationResult]],
    ) -> str:
        """Format branches into a readable comparison format."""
        parts = []
        for i, (mode, result) in enumerate(branches):
            status = "SUCCESS" if result.success else "FAILED"
            # Truncate response to avoid token explosion
            response_preview = result.response[:2000]
            if len(result.response) > 2000:
                response_preview += "\n[... truncated ...]"

            parts.append(f"""
## Branch {i}: {mode.value.upper()} [{status}]
Model: {result.model_used or 'unknown'}
Quality Score: {result.quality_score or 'N/A'}

Response:
```
{response_preview}
```
""")
        return "\n---\n".join(parts)

    def _parse_evaluation_response(
        self,
        response_text: str,
        branches: list[tuple[OrchestrationMode, OrchestrationResult]],
    ) -> BranchEvaluation:
        """Parse the critic's JSON evaluation response."""
        # Extract JSON from response (may be wrapped in markdown)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            raise ValueError("No JSON found in critic response")

        data = json.loads(json_match.group())

        # Parse scores
        scores: list[BranchScore] = []
        for s in data.get("scores", []):
            try:
                mode = OrchestrationMode(s["mode"])
            except ValueError:
                # Mode name might not match exactly, try to find it
                mode = branches[len(scores)][0] if len(scores) < len(branches) else OrchestrationMode.NONE

            scores.append(BranchScore(
                mode=mode,
                correctness=float(s.get("correctness", 5)) / 10,
                completeness=float(s.get("completeness", 5)) / 10,
                quality=float(s.get("quality", 5)) / 10,
                confidence=float(s.get("confidence", 5)) / 10,
            ))

        winner_index = int(data.get("winner_index", 0))
        # Validate winner_index
        if winner_index < 0 or winner_index >= len(branches):
            # Find highest scoring branch
            if scores:
                winner_index = max(range(len(scores)), key=lambda i: scores[i].weighted_score)
            else:
                winner_index = 0

        return BranchEvaluation(
            winner_index=winner_index,
            winner_mode=branches[winner_index][0],
            scores=scores,
            reasoning=data.get("reasoning", ""),
            insights=data.get("insights", ""),
            raw_response=response_text,
        )

    def _fallback_evaluation(
        self,
        branches: list[tuple[OrchestrationMode, OrchestrationResult]],
    ) -> BranchEvaluation:
        """
        Fallback evaluation when critic LLM fails.

        Uses quality_score and success status to pick winner.
        """
        # Filter to successful branches
        successful = [(i, mode, res) for i, (mode, res) in enumerate(branches) if res.success]

        if not successful:
            # All failed - pick first
            return BranchEvaluation(
                winner_index=0,
                winner_mode=branches[0][0],
                scores=[],
                reasoning="Fallback: All branches failed, selecting first",
                insights="",
            )

        # Pick by quality_score
        best_idx, best_mode, best_res = max(
            successful,
            key=lambda x: x[2].quality_score or 0.5
        )

        return BranchEvaluation(
            winner_index=best_idx,
            winner_mode=best_mode,
            scores=[],
            reasoning=f"Fallback: Selected by quality_score ({best_res.quality_score})",
            insights=""
        )
