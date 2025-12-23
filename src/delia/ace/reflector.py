# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
ACE Reflector: Task Outcome Analysis.

Analyzes task execution outcomes to extract structured insights.
Uses existing LLM infrastructure via delegation module.

Based on ACE Framework research - separates evaluation/insight extraction
from curation (handled by Curator).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger()


class InsightType(str, Enum):
    """Type of insight extracted from task execution."""
    STRATEGY = "strategy"        # What approach worked
    ANTI_PATTERN = "anti_pattern"  # What to avoid
    FAILURE_MODE = "failure_mode"  # Common failure pattern
    TOOL_USAGE = "tool_usage"     # How to use specific tools
    CONTEXT_HINT = "context_hint"  # What context was needed


@dataclass
class ExtractedInsight:
    """A single insight extracted by the Reflector."""
    content: str  # The actual strategic bullet content
    insight_type: InsightType
    task_type: str  # coding, testing, debugging, etc.
    confidence: float  # 0.0-1.0 how confident in this insight
    source_context: str | None = None  # What led to this insight
    related_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class ReflectionResult:
    """Complete result of reflecting on a task execution."""
    task_succeeded: bool
    task_type: str
    insights: list[ExtractedInsight]
    errors_identified: list[str]
    root_causes: list[str]
    correct_approaches: list[str]
    bullets_to_tag_helpful: list[str]  # Bullet IDs that helped
    bullets_to_tag_harmful: list[str]  # Bullet IDs that hurt
    raw_reasoning: str


# Reflector prompt template (adapted from ACE paper Figure 10)
REFLECTOR_PROMPT = """You are an expert diagnostician analyzing a completed AI task.

## Task Context
- **Task Type**: {task_type}
- **Task Description**: {task_description}
- **Outcome**: {"SUCCESS" if task_succeeded else "FAILURE"}

## Execution Details
{execution_details}

## Applied Playbook Bullets
{applied_bullets}

## Your Analysis

Analyze this execution and provide structured insights in JSON format:

```json
{{
  "task_succeeded": true/false,
  "errors_identified": ["specific error 1", "specific error 2"],
  "root_causes": ["why error 1 happened", "why error 2 happened"],
  "correct_approaches": ["what should have been done instead"],
  "insights": [
    {{
      "content": "Reusable strategy or pattern (imperative form)",
      "type": "strategy|anti_pattern|failure_mode|tool_usage|context_hint",
      "confidence": 0.0-1.0,
      "tags": ["relevant", "tags"]
    }}
  ],
  "bullets_helpful": ["bullet_id_1", "bullet_id_2"],
  "bullets_harmful": ["bullet_id_3"],
  "reasoning": "Brief explanation of your analysis"
}}
```

Focus on:
1. **Extracting reusable patterns** - What worked that could help future tasks?
2. **Identifying anti-patterns** - What should be avoided?
3. **Tagging bullet effectiveness** - Which applied bullets helped or hurt?

Output ONLY valid JSON, no additional text."""


class Reflector:
    """
    ACE Reflector: Distills concrete insights from task executions.

    The Reflector analyzes:
    1. Task outcome (success/failure)
    2. Tool calls and their results
    3. Applied playbook bullets
    4. User feedback signals

    Outputs structured insights for the Curator.

    Uses existing LLM infrastructure - no new backends needed.
    """

    def __init__(self, model_tier: str = "quick"):
        """
        Initialize Reflector.

        Args:
            model_tier: LLM tier for reflection calls (default: "quick" for speed)
        """
        self.model_tier = model_tier

    async def reflect(
        self,
        task_description: str,
        task_type: str,
        task_succeeded: bool,
        outcome: str | None = None,
        tool_calls: list[dict] | None = None,
        applied_bullets: list[dict] | None = None,
        error_trace: str | None = None,
        user_feedback: str | None = None,
    ) -> ReflectionResult:
        """
        Reflect on a completed task to extract learnings.

        Args:
            task_description: What the user asked for
            task_type: Detected task type (coding, testing, etc.)
            task_succeeded: Whether task completed successfully
            outcome: The final response given (optional, can be long)
            tool_calls: List of tools called during execution
            applied_bullets: Playbook bullets that were applied [{id, content}, ...]
            error_trace: Any errors encountered
            user_feedback: Explicit user feedback if available

        Returns:
            ReflectionResult with extracted insights
        """
        # Build execution details summary
        execution_details = self._build_execution_details(
            outcome=outcome,
            tool_calls=tool_calls,
            error_trace=error_trace,
            user_feedback=user_feedback,
        )

        # Format applied bullets
        bullets_text = "None applied"
        if applied_bullets:
            bullets_text = "\n".join(
                f"- [{b.get('id', 'unknown')}] {b.get('content', '')}"
                for b in applied_bullets
            )

        # Build prompt
        prompt = REFLECTOR_PROMPT.format(
            task_type=task_type,
            task_description=task_description[:500],  # Truncate if too long
            task_succeeded=task_succeeded,
            execution_details=execution_details,
            applied_bullets=bullets_text,
        )

        # Call LLM using existing infrastructure
        try:
            response = await self._call_llm(prompt)
            return self._parse_response(response, task_type, task_succeeded)
        except Exception as e:
            log.warning("reflection_failed", error=str(e))
            # Return minimal result on failure
            return ReflectionResult(
                task_succeeded=task_succeeded,
                task_type=task_type,
                insights=[],
                errors_identified=[str(e)] if not task_succeeded else [],
                root_causes=[],
                correct_approaches=[],
                bullets_to_tag_helpful=[],
                bullets_to_tag_harmful=[],
                raw_reasoning=f"Reflection failed: {e}",
            )

    def _build_execution_details(
        self,
        outcome: str | None,
        tool_calls: list[dict] | None,
        error_trace: str | None,
        user_feedback: str | None,
    ) -> str:
        """Build a summary of execution details."""
        parts = []

        if outcome:
            # Truncate outcome to avoid context overflow
            truncated = outcome[:1000] + "..." if len(outcome) > 1000 else outcome
            parts.append(f"### Outcome\n{truncated}")

        if tool_calls:
            tool_summary = []
            for tc in tool_calls[:10]:  # Limit to 10 tools
                name = tc.get("name", "unknown")
                success = tc.get("success", True)
                tool_summary.append(f"- {name}: {'OK' if success else 'FAILED'}")
            parts.append(f"### Tool Calls\n" + "\n".join(tool_summary))

        if error_trace:
            parts.append(f"### Error Trace\n{error_trace[:500]}")

        if user_feedback:
            parts.append(f"### User Feedback\n{user_feedback}")

        return "\n\n".join(parts) if parts else "No additional details available."

    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM using existing Delia infrastructure.

        Reuses the llm module - no new LLM connections.
        """
        from delia import llm
        from delia.backend_manager import backend_manager
        from delia.routing import select_model

        # Get active backend
        backend = backend_manager.get_active_backend()
        if not backend:
            raise RuntimeError("No active backend available")

        # Select appropriate model for quick analysis
        model = await select_model(
            task_type=self.model_tier,
            content_length=len(prompt),
            model_override=None,
            content=prompt,
        )

        # Make the call
        response = await llm.call_llm(
            model=model,
            content=prompt,
            system="You are a task analysis expert. Output only valid JSON.",
            stream=False,
            task_type="analyze",
            original_task="reflection",
            language="json",
            backend=backend,
        )

        return response

    def _parse_response(
        self,
        response: str,
        task_type: str,
        task_succeeded: bool,
    ) -> ReflectionResult:
        """Parse LLM response into ReflectionResult."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            # Parse insights
            insights = []
            for insight_data in data.get("insights", []):
                try:
                    insight_type = InsightType(insight_data.get("type", "strategy"))
                except ValueError:
                    insight_type = InsightType.STRATEGY

                insights.append(ExtractedInsight(
                    content=insight_data.get("content", ""),
                    insight_type=insight_type,
                    task_type=task_type,
                    confidence=float(insight_data.get("confidence", 0.5)),
                    tags=insight_data.get("tags", []),
                ))

            return ReflectionResult(
                task_succeeded=data.get("task_succeeded", task_succeeded),
                task_type=task_type,
                insights=insights,
                errors_identified=data.get("errors_identified", []),
                root_causes=data.get("root_causes", []),
                correct_approaches=data.get("correct_approaches", []),
                bullets_to_tag_helpful=data.get("bullets_helpful", []),
                bullets_to_tag_harmful=data.get("bullets_harmful", []),
                raw_reasoning=data.get("reasoning", ""),
            )

        except json.JSONDecodeError as e:
            log.warning("reflection_parse_failed", error=str(e))
            return ReflectionResult(
                task_succeeded=task_succeeded,
                task_type=task_type,
                insights=[],
                errors_identified=[],
                root_causes=[],
                correct_approaches=[],
                bullets_to_tag_helpful=[],
                bullets_to_tag_harmful=[],
                raw_reasoning=response,  # Store raw response for debugging
            )

    async def quick_reflect(
        self,
        task_type: str,
        bullet_ids: list[str],
        was_helpful: bool,
        brief_reason: str | None = None,
    ) -> list[tuple[str, bool]]:
        """
        Quick reflection without LLM - just records feedback.

        Useful for high-throughput scenarios where full analysis isn't needed.

        Args:
            task_type: The task type (for logging)
            bullet_ids: List of bullet IDs that were applied
            was_helpful: Whether the overall outcome was helpful
            brief_reason: Optional brief reason

        Returns:
            List of (bullet_id, helpful) tuples for batch feedback
        """
        log.debug(
            "quick_reflection",
            task_type=task_type,
            bullet_count=len(bullet_ids),
            helpful=was_helpful,
            reason=brief_reason,
        )

        # All bullets get the same feedback based on overall outcome
        return [(bullet_id, was_helpful) for bullet_id in bullet_ids]

    async def reflect_on_failure(
        self,
        task_description: str,
        task_type: str,
        error: str,
        attempted_approaches: list[str] | None = None,
    ) -> ReflectionResult:
        """
        Specialized reflection for failed tasks.

        Focuses on extracting failure modes and anti-patterns.
        """
        return await self.reflect(
            task_description=task_description,
            task_type=task_type,
            task_succeeded=False,
            error_trace=error,
            outcome=f"FAILED. Attempted: {attempted_approaches}" if attempted_approaches else None,
        )


# Singleton instance
_reflector: Reflector | None = None


def get_reflector() -> Reflector:
    """Get or create the global reflector instance."""
    global _reflector
    if _reflector is None:
        _reflector = Reflector()
    return _reflector
