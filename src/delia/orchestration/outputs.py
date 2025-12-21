# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Structured Output Types for Orchestration.

Provides Pydantic models for type-safe LLM responses.
When an output_type is specified, Delia will:
1. Request JSON output from the model
2. Parse the response into the specified Pydantic model
3. Validate the structure and types
4. Return a typed result

Usage:
    from delia.orchestration.outputs import CodeReview
    
    result = await service.process(
        message="Review this code: ...",
        output_type=CodeReview,
    )
    # result.structured is a CodeReview instance
    print(result.structured.issues)
    print(result.structured.score)

Inspired by OpenAI Agents SDK structured outputs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Severity and Priority Enums
# =============================================================================

class Severity(str, Enum):
    """Severity level for issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Priority(str, Enum):
    """Priority level for suggestions."""
    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    WONT = "wont"


# =============================================================================
# Code Review Output
# =============================================================================

class CodeIssue(BaseModel):
    """A single issue found in code review."""
    
    line: int | None = Field(
        default=None,
        description="Line number where the issue occurs (if applicable)",
    )
    severity: Severity = Field(
        default=Severity.MEDIUM,
        description="Severity of the issue",
    )
    category: str = Field(
        description="Category of issue: bug, security, performance, style, etc.",
    )
    description: str = Field(
        description="Clear description of the issue",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested fix for the issue",
    )


class CodeReview(BaseModel):
    """Structured output for code review tasks."""
    
    summary: str = Field(
        description="Brief overall summary of the code review",
    )
    issues: list[CodeIssue] = Field(
        default_factory=list,
        description="List of issues found",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="General improvement suggestions",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score from 0.0 to 1.0",
    )
    approved: bool = Field(
        default=False,
        description="Whether the code is approved for merge",
    )


# =============================================================================
# Analysis Output
# =============================================================================

class Finding(BaseModel):
    """A single finding from analysis."""
    
    title: str = Field(
        description="Short title for the finding",
    )
    description: str = Field(
        description="Detailed description of the finding",
    )
    evidence: str | None = Field(
        default=None,
        description="Supporting evidence or examples",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level in this finding",
    )


class Analysis(BaseModel):
    """Structured output for analysis tasks."""
    
    summary: str = Field(
        description="Executive summary of the analysis",
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="Key findings from the analysis",
    )
    conclusion: str = Field(
        description="Overall conclusion",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the analysis",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommended actions based on analysis",
    )


# =============================================================================
# Plan Output
# =============================================================================

class PlanStep(BaseModel):
    """A single step in a plan."""
    
    step_number: int = Field(
        description="Step number in sequence",
    )
    action: str = Field(
        description="What action to take",
    )
    details: str | None = Field(
        default=None,
        description="Additional details or context",
    )
    dependencies: list[int] = Field(
        default_factory=list,
        description="Step numbers this step depends on",
    )
    estimated_effort: str | None = Field(
        default=None,
        description="Estimated effort (e.g., '2 hours', '1 day')",
    )


class Plan(BaseModel):
    """Structured output for planning tasks."""
    
    goal: str = Field(
        description="The goal this plan achieves",
    )
    approach: str = Field(
        description="High-level approach description",
    )
    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Ordered list of steps to execute",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Potential risks and mitigation strategies",
    )
    success_criteria: list[str] = Field(
        default_factory=list,
        description="How to measure success",
    )


# =============================================================================
# Comparison Output
# =============================================================================

class ComparisonItem(BaseModel):
    """A single item being compared."""
    
    name: str = Field(
        description="Name/identifier of the item",
    )
    pros: list[str] = Field(
        default_factory=list,
        description="Advantages of this option",
    )
    cons: list[str] = Field(
        default_factory=list,
        description="Disadvantages of this option",
    )
    score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative score for this option",
    )


class Comparison(BaseModel):
    """Structured output for comparison tasks."""
    
    question: str = Field(
        description="The comparison question being answered",
    )
    items: list[ComparisonItem] = Field(
        default_factory=list,
        description="Items being compared",
    )
    recommendation: str = Field(
        description="Recommended choice and reasoning",
    )
    winner: str | None = Field(
        default=None,
        description="Name of the winning option (if clear winner)",
    )


# =============================================================================
# Question Answer Output
# =============================================================================

class QuestionAnswer(BaseModel):
    """Structured output for Q&A tasks."""
    
    question: str = Field(
        description="The question being answered",
    )
    answer: str = Field(
        description="The direct answer",
    )
    explanation: str | None = Field(
        default=None,
        description="Detailed explanation",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Sources or references supporting the answer",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Important caveats or limitations",
    )


# =============================================================================
# Generic Structured Output
# =============================================================================

class StructuredResponse(BaseModel):
    """Generic structured response for any task."""
    
    content: str = Field(
        description="Main response content",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the response",
    )


# =============================================================================
# Output Type Registry
# =============================================================================

# Map task types to default output types
DEFAULT_OUTPUT_TYPES: dict[str, type[BaseModel]] = {
    "review": CodeReview,
    "analyze": Analysis,
    "plan": Plan,
    "compare": Comparison,
    "question": QuestionAnswer,
}


def get_default_output_type(task_type: str) -> type[BaseModel] | None:
    """Get the default structured output type for a task."""
    return DEFAULT_OUTPUT_TYPES.get(task_type)


def get_json_schema_prompt(output_type: type[BaseModel]) -> str:
    """
    Generate a prompt instruction for JSON output.
    
    Includes the JSON schema so the model knows the expected structure.
    """
    schema = output_type.model_json_schema()
    
    return f"""Respond with a valid JSON object matching this schema:

```json
{schema}
```

Important:
- Output ONLY valid JSON, no other text
- Follow the schema exactly
- All required fields must be present
- Use appropriate types (strings, numbers, booleans, arrays)
"""


def _clean_json_artifacts(json_str: str) -> str:
    """
    Remove markdown formatting artifacts from JSON strings.

    Some models (especially smaller ones) add markdown formatting
    like **bold** or *italic* inside JSON, breaking parsing.
    """
    import re

    # Remove bold markers: **text** -> text
    json_str = re.sub(r'\*\*([^*]+)\*\*', r'\1', json_str)
    # Remove italic markers: *text* -> text (but not inside strings)
    json_str = re.sub(r'(?<![\\"])\*([^*]+)\*(?![\\"])', r'\1', json_str)
    # Remove underline bold: __text__ -> text
    json_str = re.sub(r'__([^_]+)__', r'\1', json_str)
    # Remove strikethrough: ~~text~~ -> text
    json_str = re.sub(r'~~([^~]+)~~', r'\1', json_str)
    # Remove inline code in keys: `key` -> key (only outside quotes)
    json_str = re.sub(r'`([^`]+)`', r'\1', json_str)
    # Fix common issue: trailing commas before closing bracket
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    return json_str


def parse_structured_output(
    response: str,
    output_type: type[T],
) -> T:
    """
    Parse an LLM response into a structured Pydantic model.

    Args:
        response: The raw LLM response (should be JSON)
        output_type: The Pydantic model class to parse into

    Returns:
        An instance of the output_type

    Raises:
        ValueError: If the response cannot be parsed
    """
    import json
    import re

    # Try to extract JSON from the response
    # Models sometimes wrap JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response.strip()

    # Clean markdown artifacts that break JSON parsing
    json_str = _clean_json_artifacts(json_str)

    try:
        data = json.loads(json_str)
        return output_type.model_validate(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response[:500]}")
    except Exception as e:
        raise ValueError(f"Failed to validate structure: {e}\nResponse: {response[:500]}")

