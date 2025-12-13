# Copyright (C) 2023 the project owner
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
Structured request schemas for LLM-to-LLM communication.

These schemas replace natural language inference with explicit typed fields,
optimized for consumption by AI assistants like Claude.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .enums import (
    AnalysisType,
    BackendPreference,
    ContentType,
    Language,
    ModelTier,
    ReasoningDepth,
    Severity,
    TaskType,
)


class StructuredRequest(BaseModel):
    """Base request schema for all structured tools."""

    content: str = Field(
        ...,
        description="The content to process (code, text, or mixed)",
    )
    content_type: ContentType = Field(
        default=ContentType.MIXED,
        description="Type of content being processed",
    )
    language: Language | None = Field(
        default=None,
        description="Programming language (explicit, not inferred)",
    )
    model_tier: ModelTier | None = Field(
        default=None,
        description="Force specific model tier (quick/coder/moe/thinking)",
    )
    backend: BackendPreference = Field(
        default=BackendPreference.AUTO,
        description="Backend routing preference",
    )
    file_path: str | None = Field(
        default=None,
        description="Path to file being processed (for context)",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum output tokens (optional limit)",
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1000,
        description="Request timeout in milliseconds",
    )


class CodeReviewRequest(StructuredRequest):
    """Structured input for code review tasks."""

    task_type: Literal[TaskType.REVIEW] = Field(
        default=TaskType.REVIEW,
        description="Task type (always 'review' for this schema)",
    )
    focus_areas: list[str] | None = Field(
        default=None,
        description="Specific areas to focus on (e.g., ['security', 'performance', 'style'])",
    )
    symbols: list[str] | None = Field(
        default=None,
        description="Specific symbols/functions to review",
    )
    line_range: tuple[int, int] | None = Field(
        default=None,
        description="Line range to focus on as (start, end)",
    )
    severity_threshold: Severity | None = Field(
        default=None,
        description="Minimum severity to report",
    )
    include_suggestions: bool = Field(
        default=True,
        description="Whether to include fix suggestions",
    )


class CodeGenerateRequest(StructuredRequest):
    """Structured input for code generation tasks."""

    task_type: Literal[TaskType.GENERATE] = Field(
        default=TaskType.GENERATE,
        description="Task type (always 'generate' for this schema)",
    )
    output_format: str | None = Field(
        default=None,
        description="Expected output format (e.g., 'function', 'class', 'module')",
    )
    requirements: list[str] | None = Field(
        default=None,
        description="List of requirements the generated code must satisfy",
    )
    style_guide: str | None = Field(
        default=None,
        description="Coding style to follow (e.g., 'PEP8', 'Google')",
    )
    include_tests: bool = Field(
        default=False,
        description="Whether to include unit tests",
    )
    include_docstrings: bool = Field(
        default=True,
        description="Whether to include documentation",
    )
    target_framework: str | None = Field(
        default=None,
        description="Target framework (e.g., 'fastapi', 'react', 'django')",
    )


class AnalyzeRequest(StructuredRequest):
    """Structured input for analysis tasks."""

    task_type: Literal[TaskType.ANALYZE] = Field(
        default=TaskType.ANALYZE,
        description="Task type (always 'analyze' for this schema)",
    )
    analysis_type: AnalysisType = Field(
        default=AnalysisType.GENERAL,
        description="Type of analysis to perform",
    )
    depth: ReasoningDepth = Field(
        default=ReasoningDepth.NORMAL,
        description="Analysis depth",
    )
    include_metrics: bool = Field(
        default=False,
        description="Include quantitative metrics where applicable",
    )
    symbols: list[str] | None = Field(
        default=None,
        description="Specific symbols to analyze",
    )


class ThinkRequest(BaseModel):
    """Structured input for extended reasoning tasks."""

    problem: str = Field(
        ...,
        description="The problem to reason about",
    )
    context: str | None = Field(
        default=None,
        description="Supporting context information",
    )
    constraints: list[str] | None = Field(
        default=None,
        description="Constraints to consider in reasoning",
    )
    depth: ReasoningDepth = Field(
        default=ReasoningDepth.NORMAL,
        description="Reasoning depth (quick/normal/deep)",
    )
    model_tier: ModelTier | None = Field(
        default=None,
        description="Force specific model tier",
    )
    backend: BackendPreference = Field(
        default=BackendPreference.AUTO,
        description="Backend routing preference",
    )


class SummarizeRequest(StructuredRequest):
    """Structured input for summarization tasks."""

    task_type: Literal[TaskType.SUMMARIZE] = Field(
        default=TaskType.SUMMARIZE,
        description="Task type (always 'summarize' for this schema)",
    )
    max_length: int | None = Field(
        default=None,
        description="Maximum summary length in words",
    )
    format: Literal["paragraph", "bullets", "structured"] = Field(
        default="paragraph",
        description="Output format for summary",
    )
    focus: str | None = Field(
        default=None,
        description="Specific aspect to focus summary on",
    )


class CritiqueRequest(StructuredRequest):
    """Structured input for critique tasks."""

    task_type: Literal[TaskType.CRITIQUE] = Field(
        default=TaskType.CRITIQUE,
        description="Task type (always 'critique' for this schema)",
    )
    critique_aspects: list[str] | None = Field(
        default=None,
        description="Specific aspects to critique (e.g., ['design', 'scalability', 'maintainability'])",
    )
    include_alternatives: bool = Field(
        default=True,
        description="Whether to suggest alternative approaches",
    )
    severity_level: Literal["gentle", "balanced", "strict"] = Field(
        default="balanced",
        description="How critical the critique should be",
    )


class PlanRequest(StructuredRequest):
    """Structured input for planning tasks."""

    task_type: Literal[TaskType.PLAN] = Field(
        default=TaskType.PLAN,
        description="Task type (always 'plan' for this schema)",
    )
    scope: str | None = Field(
        default=None,
        description="Scope of the plan (e.g., 'feature', 'refactor', 'migration')",
    )
    constraints: list[str] | None = Field(
        default=None,
        description="Constraints to consider in planning",
    )
    include_estimates: bool = Field(
        default=False,
        description="Whether to include effort estimates",
    )
    include_risks: bool = Field(
        default=True,
        description="Whether to include risk assessment",
    )


class BatchRequest(BaseModel):
    """Structured input for batch processing."""

    requests: list[StructuredRequest] = Field(
        ...,
        min_length=1,
        description="List of requests to process in parallel",
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop on first failure",
    )
    max_parallel: int | None = Field(
        default=None,
        ge=1,
        description="Maximum parallel requests",
    )
