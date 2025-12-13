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
Structured response schemas for LLM-to-LLM communication.

These schemas provide typed, predictable responses that AI assistants
can reliably parse and act upon.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .enums import Language, ModelTier, Severity


class UsageMetrics(BaseModel):
    """Token and timing metrics for a request."""

    input_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens in prompt",
    )
    output_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens in response",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens used",
    )
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Total request time in milliseconds",
    )
    queue_wait_ms: int | None = Field(
        default=None,
        description="Time waiting in queue (if queued)",
    )


class ExecutionInfo(BaseModel):
    """Information about the execution context."""

    model: str = Field(
        default="",
        description="Model that processed the request",
    )
    model_tier: ModelTier = Field(
        default=ModelTier.QUICK,
        description="Model tier used",
    )
    backend_id: str = Field(
        default="",
        description="Backend ID that handled request",
    )
    backend_type: str = Field(
        default="local",
        description="Backend type (local or remote)",
    )
    provider: str = Field(
        default="",
        description="Provider name (ollama, llamacpp, gemini)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Request timestamp (UTC)",
    )


class StructuredResponse(BaseModel):
    """Base response schema for all structured tools."""

    success: bool = Field(
        ...,
        description="Whether the request succeeded",
    )
    content: str = Field(
        default="",
        description="The main response content (raw LLM output)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if request failed",
    )
    usage: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Token and timing metrics",
    )
    execution: ExecutionInfo = Field(
        default_factory=ExecutionInfo,
        description="Execution context information",
    )
    request_id: str = Field(
        default="",
        description="Unique request identifier",
    )
    warnings: list[str] | None = Field(
        default=None,
        description="Non-fatal warnings about the response",
    )


class CodeFinding(BaseModel):
    """A single finding from code review."""

    severity: Severity = Field(
        ...,
        description="Issue severity level",
    )
    category: str = Field(
        ...,
        description="Category (e.g., 'security', 'performance', 'style')",
    )
    message: str = Field(
        ...,
        description="Description of the issue",
    )
    line_start: int | None = Field(
        default=None,
        ge=1,
        description="Starting line number",
    )
    line_end: int | None = Field(
        default=None,
        ge=1,
        description="Ending line number",
    )
    column: int | None = Field(
        default=None,
        ge=1,
        description="Column number",
    )
    code_snippet: str | None = Field(
        default=None,
        description="Relevant code snippet",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested fix",
    )


class CodeReviewResponse(StructuredResponse):
    """Structured output for code review tasks."""

    findings: list[CodeFinding] = Field(
        default_factory=list,
        description="List of code findings (may be empty if none found)",
    )
    summary: str = Field(
        default="",
        description="Brief summary of review",
    )
    metrics: dict[str, int] | None = Field(
        default=None,
        description="Counts by severity: {'error': 2, 'warning': 5}",
    )
    reviewed_lines: int | None = Field(
        default=None,
        description="Lines of code reviewed",
    )


class GeneratedCode(BaseModel):
    """A unit of generated code."""

    code: str = Field(
        ...,
        description="The generated code",
    )
    language: Language = Field(
        ...,
        description="Programming language",
    )
    file_path: str | None = Field(
        default=None,
        description="Suggested file path",
    )
    description: str | None = Field(
        default=None,
        description="Description of what this code does",
    )


class CodeGenerateResponse(StructuredResponse):
    """Structured output for code generation tasks."""

    generated: list[GeneratedCode] = Field(
        default_factory=list,
        description="List of generated code blocks",
    )
    imports_needed: list[str] | None = Field(
        default=None,
        description="Required imports",
    )
    dependencies: list[str] | None = Field(
        default=None,
        description="Package dependencies",
    )
    tests: list[GeneratedCode] | None = Field(
        default=None,
        description="Generated tests (if requested)",
    )


class AnalysisSection(BaseModel):
    """A section of analysis output."""

    title: str = Field(
        ...,
        description="Section title",
    )
    content: str = Field(
        ...,
        description="Section content",
    )
    importance: str = Field(
        default="medium",
        description="Importance level (low/medium/high)",
    )


class AnalyzeResponse(StructuredResponse):
    """Structured output for analysis tasks."""

    sections: list[AnalysisSection] = Field(
        default_factory=list,
        description="Analysis sections",
    )
    metrics: dict[str, float | int | str] | None = Field(
        default=None,
        description="Quantitative metrics",
    )
    recommendations: list[str] | None = Field(
        default=None,
        description="Recommendations based on analysis",
    )


class ReasoningStep(BaseModel):
    """A single step in extended reasoning."""

    step_number: int = Field(
        ...,
        ge=1,
        description="Step number in sequence",
    )
    thought: str = Field(
        ...,
        description="The reasoning thought",
    )
    conclusion: str | None = Field(
        default=None,
        description="Conclusion from this step",
    )


class ThinkResponse(StructuredResponse):
    """Structured output for extended reasoning tasks."""

    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="Steps in the reasoning process",
    )
    final_answer: str = Field(
        default="",
        description="Final answer/conclusion",
    )
    alternatives_considered: list[str] | None = Field(
        default=None,
        description="Alternative approaches considered",
    )
    confidence_explanation: str | None = Field(
        default=None,
        description="Explanation of confidence level",
    )


class SummarizeResponse(StructuredResponse):
    """Structured output for summarization tasks."""

    summary: str = Field(
        default="",
        description="The summary",
    )
    key_points: list[str] | None = Field(
        default=None,
        description="Key points extracted",
    )
    word_count: int | None = Field(
        default=None,
        description="Word count of summary",
    )


class CritiqueResponse(StructuredResponse):
    """Structured output for critique tasks."""

    strengths: list[str] = Field(
        default_factory=list,
        description="Identified strengths",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Identified weaknesses",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Improvement suggestions",
    )
    alternatives: list[str] | None = Field(
        default=None,
        description="Alternative approaches",
    )
    overall_assessment: str = Field(
        default="",
        description="Overall assessment summary",
    )


class PlanStep(BaseModel):
    """A step in a plan."""

    step_number: int = Field(
        ...,
        ge=1,
        description="Step number",
    )
    title: str = Field(
        ...,
        description="Step title",
    )
    description: str = Field(
        ...,
        description="Step description",
    )
    dependencies: list[int] | None = Field(
        default=None,
        description="Dependent step numbers",
    )


class PlanResponse(StructuredResponse):
    """Structured output for planning tasks."""

    steps: list[PlanStep] = Field(
        default_factory=list,
        description="Plan steps",
    )
    risks: list[str] | None = Field(
        default=None,
        description="Identified risks",
    )
    assumptions: list[str] | None = Field(
        default=None,
        description="Assumptions made",
    )
    overview: str = Field(
        default="",
        description="Plan overview",
    )


class BatchResponseItem(BaseModel):
    """A single response in a batch."""

    index: int = Field(
        ...,
        ge=0,
        description="Index in original request array",
    )
    response: StructuredResponse = Field(
        ...,
        description="The response for this item",
    )


class BatchResponse(BaseModel):
    """Structured output for batch processing."""

    success: bool = Field(
        ...,
        description="Whether all requests succeeded",
    )
    results: list[BatchResponseItem] = Field(
        default_factory=list,
        description="Individual results",
    )
    total_usage: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Aggregated usage metrics",
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of failed requests",
    )
    request_id: str = Field(
        default="",
        description="Batch request identifier",
    )
