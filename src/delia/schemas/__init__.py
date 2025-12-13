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
Structured JSON schemas for LLM-to-LLM communication.

This module provides typed request/response schemas optimized for
AI assistants to communicate with Delia's MCP tools.

Usage:
    from delia.schemas import (
        CodeReviewRequest, CodeReviewResponse,
        ModelTier, Language, ContentType,
    )
"""

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
from .requests import (
    AnalyzeRequest,
    BatchRequest,
    CodeGenerateRequest,
    CodeReviewRequest,
    CritiqueRequest,
    PlanRequest,
    StructuredRequest,
    SummarizeRequest,
    ThinkRequest,
)
from .responses import (
    AnalysisSection,
    AnalyzeResponse,
    BatchResponse,
    BatchResponseItem,
    CodeFinding,
    CodeGenerateResponse,
    CodeReviewResponse,
    CritiqueResponse,
    ExecutionInfo,
    GeneratedCode,
    PlanResponse,
    PlanStep,
    ReasoningStep,
    StructuredResponse,
    SummarizeResponse,
    ThinkResponse,
    UsageMetrics,
)

__all__ = [  # noqa: RUF022 - organized by category, not alphabetically
    # Enums
    "TaskType",
    "ModelTier",
    "ContentType",
    "Language",
    "BackendPreference",
    "Severity",
    "AnalysisType",
    "ReasoningDepth",
    # Request schemas
    "StructuredRequest",
    "CodeReviewRequest",
    "CodeGenerateRequest",
    "AnalyzeRequest",
    "ThinkRequest",
    "SummarizeRequest",
    "CritiqueRequest",
    "PlanRequest",
    "BatchRequest",
    # Response schemas
    "UsageMetrics",
    "ExecutionInfo",
    "StructuredResponse",
    "CodeFinding",
    "CodeReviewResponse",
    "GeneratedCode",
    "CodeGenerateResponse",
    "AnalysisSection",
    "AnalyzeResponse",
    "ReasoningStep",
    "ThinkResponse",
    "SummarizeResponse",
    "CritiqueResponse",
    "PlanStep",
    "PlanResponse",
    "BatchResponseItem",
    "BatchResponse",
]
