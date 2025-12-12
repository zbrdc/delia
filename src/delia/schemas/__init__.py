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
    TaskType,
    ModelTier,
    ContentType,
    Language,
    BackendPreference,
    Severity,
    AnalysisType,
    ReasoningDepth,
)
from .requests import (
    StructuredRequest,
    CodeReviewRequest,
    CodeGenerateRequest,
    AnalyzeRequest,
    ThinkRequest,
    SummarizeRequest,
    CritiqueRequest,
    PlanRequest,
    BatchRequest,
)
from .responses import (
    UsageMetrics,
    ExecutionInfo,
    StructuredResponse,
    CodeFinding,
    CodeReviewResponse,
    GeneratedCode,
    CodeGenerateResponse,
    AnalysisSection,
    AnalyzeResponse,
    ReasoningStep,
    ThinkResponse,
    SummarizeResponse,
    CritiqueResponse,
    PlanStep,
    PlanResponse,
    BatchResponseItem,
    BatchResponse,
)

__all__ = [
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
