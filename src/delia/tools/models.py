# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Pydantic models for MCP tool responses.

These models provide schema validation and documentation for tool outputs.
Use with FastMCP's output_schema parameter for type-safe responses.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# SCAN CODEBASE MODELS
# =============================================================================

class DirectoryInfo(BaseModel):
    """Information about a directory in the project."""
    name: str
    file_count: int = Field(ge=0)


class FilePreview(BaseModel):
    """A file with size and optional preview content."""
    path: str
    size: int = Field(ge=0)
    lines: int = Field(ge=0)
    preview: str = ""


class ScanStructure(BaseModel):
    """Project directory structure."""
    dirs: list[DirectoryInfo] = Field(default_factory=list)
    root_files: list[str] = Field(default_factory=list)


class ScanOverviewResponse(BaseModel):
    """Response from scan_codebase phase='overview'."""
    project: str
    path: str
    phase: Literal["overview"] = "overview"
    structure: ScanStructure
    top_extensions: dict[str, int] = Field(default_factory=dict)
    total_files: int = Field(ge=0)
    has_package_json: bool = False
    has_pyproject: bool = False
    has_src: bool = False
    has_tests: bool = False
    NEXT_PHASES: list[str] = Field(default_factory=list)


class ScanManifestsResponse(BaseModel):
    """Response from scan_codebase phase='manifests'."""
    project: str
    path: str
    phase: Literal["manifests"] = "manifests"
    manifest_files: list[FilePreview] = Field(default_factory=list)
    NEXT: str = ""


class ScanEntryPointsResponse(BaseModel):
    """Response from scan_codebase phase='entry_points'."""
    project: str
    path: str
    phase: Literal["entry_points"] = "entry_points"
    entry_files: list[FilePreview] = Field(default_factory=list)
    NEXT: str = ""


class ScanSamplesResponse(BaseModel):
    """Response from scan_codebase phase='samples'."""
    project: str
    path: str
    phase: Literal["samples"] = "samples"
    sample_files: list[FilePreview] = Field(default_factory=list)
    detected_patterns: list[str] = Field(default_factory=list)
    NEXT: str = ""


# =============================================================================
# HEALTH CHECK MODELS
# =============================================================================

class BackendHealth(BaseModel):
    """Health status for a single backend."""
    id: str
    name: str
    provider: str
    type: Literal["local", "remote"]
    enabled: bool
    available: bool
    latency_ms: int | None = None
    error: str | None = None
    score: float = Field(ge=0)


class HealthResponse(BaseModel):
    """Response from health check tool."""
    status: Literal["ok", "degraded", "error"]
    backends: list[BackendHealth]
    active_sessions: int = Field(ge=0)
    uptime: str = ""


# =============================================================================
# INIT PROJECT MODELS
# =============================================================================

class TechStack(BaseModel):
    """Detected technology stack."""
    primary_language: str | None = None
    frameworks: list[str] = Field(default_factory=list)


class ProfileRecommendationOutput(BaseModel):
    """Profile recommendations from init_project."""
    high_priority: list[str] = Field(default_factory=list)
    total: int = Field(ge=0)


class InitProjectResponse(BaseModel):
    """Response from init_project tool."""
    project: str
    path: str
    status: str
    manual_mode: bool = False
    tech_stack: TechStack | None = None
    profile_recommendations: ProfileRecommendationOutput | None = None
    detected_agents: list[str] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# =============================================================================
# PLAYBOOK MODELS (re-exported from playbook.py for convenience)
# =============================================================================

from ..playbook import (
    PlaybookBullet,
    ProfileRecommendation,
    PatternGap,
    EvaluationState,
)

__all__ = [
    # Scan models
    "DirectoryInfo",
    "FilePreview",
    "ScanStructure",
    "ScanOverviewResponse",
    "ScanManifestsResponse",
    "ScanEntryPointsResponse",
    "ScanSamplesResponse",
    # Health models
    "BackendHealth",
    "HealthResponse",
    # Init models
    "TechStack",
    "ProfileRecommendationOutput",
    "InitProjectResponse",
    # Playbook models
    "PlaybookBullet",
    "ProfileRecommendation",
    "PatternGap",
    "EvaluationState",
]
