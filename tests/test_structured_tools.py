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
Tests for structured JSON tools.

These tests verify the schema validation and response structure
of the LLM-to-LLM optimized tools.
"""
import json
import pytest

from delia.schemas import (
    # Enums
    TaskType,
    ModelTier,
    ContentType,
    Language,
    BackendPreference,
    Severity,
    AnalysisType,
    ReasoningDepth,
    # Requests
    StructuredRequest,
    CodeReviewRequest,
    CodeGenerateRequest,
    AnalyzeRequest,
    ThinkRequest,
    # Responses
    UsageMetrics,
    ExecutionInfo,
    StructuredResponse,
    CodeReviewResponse,
    CodeGenerateResponse,
    AnalyzeResponse,
    ThinkResponse,
)


class TestEnums:
    """Test enum definitions."""

    def test_task_type_values(self):
        """Verify all task types are defined."""
        assert TaskType.REVIEW.value == "review"
        assert TaskType.ANALYZE.value == "analyze"
        assert TaskType.GENERATE.value == "generate"
        assert TaskType.SUMMARIZE.value == "summarize"
        assert TaskType.CRITIQUE.value == "critique"
        assert TaskType.QUICK.value == "quick"
        assert TaskType.PLAN.value == "plan"
        assert TaskType.THINK.value == "think"

    def test_model_tier_values(self):
        """Verify all model tiers are defined."""
        assert ModelTier.QUICK.value == "quick"
        assert ModelTier.CODER.value == "coder"
        assert ModelTier.MOE.value == "moe"
        assert ModelTier.THINKING.value == "thinking"

    def test_content_type_values(self):
        """Verify all content types are defined."""
        assert ContentType.CODE.value == "code"
        assert ContentType.TEXT.value == "text"
        assert ContentType.MIXED.value == "mixed"

    def test_language_values(self):
        """Verify common languages are defined."""
        assert Language.PYTHON.value == "python"
        assert Language.TYPESCRIPT.value == "typescript"
        assert Language.RUST.value == "rust"
        assert Language.GO.value == "go"


class TestRequestSchemas:
    """Test request schema validation."""

    def test_structured_request_minimal(self):
        """Test minimal StructuredRequest."""
        req = StructuredRequest(content="Hello world")
        assert req.content == "Hello world"
        assert req.content_type == ContentType.MIXED
        assert req.language is None
        assert req.model_tier is None
        assert req.backend == BackendPreference.AUTO

    def test_structured_request_full(self):
        """Test StructuredRequest with all fields."""
        req = StructuredRequest(
            content="def foo(): pass",
            content_type=ContentType.CODE,
            language=Language.PYTHON,
            model_tier=ModelTier.CODER,
            backend=BackendPreference.LOCAL,
            file_path="/path/to/file.py",
            max_tokens=1000,
            timeout_ms=5000,
        )
        assert req.content_type == ContentType.CODE
        assert req.language == Language.PYTHON
        assert req.model_tier == ModelTier.CODER
        assert req.backend == BackendPreference.LOCAL
        assert req.file_path == "/path/to/file.py"
        assert req.max_tokens == 1000
        assert req.timeout_ms == 5000

    def test_code_review_request(self):
        """Test CodeReviewRequest with focus areas."""
        req = CodeReviewRequest(
            content="def foo(): pass",
            content_type=ContentType.CODE,
            language=Language.PYTHON,
            focus_areas=["security", "performance"],
            symbols=["foo"],
            severity_threshold=Severity.WARNING,
        )
        assert req.task_type == TaskType.REVIEW
        assert req.focus_areas == ["security", "performance"]
        assert req.symbols == ["foo"]
        assert req.severity_threshold == Severity.WARNING

    def test_code_generate_request(self):
        """Test CodeGenerateRequest."""
        req = CodeGenerateRequest(
            content="Create a fibonacci function",
            language=Language.PYTHON,
            include_tests=True,
            include_docstrings=True,
            target_framework="pytest",
        )
        assert req.task_type == TaskType.GENERATE
        assert req.include_tests is True
        assert req.target_framework == "pytest"

    def test_analyze_request(self):
        """Test AnalyzeRequest."""
        req = AnalyzeRequest(
            content="class MyClass: pass",
            analysis_type=AnalysisType.COMPLEXITY,
            depth=ReasoningDepth.DEEP,
            include_metrics=True,
        )
        assert req.task_type == TaskType.ANALYZE
        assert req.analysis_type == AnalysisType.COMPLEXITY
        assert req.depth == ReasoningDepth.DEEP
        assert req.include_metrics is True

    def test_think_request(self):
        """Test ThinkRequest."""
        req = ThinkRequest(
            problem="How to design a cache?",
            context="Distributed system",
            constraints=["Must handle 10k QPS"],
            depth=ReasoningDepth.DEEP,
        )
        assert req.problem == "How to design a cache?"
        assert req.context == "Distributed system"
        assert req.constraints == ["Must handle 10k QPS"]
        assert req.depth == ReasoningDepth.DEEP

    def test_request_json_serialization(self):
        """Test that requests can be serialized to JSON."""
        req = CodeReviewRequest(
            content="def foo(): pass",
            language=Language.PYTHON,
            focus_areas=["security"],
        )
        json_str = req.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["content"] == "def foo(): pass"
        assert parsed["language"] == "python"
        assert parsed["focus_areas"] == ["security"]

    def test_request_json_deserialization(self):
        """Test that requests can be parsed from JSON."""
        json_str = json.dumps({
            "content": "def foo(): pass",
            "content_type": "code",
            "language": "python",
            "model_tier": "coder",
        })
        req = StructuredRequest.model_validate_json(json_str)
        assert req.content == "def foo(): pass"
        assert req.content_type == ContentType.CODE
        assert req.language == Language.PYTHON
        assert req.model_tier == ModelTier.CODER


class TestResponseSchemas:
    """Test response schema validation."""

    def test_usage_metrics(self):
        """Test UsageMetrics defaults."""
        metrics = UsageMetrics()
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.total_tokens == 0
        assert metrics.latency_ms == 0
        assert metrics.queue_wait_ms is None

    def test_execution_info(self):
        """Test ExecutionInfo."""
        info = ExecutionInfo(
            model="qwen2.5-coder:14b",
            model_tier=ModelTier.CODER,
            backend_id="ollama-local",
            backend_type="local",
            provider="ollama",
        )
        assert info.model == "qwen2.5-coder:14b"
        assert info.model_tier == ModelTier.CODER
        assert info.backend_id == "ollama-local"
        assert info.provider == "ollama"

    def test_structured_response_success(self):
        """Test successful StructuredResponse."""
        resp = StructuredResponse(
            success=True,
            content="The code looks good.",
            usage=UsageMetrics(total_tokens=100, latency_ms=500),
            execution=ExecutionInfo(model="test", model_tier=ModelTier.QUICK),
            request_id="abc123",
        )
        assert resp.success is True
        assert resp.content == "The code looks good."
        assert resp.error is None
        assert resp.usage.total_tokens == 100

    def test_structured_response_error(self):
        """Test error StructuredResponse."""
        resp = StructuredResponse(
            success=False,
            content="",
            error="Model not available",
            request_id="abc123",
        )
        assert resp.success is False
        assert resp.error == "Model not available"

    def test_code_review_response(self):
        """Test CodeReviewResponse."""
        resp = CodeReviewResponse(
            success=True,
            content="Found issues...",
            findings=[],
            summary="2 issues found",
            metrics={"error": 1, "warning": 1},
            reviewed_lines=100,
            request_id="abc123",
        )
        assert resp.summary == "2 issues found"
        assert resp.metrics == {"error": 1, "warning": 1}
        assert resp.reviewed_lines == 100

    def test_response_json_serialization(self):
        """Test that responses can be serialized to JSON."""
        resp = StructuredResponse(
            success=True,
            content="Test response",
            usage=UsageMetrics(total_tokens=50),
            execution=ExecutionInfo(model="test", model_tier=ModelTier.QUICK),
            request_id="test123",
        )
        json_str = resp.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["success"] is True
        assert parsed["content"] == "Test response"
        assert parsed["usage"]["total_tokens"] == 50
        assert parsed["request_id"] == "test123"


class TestSchemaImports:
    """Test that all schemas can be imported."""

    def test_all_exports(self):
        """Verify __all__ exports are accessible."""
        from delia.schemas import __all__
        assert "TaskType" in __all__
        assert "ModelTier" in __all__
        assert "StructuredRequest" in __all__
        assert "StructuredResponse" in __all__
        assert "CodeReviewRequest" in __all__
        assert "CodeReviewResponse" in __all__
