# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for structured output types."""

import json
import pytest

from delia.orchestration.outputs import (
    CodeReview,
    CodeIssue,
    Analysis,
    Finding,
    Plan,
    PlanStep,
    Comparison,
    ComparisonItem,
    QuestionAnswer,
    StructuredResponse,
    Severity,
    Priority,
    get_json_schema_prompt,
    parse_structured_output,
    get_default_output_type,
)


class TestCodeReview:
    """Tests for CodeReview structured output."""
    
    def test_basic_code_review(self):
        """Test creating a basic code review."""
        review = CodeReview(
            summary="Good code with minor issues",
            issues=[
                CodeIssue(
                    line=10,
                    severity=Severity.LOW,
                    category="style",
                    description="Missing docstring",
                )
            ],
            suggestions=["Add type hints", "Consider using dataclasses"],
            score=0.85,
            approved=True,
        )
        
        assert review.score == 0.85
        assert review.approved is True
        assert len(review.issues) == 1
        assert len(review.suggestions) == 2
        assert review.issues[0].severity == Severity.LOW
    
    def test_code_review_score_bounds(self):
        """Test that score is bounded 0-1."""
        with pytest.raises(ValueError):
            CodeReview(
                summary="Test",
                score=1.5,  # Invalid
                approved=False,
            )
    
    def test_code_review_minimal(self):
        """Test minimal code review with defaults."""
        review = CodeReview(
            summary="LGTM",
            score=1.0,
        )
        
        assert review.issues == []
        assert review.suggestions == []
        assert review.approved is False  # Default


class TestAnalysis:
    """Tests for Analysis structured output."""
    
    def test_basic_analysis(self):
        """Test creating a basic analysis."""
        analysis = Analysis(
            summary="System architecture is well-designed",
            findings=[
                Finding(
                    title="Good separation of concerns",
                    description="The codebase follows clean architecture principles",
                    confidence=0.9,
                )
            ],
            conclusion="The system is maintainable and extensible",
            confidence=0.85,
            recommendations=["Consider adding more tests"],
        )
        
        assert analysis.confidence == 0.85
        assert len(analysis.findings) == 1
        assert analysis.findings[0].confidence == 0.9


class TestPlan:
    """Tests for Plan structured output."""
    
    def test_basic_plan(self):
        """Test creating a basic plan."""
        plan = Plan(
            goal="Implement user authentication",
            approach="Use JWT tokens with refresh mechanism",
            steps=[
                PlanStep(
                    step_number=1,
                    action="Create User model",
                    estimated_effort="2 hours",
                ),
                PlanStep(
                    step_number=2,
                    action="Implement JWT generation",
                    dependencies=[1],
                    estimated_effort="4 hours",
                ),
            ],
            risks=["Token expiration handling complexity"],
            success_criteria=["Users can log in", "Tokens refresh correctly"],
        )
        
        assert plan.goal == "Implement user authentication"
        assert len(plan.steps) == 2
        assert plan.steps[1].dependencies == [1]


class TestComparison:
    """Tests for Comparison structured output."""
    
    def test_basic_comparison(self):
        """Test creating a basic comparison."""
        comparison = Comparison(
            question="Which database should we use?",
            items=[
                ComparisonItem(
                    name="PostgreSQL",
                    pros=["ACID compliance", "Rich features"],
                    cons=["Higher resource usage"],
                    score=0.85,
                ),
                ComparisonItem(
                    name="SQLite",
                    pros=["Zero configuration", "Lightweight"],
                    cons=["No concurrency"],
                    score=0.6,
                ),
            ],
            recommendation="Use PostgreSQL for production, SQLite for development",
            winner="PostgreSQL",
        )
        
        assert comparison.winner == "PostgreSQL"
        assert len(comparison.items) == 2


class TestParsing:
    """Tests for JSON parsing utilities."""
    
    def test_parse_direct_json(self):
        """Test parsing direct JSON."""
        json_str = json.dumps({
            "summary": "Good code",
            "issues": [],
            "suggestions": ["Add tests"],
            "score": 0.9,
            "approved": True,
        })
        
        result = parse_structured_output(json_str, CodeReview)
        
        assert isinstance(result, CodeReview)
        assert result.score == 0.9
    
    def test_parse_json_in_markdown(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = """Here's my review:

```json
{
    "summary": "Good code",
    "issues": [],
    "suggestions": [],
    "score": 0.8,
    "approved": true
}
```

That's my assessment."""
        
        result = parse_structured_output(response, CodeReview)
        
        assert isinstance(result, CodeReview)
        assert result.score == 0.8
    
    def test_parse_json_no_language_tag(self):
        """Test parsing JSON in code block without language tag."""
        response = """```
{"summary": "Test", "score": 0.5}
```"""
        
        result = parse_structured_output(response, CodeReview)
        
        assert result.score == 0.5
    
    def test_parse_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_structured_output("not json at all", CodeReview)
    
    def test_parse_missing_required_field(self):
        """Test that missing required fields raise ValueError."""
        json_str = json.dumps({
            "issues": [],
            # Missing: summary, score
        })
        
        with pytest.raises(ValueError, match="Failed to validate"):
            parse_structured_output(json_str, CodeReview)


class TestSchemaGeneration:
    """Tests for JSON schema prompt generation."""
    
    def test_schema_prompt_contains_schema(self):
        """Test that schema prompt contains the JSON schema."""
        prompt = get_json_schema_prompt(CodeReview)
        
        assert "json" in prompt.lower()
        assert "schema" in prompt.lower()
        assert "summary" in prompt
        assert "score" in prompt
    
    def test_schema_prompt_for_different_types(self):
        """Test schema prompts for different output types."""
        for output_type in [CodeReview, Analysis, Plan, Comparison]:
            prompt = get_json_schema_prompt(output_type)
            assert len(prompt) > 100
            assert "JSON" in prompt


class TestDefaultOutputTypes:
    """Tests for default output type mapping."""
    
    def test_review_maps_to_code_review(self):
        """Test that 'review' task maps to CodeReview."""
        assert get_default_output_type("review") == CodeReview
    
    def test_analyze_maps_to_analysis(self):
        """Test that 'analyze' task maps to Analysis."""
        assert get_default_output_type("analyze") == Analysis
    
    def test_plan_maps_to_plan(self):
        """Test that 'plan' task maps to Plan."""
        assert get_default_output_type("plan") == Plan
    
    def test_unknown_returns_none(self):
        """Test that unknown task returns None."""
        assert get_default_output_type("unknown_task") is None

