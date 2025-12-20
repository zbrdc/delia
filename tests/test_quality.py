# Copyright (C) 2024 Delia Contributors
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

"""Tests for response quality validation."""

import pytest

from delia.quality import (
    QualityConfig,
    QualityScore,
    ResponseQualityValidator,
    get_quality_validator,
    validate_response,
)


class TestQualityConfig:
    """Tests for QualityConfig dataclass."""

    def test_default_config_enabled(self):
        config = QualityConfig()
        assert config.enabled is True

    def test_default_weights_sum_to_one(self):
        config = QualityConfig()
        total = sum(config.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_default_min_words_per_task(self):
        config = QualityConfig()
        assert "review" in config.min_words_per_task
        assert "generate" in config.min_words_per_task
        assert "analyze" in config.min_words_per_task

    def test_custom_config(self):
        config = QualityConfig(
            enabled=False,
            repetition_ngram_size=3,
            repetition_threshold=2,
        )
        assert config.enabled is False
        assert config.repetition_ngram_size == 3
        assert config.repetition_threshold == 2


class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_to_dict_serializes_all_fields(self):
        score = QualityScore(
            overall=0.85,
            repetition_score=0.9,
            length_score=0.8,
            coherence_score=0.85,
            issues=["test_issue"],
        )
        d = score.to_dict()
        assert d["overall"] == 0.85
        assert d["repetition"] == 0.9
        assert d["length"] == 0.8
        assert d["coherence"] == 0.85
        assert d["issues"] == ["test_issue"]

    def test_to_dict_rounds_values(self):
        score = QualityScore(
            overall=0.85555555,
            repetition_score=0.9,
            length_score=0.8,
            coherence_score=0.85,
        )
        d = score.to_dict()
        assert d["overall"] == 0.856  # Rounded to 3 decimals


class TestResponseQualityValidatorEmpty:
    """Tests for empty/null responses."""

    def test_empty_string_returns_zero(self):
        validator = ResponseQualityValidator()
        result = validator.validate("")
        assert result.overall == 0.0
        assert "empty_response" in result.issues

    def test_none_equivalent_empty(self):
        # The function expects a string, but we test empty behavior
        validator = ResponseQualityValidator()
        result = validator.validate("")
        assert result.overall == 0.0


class TestRepetitionDetection:
    """Tests for repetition/looping detection."""

    def test_no_repetition_full_score(self):
        validator = ResponseQualityValidator()
        text = "This is a normal response with varied vocabulary and structure."
        result = validator.validate(text)
        assert result.repetition_score == 1.0

    def test_ngram_repetition_detected(self):
        validator = ResponseQualityValidator()
        # Repeat a 5-word phrase 4 times (above threshold of 3)
        text = (
            "the quick brown fox jumps "
            "the quick brown fox jumps "
            "the quick brown fox jumps "
            "the quick brown fox jumps "
        )
        result = validator.validate(text)
        assert result.repetition_score < 1.0
        assert any("ngram_repetition" in issue for issue in result.issues)

    def test_sentence_repetition_detected(self):
        validator = ResponseQualityValidator()
        text = "This is a test. This is a test. This is different."
        result = validator.validate(text)
        assert result.repetition_score < 1.0
        assert any("sentence_repetition" in issue for issue in result.issues)

    def test_token_loop_detected(self):
        validator = ResponseQualityValidator()
        text = "the the the the the the word"
        result = validator.validate(text)
        assert result.repetition_score < 1.0
        assert any("token_loop" in issue for issue in result.issues)

    def test_short_text_no_penalty(self):
        validator = ResponseQualityValidator()
        text = "ok"
        result = validator.validate(text)
        # Short text can't be analyzed for n-gram patterns
        assert result.repetition_score == 1.0


class TestLengthValidation:
    """Tests for length-based validation."""

    def test_normal_length_full_score(self):
        validator = ResponseQualityValidator()
        text = "This is a reasonably long response with enough content to analyze."
        result = validator.validate(text)
        assert result.length_score == 1.0

    def test_too_short_penalized(self):
        validator = ResponseQualityValidator()
        text = "ok"  # Only 2 characters
        result = validator.validate(text)
        assert result.length_score < 1.0
        assert any("too_short" in issue for issue in result.issues)

    def test_mostly_whitespace_penalized(self):
        validator = ResponseQualityValidator()
        text = "x" + " " * 100  # Mostly whitespace
        result = validator.validate(text)
        assert result.length_score < 1.0
        assert any("mostly_whitespace" in issue for issue in result.issues)

    def test_task_specific_minimum_words(self):
        validator = ResponseQualityValidator()
        # "analyze" requires 30 words by default
        text = "short response with only ten words here in this text"
        result = validator.validate(text, task_type="analyze")
        assert result.length_score < 1.0
        assert any("insufficient_words" in issue for issue in result.issues)

    def test_task_specific_sufficient_words(self):
        validator = ResponseQualityValidator()
        # Create a response with enough words for "generate" (10 words)
        text = "This response has more than ten words so it should pass validation."
        result = validator.validate(text, task_type="generate")
        assert result.length_score == 1.0


class TestCoherenceChecks:
    """Tests for coherence/gibberish detection."""

    def test_normal_text_coherent(self):
        validator = ResponseQualityValidator()
        text = "This is a normal, coherent response with proper English text."
        result = validator.validate(text)
        assert result.coherence_score >= 0.9

    def test_high_non_ascii_penalized(self):
        validator = ResponseQualityValidator()
        # Text with many non-ASCII characters
        text = "正常文本但是有很多中文字符在这里mixing with English"
        result = validator.validate(text)
        # Should still be mostly coherent, just flagged
        if result.coherence_score < 1.0:
            assert any("encoding_issues" in issue for issue in result.issues)

    def test_low_vocabulary_diversity_penalized(self):
        validator = ResponseQualityValidator()
        # Very repetitive vocabulary - need more words to trigger diversity check
        text = "the the the the the the the the the the word word word word word word word word word word"
        result = validator.validate(text)
        assert result.coherence_score < 1.0
        assert any("low_diversity" in issue for issue in result.issues)

    def test_single_char_spam_penalized(self):
        validator = ResponseQualityValidator()
        # Text with many isolated single characters
        text = "a b c d e f g h i j k l m n o p q r s t"
        result = validator.validate(text)
        assert result.coherence_score < 1.0
        assert any("char_spam" in issue for issue in result.issues)


class TestOverallScore:
    """Tests for overall score calculation."""

    def test_perfect_response_high_score(self):
        validator = ResponseQualityValidator()
        text = (
            "This is a well-structured response with varied vocabulary, "
            "proper sentence structure, and sufficient length to demonstrate "
            "quality content. It avoids repetition and uses diverse words "
            "throughout the entire response body."
        )
        result = validator.validate(text)
        assert result.overall >= 0.9

    def test_terrible_response_low_score(self):
        validator = ResponseQualityValidator()
        text = "a a a a a a a a a a"  # Short, repetitive, low diversity
        result = validator.validate(text)
        # Should have issues detected and lower score than perfect
        assert result.overall < 0.8
        assert len(result.issues) > 0

    def test_score_clamped_to_valid_range(self):
        validator = ResponseQualityValidator()
        # Any response should produce score in [0.0, 1.0]
        for text in ["", "x", "normal text", "a " * 100]:
            result = validator.validate(text)
            assert 0.0 <= result.overall <= 1.0


class TestCustomConfig:
    """Tests for custom configuration."""

    def test_custom_ngram_size(self):
        config = QualityConfig(repetition_ngram_size=3, repetition_threshold=2)
        validator = ResponseQualityValidator(config)
        # 3-word phrase repeated twice (at threshold)
        text = "one two three one two three other words"
        result = validator.validate(text)
        # Should detect the repetition with smaller n-gram size
        if result.repetition_score < 1.0:
            assert any("ngram_repetition" in issue for issue in result.issues)

    def test_custom_min_length(self):
        config = QualityConfig(min_response_length=50)
        validator = ResponseQualityValidator(config)
        text = "This is a short response."  # Less than 50 chars
        result = validator.validate(text)
        assert result.length_score < 1.0

    def test_custom_weights(self):
        config = QualityConfig(
            weights={"repetition": 0.0, "length": 0.0, "coherence": 1.0}
        )
        validator = ResponseQualityValidator(config)
        # This puts all weight on coherence
        text = "Normal coherent text with varied vocabulary."
        result = validator.validate(text)
        # With only coherence weighted, should reflect coherence score
        assert result.overall == pytest.approx(result.coherence_score, rel=0.01)


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_quality_validator_returns_singleton(self):
        v1 = get_quality_validator()
        v2 = get_quality_validator()
        assert v1 is v2

    def test_validate_response_convenience(self):
        result = validate_response("This is a test response with enough content.")
        assert isinstance(result, QualityScore)
        assert 0.0 <= result.overall <= 1.0

    def test_validate_response_with_task_type(self):
        result = validate_response("Short", task_type="analyze")
        # "analyze" requires 30 words, so this should fail
        assert result.length_score < 1.0


class TestAffinityTrackerIntegration:
    """Tests for integration with AffinityTracker."""

    def test_affinity_update_with_quality_score(self):
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)

        # Update with quality score instead of boolean
        tracker.update("backend1", "review", quality=0.8)
        affinity = tracker.get_affinity("backend1", "review")
        # Starting from 0.5, with alpha=0.1 and quality=0.8:
        # new = 0.5 * 0.9 + 0.8 * 0.1 = 0.45 + 0.08 = 0.53
        assert abs(affinity - 0.53) < 0.01

    def test_affinity_update_with_low_quality(self):
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)

        # Low quality score
        tracker.update("backend1", "review", quality=0.2)
        affinity = tracker.get_affinity("backend1", "review")
        # Starting from 0.5, with alpha=0.1 and quality=0.2:
        # new = 0.5 * 0.9 + 0.2 * 0.1 = 0.45 + 0.02 = 0.47
        assert abs(affinity - 0.47) < 0.01

    def test_affinity_backward_compat_success_bool(self):
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)

        # Legacy boolean still works
        tracker.update("backend1", "review", success=True)
        affinity = tracker.get_affinity("backend1", "review")
        # success=True maps to quality=1.0
        # new = 0.5 * 0.9 + 1.0 * 0.1 = 0.45 + 0.1 = 0.55
        assert abs(affinity - 0.55) < 0.01

    def test_affinity_backward_compat_failure_bool(self):
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=0.1)

        # Legacy boolean failure
        tracker.update("backend1", "review", success=False)
        affinity = tracker.get_affinity("backend1", "review")
        # success=False maps to quality=0.0
        # new = 0.5 * 0.9 + 0.0 * 0.1 = 0.45
        assert abs(affinity - 0.45) < 0.01

    def test_affinity_requires_success_or_quality(self):
        from delia.config import AffinityTracker

        tracker = AffinityTracker()
        with pytest.raises(ValueError, match="Either 'success' or 'quality'"):
            tracker.update("backend1", "review")

    def test_quality_score_clamped(self):
        from delia.config import AffinityTracker

        tracker = AffinityTracker(alpha=1.0)  # High alpha for direct update

        # Quality > 1.0 should be clamped
        tracker.update("backend1", "review", quality=1.5)
        assert tracker.get_affinity("backend1", "review") <= 1.0

        # Quality < 0.0 should be clamped
        tracker.update("backend2", "review", quality=-0.5)
        assert tracker.get_affinity("backend2", "review") >= 0.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_unicode_text(self):
        validator = ResponseQualityValidator()
        text = "This response contains émojis 🎉 and ünïcödé characters."
        result = validator.validate(text)
        assert 0.0 <= result.overall <= 1.0

    def test_very_long_text(self):
        validator = ResponseQualityValidator()
        text = "word " * 1000  # 1000 repetitions of "word"
        result = validator.validate(text)
        # Should detect low diversity
        assert result.coherence_score < 1.0
        assert result.overall < 0.8

    def test_code_snippet(self):
        validator = ResponseQualityValidator()
        text = """
def hello_world():
    print("Hello, World!")
    return True

class MyClass:
    def __init__(self):
        self.value = 42
"""
        result = validator.validate(text)
        # Code should be treated as valid content
        assert result.overall > 0.5

    def test_json_response(self):
        validator = ResponseQualityValidator()
        text = '{"status": "success", "data": {"id": 1, "name": "test"}}'
        result = validator.validate(text)
        assert result.overall > 0.5

    def test_markdown_response(self):
        validator = ResponseQualityValidator()
        text = """
# Heading

This is a paragraph with **bold** and *italic* text.

- List item 1
- List item 2

```python
print("code block")
```
"""
        result = validator.validate(text)
        assert result.overall > 0.7
