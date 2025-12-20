# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for web search quality validation."""

import pytest

from delia.tools.web_search import (
    SearchResultQuality,
    validate_search_result,
    filter_quality_results,
)


class TestValidateSearchResult:
    """Tests for individual search result validation."""

    def test_valid_result_passes(self):
        """A complete, relevant result should pass validation."""
        result = validate_search_result(
            title="Python Tutorial - Learn Python Programming",
            url="https://example.com/python-tutorial",
            body="Learn Python programming with this comprehensive tutorial. Python is a versatile language used for web development, data science, and more.",
            query="python tutorial",
        )
        assert result.is_valid
        assert result.score >= 0.7
        assert len(result.issues) == 0

    def test_missing_title_fails(self):
        """Result without title should fail."""
        result = validate_search_result(
            title="",
            url="https://example.com/page",
            body="Some content here about the topic.",
            query="test query",
        )
        assert not result.is_valid
        assert "missing_title" in result.issues

    def test_short_title_fails(self):
        """Very short title should fail."""
        result = validate_search_result(
            title="Hi",
            url="https://example.com/page",
            body="Some content here about the topic.",
            query="test query",
        )
        assert not result.is_valid
        assert "missing_title" in result.issues

    def test_invalid_url_fails(self):
        """Result without valid URL should fail."""
        result = validate_search_result(
            title="Good Title Here",
            url="not-a-valid-url",
            body="Some content here about the topic.",
            query="test query",
        )
        assert not result.is_valid
        assert "invalid_url" in result.issues

    def test_empty_url_fails(self):
        """Result with empty URL should fail."""
        result = validate_search_result(
            title="Good Title Here",
            url="",
            body="Some content here about the topic.",
            query="test query",
        )
        assert not result.is_valid
        assert "invalid_url" in result.issues

    def test_missing_body_partial_penalty(self):
        """Result without body should get partial penalty but may pass."""
        result = validate_search_result(
            title="Good Title Here",
            url="https://example.com/page",
            body="",
            query="test query",
        )
        # May or may not be valid depending on other factors
        assert "missing_body" in result.issues
        assert result.score < 1.0

    def test_repetitive_content_detected(self):
        """Repetitive spam content should be penalized."""
        result = validate_search_result(
            title="Spam Page",
            url="https://example.com/spam",
            body="buy buy buy buy buy cheap cheap cheap cheap cheap now now now now now",
            query="test query",
        )
        assert "repetitive_content" in result.issues
        # Score is lowered but may still pass due to other factors being OK
        assert result.score < 0.9

    def test_excessive_special_chars_detected(self):
        """Content with too many special characters should be penalized."""
        result = validate_search_result(
            title="Normal Title",
            url="https://example.com/page",
            body="!@#$%^&*()_+{}|:<>?~`-=[]\\;',./!@#$%^&*()",
            query="test",
        )
        assert "excessive_special_chars" in result.issues

    def test_low_relevance_detected(self):
        """Content unrelated to query should be flagged."""
        result = validate_search_result(
            title="Weather Report",
            url="https://example.com/weather",
            body="Today will be sunny with temperatures reaching 75 degrees. Expect clear skies throughout the evening.",
            query="python programming tutorial",
        )
        assert "low_relevance" in result.issues

    def test_relevant_content_not_flagged(self):
        """Content matching query should not be flagged for relevance."""
        result = validate_search_result(
            title="Python Programming Guide",
            url="https://example.com/python",
            body="This tutorial covers Python programming basics including variables, loops, and functions.",
            query="python programming tutorial",
        )
        assert "low_relevance" not in result.issues

    def test_gibberish_detected(self):
        """Content with too many short words should be flagged."""
        result = validate_search_result(
            title="Normal Title",
            url="https://example.com/page",
            body="a b c d e f g h i j k l m n o p q r s t u v w x y z",
            query="test",
        )
        assert "possible_gibberish" in result.issues


class TestFilterQualityResults:
    """Tests for batch filtering of search results."""

    def test_filter_keeps_good_results(self):
        """Good results should be kept."""
        results = [
            {
                "title": "Python Tutorial",
                "href": "https://example.com/python",
                "body": "Learn Python programming with examples and exercises.",
            },
            {
                "title": "Python Guide",
                "href": "https://example.com/guide",
                "body": "Comprehensive Python guide for beginners and experts.",
            },
        ]
        filtered, rejected = filter_quality_results(results, "python tutorial")
        assert len(filtered) == 2
        assert rejected == 0

    def test_filter_removes_bad_results(self):
        """Bad results should be filtered out."""
        results = [
            {
                "title": "Python Tutorial",
                "href": "https://example.com/python",
                "body": "Learn Python programming with examples and exercises.",
            },
            {
                "title": "",  # Missing title
                "href": "https://example.com/spam",
                "body": "Some content",
            },
            {
                "title": "Spam",
                "href": "invalid-url",  # Invalid URL
                "body": "More content here",
            },
        ]
        filtered, rejected = filter_quality_results(results, "python tutorial")
        assert len(filtered) == 1
        assert rejected == 2

    def test_filter_adds_quality_score(self):
        """Filtered results should have quality score attached."""
        results = [
            {
                "title": "Python Tutorial",
                "href": "https://example.com/python",
                "body": "Learn Python programming with examples and exercises.",
            },
        ]
        filtered, _ = filter_quality_results(results, "python tutorial")
        assert len(filtered) == 1
        assert "_quality_score" in filtered[0]
        assert 0 <= filtered[0]["_quality_score"] <= 1

    def test_filter_respects_min_score(self):
        """Results below min_score should be filtered."""
        results = [
            {
                "title": "Good Result",
                "href": "https://example.com/good",
                "body": "This is excellent content about Python programming tutorials and guides.",
            },
            {
                "title": "Mediocre Result",
                "href": "https://example.com/ok",
                "body": "Short content.",  # Short body will lower score
            },
        ]
        # With high min_score, mediocre results may be filtered
        filtered_high, rejected_high = filter_quality_results(results, "python", min_score=0.8)
        filtered_low, rejected_low = filter_quality_results(results, "python", min_score=0.3)

        # Lower min_score should keep more results
        assert len(filtered_low) >= len(filtered_high)

    def test_empty_results_handled(self):
        """Empty result list should return empty."""
        filtered, rejected = filter_quality_results([], "test query")
        assert filtered == []
        assert rejected == 0

    def test_all_bad_results_returns_empty(self):
        """If all results are bad, return empty list."""
        results = [
            {"title": "", "href": "", "body": ""},
            {"title": "x", "href": "bad", "body": ""},
        ]
        filtered, rejected = filter_quality_results(results, "test")
        assert len(filtered) == 0
        assert rejected == 2

    def test_news_url_field_handled(self):
        """News results use 'url' instead of 'href' - should still work."""
        results = [
            {
                "title": "Breaking News",
                "url": "https://news.example.com/story",
                "href": "https://news.example.com/story",  # Normalized
                "body": "Important news story about recent events and developments.",
            },
        ]
        filtered, rejected = filter_quality_results(results, "news")
        assert len(filtered) == 1
