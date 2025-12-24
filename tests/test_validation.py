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

"""
Comprehensive unit tests for input validation functions in mcp_server.py.

Tests cover boundary conditions, edge cases, and security considerations.
"""
import pytest
from hypothesis import given, strategies as st, assume

from pathlib import Path

from delia.validation import (
    validate_task, validate_content, validate_file_path, validate_model_hint,
    MAX_CONTENT_LENGTH, MAX_FILE_PATH_LENGTH
)
from delia.types import VALID_TASKS, VALID_MODELS


class TestValidateTask:
    """Tests for task type validation."""

    @pytest.mark.parametrize("task", list(VALID_TASKS))
    def test_valid_tasks_accepted(self, task):
        """All valid task types should be accepted."""
        is_valid, error = validate_task(task)
        assert is_valid, f"Task '{task}' should be valid"
        assert error == ""

    def test_empty_task_rejected(self):
        """Empty task should be rejected."""
        is_valid, error = validate_task("")
        assert not is_valid
        assert "required" in error.lower()

    def test_none_task_rejected(self):
        """None task should be rejected."""
        is_valid, error = validate_task(None)
        assert not is_valid

    @pytest.mark.parametrize("invalid_task", [
        "invalid", "REVIEW", "Review", "code", "test", "debug",
        "review ", " review", "review\n", "\treview"
    ])
    def test_invalid_tasks_rejected(self, invalid_task):
        """Invalid task types should be rejected with helpful error."""
        is_valid, error = validate_task(invalid_task)
        assert not is_valid
        assert "Invalid task type" in error
        assert invalid_task.strip() in error or "Invalid" in error

    @pytest.mark.fuzz
    @given(st.text(min_size=1, max_size=50))
    def test_arbitrary_strings_handled(self, task):
        """Arbitrary strings should be handled without crashing."""
        assume(task not in VALID_TASKS)
        is_valid, error = validate_task(task)
        assert not is_valid
        assert len(error) > 0


class TestValidateContent:
    """Tests for content validation."""

    def test_valid_content_accepted(self):
        """Normal content should be accepted."""
        is_valid, error = validate_content("Hello, this is test content.")
        assert is_valid
        assert error == ""

    def test_empty_content_rejected(self):
        """Empty content should be rejected."""
        is_valid, error = validate_content("")
        assert not is_valid
        assert "required" in error.lower()

    def test_none_content_rejected(self):
        """None content should be rejected."""
        is_valid, error = validate_content(None)
        assert not is_valid

    def test_max_content_boundary(self):
        """Content at max boundary should be accepted."""
        # Create content exactly at limit
        content = "a" * MAX_CONTENT_LENGTH
        is_valid, error = validate_content(content)
        assert is_valid, f"Content at {MAX_CONTENT_LENGTH} bytes should be valid"

    def test_over_max_content_rejected(self):
        """Content over max should be rejected."""
        content = "a" * (MAX_CONTENT_LENGTH + 1)
        is_valid, error = validate_content(content)
        assert not is_valid
        assert "too large" in error.lower()

    def test_unicode_content_counted_correctly(self):
        """Unicode content should be measured in bytes, not chars."""
        # UTF-8: emoji = 4 bytes, Chinese char = 3 bytes
        emoji_content = "🍉" * (MAX_CONTENT_LENGTH // 4)  # Should be at limit
        is_valid, error = validate_content(emoji_content)
        assert is_valid

        # Slightly over limit
        emoji_over = "🍉" * ((MAX_CONTENT_LENGTH // 4) + 1)
        is_valid, error = validate_content(emoji_over)
        assert not is_valid

    def test_whitespace_only_accepted(self):
        """Whitespace-only content is technically valid (non-empty)."""
        is_valid, error = validate_content("   \n\t  ")
        assert is_valid

    @pytest.mark.fuzz
    @given(st.text(min_size=1, max_size=1000))
    def test_arbitrary_content_handled(self, content):
        """Arbitrary content should be handled without crashing."""
        is_valid, error = validate_content(content)
        # Should either be valid or have an error message
        assert is_valid or len(error) > 0


class TestValidateFilePath:
    """Tests for file path validation."""

    def test_none_path_accepted(self):
        """None path should be accepted (optional field)."""
        is_valid, error = validate_file_path(None)
        assert is_valid
        assert error == ""

    def test_valid_paths_accepted(self):
        """Valid file paths should be accepted."""
        valid_paths = [
            "/home/user/file.py",
            "relative/path/file.txt",
            "~/documents/code.js",
            "/tmp/test",
            "file.py",
            "./src/main.rs",
        ]
        for path in valid_paths:
            is_valid, error = validate_file_path(path)
            assert is_valid, f"Path '{path}' should be valid"

    def test_empty_string_rejected(self):
        """Empty string path should be rejected."""
        is_valid, error = validate_file_path("")
        assert not is_valid
        assert "empty" in error.lower()

    def test_path_traversal_rejected(self):
        """Path traversal attempts should be rejected."""
        dangerous_paths = [
            "../../../etc/passwd",
            "/home/../../../etc/shadow",
            "foo/../../bar",
            "..\\..\\windows\\system32",
            "~/../../root/.ssh/id_rsa",
        ]
        for path in dangerous_paths:
            is_valid, error = validate_file_path(path)
            assert not is_valid, f"Path traversal '{path}' should be rejected"
            assert "traversal" in error.lower() or ".." in error

    def test_long_path_rejected(self):
        """Overly long paths should be rejected."""
        long_path = "a" * (MAX_FILE_PATH_LENGTH + 1)
        is_valid, error = validate_file_path(long_path)
        assert not is_valid
        assert "too long" in error.lower()

    def test_path_at_max_length(self):
        """Path at max length should be accepted."""
        path = "a" * MAX_FILE_PATH_LENGTH
        is_valid, error = validate_file_path(path)
        assert is_valid

    def test_tilde_expansion_allowed(self):
        """Tilde (~) should be allowed for home directory."""
        is_valid, error = validate_file_path("~/Documents/file.txt")
        assert is_valid

    @pytest.mark.fuzz
    @given(st.text(min_size=1, max_size=500))
    def test_arbitrary_paths_handled(self, path):
        """Arbitrary paths should be handled without crashing."""
        is_valid, error = validate_file_path(path)
        # Should return a valid tuple
        assert isinstance(is_valid, bool)
        assert isinstance(error, str)


class TestValidateModelHint:
    """Tests for model hint validation."""

    @pytest.mark.parametrize("model", list(VALID_MODELS))
    def test_valid_models_accepted(self, model):
        """All valid model hints should be accepted."""
        is_valid, error = validate_model_hint(model)
        assert is_valid, f"Model '{model}' should be valid"
        assert error == ""

    def test_empty_hint_accepted(self):
        """Empty model hint should be accepted (optional)."""
        is_valid, error = validate_model_hint("")
        assert is_valid
        assert error == ""

    def test_none_hint_accepted(self):
        """None model hint should be accepted (optional)."""
        is_valid, error = validate_model_hint(None)
        assert is_valid
        assert error == ""

    @pytest.mark.parametrize("invalid_model", [
        "invalid", "QUICK", "Quick", "gpt-4", "claude", "fast",
        "quick ", " quick", "quick\n"
    ])
    def test_invalid_models_rejected(self, invalid_model):
        """Invalid model hints should be rejected."""
        is_valid, error = validate_model_hint(invalid_model)
        assert not is_valid
        assert "Invalid model" in error

    @pytest.mark.fuzz
    @given(st.text(min_size=1, max_size=50))
    def test_arbitrary_hints_handled(self, hint):
        """Arbitrary model hints should be handled without crashing."""
        assume(hint not in VALID_MODELS and hint.strip() != "")
        is_valid, error = validate_model_hint(hint)
        assert not is_valid
        assert len(error) > 0


class TestValidationCombinations:
    """Test combinations of validation functions."""

    def test_all_valid_inputs(self):
        """All valid inputs should pass validation."""
        assert validate_task("review")[0]
        assert validate_content("Some code to review")[0]
        assert validate_file_path("/path/to/file.py")[0]
        assert validate_model_hint("coder")[0]

    def test_minimal_valid_request(self):
        """Minimal valid request: task + content only."""
        assert validate_task("quick")[0]
        assert validate_content("Hello world")[0]
        assert validate_file_path(None)[0]
        assert validate_model_hint(None)[0]

    @pytest.mark.parametrize("content_size", [
        100,           # Small
        10_000,        # Medium
        100_000,       # Large
        499_999,       # Near max
        MAX_CONTENT_LENGTH,  # At max
    ])
    def test_various_content_sizes(self, content_size):
        """Various content sizes up to max should be valid."""
        content = "x" * content_size
        is_valid, error = validate_content(content)
        assert is_valid, f"Content of size {content_size} should be valid"

    def test_security_validation_combinations(self):
        """Security-sensitive validations should all reject dangerous input."""
        # Path traversal
        assert not validate_file_path("../../../etc/passwd")[0]

        # Oversized content
        assert not validate_content("x" * (MAX_CONTENT_LENGTH + 1))[0]

        # Unknown task types
        assert not validate_task("exec")[0]
        assert not validate_task("shell")[0]
        assert not validate_task("system")[0]
