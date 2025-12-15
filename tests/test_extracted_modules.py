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

"""Tests for newly extracted modules to validate refactoring correctness."""

import asyncio
import re
from datetime import datetime
from unittest.mock import patch

import pytest
from hypothesis import given, strategies as st

from delia.queue import ModelQueue, QueuedRequest
from delia.routing import CODE_INDICATORS, detect_code_content
from delia.tokens import count_tokens, estimate_tokens, get_tiktoken_encoder
from delia.validation import (
    MAX_CONTENT_LENGTH,
    VALID_MODELS,
    VALID_TASKS,
    validate_content,
    validate_file_path,
    validate_model_hint,
    validate_task,
)


# ============================================================
# TOKENS MODULE TESTS
# ============================================================


class TestTokensModule:
    """Tests for tokens.py extracted module."""

    def test_count_tokens_empty(self):
        """Empty string returns 0 tokens."""
        assert count_tokens("") == 0

    def test_count_tokens_simple(self):
        """Simple text returns reasonable token count."""
        result = count_tokens("Hello world")
        assert result > 0
        assert result < 10  # Should be ~2-3 tokens

    def test_count_tokens_code(self):
        """Code content tokenizes correctly."""
        code = "def hello():\n    return 'world'"
        result = count_tokens(code)
        assert result > 5  # Code has more tokens

    def test_estimate_tokens_empty(self):
        """Estimate returns 0 for empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_approximation(self):
        """Estimate is roughly 1/4 of character count."""
        text = "a" * 100
        assert estimate_tokens(text) == 25  # 100 / 4

    def test_encoder_singleton(self):
        """Encoder is cached correctly."""
        enc1 = get_tiktoken_encoder()
        enc2 = get_tiktoken_encoder()
        assert enc1 is enc2  # Same instance

    @given(st.text(max_size=1000))
    def test_count_tokens_never_crashes(self, text):
        """Token counting never crashes on arbitrary input."""
        result = count_tokens(text)
        assert isinstance(result, int)
        assert result >= 0

    def test_count_tokens_unicode(self):
        """Unicode content handled correctly."""
        # Emoji and CJK characters
        text = "Hello 🌍 世界 مرحبا"
        result = count_tokens(text)
        assert result > 0

    def test_count_tokens_binary_like(self):
        """Binary-like content doesn't crash."""
        binary = bytes(range(256)).decode("latin-1")
        result = count_tokens(binary)
        assert result >= 0


# ============================================================
# VALIDATION MODULE TESTS
# ============================================================


class TestValidationModuleEdgeCases:
    """Edge case tests for validation.py."""

    def test_validate_task_case_sensitivity(self):
        """Tasks are case-sensitive."""
        valid, _ = validate_task("review")
        assert valid
        valid, _ = validate_task("REVIEW")
        assert not valid
        valid, _ = validate_task("Review")
        assert not valid

    def test_validate_content_exactly_at_limit(self):
        """Content exactly at byte limit is accepted."""
        content = "a" * MAX_CONTENT_LENGTH
        valid, _ = validate_content(content)
        assert valid

    def test_validate_content_one_over_limit(self):
        """Content one byte over limit is rejected."""
        content = "a" * (MAX_CONTENT_LENGTH + 1)
        valid, _ = validate_content(content)
        assert not valid

    def test_validate_content_multibyte_unicode(self):
        """Multi-byte unicode counted correctly."""
        # Each emoji is 4 bytes in UTF-8
        emoji = "🎉"
        assert len(emoji.encode("utf-8")) == 4
        # Fill to just under limit with emojis
        count = MAX_CONTENT_LENGTH // 4
        content = emoji * count
        valid, _ = validate_content(content)
        assert valid

    def test_validate_file_path_null_bytes(self):
        """Null bytes in path are handled."""
        valid, msg = validate_file_path("/path/with\x00null")
        # Should either accept or reject gracefully, not crash
        assert isinstance(valid, bool)

    def test_validate_file_path_unicode_normalization(self):
        """Unicode paths work correctly."""
        valid, _ = validate_file_path("/path/to/文件.txt")
        assert valid

    @given(st.text(max_size=100))
    def test_validate_task_never_crashes(self, task):
        """Task validation never crashes."""
        valid, msg = validate_task(task)
        assert isinstance(valid, bool)
        assert isinstance(msg, str)

    @given(st.text(max_size=10000))
    def test_validate_content_never_crashes(self, content):
        """Content validation never crashes."""
        valid, msg = validate_content(content)
        assert isinstance(valid, bool)
        assert isinstance(msg, str)


# ============================================================
# ROUTING MODULE TESTS
# ============================================================


class TestRoutingModule:
    """Tests for routing.py code detection."""

    def test_code_indicators_compiled(self):
        """All code indicators are pre-compiled regex."""
        for category in ["strong", "medium", "weak"]:
            for pattern in CODE_INDICATORS[category]:
                assert isinstance(pattern, re.Pattern)

    def test_detect_code_python(self):
        """Python code detected correctly."""
        code = """
def hello(name):
    return f"Hello, {name}!"

class Greeter:
    def __init__(self):
        self.count = 0
"""
        is_code, confidence, _ = detect_code_content(code)
        assert is_code
        assert confidence > 0.5

    def test_detect_code_javascript(self):
        """JavaScript code detected correctly."""
        code = """
const hello = (name) => {
    return `Hello, ${name}!`;
};

export function greet() {
    console.log("Hello");
}
"""
        is_code, confidence, _ = detect_code_content(code)
        assert is_code
        assert confidence > 0.3

    def test_detect_text_prose(self):
        """Prose text detected as not code."""
        # Avoid words that look like code keywords (function, import, etc.)
        text = """
This is a paragraph of normal text. It talks about various things
like the weather, philosophy, and cooking recipes. The chef prepared
a delicious meal with fresh ingredients from the garden.

The quick brown fox jumps over the lazy dog. This sentence contains
every letter of the alphabet. Pretty neat, right? We enjoyed the
sunset over the mountains while sipping warm tea.
"""
        is_code, confidence, _ = detect_code_content(text)
        assert not is_code
        assert confidence < 0.5

    def test_detect_code_short_content(self):
        """Very short content returns early."""
        is_code, confidence, reason = detect_code_content("x")
        assert not is_code
        assert confidence == 0.0
        assert "too short" in reason.lower()

    def test_detect_code_empty(self):
        """Empty content handled."""
        is_code, confidence, _ = detect_code_content("")
        assert not is_code
        assert confidence == 0.0

    def test_detect_code_mixed_content(self):
        """Mixed code and prose returns reasonable result."""
        mixed = """
# Documentation

This module provides utilities for greeting users.

## Usage

```python
def greet(name):
    print(f"Hello, {name}")
```

## Notes

Remember to handle edge cases.
"""
        is_code, confidence, _ = detect_code_content(mixed)
        # Should detect some code signals
        assert isinstance(is_code, bool)
        assert 0 <= confidence <= 1

    @given(st.text(min_size=50, max_size=2000))
    def test_detect_code_never_crashes(self, content):
        """Code detection never crashes on arbitrary input."""
        is_code, confidence, reason = detect_code_content(content)
        assert isinstance(is_code, bool)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        assert isinstance(reason, str)


# ============================================================
# QUEUE MODULE TESTS
# ============================================================


class TestQueueModuleEdgeCases:
    """Edge case tests for queue.py."""

    @pytest.mark.asyncio
    async def test_queued_request_ordering(self):
        """QueuedRequests order by priority."""
        future1 = asyncio.get_event_loop().create_future()
        future2 = asyncio.get_event_loop().create_future()
        req1 = QueuedRequest(
            priority=10,
            timestamp=datetime.now(),
            request_id="req1",
            model_name="model",
            task_type="review",
            content_length=100,
            future=future1,
        )
        req2 = QueuedRequest(
            priority=5,
            timestamp=datetime.now(),
            request_id="req2",
            model_name="model",
            task_type="review",
            content_length=100,
            future=future2,
        )
        # Lower priority number = higher priority
        assert req2 < req1

    def test_model_queue_size_estimates(self):
        """Model size estimates work for various patterns."""
        queue = ModelQueue()

        # Known patterns
        assert queue.get_model_size("some-30b-model") == 16.0
        assert queue.get_model_size("qwen-14b-instruct") == 8.0
        assert queue.get_model_size("tiny-7b") == 4.0  # Default

    def test_model_queue_priority_calculation(self):
        """Priority calculated correctly for different task types."""
        queue = ModelQueue()

        think_priority = queue.calculate_priority("think", 100, "model-7b")
        review_priority = queue.calculate_priority("review", 100, "model-7b")
        quick_priority = queue.calculate_priority("quick", 100, "model-7b")

        # Thinking tasks should have lowest priority value (highest priority)
        assert think_priority < review_priority
        # Review should have lower priority value than generic quick
        assert review_priority < quick_priority

    def test_model_queue_memory_tracking(self):
        """Memory tracking works correctly."""
        queue = ModelQueue()
        queue.gpu_memory_limit_gb = 24.0
        queue.memory_buffer_gb = 2.0

        # Initially should have full available memory
        available = queue.get_available_memory()
        assert available == 24.0 - 2.0  # 22 GB

        # Simulate loading a model
        queue.loaded_models["test-14b"] = {"size_gb": 8.0}
        available = queue.get_available_memory()
        assert available == 24.0 - 2.0 - 8.0  # 14 GB

    @pytest.mark.asyncio
    async def test_model_queue_acquire_release_cycle(self):
        """Full acquire/release cycle works."""
        queue = ModelQueue()

        # Acquire a new model
        available, future = await queue.acquire_model("test-model", "review", 100)
        assert available  # Should be available (needs loading)
        assert future is None  # Not queued

        # Model should be in loading state
        assert "test-model" in queue.loading_models

        # Release the model (success)
        await queue.release_model("test-model", success=True)

        # Model should now be loaded
        assert "test-model" in queue.loaded_models
        assert "test-model" not in queue.loading_models

    def test_queue_status_structure(self):
        """Queue status returns expected structure."""
        queue = ModelQueue()
        status = queue.get_queue_status()

        assert "loaded_models" in status
        assert "loading_models" in status
        assert "queued_requests" in status
        assert "available_memory_gb" in status
        assert "queue_stats" in status
        assert "total_queued" in status["queue_stats"]


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestModuleIntegration:
    """Tests that extracted modules work together correctly."""

    def test_validation_used_by_mcp_server(self):
        """Validation functions imported correctly in mcp_server."""
        from delia.mcp_server import (
            MAX_CONTENT_LENGTH,
            validate_content,
            validate_task,
        )

        # Should be the same functions
        valid, _ = validate_task("review")
        assert valid

    def test_tokens_used_by_providers(self):
        """Token counting works in provider context."""
        from delia.tokens import count_tokens

        # Simulate what providers do
        response_text = "This is a model response with some content."
        tokens = count_tokens(response_text)
        assert tokens > 0

    def test_routing_used_by_mcp_server(self):
        """Routing functions imported correctly."""
        from delia.mcp_server import detect_code_content

        # Need >20 chars for detection to work
        is_code, _, _ = detect_code_content("def foo():\n    return 'bar'")
        assert is_code

    def test_queue_used_by_mcp_server(self):
        """Queue classes imported correctly."""
        from delia.mcp_server import ModelQueue, model_queue

        assert isinstance(model_queue, ModelQueue)


# ============================================================
# REGRESSION TESTS
# ============================================================


class TestRefactoringRegressions:
    """Tests to catch regressions from the refactoring."""

    def test_all_valid_tasks_still_work(self):
        """All previously valid tasks still validate."""
        expected_tasks = {"review", "analyze", "generate", "summarize", "critique", "quick", "plan", "think"}
        assert VALID_TASKS == expected_tasks

    def test_all_valid_models_still_work(self):
        """All previously valid model hints still validate."""
        expected_models = {"quick", "coder", "moe", "thinking"}
        assert VALID_MODELS == expected_models

    def test_code_detection_patterns_complete(self):
        """All code detection pattern categories present."""
        assert "strong" in CODE_INDICATORS
        assert "medium" in CODE_INDICATORS
        assert "weak" in CODE_INDICATORS
        # Should have multiple patterns in each
        assert len(CODE_INDICATORS["strong"]) >= 10
        assert len(CODE_INDICATORS["medium"]) >= 5
        assert len(CODE_INDICATORS["weak"]) >= 3

    def test_max_content_length_unchanged(self):
        """Content length limit unchanged."""
        assert MAX_CONTENT_LENGTH == 500_000
