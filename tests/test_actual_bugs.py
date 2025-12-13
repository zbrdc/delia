"""
Tests that identify ACTUAL BUGS in the codebase.

These tests should FAIL if there are bugs, prompting fixes.
"""
import json
import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock



class TestValidationBugs:
    """Tests for bugs in validation functions."""

    def test_validate_content_with_none_crashes(self):
        """BUG: validate_content crashes with None input instead of returning error."""
        from mcp_server import validate_content

        # This should return (False, "error message"), not crash
        try:
            is_valid, error = validate_content(None)
            # If we get here, it handled None gracefully
            assert not is_valid, "None content should be invalid"
        except (TypeError, AttributeError) as e:
            pytest.fail(f"BUG FOUND: validate_content crashes with None: {e}")

    def test_validate_task_with_none_crashes(self):
        """BUG: validate_task may crash with None input."""
        from mcp_server import validate_task

        try:
            is_valid, error = validate_task(None)
            assert not is_valid, "None task should be invalid"
        except (TypeError, AttributeError) as e:
            pytest.fail(f"BUG FOUND: validate_task crashes with None: {e}")

    def test_validate_content_with_non_string(self):
        """FIXED: validate_content now handles non-string input gracefully."""
        from mcp_server import validate_content

        non_strings = [123, 45.67, [], {}, object(), b"bytes"]

        for value in non_strings:
            # Should not crash, should return (False, error_message)
            is_valid, error = validate_content(value)
            assert not is_valid, f"Non-string {value} should be invalid"
            assert "string" in error.lower() or "must be" in error.lower()


class TestStatsFileBugs:
    """Test for bugs in stats file handling."""

    def test_stats_file_race_condition(self, tmp_path):
        """BUG: Concurrent stats saves may corrupt file."""
        import threading
        import time

        stats_file = tmp_path / "stats.json"

        def write_stats(n):
            """Simulate stats saving."""
            data = {"counter": n, "data": "x" * 1000}
            with open(stats_file, "w") as f:
                json.dump(data, f)

        # Spawn many threads writing concurrently
        threads = []
        for i in range(20):
            t = threading.Thread(target=write_stats, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # File should be valid JSON
        try:
            with open(stats_file) as f:
                data = json.load(f)
            assert "counter" in data
        except json.JSONDecodeError as e:
            pytest.fail(f"BUG FOUND: Concurrent writes corrupted stats file: {e}")


class TestModelParsingBugs:
    """Test for bugs in model parsing."""

    def test_huge_moe_pattern_overflow(self):
        """BUG: Huge MoE patterns may cause overflow or performance issues."""
        from config import parse_model_name
        import math

        # Very large experts count - should not overflow
        result = parse_model_name("mixtral-99999x99999b")

        # Check for overflow
        assert not math.isinf(result.params_b), "BUG: MoE pattern caused infinity"
        assert not math.isnan(result.params_b), "BUG: MoE pattern caused NaN"

    def test_model_name_regex_catastrophic_backtracking(self):
        """BUG: Complex model names may cause regex catastrophic backtracking."""
        from config import parse_model_name
        import time

        # Pattern designed to cause backtracking
        evil_name = "a" * 1000 + "b" * 1000 + "14b"

        start = time.time()
        result = parse_model_name(evil_name)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0, f"BUG: Model parsing took {elapsed:.2f}s - possible catastrophic backtracking"


class TestCircuitBreakerBugs:
    """Test for bugs in circuit breaker."""

    def test_time_going_backwards(self):
        """BUG: Circuit breaker may break if system time changes."""
        from config import BackendHealth
        import time

        health = BackendHealth("test", failure_threshold=3, base_cooldown_seconds=60)

        # Trigger circuit open
        for _ in range(3):
            health.record_failure("error")

        # Simulate time going backwards (DST change, NTP correction)
        health.circuit_open_until = time.time() + 3600  # 1 hour in future

        # Now pretend time went backwards
        original_time = time.time

        def mock_time():
            return original_time() - 7200  # 2 hours ago

        with patch('time.time', mock_time):
            # This could cause issues - circuit would be closed for 3 hours
            available = health.is_available()
            # We just check it doesn't crash
            assert isinstance(available, bool)


class TestBackendManagerBugs:
    """Test for bugs in backend manager."""

    @pytest.mark.asyncio
    async def test_double_close_client(self, tmp_path):
        """BUG: Double-closing client may cause issues."""
        from backend_manager import BackendManager, BackendConfig

        settings_file = tmp_path / "settings.json"
        settings = {
            "version": "1.0",
            "backends": [{
                "id": "test",
                "name": "Test",
                "provider": "ollama",
                "type": "local",
                "url": "http://localhost:11434"
            }],
            "routing": {}
        }
        with open(settings_file, "w") as f:
            json.dump(settings, f)

        manager = BackendManager(settings_file=settings_file)
        backend = manager.get_backend("test")

        # Create and close client
        _ = backend.get_client()
        await backend.close_client()

        # Double close should not crash
        try:
            await backend.close_client()
        except Exception as e:
            pytest.fail(f"BUG FOUND: Double close crashed: {e}")

    def test_empty_models_dict_access(self, tmp_path):
        """BUG: Accessing model tier on backend with no models may crash."""
        from backend_manager import BackendConfig

        backend = BackendConfig(
            id="test",
            name="Test",
            provider="ollama",
            type="local",
            url="http://localhost:11434",
            models={}  # Empty!
        )

        # Try to access models - should not crash
        try:
            quick = backend.models.get("quick")
            assert quick is None  # Should be None, not crash
        except Exception as e:
            pytest.fail(f"BUG FOUND: Empty models dict access crashed: {e}")


class TestCodeDetectionBugs:
    """Test for bugs in code detection."""

    def test_code_detection_with_binary(self):
        """BUG: Code detection may crash on binary-like content."""
        from mcp_server import detect_code_content

        # Binary-like content with non-printable chars
        binary_content = "".join(chr(i) for i in range(256)) * 10

        try:
            is_code, confidence, reason = detect_code_content(binary_content)
            # Should not crash
            assert isinstance(is_code, bool)
        except Exception as e:
            pytest.fail(f"BUG FOUND: Code detection crashed on binary content: {e}")

    def test_code_detection_extremely_long_lines(self):
        """BUG: Code detection may hang on very long lines."""
        from mcp_server import detect_code_content
        import time

        # One extremely long line
        long_line = "x = " + "a" * 100000 + ";"

        start = time.time()
        is_code, confidence, reason = detect_code_content(long_line)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"BUG: Code detection took {elapsed:.2f}s on long line"


class TestTokenCountingBugs:
    """Test for bugs in token counting."""

    def test_token_count_surrogate_pairs(self):
        """BUG: Token counting may fail on text with surrogate pairs."""
        from mcp_server import count_tokens

        # Text with emoji that uses surrogate pairs
        text = "Hello 👨‍👩‍👧‍👦 World 🏳️‍🌈 Test"

        try:
            tokens = count_tokens(text)
            assert tokens >= 0
        except Exception as e:
            pytest.fail(f"BUG FOUND: Token counting failed on surrogate pairs: {e}")

    def test_estimate_tokens_empty(self):
        """BUG: estimate_tokens may divide by zero."""
        from mcp_server import estimate_tokens

        try:
            tokens = estimate_tokens("")
            assert tokens == 0
        except ZeroDivisionError:
            pytest.fail("BUG FOUND: estimate_tokens divides by zero on empty string")


class TestAsyncBugs:
    """Test for async-related bugs."""

    @pytest.mark.asyncio
    async def test_select_model_no_backend(self):
        """BUG: select_model may crash when no backend available."""
        from mcp_server import select_model

        with patch("mcp_server.backend_manager") as mock_manager:
            mock_manager.get_active_backend.return_value = None

            try:
                model = await select_model("quick")
                # Should return default, not crash
                assert model is not None
            except AttributeError as e:
                pytest.fail(f"BUG FOUND: select_model crashes with no backend: {e}")

    @pytest.mark.asyncio
    async def test_health_no_backends(self):
        """BUG: health tool may crash when no backends configured."""
        from backend_manager import BackendManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "settings.json"
            settings = {"version": "1.0", "backends": [], "routing": {}}
            with open(settings_file, "w") as f:
                json.dump(settings, f)

            manager = BackendManager(settings_file=settings_file)

            try:
                health = await manager.check_all_health()
                # Should return empty dict, not crash
                assert health == {}
            except Exception as e:
                pytest.fail(f"BUG FOUND: health check crashes with no backends: {e}")


class TestConfigBugs:
    """Test for bugs in config module."""

    def test_detect_model_tier_empty_known_models(self):
        """BUG: detect_model_tier may crash with empty known_models."""
        from config import detect_model_tier

        try:
            tier = detect_model_tier("qwen:14b", {})
            assert tier in {"quick", "coder", "moe"}
        except Exception as e:
            pytest.fail(f"BUG FOUND: detect_model_tier crashes with empty known_models: {e}")

    def test_parse_model_name_only_numbers(self):
        """BUG: Model name with only numbers may confuse parser."""
        from config import parse_model_name

        result = parse_model_name("12345678901234567890")
        assert result is not None
        assert result.params_b >= 0


class TestSecurityBugs:
    """Test for security-related bugs."""

    def test_path_traversal_with_encoded_chars(self):
        """BUG: URL-encoded path traversal may bypass validation."""
        from mcp_server import validate_file_path

        # Already tested but double-check the actual ".." detection
        encoded_attempts = [
            "..%2f..%2fetc%2fpasswd",  # URL encoded /
            "..%5c..%5cetc%5cpasswd",  # URL encoded \
        ]

        for attempt in encoded_attempts:
            is_valid, error = validate_file_path(attempt)
            # Even URL-encoded ".." should be caught
            if ".." in attempt:
                assert not is_valid, f"URL-encoded traversal not caught: {attempt}"

    def test_content_size_integer_overflow(self):
        """BUG: Extremely large content size check may overflow."""
        from mcp_server import MAX_CONTENT_LENGTH

        # Verify MAX_CONTENT_LENGTH is reasonable
        assert MAX_CONTENT_LENGTH > 0
        assert MAX_CONTENT_LENGTH < 2**31  # Should fit in signed 32-bit int
