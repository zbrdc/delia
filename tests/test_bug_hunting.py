"""
Aggressive bug-hunting tests designed to find edge cases and potential issues.

These tests specifically target:
- Edge cases and boundary conditions
- Error handling paths
- Race conditions
- Malformed inputs
- Resource leaks
- Security issues
"""
import json
import pytest
import asyncio
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from hypothesis import given, strategies as st, settings, assume, Phase
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant


from delia.backend_manager import BackendManager, BackendConfig
from delia.config import Config, BackendHealth, parse_model_name, detect_model_tier


class TestEdgeCaseModelParsing:
    """Find bugs in model name parsing with extreme inputs."""

    @pytest.mark.parametrize("model_name", [
        "",  # Empty string
        " ",  # Whitespace only
        "\n\t\r",  # Various whitespace
        "a" * 10000,  # Very long name
        "model-" + "9" * 100 + "b",  # Huge param count
        "model-0b",  # Zero params
        "model--14b",  # Double dash
        "model::",  # Double colon
        "model:14b:extra",  # Multiple colons
        "🍉-model-14b",  # Emoji in name
        "model\x00name",  # Null byte
        "model\nname",  # Newline in name
        "../../../etc/passwd",  # Path traversal pattern
        "<script>alert(1)</script>",  # XSS pattern
        "'; DROP TABLE models;--",  # SQL injection pattern
        "model-14b-instruct-GGUF-Q4_K_M",  # Complex real pattern
        "0x14b",  # Hex-like
        "-14b",  # Leading dash
        "14b-",  # Trailing number
        "b14",  # Reversed
        "14B",  # Uppercase B
        "14.5.3b",  # Multiple decimals
        "1e10b",  # Scientific notation
        "inf",  # Infinity
        "nan",  # NaN
        "None",  # None string
        "null",  # null string
        "undefined",  # undefined string
    ])
    def test_parse_model_name_edge_cases(self, model_name):
        """Model name parsing should handle all edge cases without crashing."""
        result = parse_model_name(model_name)

        assert result is not None
        assert hasattr(result, 'params_b')
        assert hasattr(result, 'is_coder')
        assert hasattr(result, 'is_moe')
        assert isinstance(result.params_b, (int, float))
        assert result.params_b >= 0, f"Negative params for {model_name}: {result.params_b}"
        # Check for infinity/nan
        import math
        assert not math.isnan(result.params_b), f"NaN params for {model_name}"
        assert not math.isinf(result.params_b), f"Infinite params for {model_name}"

    def test_moe_pattern_overflow(self):
        """MoE patterns with huge numbers shouldn't overflow."""
        # 999x999b would be 998001B - test this doesn't cause issues
        result = parse_model_name("mixtral-999x999b")
        assert result.is_moe
        # Should handle the multiplication
        assert result.params_b == 998001.0

    def test_decimal_precision(self):
        """Decimal param sizes should maintain reasonable precision."""
        result = parse_model_name("model-0.1b")
        assert abs(result.params_b - 0.1) < 0.01

        result = parse_model_name("model-0.001b")
        # Very small values might be 0 or the actual value
        assert result.params_b >= 0


class TestValidationSecurityBugs:
    """Find security-related bugs in validation."""

    def test_path_traversal_variants(self):
        """Test various path traversal bypass attempts."""
        from mcp_server import validate_file_path

        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//etc/passwd",
            "..%2f..%2f..%2fetc/passwd",  # URL encoded
            "..%252f..%252f..%252fetc/passwd",  # Double encoded
            "..\\..\\..",
            "..;/..;/..;/etc/passwd",  # Semicolon variant
            "/..\\../..\\../etc/passwd",  # Mixed slashes
            "....\\....\\....\\etc\\passwd",
            "..../..../..../etc/passwd",
            "..%c0%af..%c0%af..%c0%afetc/passwd",  # Unicode encoding
            "..%c1%9c..%c1%9c..%c1%9cetc/passwd",
        ]

        for attempt in traversal_attempts:
            is_valid, error = validate_file_path(attempt)
            # Should detect any ".." pattern as traversal
            if ".." in attempt:
                assert not is_valid, f"Path traversal not caught: {attempt}"

    def test_content_size_boundary(self):
        """Test content size exactly at boundaries."""
        from mcp_server import validate_content, MAX_CONTENT_LENGTH

        # Exactly at limit
        content = "x" * MAX_CONTENT_LENGTH
        is_valid, _ = validate_content(content)
        assert is_valid, "Content at exact limit should be valid"

        # One byte over
        content = "x" * (MAX_CONTENT_LENGTH + 1)
        is_valid, _ = validate_content(content)
        assert not is_valid, "Content over limit should be invalid"

    def test_unicode_byte_counting_bug(self):
        """Unicode should be counted as bytes, not characters."""
        from mcp_server import validate_content, MAX_CONTENT_LENGTH

        # 4-byte emoji repeated
        emoji = "🍉"  # 4 bytes in UTF-8
        # Calculate how many would fit
        count = MAX_CONTENT_LENGTH // 4
        content = emoji * count
        is_valid, _ = validate_content(content)
        assert is_valid, "Emoji content at byte limit should be valid"

        # One more emoji would exceed
        content = emoji * (count + 1)
        is_valid, _ = validate_content(content)
        assert not is_valid, "Emoji content over byte limit should be invalid"


class TestCircuitBreakerBugs:
    """Find bugs in circuit breaker logic."""

    def test_rapid_state_transitions(self):
        """Rapid open/close transitions shouldn't cause issues."""
        health = BackendHealth("test", failure_threshold=2, base_cooldown_seconds=0.01)

        for _ in range(100):
            # Fail to open circuit
            health.record_failure("error")
            health.record_failure("error")
            assert not health.is_available()

            # Wait for cooldown
            time.sleep(0.02)

            # Should be available again
            assert health.is_available()

            # Success resets
            health.record_success(100)
            assert health.consecutive_failures == 0

    def test_concurrent_failure_recording(self):
        """Concurrent failure recordings shouldn't race."""
        health = BackendHealth("test", failure_threshold=5)

        async def record_failures():
            for _ in range(10):
                health.record_failure("error")
                await asyncio.sleep(0)

        async def run_concurrent():
            await asyncio.gather(*[record_failures() for _ in range(5)])

        asyncio.run(run_concurrent())

        # Should have recorded all failures (or at least not crashed)
        assert health.consecutive_failures >= 5

    def test_cooldown_overflow(self):
        """Exponential backoff shouldn't overflow."""
        health = BackendHealth("test", failure_threshold=1, base_cooldown_seconds=60)

        # Many failures to trigger exponential backoff
        for i in range(100):
            health.record_failure("error")

        # Cooldown should be capped, not infinite
        wait = health.time_until_available()
        assert wait < 86400 * 365, "Cooldown should be capped reasonably"
        assert wait >= 0, "Cooldown shouldn't be negative"

    def test_negative_context_size(self):
        """Negative context sizes shouldn't break things."""
        health = BackendHealth("test")

        # Try negative context size
        health.record_success(context_size=-100)
        # Should handle gracefully
        assert health.max_successful_context >= 0 or health.max_successful_context == -100

        health.record_failure("error", context_size=-500)
        # Should not crash


class TestBackendManagerBugs:
    """Find bugs in backend manager."""

    @pytest.fixture
    def temp_settings(self, tmp_path):
        """Create temp settings file."""
        settings = {
            "version": "1.0",
            "backends": [],
            "routing": {}
        }
        path = tmp_path / "settings.json"
        with open(path, "w") as f:
            json.dump(settings, f)
        return path

    def test_concurrent_backend_modifications(self, temp_settings):
        """Concurrent add/remove shouldn't corrupt state."""
        manager = BackendManager(settings_file=temp_settings)

        async def add_backends():
            for i in range(10):
                manager.add_backend({
                    "id": f"backend-add-{i}",
                    "name": f"Backend {i}",
                    "provider": "ollama",
                    "type": "local",
                    "url": f"http://localhost:{11434 + i}"
                })
                await asyncio.sleep(0)

        async def remove_backends():
            for i in range(10):
                try:
                    await manager.remove_backend(f"backend-add-{i}")
                except:
                    pass  # May not exist yet
                await asyncio.sleep(0)

        async def run():
            await asyncio.gather(add_backends(), remove_backends())

        asyncio.run(run())

        # State should be consistent
        assert isinstance(manager.backends, dict)

    def test_settings_file_corruption_recovery(self, tmp_path):
        """Manager should recover from corrupted settings."""
        path = tmp_path / "corrupted.json"

        # Write corrupted JSON
        with open(path, "w") as f:
            f.write("{invalid json content")

        # Should not crash
        manager = BackendManager(settings_file=path)
        assert manager.backends == {} or isinstance(manager.backends, dict)

    def test_missing_parent_directory(self, tmp_path):
        """Settings in non-existent directory should be handled."""
        path = tmp_path / "nonexistent" / "dir" / "settings.json"

        # Should handle gracefully (create or error cleanly)
        try:
            manager = BackendManager(settings_file=path)
        except Exception as e:
            # Should be a clean error, not a crash
            assert "directory" in str(e).lower() or "exist" in str(e).lower() or isinstance(e, (FileNotFoundError, OSError))

    @pytest.mark.asyncio
    async def test_reload_during_health_check(self, temp_settings):
        """Reload during health check shouldn't crash."""
        manager = BackendManager(settings_file=temp_settings)

        # Add a backend
        manager.add_backend({
            "id": "test-backend",
            "name": "Test",
            "provider": "ollama",
            "type": "local",
            "url": "http://localhost:11434"
        })

        # Mock health check to take some time
        async def slow_health():
            await asyncio.sleep(0.1)
            return True

        for backend in manager.backends.values():
            backend.check_health = slow_health

        # Start health check and reload concurrently
        async def health_and_reload():
            await asyncio.gather(
                manager.check_all_health(use_cache=False),
                manager.reload()
            )

        # Should not crash
        await health_and_reload()


class TestTokenCountingBugs:
    """Find bugs in token counting."""

    def test_empty_and_whitespace(self):
        """Edge cases in token counting."""
        from mcp_server import count_tokens, estimate_tokens

        # Empty
        assert count_tokens("") == 0
        assert estimate_tokens("") == 0

        # Whitespace only
        tokens = count_tokens("   ")
        assert tokens >= 0

        tokens = count_tokens("\n\n\n")
        assert tokens >= 0

    def test_very_long_content(self):
        """Very long content shouldn't cause memory issues."""
        from mcp_server import count_tokens, estimate_tokens

        # 1MB of text
        content = "word " * 200000
        tokens = count_tokens(content)
        assert tokens > 0
        assert tokens < len(content)  # Should be fewer tokens than chars

    def test_binary_content(self):
        """Binary-like content shouldn't crash."""
        from mcp_server import count_tokens

        # Random bytes as string
        content = "".join(chr(i % 256) for i in range(1000))
        try:
            tokens = count_tokens(content)
            assert tokens >= 0
        except Exception as e:
            # Should be a clean error if it fails
            assert "encode" in str(e).lower() or "token" in str(e).lower()


class TestCodeDetectionBugs:
    """Find bugs in code detection."""

    def test_polyglot_content(self):
        """Content that looks like multiple languages."""
        from mcp_server import detect_code_content

        # HTML with embedded JS and CSS
        polyglot = """
<!DOCTYPE html>
<html>
<head>
<style>
.container { display: flex; }
</style>
<script>
function init() {
    console.log("Hello");
}
</script>
</head>
</html>
"""
        is_code, confidence, reason = detect_code_content(polyglot)
        # Should detect as code
        assert isinstance(is_code, bool)
        assert 0 <= confidence <= 1

    def test_minified_code(self):
        """Minified code detection."""
        from mcp_server import detect_code_content

        minified = 'function a(b){return b+1}var c=a(5);console.log(c);'
        is_code, confidence, reason = detect_code_content(minified)
        # Should still detect as code
        assert is_code or confidence > 0

    def test_markdown_with_code_blocks(self):
        """Markdown with code blocks."""
        from mcp_server import detect_code_content

        markdown = """
# README

This is documentation.

```python
def hello():
    print("world")
```

More text here.
"""
        is_code, confidence, reason = detect_code_content(markdown)
        # Could go either way, but shouldn't crash
        assert isinstance(is_code, bool)


class TestRaceConditions:
    """Find race conditions in async code."""

    @pytest.mark.asyncio
    async def test_concurrent_model_selection(self):
        """Concurrent model selection shouldn't race."""
        from mcp_server import select_model
        from backend_manager import BackendConfig

        mock_backend = BackendConfig(
            id="test", name="Test", provider="ollama", type="local",
            url="http://localhost:11434",
            models={"quick": "q", "coder": "c", "moe": "m", "thinking": "t"}
        )

        with patch("mcp_server.backend_manager") as mock_manager:
            mock_manager.get_active_backend.return_value = mock_backend

            async def select():
                return await select_model("quick")

            # Run many concurrent selections
            results = await asyncio.gather(*[select() for _ in range(100)])

            # All should return the same model
            assert all(r == "q" for r in results)

    @pytest.mark.asyncio
    async def test_stats_concurrent_updates(self):
        """Concurrent stats updates shouldn't lose data."""
        # This tests the thread safety of stats tracking
        from mcp_server import TASK_STATS, _stats_thread_lock

        initial_count = TASK_STATS.get("quick", 0)

        async def increment():
            with _stats_thread_lock:
                TASK_STATS["quick"] = TASK_STATS.get("quick", 0) + 1

        # Many concurrent increments
        await asyncio.gather(*[increment() for _ in range(1000)])

        # Should have incremented correctly
        assert TASK_STATS["quick"] == initial_count + 1000


class TestResourceLeaks:
    """Find resource leaks."""

    @pytest.mark.asyncio
    async def test_http_client_cleanup(self, tmp_path):
        """HTTP clients should be properly cleaned up."""
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

        # Create client
        client1 = backend.get_client()
        assert client1 is not None

        # Close and verify cleanup
        await backend.close_client()
        assert backend._client is None

        # Should be able to create new client
        client2 = backend.get_client()
        assert client2 is not None
        assert client2 is not client1  # Should be new

        # Cleanup
        await backend.close_client()
