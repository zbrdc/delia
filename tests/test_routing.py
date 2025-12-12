"""Tests for model routing logic."""

import pytest
from unittest.mock import MagicMock, patch
import config
from backend_manager import BackendConfig


class TestRouting:
    """Test model selection and routing."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.mock_backend = BackendConfig(
            id="test-backend",
            name="Test",
            provider="ollama",
            type="local",
            url="http://localhost",
            models={
                "quick": "qwen-quick",
                "coder": "qwen-coder",
                "moe": "qwen-moe",
                "thinking": "qwen-think"
            }
        )

    @pytest.mark.asyncio
    async def test_select_model_defaults(self):
        """Test default model selection."""
        from mcp_server import select_model

        with patch("mcp_server.backend_manager") as mock_manager:
            mock_manager.get_active_backend.return_value = self.mock_backend

            # Simple task -> quick
            model = await select_model("quick")
            assert model == "qwen-quick"

            # Summarize -> quick
            model = await select_model("summarize")
            assert model == "qwen-quick"

    @pytest.mark.asyncio
    async def test_select_model_tasks(self):
        """Test task-based routing."""
        from mcp_server import select_model

        with patch("mcp_server.backend_manager") as mock_manager:
            mock_manager.get_active_backend.return_value = self.mock_backend

            # Plan -> moe
            model = await select_model("plan")
            assert model == "qwen-moe"

            # Critique -> moe
            model = await select_model("critique")
            assert model == "qwen-moe"

            # Generate code -> coder (with code content)
            code_content = """
def hello():
    print("Hello, world!")
    return True

class Test:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
"""
            model = await select_model("generate", content=code_content, content_size=len(code_content))
            assert model == "qwen-coder"

    @pytest.mark.asyncio
    async def test_select_model_override(self):
        """Test explicit model overrides."""
        from mcp_server import select_model

        with patch("mcp_server.backend_manager") as mock_manager:
            mock_manager.get_active_backend.return_value = self.mock_backend

            # Override with "moe"
            model = await select_model("quick", model_override="moe")
            assert model == "qwen-moe"

            # Override with "coder"
            model = await select_model("plan", model_override="coder")
            assert model == "qwen-coder"

            # Override with "thinking"
            model = await select_model("quick", model_override="thinking")
            assert model == "qwen-think"

    def test_detect_model_tier(self):
        """Test model tier detection."""
        from mcp_server import detect_model_tier

        # Known models
        known = {
            "quick": "qwen-quick",
            "coder": "qwen-coder"
        }

        assert detect_model_tier("qwen-quick", known) == "quick"
        assert detect_model_tier("qwen-coder", known) == "coder"

        # Unknown models - parsing logic
        assert detect_model_tier("mixtral-8x7b") == "moe"
        assert detect_model_tier("llama-3-70b") == "moe"
        assert detect_model_tier("deepseek-coder-33b") == "moe"  # >= 30B -> moe
        assert detect_model_tier("deepseek-coder-6.7b") == "coder"
        assert detect_model_tier("tiny-llama-1b") == "quick"


class TestCircuitBreaker:
    """Test circuit breaker logic."""

    def test_circuit_breaker_logic(self):
        from config import BackendHealth
        import time

        health = BackendHealth("test")
        assert health.is_available() is True

        # Simulate failures
        health.record_failure("timeout")
        assert health.is_available() is True  # Threshold is 3

        health.record_failure("timeout")
        health.record_failure("timeout")

        # Circuit should open
        assert health.is_available() is False

        # Wait time should be positive
        assert health.time_until_available() > 0

        # Simulate cooldown passing
        health.circuit_open_until = time.time() - 1
        assert health.is_available() is True

        # Success should reset
        health.record_success(100)
        assert health.consecutive_failures == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
