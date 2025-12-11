import unittest
import asyncio
from unittest.mock import MagicMock, patch
import config
from mcp_server import select_model, detect_model_tier
from backend_manager import BackendManager, BackendConfig

class TestRouting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Mock backend manager
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
        
    @patch("mcp_server.backend_manager")
    async def test_select_model_defaults(self, mock_manager):
        """Test default model selection."""
        mock_manager.get_active_backend.return_value = self.mock_backend
        
        # Simple task -> quick
        model = await select_model("quick")
        self.assertEqual(model, "qwen-quick")
        
        # Summarize -> quick
        model = await select_model("summarize")
        self.assertEqual(model, "qwen-quick")

    @patch("mcp_server.backend_manager")
    async def test_select_model_tasks(self, mock_manager):
        """Test task-based routing."""
        mock_manager.get_active_backend.return_value = self.mock_backend
        
        # Plan -> moe
        model = await select_model("plan")
        self.assertEqual(model, "qwen-moe")
        
        # Critique -> moe
        model = await select_model("critique")
        self.assertEqual(model, "qwen-moe")
        
        # Generate code -> coder (assuming defaults in config.py)
        # We need to ensure config.coder_tasks includes 'generate'
        model = await select_model("generate")
        # Note: select_model logic checks detect_code_content for coder tasks
        # If content is empty, it might fall back to quick or coder depending on logic
        # In current mcp_server.py: 
        # Priority 4: Code-focused tasks - check if content is actually code
        # if task_type in config.coder_tasks and code_detection: ...
        # If no content, code_detection is None.
        # Let's provide content
        
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
        self.assertEqual(model, "qwen-coder")

    @patch("mcp_server.backend_manager")
    async def test_select_model_override(self, mock_manager):
        """Test explicit model overrides."""
        mock_manager.get_active_backend.return_value = self.mock_backend
        
        # Override with "moe"
        model = await select_model("quick", model_override="moe")
        self.assertEqual(model, "qwen-moe")
        
        # Override with "coder"
        model = await select_model("plan", model_override="coder")
        self.assertEqual(model, "qwen-coder")
        
        # Override with "thinking"
        model = await select_model("quick", model_override="thinking")
        self.assertEqual(model, "qwen-think")

    def test_detect_model_tier(self):
        """Test model tier detection."""
        # Known models
        known = {
            "quick": "qwen-quick",
            "coder": "qwen-coder"
        }
        
        self.assertEqual(detect_model_tier("qwen-quick", known), "quick")
        self.assertEqual(detect_model_tier("qwen-coder", known), "coder")
        
        # Unknown models - parsing logic
        self.assertEqual(detect_model_tier("mixtral-8x7b"), "moe")
        self.assertEqual(detect_model_tier("llama-3-70b"), "moe")
        self.assertEqual(detect_model_tier("deepseek-coder-33b"), "moe") # >= 30B -> moe
        self.assertEqual(detect_model_tier("deepseek-coder-6.7b"), "coder")
        self.assertEqual(detect_model_tier("tiny-llama-1b"), "quick")

class TestCircuitBreaker(unittest.TestCase):
    def test_circuit_breaker_logic(self):
        from config import BackendHealth
        import time
        
        health = BackendHealth("test")
        self.assertTrue(health.is_available())
        
        # Simulate failures
        health.record_failure("timeout")
        self.assertTrue(health.is_available()) # Threshold is 3
        
        health.record_failure("timeout")
        health.record_failure("timeout")
        
        # Circuit should open
        self.assertFalse(health.is_available())
        
        # Wait time should be positive
        self.assertGreater(health.time_until_available(), 0)
        
        # Simulate cooldown passing (mocking time would be better but simple check is okay)
        health.circuit_open_until = time.time() - 1
        self.assertTrue(health.is_available())
        
        # Success should reset
        health.record_success(100)
        self.assertEqual(health.consecutive_failures, 0)

if __name__ == "__main__":
    unittest.main()
