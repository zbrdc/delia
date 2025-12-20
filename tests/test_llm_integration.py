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
Integration tests for actual LLM calls.

These tests require a running LLM backend (llama.cpp or Ollama).
Skip with: pytest tests/test_llm_integration.py -v -m "not integration"

Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_llm_integration.py -v
"""

import os
import sys
import json
import asyncio
import tempfile
from pathlib import Path

import pytest
import httpx


# Check if backends are available
def is_llamacpp_available():
    try:
        r = httpx.get("http://localhost:8080/health", timeout=2)
        return r.status_code == 200
    except:
        return False


def is_ollama_available():
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except:
        return False


LLAMACPP_AVAILABLE = is_llamacpp_available()
OLLAMA_AVAILABLE = is_ollama_available()
ANY_BACKEND_AVAILABLE = LLAMACPP_AVAILABLE or OLLAMA_AVAILABLE


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Use a temp directory for test data."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    # Clear cached modules
    modules_to_clear = ["delia.paths", "delia.config", "delia.backend_manager", "delia.mcp_server", "delia"]
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) or mod == m for m in modules_to_clear):
            del sys.modules[mod]

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


@pytest.mark.skipif(not LLAMACPP_AVAILABLE, reason="llama.cpp not running")
class TestLlamaCppDirect:
    """Test direct llama.cpp API calls."""

    def test_health_endpoint(self):
        """llama.cpp health endpoint should respond."""
        r = httpx.get("http://localhost:8080/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"

    def test_models_endpoint(self):
        """llama.cpp should list available models."""
        r = httpx.get("http://localhost:8080/v1/models", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_simple_completion(self):
        """llama.cpp should generate a completion."""
        # First get available models
        models_resp = httpx.get("http://localhost:8080/v1/models", timeout=5)
        models = models_resp.json()["data"]
        loaded_models = [m["id"] for m in models if m.get("status", {}).get("value") == "loaded"]

        if not loaded_models:
            pytest.skip("No loaded model in llama.cpp")

        model_id = loaded_models[0]

        r = httpx.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "max_tokens": 100,  # More tokens to let model finish
                "temperature": 0,
            },
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

        msg = data["choices"][0]["message"]
        # Check content or reasoning_content (for thinking models like Qwen3)
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        assert len(content) > 0  # Model should produce some output


@pytest.mark.skipif(not ANY_BACKEND_AVAILABLE, reason="No LLM backend available")
class TestBackendManager:
    """Test backend manager with real backends."""

    def test_detect_available_backend(self, tmp_path):
        """Backend manager should detect available backends."""
        from delia.backend_manager import BackendManager

        # Create settings with our backends
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "llamacpp-test",
                    "name": "LlamaCpp Test",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "Qwen3-4B-Q4_K_M"}
                }
            ],
            "routing": {"prefer_local": True}
        }

        settings_file = tmp_path / "settings.json"
        with open(settings_file, "w") as f:
            json.dump(settings, f)

        manager = BackendManager(settings_file=settings_file)

        # Should have our backend
        assert "llamacpp-test" in manager.backends
        backend = manager.get_backend("llamacpp-test")
        assert backend.provider == "llamacpp"

    @pytest.mark.asyncio
    async def test_backend_health_check(self, tmp_path):
        """Backend manager should check backend health."""
        from delia.backend_manager import BackendManager
        from unittest.mock import AsyncMock, patch

        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "llamacpp-test",
                    "name": "LlamaCpp Test",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {}
                }
            ],
            "routing": {"prefer_local": True}
        }

        settings_file = tmp_path / "settings.json"
        with open(settings_file, "w") as f:
            json.dump(settings, f)

        manager = BackendManager(settings_file=settings_file)

        # Mock the HTTP call to return 200 OK
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"status": "ok"}
            
            # Check health of all backends
            await manager.check_all_health()

        # Backend should be marked as available
        backend = manager.get_backend("llamacpp-test")
        assert backend is not None
        assert backend._available is True


@pytest.mark.skipif(not LLAMACPP_AVAILABLE, reason="llama.cpp not running")
class TestDelegateFunction:
    """Test the delegate function with real LLM calls."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings for all tests in this class."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "llamacpp-test",
                    "name": "LlamaCpp Test",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {
                        "quick": "Qwen3-4B-Q4_K_M",
                        "coder": "Qwen3-4B-Q4_K_M",
                        "moe": "Qwen3-4B-Q4_K_M",
                        "thinking": "Qwen3-4B-Q4_K_M"
                    }
                }
            ],
            "routing": {"prefer_local": True}
        }

        from delia import paths
        paths.ensure_directories()
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_delegate_simple_task(self):
        """delegate() should successfully call the LLM."""
        from delia import mcp_server

        # Reload backend manager to pick up settings
        await mcp_server.backend_manager.reload()

        # Call delegate via .fn (the actual function, not the FunctionTool wrapper)
        result = await mcp_server.delegate.fn(
            task="quick",
            content="What is 2+2? Reply with just the number.",
            model="quick"
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_delegate_code_task(self):
        """delegate() should handle code generation tasks."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        # Call delegate via .fn
        result = await mcp_server.delegate.fn(
            task="generate",
            content="Write a Python function that adds two numbers. Just the function, no explanation.",
            model="coder",
            language="python"
        )

        assert result is not None
        assert len(result) > 0


@pytest.mark.skipif(not LLAMACPP_AVAILABLE, reason="llama.cpp not running")
class TestHealthAndModels:
    """Test health and models MCP tools."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings for all tests in this class."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "llamacpp-test",
                    "name": "LlamaCpp Test",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "Qwen3-4B-Q4_K_M"}
                }
            ],
            "routing": {"prefer_local": True}
        }

        from delia import paths
        paths.ensure_directories()
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_health_tool(self):
        """health() tool should return backend status."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.health.fn()

        assert result is not None
        # Should return some status info
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_models_tool(self):
        """models() tool should return available models."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.models.fn()

        assert result is not None
        assert len(result) > 0


@pytest.mark.skipif(not LLAMACPP_AVAILABLE, reason="llama.cpp not running")
class TestThinkFunction:
    """Test the think function with real LLM calls."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings for all tests in this class."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "llamacpp-test",
                    "name": "LlamaCpp Test",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {
                        "quick": "Qwen3-4B-Q4_K_M",
                        "coder": "Qwen3-4B-Q4_K_M",
                        "moe": "Qwen3-4B-Q4_K_M",
                        "thinking": "Qwen3-4B-Q4_K_M"
                    }
                }
            ],
            "routing": {"prefer_local": True}
        }

        from delia import paths
        paths.ensure_directories()
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_think_simple(self):
        """think() should process reasoning tasks."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.think.fn(
            problem="What is the sum of 5 and 7?",
            depth="quick"
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
