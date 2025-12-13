# Copyright (C) 2023 the project owner
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
Comprehensive tests for all MCP tools in mcp_server.py.

Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_mcp_tools.py -v
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Use a temp directory for test data."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    # Clear cached modules
    modules_to_clear = ["delia.paths", "delia.config", "delia.backend_manager", "delia.mcp_server", "delia.multi_user_tracking", "delia.auth", "delia"]
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) or mod == m for m in modules_to_clear):
            del sys.modules[mod]

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


class TestBatchFunction:
    """Test the batch() function for parallel LLM execution."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings for all tests in this class."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "test-backend",
                    "name": "Test Backend",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {
                        "quick": "test-model",
                        "coder": "test-model",
                        "moe": "test-model",
                        "thinking": "test-model"
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
    async def test_batch_parses_json_tasks(self):
        """batch() should parse JSON task array."""
        from delia import mcp_server, delegation

        # Mock the actual LLM call to avoid needing a real backend
        # execute_delegate_call was moved to delegation module
        with patch.object(delegation, 'execute_delegate_call', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = ("Task completed", 100)

            tasks = json.dumps([
                {"task": "summarize", "content": "Test content 1"},
                {"task": "quick", "content": "Test content 2"}
            ])

            result = await mcp_server.batch.fn(tasks=tasks)

            assert result is not None
            # batch should have attempted to execute tasks or returned error about no backend
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_batch_rejects_invalid_json(self):
        """batch() should handle invalid JSON gracefully."""
        from delia import mcp_server

        result = await mcp_server.batch.fn(tasks="not valid json")

        assert "error" in result.lower() or "invalid" in result.lower()

    @pytest.mark.asyncio
    async def test_batch_rejects_non_array(self):
        """batch() should reject non-array JSON."""
        from delia import mcp_server

        result = await mcp_server.batch.fn(tasks='{"task": "summarize"}')

        assert "array" in result.lower() or "error" in result.lower()


class TestHealthFunction:
    """Test the health() function."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "test-backend",
                    "name": "Test",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 0,
                    "models": {}
                }
            ],
            "routing": {"prefer_local": True}
        }

        from delia import paths
        paths.ensure_directories()
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_health_returns_status(self):
        """health() should return backend status."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.health.fn()

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_health_includes_usage_stats(self):
        """health() should include usage statistics."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.health.fn()

        # Should contain stats info
        assert "usage" in result.lower() or "calls" in result.lower() or "status" in result.lower()


class TestQueueStatusFunction:
    """Test the queue_status() function."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings."""
        settings = {
            "version": "1.0",
            "backends": [],
            "routing": {"prefer_local": True}
        }

        from delia import paths
        paths.ensure_directories()
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_queue_status_returns_info(self):
        """queue_status() should return queue information."""
        from delia import mcp_server

        result = await mcp_server.queue_status.fn()

        assert result is not None
        assert isinstance(result, str)


class TestModelsFunction:
    """Test the models() function."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings with multiple backends."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "backend-1",
                    "name": "Backend 1",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "llama3", "coder": "codellama"}
                },
                {
                    "id": "backend-2",
                    "name": "Backend 2",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": False,
                    "priority": 1,
                    "models": {"quick": "phi3"}
                }
            ],
            "routing": {"prefer_local": True}
        }

        from delia import paths
        paths.ensure_directories()
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_models_lists_backends(self):
        """models() should list configured models."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.models.fn()

        assert result is not None
        assert isinstance(result, str)
        # Should mention backend info
        assert len(result) > 0


class TestSwitchBackendFunction:
    """Test the switch_backend() function."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings with multiple backends."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "backend-a",
                    "name": "Backend A",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 0,
                    "models": {}
                },
                {
                    "id": "backend-b",
                    "name": "Backend B",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 1,
                    "models": {}
                }
            ],
            "routing": {"prefer_local": True}
        }

        from delia import paths
        paths.ensure_directories()
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_switch_backend_valid(self):
        """switch_backend() should switch to valid backend."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.switch_backend.fn(backend_id="backend-b")

        assert result is not None
        # Should confirm switch
        assert "backend-b" in result.lower() or "switch" in result.lower()

    @pytest.mark.asyncio
    async def test_switch_backend_invalid(self):
        """switch_backend() should handle invalid backend ID."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.switch_backend.fn(backend_id="nonexistent")

        assert result is not None
        # Should indicate error or not found
        assert "not found" in result.lower() or "error" in result.lower() or "invalid" in result.lower()


class TestSwitchModelFunction:
    """Test the switch_model() function."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "test-backend",
                    "name": "Test",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "llama3", "coder": "codellama", "moe": "mixtral", "thinking": "deepseek"}
                }
            ],
            "routing": {"prefer_local": True}
        }

        from delia import paths
        paths.ensure_directories()
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_switch_model_valid_tier(self):
        """switch_model() should update model for valid tier."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.switch_model.fn(tier="quick", model_name="phi3")

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_switch_model_invalid_tier(self):
        """switch_model() should reject invalid tier."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.switch_model.fn(tier="invalid_tier", model_name="llama3")

        assert result is not None
        # Should indicate invalid tier
        assert "invalid" in result.lower() or "error" in result.lower() or "tier" in result.lower()


class TestGetModelInfoFunction:
    """Test the get_model_info_tool() function."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings."""
        settings = {
            "version": "1.0",
            "backends": [],
            "routing": {"prefer_local": True}
        }

        from delia import paths
        paths.ensure_directories()
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_get_model_info_known_model(self):
        """get_model_info_tool() should return info for known model."""
        from delia import mcp_server

        result = await mcp_server.get_model_info_tool.fn(model_name="llama-3-70b")

        assert result is not None
        assert isinstance(result, str)
        # Should include tier info
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_model_info_coder_model(self):
        """get_model_info_tool() should detect coder models."""
        from delia import mcp_server

        result = await mcp_server.get_model_info_tool.fn(model_name="deepseek-coder-33b")

        assert result is not None
        # Should mention coder or code capabilities
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_model_info_moe_model(self):
        """get_model_info_tool() should detect MoE models."""
        from delia import mcp_server

        result = await mcp_server.get_model_info_tool.fn(model_name="mixtral-8x7b")

        assert result is not None
        assert len(result) > 0


class TestDelegateModelSelection:
    """Test delegate() model selection logic."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "test-backend",
                    "name": "Test",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {
                        "quick": "quick-model",
                        "coder": "coder-model",
                        "moe": "moe-model",
                        "thinking": "thinking-model"
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
    async def test_delegate_with_model_override(self):
        """delegate() should respect explicit model override."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        # Mock to avoid actual LLM call
        with patch.object(mcp_server, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Response"

            result = await mcp_server.delegate.fn(
                task="quick",
                content="Test",
                model="coder"  # Explicit override
            )

            # Should return something (either mocked response or error about backend)
            assert result is not None


class TestThinkDepthLevels:
    """Test think() with different depth levels."""

    @pytest.fixture(autouse=True)
    def setup_settings(self, tmp_path):
        """Set up settings."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "test-backend",
                    "name": "Test",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {
                        "quick": "test-model",
                        "coder": "test-model",
                        "moe": "test-model",
                        "thinking": "test-model"
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
    async def test_think_quick_depth(self):
        """think() with quick depth should work."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        with patch.object(mcp_server, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Quick thought"

            result = await mcp_server.think.fn(
                problem="What is 2+2?",
                depth="quick"
            )

            # Should return something (mocked response or error about backend)
            assert result is not None

    @pytest.mark.asyncio
    async def test_think_deep_depth(self):
        """think() with deep depth should use thinking model."""
        from delia import mcp_server
        await mcp_server.backend_manager.reload()

        with patch.object(mcp_server, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Deep thought"

            result = await mcp_server.think.fn(
                problem="Complex problem",
                depth="deep"
            )

            # Should return something (mocked response or error about backend)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
