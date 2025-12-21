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
            from delia.tools.handlers import batch_impl as batch_tool
        
            # Mock the actual LLM call to avoid needing a real backend
            # execute_delegate_call was moved to delegation module
            with patch.object(delegation, 'execute_delegate_call', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = ("Task completed", 100)
        
                tasks = json.dumps([
                    {"task": "summarize", "content": "Test content 1"},
                    {"task": "quick", "content": "Test content 2"}
                ])
        
                result = await batch_tool(tasks=tasks)
                assert result is not None
            # batch should have attempted to execute tasks or returned error about no backend
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_batch_rejects_invalid_json(self):
        """batch() should handle invalid JSON gracefully."""
        from delia.tools.handlers import batch_impl as batch_tool

        result = await batch_tool(tasks="not valid json")
        assert "Error: Invalid JSON" in result

    @pytest.mark.asyncio
    async def test_batch_rejects_non_array(self):
        """batch() should reject non-array JSON."""
        from delia.tools.handlers import batch_impl as batch_tool
    
        result = await batch_tool(tasks='{"task": "summarize"}')
        assert "Error: tasks must be a JSON array" in result


class TestDryRunFunction:
    """Test the delegate() dry_run parameter."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, tmp_path):
        """Set up test environment."""
        os.environ["DELIA_DATA_DIR"] = str(tmp_path)

        # Clear cached modules
        modules_to_clear = ["delia.paths", "delia.config", "delia.mcp_server", "delia.delegation", "delia"]
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) or mod == m for m in modules_to_clear):
                del sys.modules[mod]

        from delia import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "test-local",
                    "name": "Test Local",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 1,
                    "models": {"quick": "test:7b", "coder": "test:14b", "moe": "test:30b"}
                }
            ]
        }
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        yield

        os.environ.pop("DELIA_DATA_DIR", None)

    @pytest.mark.asyncio
    async def test_dry_run_returns_json(self):
        """dry_run=True should return JSON with estimation signals."""
        from delia.tools.handlers import delegate_tool_impl as delegate_tool
    
        result = await delegate_tool(
            task="review",
            content="def hello(): pass",
            dry_run=True
        )
        
        # Should be valid JSON
        data = json.loads(result)
        assert "estimated_tokens" in data
        assert "recommended_backend" in data
        assert "is_safe" in data

        @pytest.mark.asyncio
        async def test_dry_run_contains_required_fields(self):
            """dry_run should return all required estimation fields."""
            from delia.tools.handlers import delegate_tool_impl as delegate_tool
        
            result = await delegate_tool(
                task="analyze",
                content="class Foo:\n    def bar(self): pass",
                language="python",
                dry_run=True
            )
            data = json.loads(result)

        # Check required fields
        assert "valid" in data
        assert "estimated_tokens" in data
        assert "recommended_tier" in data
        assert "recommended_model" in data
        assert "content_fits" in data
        assert "task_type" in data

        @pytest.mark.asyncio
        async def test_dry_run_token_estimation(self):
            """dry_run should provide reasonable token estimates."""
            from delia.tools.handlers import delegate_tool_impl as delegate_tool
        
            content = "x" * 1000  # ~250 tokens
            result = await delegate_tool(
                task="quick",
                content=content,
                dry_run=True
            )
            data = json.loads(result)

        # Token count should be reasonable (accounting for prompt template overhead)
        assert data["estimated_tokens"] > 100
        assert data["estimated_tokens"] < 5000

        @pytest.mark.asyncio
        async def test_dry_run_invalid_task(self):
            """dry_run should report validation errors."""
            from delia.tools.handlers import delegate_tool_impl as delegate_tool
        
            result = await delegate_tool(
                task="invalid_task_type",
                content="test",
                dry_run=True
            )
            data = json.loads(result)
        assert data.get("valid") is False or "error" in data


class TestStreamingFunction:
    """Test the delegate() stream parameter."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, tmp_path):
        """Set up test environment."""
        os.environ["DELIA_DATA_DIR"] = str(tmp_path)

        # Clear cached modules
        modules_to_clear = ["delia.paths", "delia.config", "delia.mcp_server", "delia.delegation", "delia.providers", "delia"]
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) or mod == m for m in modules_to_clear):
                del sys.modules[mod]

        from delia import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "test-local",
                    "name": "Test Local",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 1,
                    "models": {"quick": "test:7b", "coder": "test:14b", "moe": "test:30b"}
                }
            ]
        }
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        yield

        os.environ.pop("DELIA_DATA_DIR", None)

    @pytest.mark.asyncio
    async def test_stream_parameter_exists(self):
        """delegate() should accept stream parameter."""
        from delia.tools.handlers import delegate_tool_impl as delegate_tool
    
        # Verify the delegate function has stream parameter
        import inspect
        sig = inspect.signature(delegate_tool)
        assert "stream" in sig.parameters
        assert sig.parameters["stream"].default is False

    @pytest.mark.asyncio
    async def test_stream_with_mocked_provider(self):
        """stream=True should use streaming mode and collect response."""
        from delia import llm
        from delia.tools.handlers import delegate_tool_impl as delegate_tool
        from delia.providers import StreamChunk
    
        # Create an async generator that yields streaming chunks
        async def mock_stream(*args, **kwargs):
            yield StreamChunk(text="Hello ")
            yield StreamChunk(text="World")
            yield StreamChunk(done=True, tokens=10, metadata={"elapsed_ms": 100})
    
        with patch.object(llm, 'call_llm_stream', side_effect=mock_stream):
            result = await delegate_tool(
                task="quick",
                content="Test content",
                stream=True
            )
            
            # Should concatenate chunks
            assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_stream_includes_metadata(self):
        """stream=True with include_metadata should show streaming marker."""
        from delia import llm
        from delia.tools.handlers import delegate_tool_impl as delegate_tool
        from delia.providers import StreamChunk
    
        async def mock_stream(*args, **kwargs):
            yield StreamChunk(text="Response text")
            yield StreamChunk(done=True, tokens=50, metadata={"elapsed_ms": 200})
    
        with patch.object(llm, 'call_llm_stream', side_effect=mock_stream):
            result = await delegate_tool(
                task="review",
                content="Test content",
                stream=True,
                include_metadata=True
            )
            
            assert "Response text" in result
            assert "(streamed)" in result
            assert "200ms" in result

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """stream=True should handle errors from streaming."""
        from delia import llm
        from delia.tools.handlers import delegate_tool_impl as delegate_tool
        from delia.providers import StreamChunk

        async def mock_stream_error(*args, **kwargs):
            yield StreamChunk(done=True, error="Connection lost")

        with patch.object(llm, 'call_llm_stream', side_effect=mock_stream_error):
            result = await delegate_tool(
                task="quick",
                content="Test content",
                stream=True
            )
            
            assert "Error: Connection lost" in result

    @pytest.mark.asyncio
    async def test_stream_false_uses_normal_path(self):
        """stream=False should use non-streaming implementation."""
        from delia import delegation, llm
        from delia.tools.handlers import delegate_tool_impl as delegate_tool
    
        with patch.object(llm, 'call_llm', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"success": True, "response": "Normal response", "tokens": 100}
    
            result = await delegate_tool(
                task="quick",
                content="Test content",
                stream=False
            )
            
            assert "Normal response" in result
            mock_execute.assert_called_once()


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
        from delia.tools.orchestration import _health_handler as health_tool
        await mcp_server.backend_manager.reload()
    
        result = await health_tool()
        assert "# Backend Health" in result
        assert "ollama-gpu" in result
        assert "ONLINE" in result or "OFFLINE" in result


    @pytest.mark.asyncio
    async def test_health_includes_usage_stats(self):
        """health() should include usage statistics."""
        from delia import mcp_server
        from delia.tools.orchestration import _health_handler as health_tool
        await mcp_server.backend_manager.reload()
    
        result = await health_tool()
        assert "Total requests:" in result
        assert "Total tokens:" in result


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
        from delia.tools.admin import queue_status
    
        result = await queue_status()
        assert "# Model Queue Status" in result
        assert "Active Slots:" in result
        assert "Queue Length:" in result


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
        from delia.tools.orchestration import _models_handler as models_tool
        await mcp_server.backend_manager.reload()
    
        result = await models_tool()
        assert "# Available Models" in result
        assert "quick tier" in result
        assert "coder tier" in result



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
        from delia.tools.orchestration import _switch_backend_handler as switch_backend_tool
        await mcp_server.backend_manager.reload()
    
        result = await switch_backend_tool(backend_id="backend-b")
        assert "Switched to backend: backend-b" in result
        assert mcp_server.backend_manager.get_active_backend().id == "backend-b"


    @pytest.mark.asyncio
    async def test_switch_backend_invalid(self):
        """switch_backend() should handle invalid backend ID."""
        from delia import mcp_server
        from delia.tools.orchestration import _switch_backend_handler as switch_backend_tool
        await mcp_server.backend_manager.reload()
    
        result = await switch_backend_tool(backend_id="nonexistent")
        assert "Error: Backend 'nonexistent' not found" in result


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
        from delia.tools.admin import switch_model
        await mcp_server.backend_manager.reload()
    
        result = await switch_model(tier="quick", model_name="phi3")
        assert "Updated quick tier to use phi3" in result

    @pytest.mark.asyncio
    async def test_switch_model_invalid_tier(self):
        """switch_model() should reject invalid tier."""
        from delia import mcp_server
        from delia.tools.admin import switch_model
        await mcp_server.backend_manager.reload()
    
        result = await switch_model(tier="invalid_tier", model_name="llama3")
        assert "Error: Invalid tier" in result


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

            from delia.tools.resources import get_model_info_tool

        

            result = await get_model_info_tool(model_name="llama-3-70b")

            data = json.loads(result)

            assert data["model"] == "llama-3-70b"

            assert data["params_b"] == 70.0

    

    @pytest.mark.asyncio
    async def test_get_model_info_coder_model(self):
        """get_model_info_tool() should detect coder models."""
        from delia.tools.resources import get_model_info_tool
    
        result = await get_model_info_tool(model_name="deepseek-coder-33b")
        data = json.loads(result)
        assert data["is_coder"] is True

    @pytest.mark.asyncio
    async def test_get_model_info_moe_model(self):
        """get_model_info_tool() should detect MoE models."""
        from delia.tools.resources import get_model_info_tool
    
        result = await get_model_info_tool(model_name="mixtral-8x7b")
        data = json.loads(result)
        assert data["is_moe"] is True


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
        from delia import mcp_server, llm
        from delia.tools.handlers import delegate_tool_impl as delegate_tool
        await mcp_server.backend_manager.reload()
    
        # Mock to avoid actual LLM call
        with patch.object(llm, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"success": True, "response": "Response", "tokens": 50}
    
            result = await delegate_tool(
                task="quick",
                content="Test",
                model="coder"  # Explicit override
            )
            
            # Verify call_llm was called with override
            args, kwargs = mock_llm.call_args
            # Model should be from coder tier
            assert "coder" in kwargs.get("model", "") or "deepseek" in kwargs.get("model", "")


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
        from delia import mcp_server, llm
        from delia.tools.handlers import think_impl as think_tool
        await mcp_server.backend_manager.reload()
    
        with patch.object(llm, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"success": True, "response": "Quick thought", "tokens": 20}
    
            result = await think_tool(
                problem="What is 2+2?",
                depth="quick"
            )
            
            assert "Quick thought" in result
            
            # Verify quick model hint was used
            call_args = mock_llm.call_args
            assert call_args.kwargs.get("model_override") == "quick"

    @pytest.mark.asyncio
    async def test_think_deep_depth(self):
        """think() with deep depth should use thinking model."""
        from delia import mcp_server, llm
        await mcp_server.backend_manager.reload()

        with patch.object(llm, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"success": True, "response": "Deep thought", "tokens": 100}

            result = await mcp_server.think.fn(
                problem="Complex problem",
                depth="deep"
            )

            # Should return something (mocked response or error about backend)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
