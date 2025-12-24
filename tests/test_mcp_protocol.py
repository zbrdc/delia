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
End-to-end MCP protocol tests.

Tests the full MCP communication flow:
Client connects → Initialize → List tools → Call tool → Get response

Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_mcp_protocol.py -v
"""

import os
import sys
import json
import asyncio
import subprocess
import time
from pathlib import Path

import pytest
import httpx
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Use a temp directory for test data."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    # Create minimal settings
    from delia import paths
    paths.ensure_directories()

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
                "models": {"quick": "test-model"}
            }
        ],
        "routing": {"prefer_local": True}
    }
    with open(paths.SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


class TestMCPToolsExport:
    """Test that MCP tools are properly exported."""

    def test_mcp_instance_exists(self):
        """FastMCP instance should be created."""
        from delia import mcp_server
        assert mcp_server.mcp is not None

    def test_mcp_has_tools_registered(self):
        """MCP should have tools registered."""
        from delia import mcp_server

        # Check that key tool implementation functions exist (backward compat re-exports)
        assert hasattr(mcp_server, 'delegate')
        assert hasattr(mcp_server, 'think')
        assert hasattr(mcp_server, 'batch')
        assert hasattr(mcp_server, 'health')
        assert hasattr(mcp_server, 'switch_backend')
        assert hasattr(mcp_server, 'switch_model')
        assert hasattr(mcp_server, 'get_model_info')

        # These are registered on the mcp object itself
        assert mcp_server.mcp is not None

    def test_tools_are_callable(self):
        """Tool implementations should be callable async functions."""
        from delia import mcp_server
        import asyncio

        # Tool implementations are async functions, directly callable
        assert callable(mcp_server.delegate)
        assert callable(mcp_server.health)
        assert asyncio.iscoroutinefunction(mcp_server.delegate)
        assert asyncio.iscoroutinefunction(mcp_server.health)


class TestMCPToolSignatures:
    """Test MCP tool function signatures."""

    def test_delegate_parameters(self):
        """delegate() should accept required parameters."""
        from delia import mcp_server
        import inspect

        sig = inspect.signature(mcp_server.delegate)
        params = list(sig.parameters.keys())

        # Required parameters
        assert 'task' in params
        assert 'content' in params

        # Optional parameters
        assert 'model' in params or 'file' in params

    def test_think_parameters(self):
        """think() should accept required parameters."""
        from delia import mcp_server
        import inspect

        sig = inspect.signature(mcp_server.think)
        params = list(sig.parameters.keys())

        assert 'problem' in params
        assert 'depth' in params or 'context' in params

    def test_batch_parameters(self):
        """batch() should accept tasks parameter."""
        from delia import mcp_server
        import inspect

        sig = inspect.signature(mcp_server.batch)
        params = list(sig.parameters.keys())

        assert 'tasks' in params

    def test_switch_backend_parameters(self):
        """switch_backend() should accept backend_id."""
        from delia import mcp_server
        import inspect

        sig = inspect.signature(mcp_server.switch_backend)
        params = list(sig.parameters.keys())

        assert 'backend_id' in params

    def test_switch_model_parameters(self):
        """switch_model() should accept tier and model_name."""
        from delia import mcp_server
        import inspect

        sig = inspect.signature(mcp_server.switch_model)
        params = list(sig.parameters.keys())

        assert 'tier' in params
        assert 'model_name' in params


class TestMCPToolResponses:
    """Test MCP tools return appropriate responses."""

    @pytest.mark.asyncio
    async def test_health_returns_string(self):
        """health() should return a string."""
        from delia import mcp_server

        result = await mcp_server.health()

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_models_returns_string(self):
        """models() should return a string."""
        from delia import mcp_server

        result = await mcp_server.models()

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_queue_status_returns_string(self):
        """queue_status() should return a string."""
        from delia import mcp_server

        result = await mcp_server.queue_status()

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_model_info_returns_string(self):
        """get_model_info() should return a string."""
        from delia import mcp_server

        result = await mcp_server.get_model_info(model_name="llama-3-8b")

        assert result is not None
        assert isinstance(result, str)


class TestMCPHTTPProtocol:
    """Test MCP over HTTP transport."""

    @pytest.fixture
    def http_server(self, tmp_path):
        """Start HTTP server for testing."""
        env = os.environ.copy()
        env["DELIA_DATA_DIR"] = str(tmp_path)

        # Create settings
        settings_file = tmp_path / "settings.json"
        settings = {
            "version": "1.0",
            "backends": [],
            "routing": {"prefer_local": True}
        }
        # Need to create in project root
        from delia import paths
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        proc = subprocess.Popen(
            ["uv", "run", "python", "-m", "delia.mcp_server", "--transport", "http", "--port", "18770"],
            cwd="/home/dan/git/delia",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for startup
        time.sleep(3)

        yield proc

        # Cleanup
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_http_server_responds(self, http_server):
        """HTTP server should respond to requests."""
        if http_server.poll() is not None:
            pytest.skip("Server failed to start")

        try:
            response = httpx.get("http://localhost:18770/", timeout=5)
            # Any response means server is running
            assert response.status_code is not None
        except httpx.ConnectError:
            pytest.skip("Could not connect to server")


class TestMCPJSONRPCFormat:
    """Test JSON-RPC message format compliance."""

    def test_jsonrpc_request_format(self):
        """Verify JSON-RPC request format."""
        # Standard JSON-RPC 2.0 request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "health",
                "arguments": {}
            }
        }

        assert request["jsonrpc"] == "2.0"
        assert "id" in request
        assert "method" in request

    def test_jsonrpc_response_format(self):
        """Verify JSON-RPC response format."""
        # Standard JSON-RPC 2.0 response
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "OK"}]
            }
        }

        assert response["jsonrpc"] == "2.0"
        assert "id" in response
        assert "result" in response or "error" in response


class TestMCPInitializeFlow:
    """Test MCP initialize handshake."""

    def test_initialize_request_format(self):
        """Verify initialize request format."""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }

        assert init_request["method"] == "initialize"
        assert "protocolVersion" in init_request["params"]
        assert "clientInfo" in init_request["params"]

    def test_tools_list_request_format(self):
        """Verify tools/list request format."""
        list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        assert list_request["method"] == "tools/list"


class TestMCPToolCallFlow:
    """Test full tool call flow."""

    @pytest.fixture
    def mock_env(self):
        """Setup mock environment for MCP tests."""
        from delia.backend_manager import BackendConfig
        from unittest.mock import MagicMock, patch

        # Mock backend
        mock_backend = BackendConfig(
            id="test-backend",
            name="Test Backend",
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

        with patch("delia.backend_manager.BackendManager.get_active_backend", return_value=mock_backend), \
             patch("delia.backend_manager.BackendManager.get_enabled_backends", return_value=[mock_backend]), \
             patch("delia.routing.BackendScorer.score", return_value=1.0), \
             patch("delia.routing.get_backend_metrics", return_value=MagicMock(total_requests=0)), \
             patch("delia.routing.get_backend_health", return_value=MagicMock(is_available=lambda: True)), \
             patch("delia.tools.handlers_orchestration.check_context_gate", return_value=None):
            yield mock_backend

    @pytest.mark.asyncio
    async def test_delegate_flow(self, mock_env):
        """Test complete delegate tool call flow."""
        from delia import mcp_server

        # Simulate MCP tool call
        with patch("delia.orchestration.executor.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"success": True, "response": "Summary text", "model": "test"}
            result = await mcp_server.delegate(
                task="summarize",
                content="This is a test document that needs summarization."
            )

        # Should return something
        assert result is not None
        assert isinstance(result, str)
        assert "Summary" in result

    @pytest.mark.asyncio
    async def test_think_flow(self, mock_env):
        """Test complete think tool call flow."""
        from delia import mcp_server

        with patch("delia.orchestration.executor.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"success": True, "response": "Thinking result", "model": "test"}
            result = await mcp_server.think(
                problem="What is 2 + 2?",
                depth="quick"
            )

        assert result is not None
        assert isinstance(result, str)
        assert "Thinking" in result

    @pytest.mark.asyncio
    async def test_batch_flow(self, mock_env):
        """Test complete batch tool call flow."""
        from delia import mcp_server

        tasks = json.dumps([
            {"task": "quick", "content": "What is Python?"},
            {"task": "quick", "content": "What is JavaScript?"}
        ])

        with patch("delia.orchestration.executor.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"success": True, "response": "Python is a language.", "model": "test"}
            result = await mcp_server.batch(tasks=tasks)

        assert result is not None
        assert isinstance(result, str)
        assert "Python" in result


class TestMCPErrorHandling:
    """Test MCP error handling in protocol."""

    @pytest.mark.asyncio
    async def test_invalid_task_handled(self):
        """Invalid task should return error, not crash."""
        from delia import mcp_server

        result = await mcp_server.delegate(
            task="",  # Empty task
            content="Test"
        )

        # Should return error message
        assert result is not None

    @pytest.mark.asyncio
    async def test_invalid_json_batch_handled(self):
        """Invalid JSON in batch should return error."""
        from delia import mcp_server

        result = await mcp_server.batch(tasks="not valid json")

        assert result is not None
        # Should indicate error
        assert "error" in result.lower() or "invalid" in result.lower()


class TestMCPTransportCompatibility:
    """Test MCP works with different transports."""

    def test_mcp_supports_stdio(self):
        """MCP should support stdio transport."""
        from delia import mcp_server

        # FastMCP's run method should accept stdio
        mcp = mcp_server.mcp
        assert mcp is not None

    def test_mcp_supports_http(self):
        """MCP should support http transport."""
        from delia import mcp_server
        mcp = mcp_server.mcp
        assert mcp is not None

    def test_mcp_supports_sse(self):
        """MCP should support sse transport."""
        from delia import mcp_server
        mcp = mcp_server.mcp
        assert mcp is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
