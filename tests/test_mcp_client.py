# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the MCP client module (external tool bridge)."""

from unittest.mock import MagicMock

import pytest

from delia.tools.mcp_client import (
    MCPClient,
    MCPClientManager,
    MCPError,
    MCPServerConfig,
    MCPTool,
)


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        config = MCPServerConfig(
            id="test-server",
            name="Test Server",
            command=["python", "-m", "test_server"],
        )
        assert config.id == "test-server"
        assert config.name == "Test Server"
        assert config.command == ["python", "-m", "test_server"]
        assert config.enabled is True
        assert config.env == {}
        assert config.timeout_seconds == 30.0

    def test_config_with_all_options(self):
        """Test configuration with all options."""
        config = MCPServerConfig(
            id="full-server",
            name="Full Server",
            command=["node", "server.js"],
            env={"NODE_ENV": "production"},
            enabled=False,
            timeout_seconds=60.0,
        )
        assert config.id == "full-server"
        assert config.name == "Full Server"
        assert config.command == ["node", "server.js"]
        assert config.env == {"NODE_ENV": "production"}
        assert config.enabled is False
        assert config.timeout_seconds == 60.0

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "id": "dict-server",
            "name": "Dict Server",
            "command": ["bash", "-c", "echo"],
            "enabled": True,
        }
        config = MCPServerConfig.from_dict(data)
        assert config.id == "dict-server"
        assert config.name == "Dict Server"
        assert config.command == ["bash", "-c", "echo"]
        assert config.enabled is True

    def test_from_dict_with_defaults(self):
        """Test from_dict with minimal data uses defaults."""
        data = {}
        config = MCPServerConfig.from_dict(data)
        # Should use defaults for missing fields
        assert config.name == "Unknown MCP Server"
        assert config.command == []
        assert config.enabled is True
        assert config.timeout_seconds == 30.0

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = MCPServerConfig(
            id="to-dict",
            name="To Dict Server",
            command=["test", "arg1"],
            env={"KEY": "value"},
            timeout_seconds=45.0,
        )
        result = config.to_dict()
        assert result["id"] == "to-dict"
        assert result["name"] == "To Dict Server"
        assert result["command"] == ["test", "arg1"]
        assert result["env"] == {"KEY": "value"}
        assert result["enabled"] is True
        assert result["timeout_seconds"] == 45.0


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_basic_tool(self):
        """Test basic tool creation."""
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_id="test-server",
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object"}
        assert tool.server_id == "test-server"

    def test_tool_with_complex_schema(self):
        """Test tool with complex JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "recursive": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        }
        tool = MCPTool(
            name="list_files",
            description="List files in directory",
            input_schema=schema,
            server_id="fs-server",
        )
        assert tool.input_schema["properties"]["path"]["type"] == "string"
        assert tool.input_schema["required"] == ["path"]


class TestMCPClient:
    """Tests for MCPClient class."""

    def test_init(self):
        """Test client initialization."""
        config = MCPServerConfig(
            id="test",
            name="Test",
            command=["python", "-m", "server"],
        )
        client = MCPClient(config)
        assert client.config == config
        assert client._process is None
        assert client._tools == []

    @pytest.mark.asyncio
    async def test_start_server_failure(self):
        """Test handling of server start failure."""
        config = MCPServerConfig(
            id="fail-server",
            name="Fail Server",
            command=["nonexistent_command_xyz_123"],
        )
        client = MCPClient(config)

        # start() returns False on failure, doesn't raise
        result = await client.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        """Test stopping a client that's not running."""
        config = MCPServerConfig(
            id="not-running",
            name="Not Running",
            command=["test"],
        )
        client = MCPClient(config)
        # Should not raise
        await client.stop()
        assert not client.is_running

    def test_is_running_when_not_started(self):
        """Test is_running when client hasn't been started."""
        config = MCPServerConfig(
            id="test",
            name="Test",
            command=["test"],
        )
        client = MCPClient(config)
        assert not client.is_running

    def test_tools_property_when_not_connected(self):
        """Test tools property when not connected."""
        config = MCPServerConfig(
            id="test",
            name="Test",
            command=["test"],
        )
        client = MCPClient(config)
        assert client.tools == []


class TestMCPClientManager:
    """Tests for MCPClientManager class."""

    def test_init(self):
        """Test manager initialization."""
        manager = MCPClientManager()
        assert manager._clients == {}
        assert manager._configs == []

    def test_load_config_empty(self):
        """Test loading empty configuration."""
        manager = MCPClientManager()
        manager.load_config([])
        assert len(manager._configs) == 0

    def test_load_config_single_server(self):
        """Test loading a single server config."""
        manager = MCPClientManager()
        manager.load_config([
            {
                "id": "test-server",
                "name": "Test Server",
                "command": ["python", "-m", "server"],
                "enabled": True,
            }
        ])
        assert len(manager._configs) == 1
        assert manager._configs[0].id == "test-server"
        assert manager._configs[0].command == ["python", "-m", "server"]

    def test_load_config_disabled_server(self):
        """Test that disabled servers are loaded but not started automatically."""
        manager = MCPClientManager()
        manager.load_config([
            {
                "id": "disabled-server",
                "name": "Disabled",
                "command": ["python"],
                "enabled": False,
            }
        ])
        assert len(manager._configs) == 1
        assert manager._configs[0].enabled is False

    def test_load_config_multiple_servers(self):
        """Test loading multiple server configs."""
        manager = MCPClientManager()
        manager.load_config([
            {"id": "server1", "name": "Server 1", "command": ["cmd1"]},
            {"id": "server2", "name": "Server 2", "command": ["cmd2", "arg"]},
            {"id": "server3", "name": "Server 3", "command": ["cmd3"], "enabled": False},
        ])
        assert len(manager._configs) == 3

    def test_get_status_empty(self):
        """Test get_status when no servers configured."""
        manager = MCPClientManager()
        status = manager.get_status()
        assert status["servers"] == []
        assert status["total_servers"] == 0
        assert status["running_servers"] == 0
        assert status["total_tools"] == 0

    def test_get_status_configured_not_running(self):
        """Test get_status for configured but not running servers."""
        manager = MCPClientManager()
        manager.load_config([
            {"id": "test-server", "name": "Test", "command": ["test"]},
        ])
        status = manager.get_status()
        assert status["total_servers"] == 1
        assert status["running_servers"] == 0
        assert len(status["servers"]) == 1
        assert status["servers"][0]["id"] == "test-server"
        assert status["servers"][0]["running"] is False

    def test_get_all_tools_empty(self):
        """Test getting all tools when none configured."""
        manager = MCPClientManager()
        tools = manager.get_all_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_start_server_not_configured(self):
        """Test starting a server that isn't configured."""
        manager = MCPClientManager()
        # Returns False for non-existent server, doesn't raise
        result = await manager.start_server("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_server_not_running(self):
        """Test stopping a server that isn't running."""
        manager = MCPClientManager()
        manager.load_config([
            {"id": "test", "name": "Test", "command": ["test"]},
        ])
        # Should return False when not running
        result = await manager.stop_server("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_all_servers(self):
        """Test stopping all servers."""
        manager = MCPClientManager()
        manager.load_config([
            {"id": "server1", "name": "S1", "command": ["test1"]},
            {"id": "server2", "name": "S2", "command": ["test2"]},
        ])
        # Should not raise even if no servers running
        await manager.stop_all()


class TestMCPError:
    """Tests for MCPError exception."""

    def test_error_message(self):
        """Test error message."""
        error = MCPError("Test error message")
        assert str(error) == "Test error message"
        # Default code is -1
        assert error.code == -1

    def test_error_with_code(self):
        """Test error with code."""
        error = MCPError("Not found", code=-32001)
        assert str(error) == "Not found"
        assert error.code == -32001

    def test_error_inheritance(self):
        """Test that MCPError inherits from Exception."""
        error = MCPError("test")
        assert isinstance(error, Exception)


class TestToolRegistration:
    """Tests for tool registration with ToolRegistry."""

    def test_register_tools_empty(self):
        """Test registering tools when no tools available."""
        from delia.tools import ToolRegistry

        manager = MCPClientManager()
        registry = ToolRegistry()

        count = manager.register_tools(registry)
        assert count == 0

    def test_register_tools_with_mock_client(self):
        """Test that registered tools have prefixed names."""
        from delia.tools import ToolRegistry

        manager = MCPClientManager()
        registry = ToolRegistry()

        # Create a proper config and mock client
        config = MCPServerConfig(
            id="mock-server",
            name="Mock Server",
            command=["mock"],
        )
        manager._configs.append(config)

        # Mock a running client with tools
        mock_client = MagicMock()
        mock_client.is_running = True
        mock_client.tools = [
            MCPTool(
                name="test_tool",
                description="Test",
                input_schema={"type": "object"},
                server_id="mock-server",
            )
        ]
        manager._clients["mock-server"] = mock_client

        count = manager.register_tools(registry)
        assert count == 1
        assert "mcp_mock-server_test_tool" in registry

    def test_register_multiple_tools_from_multiple_servers(self):
        """Test registering tools from multiple servers."""
        from delia.tools import ToolRegistry

        manager = MCPClientManager()
        registry = ToolRegistry()

        # Mock first server
        mock_client1 = MagicMock()
        mock_client1.is_running = True
        mock_client1.tools = [
            MCPTool(name="tool1", description="Tool 1", input_schema={}, server_id="server1"),
            MCPTool(name="tool2", description="Tool 2", input_schema={}, server_id="server1"),
        ]
        manager._clients["server1"] = mock_client1

        # Mock second server
        mock_client2 = MagicMock()
        mock_client2.is_running = True
        mock_client2.tools = [
            MCPTool(name="tool3", description="Tool 3", input_schema={}, server_id="server2"),
        ]
        manager._clients["server2"] = mock_client2

        count = manager.register_tools(registry)
        assert count == 3
        assert "mcp_server1_tool1" in registry
        assert "mcp_server1_tool2" in registry
        assert "mcp_server2_tool3" in registry
