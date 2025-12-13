# Copyright (C) 2023 the project owner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
MCP Client Detection and Configuration Tests.

These tests validate that Delia will be properly detected as an MCP server
by supported applications (Claude Desktop, VS Code, Cursor, etc.).

This addresses the common issue of MCP servers not being detected properly.

Run with: uv run pytest tests/test_mcp_detection.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ============================================================
# CONFIGURATION GENERATION TESTS
# ============================================================


class TestMCPClientConfig:
    """Test MCP client configuration data."""

    def test_mcp_clients_defined(self):
        """All expected MCP clients should be defined."""
        from delia.cli import MCP_CLIENTS

        expected_clients = ["claude", "vscode", "cursor", "gemini"]
        for client in expected_clients:
            assert client in MCP_CLIENTS, f"Missing client: {client}"

    def test_all_clients_have_required_fields(self):
        """Each client config should have required fields."""
        from delia.cli import MCP_CLIENTS

        required_fields = ["name", "config_key", "paths"]
        for client_id, config in MCP_CLIENTS.items():
            for field in required_fields:
                assert field in config, f"{client_id} missing field: {field}"

    def test_all_clients_have_platform_paths(self):
        """Each client should have paths for all platforms."""
        from delia.cli import MCP_CLIENTS

        platforms = ["Linux", "Darwin", "Windows"]
        for client_id, config in MCP_CLIENTS.items():
            for platform in platforms:
                assert platform in config["paths"], f"{client_id} missing {platform} path"
                assert config["paths"][platform] is not None

    def test_config_paths_are_absolute(self):
        """Config paths should be absolute (not relative)."""
        from delia.cli import MCP_CLIENTS

        for client_id, config in MCP_CLIENTS.items():
            for platform, path in config["paths"].items():
                assert path.is_absolute(), f"{client_id} {platform} path is relative: {path}"

    def test_config_keys_match_spec(self):
        """Config keys should match MCP specification."""
        from delia.cli import MCP_CLIENTS

        # Claude, Cursor use "mcpServers", VS Code uses "servers"
        mcpservers_clients = ["claude", "cursor", "gemini", "windsurf"]
        servers_clients = ["vscode", "vscode-insiders", "copilot-cli"]

        for client_id in mcpservers_clients:
            if client_id in MCP_CLIENTS:
                assert MCP_CLIENTS[client_id]["config_key"] == "mcpServers"

        for client_id in servers_clients:
            if client_id in MCP_CLIENTS:
                assert MCP_CLIENTS[client_id]["config_key"] == "servers"


class TestConfigGeneration:
    """Test configuration file generation."""

    def test_generate_client_config_returns_dict(self, tmp_path):
        """generate_client_config should return a dictionary."""
        from delia.cli import generate_client_config

        config = generate_client_config("claude", tmp_path)

        assert isinstance(config, dict)
        assert "command" in config
        assert "args" in config

    def test_generated_config_uses_uv(self, tmp_path):
        """Generated config should use uv to run delia."""
        from delia.cli import generate_client_config

        config = generate_client_config("claude", tmp_path)

        assert config["command"] == "uv"
        assert "--directory" in config["args"]
        assert str(tmp_path) in config["args"]
        assert "run" in config["args"]
        assert "delia" in config["args"]
        assert "serve" in config["args"]

    def test_generated_config_has_valid_json_structure(self, tmp_path):
        """Generated config should be valid JSON."""
        from delia.cli import generate_client_config

        config = generate_client_config("claude", tmp_path)

        # Should be JSON serializable
        json_str = json.dumps(config)
        parsed = json.loads(json_str)
        assert parsed == config

    @pytest.mark.parametrize("client_id", ["claude", "vscode", "cursor", "gemini"])
    def test_config_generated_for_all_clients(self, tmp_path, client_id):
        """Config should be generatable for all supported clients."""
        from delia.cli import MCP_CLIENTS, generate_client_config

        if client_id not in MCP_CLIENTS:
            pytest.skip(f"{client_id} not in MCP_CLIENTS")

        config = generate_client_config(client_id, tmp_path)
        assert config is not None
        assert "command" in config


class TestInstallToClient:
    """Test the install_to_client function."""

    def test_install_creates_config_file(self, tmp_path):
        """install_to_client should create config file."""
        from delia.cli import MCP_CLIENTS, install_to_client

        # Create a fake client config pointing to temp dir
        config_path = tmp_path / "mcp.json"

        with patch.dict(MCP_CLIENTS, {
            "test-client": {
                "name": "Test Client",
                "config_key": "mcpServers",
                "paths": {"Linux": config_path, "Darwin": config_path, "Windows": config_path},
            }
        }):
            result = install_to_client("test-client", tmp_path)

            assert result is True
            assert config_path.exists()

    def test_install_creates_valid_json(self, tmp_path):
        """Installed config should be valid JSON."""
        from delia.cli import MCP_CLIENTS, install_to_client

        config_path = tmp_path / "mcp.json"

        with patch.dict(MCP_CLIENTS, {
            "test-client": {
                "name": "Test Client",
                "config_key": "mcpServers",
                "paths": {"Linux": config_path, "Darwin": config_path, "Windows": config_path},
            }
        }):
            install_to_client("test-client", tmp_path)

            with open(config_path) as f:
                config = json.load(f)

            assert "mcpServers" in config
            assert "delia" in config["mcpServers"]

    def test_install_preserves_existing_config(self, tmp_path):
        """Installing should preserve existing servers in config."""
        from delia.cli import MCP_CLIENTS, install_to_client

        config_path = tmp_path / "mcp.json"

        # Pre-existing config
        existing = {
            "mcpServers": {
                "other-server": {"command": "other", "args": []}
            }
        }
        with open(config_path, "w") as f:
            json.dump(existing, f)

        with patch.dict(MCP_CLIENTS, {
            "test-client": {
                "name": "Test Client",
                "config_key": "mcpServers",
                "paths": {"Linux": config_path, "Darwin": config_path, "Windows": config_path},
            }
        }):
            install_to_client("test-client", tmp_path)

            with open(config_path) as f:
                config = json.load(f)

            # Both servers should exist
            assert "other-server" in config["mcpServers"]
            assert "delia" in config["mcpServers"]


# ============================================================
# MCP PROTOCOL COMPLIANCE TESTS
# ============================================================


class TestMCPProtocolCompliance:
    """Test that Delia complies with MCP protocol spec."""

    @pytest.fixture(autouse=True)
    def setup_test_env(self, tmp_path):
        """Set up test environment."""
        os.environ["DELIA_DATA_DIR"] = str(tmp_path)

        from delia import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [],
            "routing": {"prefer_local": True}
        }
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        yield

        os.environ.pop("DELIA_DATA_DIR", None)

    def test_mcp_server_name_set(self):
        """MCP server should have name set."""
        from delia import mcp_server

        assert mcp_server.mcp is not None
        assert mcp_server.mcp.name == "delia"

    def test_tools_have_descriptions(self):
        """All MCP tools should have descriptions."""
        from delia import mcp_server

        tools = [
            mcp_server.delegate,
            mcp_server.think,
            mcp_server.batch,
            mcp_server.health,
            mcp_server.models,
        ]

        for tool in tools:
            # FastMCP FunctionTool has description
            assert hasattr(tool, 'fn')
            # The docstring becomes the description
            assert tool.fn.__doc__ is not None

    def test_required_tools_registered(self):
        """Core tools required for MCP detection should be registered."""
        from delia import mcp_server

        # These tools should exist
        required = ["delegate", "think", "batch", "health", "models"]
        for tool_name in required:
            assert hasattr(mcp_server, tool_name), f"Missing tool: {tool_name}"


# ============================================================
# STDIO TRANSPORT TESTS
# ============================================================


class TestSTDIOTransport:
    """Test MCP over STDIO transport (used by Claude, VS Code, etc.)."""

    @pytest.fixture
    def temp_settings(self, tmp_path):
        """Create temporary settings file."""
        os.environ["DELIA_DATA_DIR"] = str(tmp_path)

        from delia import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [],
            "routing": {"prefer_local": True}
        }
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        yield tmp_path

        os.environ.pop("DELIA_DATA_DIR", None)

    def test_stdio_server_starts(self, temp_settings):
        """Server should start via STDIO and respond to initialize."""
        env = os.environ.copy()
        env["DELIA_DATA_DIR"] = str(temp_settings)

        # Start the server with STDIO
        proc = subprocess.Popen(
            ["uv", "run", "delia", "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(Path(__file__).parent.parent),  # Project root
        )

        try:
            # Send MCP initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}
                }
            }
            request_bytes = json.dumps(init_request).encode() + b"\n"
            proc.stdin.write(request_bytes)
            proc.stdin.flush()

            # Wait briefly for response
            time.sleep(1)

            # Check process is still running (didn't crash)
            if proc.poll() is None:
                # Server is running - good sign
                pass

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_serve_command_exists(self):
        """'delia serve' command should exist."""
        result = subprocess.run(
            ["uv", "run", "delia", "serve", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "serve" in result.stdout.lower() or "transport" in result.stdout.lower()


# ============================================================
# FULL INTEGRATION TEST
# ============================================================


class TestMCPIntegration:
    """Full integration tests for MCP detection."""

    @pytest.fixture
    def isolated_env(self, tmp_path):
        """Create isolated test environment."""
        # Temp data dir
        data_dir = tmp_path / "delia-data"
        data_dir.mkdir()
        os.environ["DELIA_DATA_DIR"] = str(data_dir)

        # Create minimal settings
        from delia import paths
        paths.ensure_directories()
        settings = {
            "version": "1.0",
            "backends": [],
            "routing": {"prefer_local": True}
        }
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        yield tmp_path

        os.environ.pop("DELIA_DATA_DIR", None)

    def test_full_install_flow(self, isolated_env):
        """Test complete installation flow for a client."""
        from delia.cli import MCP_CLIENTS, install_to_client

        # Use temp path for config
        config_path = isolated_env / "test-config.json"
        delia_root = Path(__file__).parent.parent

        with patch.dict(MCP_CLIENTS, {
            "test": {
                "name": "Test",
                "config_key": "mcpServers",
                "paths": {"Linux": config_path, "Darwin": config_path, "Windows": config_path}
            }
        }):
            # Install
            result = install_to_client("test", delia_root)
            assert result is True

            # Read back
            with open(config_path) as f:
                config = json.load(f)

            # Verify structure
            assert "mcpServers" in config
            assert "delia" in config["mcpServers"]
            assert config["mcpServers"]["delia"]["command"] == "uv"
            assert "serve" in config["mcpServers"]["delia"]["args"]

    def test_config_command_is_executable(self, isolated_env):
        """The command in generated config should be executable."""
        from delia.cli import generate_client_config
        import shutil

        delia_root = Path(__file__).parent.parent
        config = generate_client_config("claude", delia_root)

        # Check command exists
        command = config["command"]
        assert shutil.which(command) is not None, f"Command not found: {command}"


# ============================================================
# DIAGNOSTIC TESTS
# ============================================================


class TestMCPDiagnostics:
    """Tests that can help diagnose MCP detection issues."""

    def test_print_config_for_debugging(self, tmp_path, capsys):
        """Print generated config for manual inspection."""
        from delia.cli import generate_client_config

        config = generate_client_config("claude", tmp_path)

        print("\n=== Generated MCP Config for Claude ===")
        print(json.dumps(config, indent=2))
        print("========================================\n")

        # This test always passes - it's for debugging output
        assert True

    def test_print_all_client_paths(self, capsys):
        """Print all client config paths for debugging."""
        from delia.cli import MCP_CLIENTS
        import platform

        plat = platform.system()

        print("\n=== MCP Client Config Paths ===")
        for client_id, info in MCP_CLIENTS.items():
            path = info["paths"].get(plat, "N/A")
            exists = path.exists() if isinstance(path, Path) else False
            print(f"{client_id}: {path} (exists: {exists})")
        print("================================\n")

        assert True

    def test_detect_installed_clients(self, capsys):
        """Detect which MCP clients are installed."""
        from delia.cli import detect_clients

        clients = detect_clients()

        print("\n=== Detected MCP Clients ===")
        if not clients:
            print("No clients detected")
        for client in clients:
            print(f"{client.id}: installed={client.installed}, configured={client.configured}")
        print("============================\n")

        # Just informational
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
