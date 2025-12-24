# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MCP Server Management Tools.

Tools for managing external MCP servers for tool passthrough.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from fastmcp import FastMCP

from ..backend_manager import backend_manager
from ..container import get_container

# Get container services
container = get_container()
mcp_client_manager = container.mcp_client_manager


def register_mcp_management_tools(mcp: FastMCP):
    """Register MCP server management tools."""

    @mcp.tool()
    async def mcp_servers(
        action: str = "status",
        server_id: str | None = None,
        command: str | None = None,
        name: str | None = None,
        env: str | None = None,
    ) -> str:
        """
        Manage external MCP servers for tool passthrough.

        This enables Delia to use tools from any MCP server.
        Configure servers in settings.json under "mcp_servers" array.

        WHEN TO USE:
        - Check status of connected MCP servers
        - Start/stop MCP servers dynamically
        - Add/remove MCP server configurations
        - List available tools from external servers

        Args:
            action: Action to perform:
                - "status" - Show all configured servers and their status (default)
                - "start" - Start a specific server (requires server_id)
                - "stop" - Stop a specific server (requires server_id)
                - "start_all" - Start all enabled servers
                - "stop_all" - Stop all running servers
                - "add" - Add a new server configuration (requires command, name optional)
                - "remove" - Remove a server configuration (requires server_id)
                - "tools" - List all available tools from running servers
            server_id: Server ID for start/stop/remove actions
            command: JSON array of command args for 'add' action (e.g., '["npx", "mcp-server"]')
            name: Human-readable name for 'add' action
            env: JSON object of environment variables for 'add' action

        Returns:
            JSON with action result and server status

        Examples:
            mcp_servers()  # Show status
            mcp_servers(action="start_all")  # Start all enabled servers
            mcp_servers(action="tools")  # List all available tools
            mcp_servers(action="add", command='["npx", "@anthropic/mcp-server-filesystem", "/home"]', name="Filesystem")
            mcp_servers(action="start", server_id="filesystem")
        """
        if action == "status":
            status = mcp_client_manager.get_status()
            return json.dumps(status, indent=2)

        elif action == "start_all":
            results = await mcp_client_manager.start_all()
            return json.dumps({
                "action": "start_all",
                "results": results,
                "status": mcp_client_manager.get_status(),
            }, indent=2)

        elif action == "stop_all":
            await mcp_client_manager.stop_all()
            return json.dumps({
                "action": "stop_all",
                "success": True,
                "status": mcp_client_manager.get_status(),
            }, indent=2)

        elif action == "start":
            if not server_id:
                return json.dumps({"error": "server_id required for start action"})
            success = await mcp_client_manager.start_server(server_id)
            return json.dumps({
                "action": "start",
                "server_id": server_id,
                "success": success,
                "status": mcp_client_manager.get_status(),
            }, indent=2)

        elif action == "stop":
            if not server_id:
                return json.dumps({"error": "server_id required for stop action"})
            success = await mcp_client_manager.stop_server(server_id)
            return json.dumps({
                "action": "stop",
                "server_id": server_id,
                "success": success,
                "status": mcp_client_manager.get_status(),
            }, indent=2)

        elif action == "add":
            if not command:
                return json.dumps({"error": "command required for add action (JSON array)"})
            try:
                cmd_list = json.loads(command)
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid command JSON - must be array"})

            if not isinstance(cmd_list, list):
                return json.dumps({"error": "command must be a JSON array"})

            # Parse optional env
            env_dict = {}
            if env:
                try:
                    env_dict = json.loads(env)
                except json.JSONDecodeError:
                    return json.dumps({"error": "Invalid env JSON - must be object"})

            # Create server config
            new_id = server_id or str(uuid.uuid4())[:8]
            config = {
                "id": new_id,
                "name": name or f"MCP Server {new_id}",
                "command": cmd_list,
                "enabled": True,
                "env": env_dict,
            }

            # Add to backend manager (persists to settings.json)
            success = backend_manager.add_mcp_server(config)
            if success:
                # Reload manager config
                mcp_client_manager.load_config(backend_manager.get_mcp_servers())

            return json.dumps({
                "action": "add",
                "success": success,
                "server_id": new_id,
                "config": config if success else None,
            }, indent=2)

        elif action == "remove":
            if not server_id:
                return json.dumps({"error": "server_id required for remove action"})

            # Stop if running
            await mcp_client_manager.stop_server(server_id)

            # Remove from config
            success = backend_manager.remove_mcp_server(server_id)
            if success:
                mcp_client_manager.load_config(backend_manager.get_mcp_servers())

            return json.dumps({
                "action": "remove",
                "server_id": server_id,
                "success": success,
            }, indent=2)

        elif action == "tools":
            tools = mcp_client_manager.get_all_tools()
            return json.dumps({
                "action": "tools",
                "total_tools": len(tools),
                "tools": [
                    {
                        "name": f"mcp_{t.server_id}_{t.name}",
                        "server_id": t.server_id,
                        "original_name": t.name,
                        "description": t.description,
                    }
                    for t in tools
                ],
            }, indent=2)

        else:
            return json.dumps({
                "error": f"Unknown action: {action}",
                "valid_actions": ["status", "start", "stop", "start_all", "stop_all", "add", "remove", "tools"],
            })
