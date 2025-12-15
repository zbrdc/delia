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

"""MCP Client for connecting to external MCP servers.

This module enables Delia to act as an MCP client, spawning external MCP servers
and importing their tools into Delia's ToolRegistry. This allows local LLMs to
use tools from any MCP server.

Architecture:
    User configures MCP servers in settings.json:
    {
        "mcp_servers": [
            {
                "id": "filesystem",
                "name": "Filesystem Tools",
                "command": ["npx", "@anthropic/mcp-server-filesystem", "/home"],
                "enabled": true,
                "env": {}
            }
        ]
    }

    On agent startup:
    1. MCPClientManager spawns configured servers as subprocesses
    2. Connects via STDIO transport (JSON-RPC 2.0)
    3. Queries tools/list to get available tools
    4. Creates proxy handlers and registers in ToolRegistry
    5. Agent loop can now use these tools seamlessly
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from .registry import ToolDefinition, ToolRegistry

log = structlog.get_logger()

# MCP Protocol Version
MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPServerConfig:
    """Configuration for an external MCP server.

    Attributes:
        id: Unique identifier for this server
        name: Human-readable name
        command: Command and arguments to spawn the server
        enabled: Whether this server should be started
        env: Additional environment variables for the subprocess
        timeout_seconds: Timeout for tool calls
    """

    id: str
    name: str
    command: list[str]
    enabled: bool = True
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerConfig":
        """Create from dictionary (settings.json format)."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Unknown MCP Server"),
            command=data.get("command", []),
            enabled=data.get("enabled", True),
            env=data.get("env", {}),
            timeout_seconds=data.get("timeout_seconds", 30.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "command": self.command,
            "enabled": self.enabled,
            "env": self.env,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class MCPTool:
    """Representation of a tool from an MCP server.

    Attributes:
        name: Tool name (e.g., "read_file")
        description: Human-readable description
        input_schema: JSON Schema for parameters
        server_id: ID of the MCP server providing this tool
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server_id: str


class MCPClient:
    """Client for a single MCP server subprocess.

    Handles spawning the server process, JSON-RPC communication over STDIO,
    and graceful shutdown.

    Example:
        client = MCPClient(config)
        await client.start()
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "/etc/hosts"})
        await client.stop()
    """

    def __init__(self, config: MCPServerConfig):
        """Initialize MCP client.

        Args:
            config: Server configuration including command to spawn
        """
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending_requests: dict[int | str, asyncio.Future[dict[str, Any]]] = {}
        self._read_task: asyncio.Task[None] | None = None
        self._initialized = False
        self._server_capabilities: dict[str, Any] = {}
        self._tools: list[MCPTool] = []

    @property
    def is_running(self) -> bool:
        """Check if the server process is running."""
        return self._process is not None and self._process.returncode is None

    async def start(self) -> bool:
        """Start the MCP server subprocess and initialize connection.

        Returns:
            True if server started and initialized successfully
        """
        if self.is_running:
            log.warning("mcp_server_already_running", server_id=self.config.id)
            return True

        if not self.config.command:
            log.error("mcp_server_no_command", server_id=self.config.id)
            return False

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.env)

            # Spawn subprocess with STDIO transport
            log.info(
                "mcp_server_starting",
                server_id=self.config.id,
                command=self.config.command,
            )

            self._process = await asyncio.create_subprocess_exec(
                *self.config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Start reading responses
            self._read_task = asyncio.create_task(self._read_loop())

            # Initialize MCP connection
            init_result = await self._initialize()
            if not init_result:
                await self.stop()
                return False

            # Fetch available tools
            await self._fetch_tools()

            log.info(
                "mcp_server_started",
                server_id=self.config.id,
                tools_count=len(self._tools),
                tool_names=[t.name for t in self._tools],
            )

            return True

        except FileNotFoundError:
            log.error(
                "mcp_server_command_not_found",
                server_id=self.config.id,
                command=self.config.command[0] if self.config.command else "unknown",
            )
            return False
        except Exception as e:
            log.error("mcp_server_start_failed", server_id=self.config.id, error=str(e))
            await self.stop()
            return False

    async def stop(self) -> None:
        """Stop the MCP server subprocess gracefully."""
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._process:
            try:
                # Send terminate signal
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # Already terminated
            finally:
                self._process = None

        self._initialized = False
        self._pending_requests.clear()

        log.info("mcp_server_stopped", server_id=self.config.id)

    async def _read_loop(self) -> None:
        """Read and dispatch JSON-RPC messages from the server."""
        if not self._process or not self._process.stdout:
            return

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    message = json.loads(line.decode("utf-8"))
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    log.warning(
                        "mcp_invalid_json",
                        server_id=self.config.id,
                        line=line.decode("utf-8", errors="replace")[:100],
                    )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("mcp_read_loop_error", server_id=self.config.id, error=str(e))

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle incoming JSON-RPC message."""
        if "id" in message:
            # Response to our request
            request_id = message["id"]
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if "error" in message:
                    future.set_exception(
                        MCPError(
                            message["error"].get("message", "Unknown error"),
                            message["error"].get("code", -1),
                        )
                    )
                else:
                    future.set_result(message.get("result", {}))
        elif "method" in message:
            # Server-initiated notification or request
            method = message["method"]
            log.debug(
                "mcp_notification",
                server_id=self.config.id,
                method=method,
            )

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for response.

        Args:
            method: RPC method name
            params: Optional parameters
            timeout: Request timeout (defaults to config timeout)

        Returns:
            Response result

        Raises:
            MCPError: On RPC error response
            asyncio.TimeoutError: On timeout
        """
        if not self._process or not self._process.stdin:
            raise MCPError("Server not running", -32000)

        self._request_id += 1
        request_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        # Create future for response
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            # Send request
            data = json.dumps(request) + "\n"
            self._process.stdin.write(data.encode("utf-8"))
            await self._process.stdin.drain()

            # Wait for response
            effective_timeout = timeout or self.config.timeout_seconds
            result = await asyncio.wait_for(future, timeout=effective_timeout)
            return result

        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise
        except Exception:
            self._pending_requests.pop(request_id, None)
            raise

    async def _initialize(self) -> bool:
        """Send MCP initialize handshake."""
        try:
            result = await self._send_request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {},
                    },
                    "clientInfo": {
                        "name": "delia",
                        "version": "0.1.0",
                    },
                },
                timeout=10.0,
            )

            self._server_capabilities = result.get("capabilities", {})
            server_info = result.get("serverInfo", {})

            log.info(
                "mcp_initialized",
                server_id=self.config.id,
                server_name=server_info.get("name"),
                server_version=server_info.get("version"),
                capabilities=list(self._server_capabilities.keys()),
            )

            # Send initialized notification
            if self._process and self._process.stdin:
                notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
                self._process.stdin.write(notification.encode("utf-8"))
                await self._process.stdin.drain()

            self._initialized = True
            return True

        except Exception as e:
            log.error("mcp_initialize_failed", server_id=self.config.id, error=str(e))
            return False

    async def _fetch_tools(self) -> None:
        """Fetch available tools from the server."""
        try:
            result = await self._send_request("tools/list", {})
            tools_data = result.get("tools", [])

            self._tools = [
                MCPTool(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {"type": "object", "properties": {}}),
                    server_id=self.config.id,
                )
                for tool in tools_data
            ]

        except Exception as e:
            log.error("mcp_tools_list_failed", server_id=self.config.id, error=str(e))
            self._tools = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        if not self._initialized:
            raise MCPError("Server not initialized", -32002)

        try:
            result = await self._send_request(
                "tools/call",
                {
                    "name": name,
                    "arguments": arguments,
                },
            )

            # Extract content from MCP response format
            content = result.get("content", [])
            if isinstance(content, list):
                # Concatenate text content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                return "\n".join(text_parts) if text_parts else json.dumps(result)
            return str(content)

        except MCPError:
            raise
        except Exception as e:
            raise MCPError(f"Tool call failed: {e}", -32603) from e

    @property
    def tools(self) -> list[MCPTool]:
        """Get list of available tools."""
        return self._tools


class MCPError(Exception):
    """Error from MCP protocol."""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.code = code


class MCPClientManager:
    """Manager for multiple MCP server connections.

    Handles starting/stopping servers and provides unified tool access.

    Example:
        manager = MCPClientManager()
        manager.load_config([{"id": "fs", "command": ["mcp-server"], ...}])
        await manager.start_all()

        # Register tools in ToolRegistry
        manager.register_tools(registry)

        await manager.stop_all()
    """

    def __init__(self) -> None:
        """Initialize the manager."""
        self._clients: dict[str, MCPClient] = {}
        self._configs: list[MCPServerConfig] = []

    def load_config(self, configs: list[dict[str, Any]]) -> None:
        """Load MCP server configurations.

        Args:
            configs: List of server config dictionaries from settings.json
        """
        self._configs = [MCPServerConfig.from_dict(c) for c in configs]
        log.info("mcp_configs_loaded", count=len(self._configs))

    async def start_all(self) -> dict[str, bool]:
        """Start all enabled MCP servers.

        Returns:
            Dict mapping server_id to success status
        """
        results = {}

        for config in self._configs:
            if not config.enabled:
                log.info("mcp_server_disabled", server_id=config.id)
                results[config.id] = False
                continue

            client = MCPClient(config)
            success = await client.start()
            results[config.id] = success

            if success:
                self._clients[config.id] = client

        return results

    async def stop_all(self) -> None:
        """Stop all running MCP servers."""
        for client in self._clients.values():
            await client.stop()
        self._clients.clear()

    async def start_server(self, server_id: str) -> bool:
        """Start a specific MCP server.

        Args:
            server_id: ID of server to start

        Returns:
            True if started successfully
        """
        config = next((c for c in self._configs if c.id == server_id), None)
        if not config:
            log.error("mcp_server_not_found", server_id=server_id)
            return False

        if server_id in self._clients:
            log.warning("mcp_server_already_running", server_id=server_id)
            return True

        client = MCPClient(config)
        success = await client.start()
        if success:
            self._clients[server_id] = client
        return success

    async def stop_server(self, server_id: str) -> bool:
        """Stop a specific MCP server.

        Args:
            server_id: ID of server to stop

        Returns:
            True if stopped successfully
        """
        client = self._clients.get(server_id)
        if not client:
            log.warning("mcp_server_not_running", server_id=server_id)
            return False

        await client.stop()
        del self._clients[server_id]
        return True

    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all running servers.

        Returns:
            List of all available MCP tools
        """
        tools = []
        for client in self._clients.values():
            tools.extend(client.tools)
        return tools

    async def call_tool(self, server_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on a specific server.

        Args:
            server_id: ID of the server
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool result as string

        Raises:
            MCPError: If server not found or tool call fails
        """
        client = self._clients.get(server_id)
        if not client:
            raise MCPError(f"Server not running: {server_id}", -32001)
        return await client.call_tool(tool_name, arguments)

    def register_tools(self, registry: "ToolRegistry") -> int:
        """Register all MCP tools into a ToolRegistry.

        Creates proxy handlers that route tool calls to the appropriate
        MCP server.

        Args:
            registry: ToolRegistry to register tools into

        Returns:
            Number of tools registered
        """
        from .registry import ToolDefinition

        count = 0

        for tool in self.get_all_tools():
            # Create proxy handler that captures server_id and tool_name
            server_id = tool.server_id
            tool_name = tool.name

            async def handler(
                _server_id: str = server_id,
                _tool_name: str = tool_name,
                **kwargs: Any,
            ) -> str:
                return await self.call_tool(_server_id, _tool_name, kwargs)

            # Register with prefixed name to avoid conflicts
            prefixed_name = f"mcp_{tool.server_id}_{tool.name}"

            try:
                registry.register(
                    ToolDefinition(
                        name=prefixed_name,
                        description=f"[MCP:{tool.server_id}] {tool.description}",
                        parameters=tool.input_schema,
                        handler=handler,
                    )
                )
                count += 1
                log.debug(
                    "mcp_tool_registered",
                    tool_name=prefixed_name,
                    server_id=tool.server_id,
                )
            except ValueError:
                # Tool already registered
                log.warning(
                    "mcp_tool_already_registered",
                    tool_name=prefixed_name,
                )

        log.info("mcp_tools_registered", count=count)
        return count

    def get_status(self) -> dict[str, Any]:
        """Get status of all configured servers.

        Returns:
            Dict with server status information
        """
        servers = []
        for config in self._configs:
            client = self._clients.get(config.id)
            servers.append({
                "id": config.id,
                "name": config.name,
                "enabled": config.enabled,
                "running": client.is_running if client else False,
                "tools_count": len(client.tools) if client else 0,
                "tools": [t.name for t in client.tools] if client else [],
            })

        return {
            "servers": servers,
            "total_servers": len(self._configs),
            "running_servers": len(self._clients),
            "total_tools": len(self.get_all_tools()),
        }
