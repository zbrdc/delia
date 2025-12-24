#!/usr/bin/env python3
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
Delia — Multi-Model LLM Delegation Server

A pure MCP server that intelligently routes tasks to optimal models.
Automatically selects quick/coder/moe tiers based on task type and content.

Usage:
    uv run mcp_server.py                    # stdio transport (default)
    uv run mcp_server.py --transport sse    # SSE transport on port 8200
"""

import os
import sys

# ============================================================ 
# EARLY LOGGING CONFIGURATION (must happen before ANY other imports)
# ============================================================ 
# Configure structlog to suppress stdout output by default.
# This is critical for STDIO transport where stdout must be pure JSON-RPC.

# Store original stdout for MCP protocol (using both sys.stdout and a file descriptor backup)
_original_stdout = sys.stdout
_original_stdout_fd = os.dup(1)

# Redirect FD 1 (stdout) to FD 2 (stderr) at the OS level
# This catches EVERYTHING, including low-level writes and external C libraries
os.dup2(2, 1)

# Also update sys.stdout for Python-level prints
sys.stdout = sys.stderr

# ============================================================ 

import asyncio
import contextlib
import atexit
import fcntl
import httpx
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

import structlog
from fastmcp import FastMCP

# Singleton lock file location
_LOCK_FILE = Path.home() / ".delia" / "server.lock"
_HTTP_PORT_FILE = Path.home() / ".delia" / "http_server.port"

# Context variables for user tracking in multi-user mode
current_client_id: ContextVar[Optional[str]] = ContextVar("current_client_id", default=None)
current_username: ContextVar[Optional[str]] = ContextVar("current_username", default=None)

from .container import get_container
from .config import config
from .llm import init_llm_module
from .logging_service import get_logging_service

# Use logging service directly for early setup to ensure stdout is clean
get_logging_service().configure_structlog(use_stderr=True)

# Use container to handle other early setup
_container = get_container()


def _configure_structlog(use_stderr: bool = True) -> None:
    """Helper to reconfigure structlog (for transport changes)."""
    get_logging_service().configure_structlog(use_stderr=use_stderr)


from . import paths
from .context import set_project_context
from .lifecycle import startup_handler, shutdown_handler, init_database
from .stats_handler import update_stats_sync, save_stats_background
from .orchestration.background import start_prewarm_task
from .auth_routes import register_auth_routes, register_oauth_routes
from .multi_user_tracking import tracker

# Global service container (single source of truth)
container = get_container()

# Configure structlog with stderr by default
container.initialize(use_stderr=True)
log = structlog.get_logger()

# Use services from container
model_queue = container.model_queue
mcp_client_manager = container.mcp_client_manager
stats_service = container.stats_service

# ============================================================
# BACKWARD COMPATIBILITY RE-EXPORTS
# ============================================================
# These re-exports maintain backward compatibility for tests and
# external code that imports from mcp_server directly.

# Tool implementations (for tests that call handlers directly)
from .tools.handlers_orchestration import (
    delegate_tool_impl as delegate,
    think_impl as think,
    batch_impl as batch,
    session_compact_impl as session_compact,
    session_stats_impl as session_stats,
)
from .tools.admin import (
    health_impl as health,
    switch_model_impl as switch_model,
    switch_backend_impl as switch_backend,
    get_model_info_impl as get_model_info,
    models_impl as models,
    queue_status_impl as queue_status,
)

# Routing utilities
from .routing import detect_code_content
from .config import detect_model_tier

# Orchestration service
from .orchestration.service import get_orchestration_service

# Stats utilities
from .stats_handler import save_all_stats_async

# Queue and routing
from .queue import ModelQueue
from .routing import ModelRouter, get_router

# Ensure all data directories exist
paths.ensure_directories()

# Initialize the LLM module with callbacks and model queue
init_llm_module(
    stats_callback=update_stats_sync,
    save_stats_callback=save_stats_background,
    model_queue=model_queue,
)

# Load MCP server configurations (but don't start servers yet - that's async)
from .backend_manager import backend_manager
mcp_client_manager.load_config(backend_manager.get_mcp_servers())


# ============================================================ 
# MCP SERVER SETUP
# ============================================================ 

def _build_dynamic_instructions(project_path: str | None = None) -> str:
    """Build MCP instructions with dynamic playbook content."""
    from .playbook import get_playbook_manager
    from .context import current_project_path
    
    path = project_path or current_project_path.get() or str(Path.cwd())
    pm = get_playbook_manager()
    pm.set_project(Path(path))
    
    # Load playbook bullets
    priority_task_types = ["coding", "testing", "architecture", "debugging", "project", "git"]
    playbook_sections = []
    
    for task_type in priority_task_types:
        bullets = pm.get_top_bullets(task_type, limit=5)
        if bullets:
            bullet_lines = [f"- {b.content}" for b in bullets]
            playbook_sections.append(f"### {task_type.title()}\n" + "\n".join(bullet_lines))
    
    parts = []
    
    if playbook_sections:
        parts.append("# Project-Specific Strategies\n\nDelia has learned the following patterns:\n")
        parts.append("\n## LEARNED PLAYBOOK\n\n")
        parts.append("\n\n".join(playbook_sections))
        parts.append("\n\n")
    
    # Load base instructions
    inst_file = Path(__file__).parent / "mcp_instructions.md"
    parts.append(inst_file.read_text() if inst_file.exists() else "")
    
    return "".join(parts)


mcp = FastMCP("delia", instructions=_build_dynamic_instructions())

# ============================================================ 
# TOOL REGISTRATION (Profile-based)
# ============================================================ 
# Profiles: minimal (~15 tools), standard (~35 tools), full (~67 tools)

from .tools.handlers import register_tool_handlers
from .tools.admin import register_admin_tools
from .tools.resources import register_resource_tools
from .tools.consolidated import register_consolidated_tools
from .tools.files import register_file_tools
from .tools.lsp import register_lsp_tools
from .tools.mcp_management import register_mcp_management_tools

tool_profile = config.tool_profile
log.info("tool_profile_selected", profile=tool_profile)

# Profile-based tool registration
# minimal: ~20 tools (file ops, LSP, delegate)
# standard: ~45 tools (+ framework, consolidated, admin)  
# full: ~67 tools (everything)

# MINIMAL: Always register core tools
register_file_tools(mcp)
register_lsp_tools(mcp)

if tool_profile in ("standard", "full"):
    # STANDARD: Add framework and admin tools
    register_consolidated_tools(mcp)
    register_admin_tools(mcp)
    register_tool_handlers(mcp)  # orchestration + framework

if tool_profile == "full":
    # FULL: Add resources and MCP management
    register_resource_tools(mcp)
    register_mcp_management_tools(mcp)

log.info("tools_registered", profile=tool_profile, count=len(mcp._tool_manager._tools))

# Register Middleware
from .middleware import UserTrackingMiddleware
mcp.add_middleware(UserTrackingMiddleware())

# Register Auth Routes
if config.auth_enabled:
    register_auth_routes(mcp, tracker)
    register_oauth_routes(mcp)
else:
    log.info("auth_disabled", message="Authentication routes not registered.")


# ============================================================ 
# SINGLETON & PROXY SUPPORT
# ============================================================ 

def _acquire_singleton_lock() -> bool:
    """Try to acquire singleton lock for stdio mode.
    
    Returns:
        True if lock acquired (we're the only instance)
        False if another instance is running
    """
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        lock_fd = open(_LOCK_FILE, "w")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Write PID for debugging
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        # Keep file open to hold lock
        return True
    except (IOError, OSError):
        return False


def _get_http_server_port() -> int | None:
    """Get port of running HTTP server if any."""
    if not _HTTP_PORT_FILE.exists():
        return None
    try:
        port = int(_HTTP_PORT_FILE.read_text().strip())
        # Verify server is actually running
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                return port
    except Exception:
        pass
    return None


def _save_http_server_port(port: int) -> None:
    """Save HTTP server port for proxy clients."""
    _HTTP_PORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HTTP_PORT_FILE.write_text(str(port))


def _run_proxy_mode(port: int) -> None:
    """Run as a proxy, forwarding stdio to HTTP server.
    
    This allows multiple AI tools to share a single Delia HTTP backend.
    """
    import json
    import select
    
    # Restore stdout for JSON-RPC
    os.dup2(_original_stdout_fd, 1)
    sys.stdout = _original_stdout
    
    base_url = f"http://localhost:{port}"
    
    print(json.dumps({
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "message": f"Delia proxy mode: forwarding to HTTP server on port {port}"
        }
    }), file=sys.stderr)
    
    with httpx.Client(timeout=30.0) as client:
        while True:
            # Read JSON-RPC request from stdin
            if select.select([sys.stdin], [], [], 0.1)[0]:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    
                    request = json.loads(line)
                    
                    # Forward to HTTP server
                    resp = client.post(
                        f"{base_url}/mcp",
                        json=request,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    # Return response
                    print(resp.text, flush=True)
                    
                except json.JSONDecodeError:
                    continue
                except httpx.RequestError as e:
                    # Connection lost, exit proxy
                    print(json.dumps({
                        "jsonrpc": "2.0",
                        "error": {"code": -32000, "message": f"Proxy error: {e}"}
                    }), flush=True)
                    break


# ============================================================ 
# RUN SERVER
# ============================================================ 

def run_server(
    transport: str = "stdio",
    port: int = 8200,
    host: str = "0.0.0.0",
) -> None:
    """Run the Delia MCP server."""
    global log
    transport = transport.lower().strip()

    if transport == "stdio":
        # Check if HTTP server is running - use proxy mode if so
        http_port = _get_http_server_port()
        if http_port:
            log.info("proxy_mode", port=http_port, reason="HTTP server detected")
            _run_proxy_mode(http_port)
            return
        
        # Check singleton lock to prevent multiple heavy instances
        if not _acquire_singleton_lock():
            # Another stdio instance is running - warn and continue anyway
            # (we can't fully prevent this without breaking some use cases)
            log.warning(
                "multiple_instances_detected",
                message="Another Delia stdio server is running. Consider using HTTP mode for shared access.",
                hint="Run 'delia serve --transport http' and configure AI tools to connect to it"
            )
        
        # Ensure we are currently redirecting everything to stderr
        # This re-applies the redirection in case it was somehow reset, 
        # but relies primarily on the early setup.
        with contextlib.redirect_stdout(sys.stderr):
            _configure_structlog(use_stderr=True)
            log = structlog.get_logger()
            log.info("stdio_logging_configured", destination="stderr")

            asyncio.run(startup_handler())
            atexit.register(lambda: asyncio.run(shutdown_handler()))
            start_prewarm_task()

            log.info("server_starting", transport="stdio", auth_enabled=False)
            
        # CRITICAL: Restore stdout ONLY for the MCP protocol
        # Restore both the Python object and the underlying file descriptor
        os.dup2(_original_stdout_fd, 1)
        sys.stdout = _original_stdout
        
        # CRITICAL: show_banner=False keeps stdout clean for JSON-RPC protocol
        mcp.run(show_banner=False)

    elif transport in ("http", "streamable-http"):
        # For HTTP, we can leave stdout redirected to stderr (logs)
        _configure_structlog(use_stderr=True)
        log = structlog.get_logger()
        log.info("http_logging_configured", destination="stderr")

        asyncio.run(init_database())
        asyncio.run(startup_handler())
        atexit.register(lambda: asyncio.run(shutdown_handler()))
        start_prewarm_task()

        # Save port for proxy clients to discover
        _save_http_server_port(port)
        atexit.register(lambda: _HTTP_PORT_FILE.unlink(missing_ok=True))

        auth_endpoints = ["/auth/register"] if config.auth_enabled else []
        log.info("server_starting", transport="http", host=host, port=port, endpoints=auth_endpoints, proxy_enabled=True)
        mcp.run(transport="http", host=host, port=port)

    elif transport == "sse":
        _configure_structlog(use_stderr=True)
        log = structlog.get_logger()
        log.info("sse_logging_configured", destination="stderr")

        asyncio.run(init_database())
        asyncio.run(startup_handler())
        atexit.register(lambda: asyncio.run(shutdown_handler()))
        start_prewarm_task()

        log.info("server_starting", transport="sse", host=host, port=port)
        mcp.run(transport="sse", host=host, port=port)

    else:
        raise ValueError(f"Unknown transport: {transport}")


if __name__ == "__main__":
    # Minimal entry point for direct execution
    import argparse
    parser = argparse.ArgumentParser(description="Delia MCP Server")
    parser.add_argument("--transport", "-t", default="stdio", help="Transport: stdio, sse, http")
    parser.add_argument("--port", "-p", type=int, default=8200, help="Port for HTTP/SSE")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()
    
    run_server(transport=args.transport, port=args.port, host=args.host)