# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Lightweight MCP Proxy Mode.

This is the PRIMARY way Delia runs for AI tools. Instead of each tool spawning
a heavy MCP server (~1GB), all tools share a single HTTP backend via lightweight
proxies (~20MB each).

Architecture:
    [Claude Code] ─stdio──> [Proxy] ─http──┐
    [Gemini]      ─stdio──> [Proxy] ─http──┼──> [HTTP Backend]
    [Copilot]     ─stdio──> [Proxy] ─http──┘

Memory: 3 AI tools = ~60MB (proxies) + ~1GB (backend) = ~1.1GB
vs old: 3 AI tools = ~3GB (3 heavy servers)
"""

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Port file for discovering running HTTP server
_HTTP_PORT_FILE = Path.home() / ".delia" / "http_server.port"
_DEFAULT_HTTP_PORT = 8765


def get_http_server_port() -> int | None:
    """Get port of running HTTP server if any."""
    if not _HTTP_PORT_FILE.exists():
        return None
    try:
        port = int(_HTTP_PORT_FILE.read_text().strip())
        # Verify server is actually running by checking if port is open
        # FastMCP doesn't have /health, so we just check connectivity
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex(("localhost", port))
            if result == 0:
                return port
    except Exception:
        pass
    # Port file exists but server not responding - clean up stale file
    _HTTP_PORT_FILE.unlink(missing_ok=True)
    return None


def start_http_backend(port: int = _DEFAULT_HTTP_PORT) -> int | None:
    """Start HTTP backend as a daemon process.

    Returns:
        Port number if started successfully, None if failed.
    """
    import shutil

    # Find delia executable
    delia_exe = shutil.which("delia")
    if not delia_exe:
        # Try the current venv
        venv_delia = Path(sys.prefix) / "bin" / "delia"
        if venv_delia.exists():
            delia_exe = str(venv_delia)
        else:
            return None

    # Start as detached daemon
    log_file = Path.home() / ".delia" / "backend.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as log:
        subprocess.Popen(
            [delia_exe, "serve", "--transport", "http", "--port", str(port)],
            stdout=log,
            stderr=log,
            start_new_session=True,  # Detach from parent
            close_fds=True,
        )

    # Wait for server to start (up to 10 seconds)
    for _ in range(20):
        time.sleep(0.5)
        server_port = get_http_server_port()
        if server_port:
            return server_port

    return None


def run_proxy(port: int) -> None:
    """Run as a lightweight proxy, forwarding stdio to HTTP server.

    This allows multiple AI tools to share a single Delia HTTP backend
    without each spawning a full heavy MCP server.
    """
    base_url = f"http://localhost:{port}"

    # Send notification that we're in proxy mode (to stderr, not stdout)
    print(
        json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {
                "level": "info",
                "message": f"Delia proxy mode: connected to backend on port {port}"
            }
        }),
        file=sys.stderr
    )

    with httpx.Client(timeout=120.0) as client:
        while True:
            # Wait for input with timeout
            if select.select([sys.stdin], [], [], 0.5)[0]:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        # EOF - stdin closed
                        break

                    line = line.strip()
                    if not line:
                        continue

                    request = json.loads(line)

                    # Forward to HTTP server's MCP endpoint
                    resp = client.post(
                        f"{base_url}/mcp",
                        json=request,
                        headers={"Content-Type": "application/json"}
                    )

                    # Return response to stdout
                    print(resp.text, flush=True)

                except json.JSONDecodeError as e:
                    # Invalid JSON - send error response
                    error_resp = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": f"Parse error: {e}"},
                        "id": None
                    }
                    print(json.dumps(error_resp), flush=True)

                except httpx.RequestError as e:
                    # HTTP error - connection to backend failed
                    error_resp = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32000, "message": f"Backend error: {e}"},
                        "id": None
                    }
                    print(json.dumps(error_resp), flush=True)
                    # Don't exit - backend might recover


def run_stdio_via_proxy() -> bool:
    """Attempt to run as lightweight proxy to HTTP backend.

    If an HTTP backend is running, connects as a thin proxy.
    Otherwise, returns False so caller can run full server.

    Note: FastMCP uses SSE for HTTP responses, which our simple proxy
    doesn't fully support yet. For now, we just check if backend exists
    and let the user know to use HTTP mode for multi-instance scenarios.

    Returns:
        True if running as proxy (caller should exit)
        False if no backend found (caller should start full server)
    """
    port = get_http_server_port()

    if port:
        # HTTP backend is running - we could proxy but SSE is complex
        # For now, just run the full server with lazy loading
        # The lazy container makes each instance much lighter
        print(
            f"Note: HTTP backend running on port {port}. "
            "For best memory efficiency, configure AI tools to use "
            f"http://localhost:{port}/mcp directly.",
            file=sys.stderr
        )
        return False  # Fall back to full server with lazy loading

    # No backend - also run full server
    return False


if __name__ == "__main__":
    if not run_stdio_via_proxy():
        sys.exit(1)
