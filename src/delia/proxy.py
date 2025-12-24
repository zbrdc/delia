# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Lightweight MCP Proxy Mode.

When another Delia HTTP server is running, this module provides a minimal
proxy that forwards MCP stdio requests to the HTTP server, avoiding the
heavy initialization of the full MCP server.

Memory footprint: ~20MB vs ~1GB for full server.
"""

import json
import os
import select
import sys
from pathlib import Path

import httpx

# Port file for discovering running HTTP server
_HTTP_PORT_FILE = Path.home() / ".delia" / "http_server.port"


def get_http_server_port() -> int | None:
    """Get port of running HTTP server if any."""
    if not _HTTP_PORT_FILE.exists():
        return None
    try:
        port = int(_HTTP_PORT_FILE.read_text().strip())
        # Verify server is actually running
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                return port
    except Exception:
        pass
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
                "message": f"Delia proxy mode: forwarding to HTTP server on port {port}"
            }
        }),
        file=sys.stderr
    )

    with httpx.Client(timeout=60.0) as client:
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
                        "error": {"code": -32000, "message": f"Proxy error: {e}"},
                        "id": None
                    }
                    print(json.dumps(error_resp), flush=True)
                    # Backend might be down, exit proxy
                    break


def maybe_run_proxy() -> bool:
    """Check if we should run in proxy mode and do so if yes.

    Returns:
        True if running as proxy (caller should exit)
        False if no HTTP server found (caller should start full server)
    """
    port = get_http_server_port()
    if port:
        run_proxy(port)
        return True
    return False


if __name__ == "__main__":
    port = get_http_server_port()
    if port:
        run_proxy(port)
    else:
        print("No HTTP server found. Start one with: delia serve --transport http", file=sys.stderr)
        sys.exit(1)
