# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Lifecycle management for Delia MCP server.
Handles startup, shutdown, and background processes.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import threading
import time
import webbrowser
from typing import Any

import structlog

from . import paths
from .backend_manager import backend_manager
from .config import config
from .container import get_container
from .session_manager import get_session_manager
from .multi_user_tracking import tracker
from .auth import create_db_and_tables

log = structlog.get_logger()

# Dashboard subprocess handle (global for cleanup)
_dashboard_process: subprocess.Popen | None = None
_dashboard_port: int | None = None


def _find_free_port(start: int = 3001, end: int = 3100) -> int:
    """Find a free port for the dashboard."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


def _launch_dashboard() -> tuple[subprocess.Popen, int] | None:
    """
    Launch the Next.js dashboard in production mode.

    Returns (process, port) tuple or None if dashboard not available.
    """
    global _dashboard_process, _dashboard_port

    # Find dashboard directory relative to project root
    project_root = paths.PROJECT_ROOT
    dashboard_dir = project_root / "dashboard"

    if not dashboard_dir.exists():
        log.debug("dashboard_not_found", path=str(dashboard_dir))
        return None

    # Check if we have a built dashboard
    next_dir = dashboard_dir / ".next"
    if not next_dir.exists():
        log.warning("dashboard_not_built", hint="Run 'npm run build' in dashboard/")
        return None

    port = _find_free_port()

    try:
        # Launch Next.js in production mode
        env = os.environ.copy()
        env["PORT"] = str(port)
        env["DELIA_PROJECT_ROOT"] = str(project_root)

        proc = subprocess.Popen(
            ["npm", "run", "start"],
            cwd=str(dashboard_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        _dashboard_process = proc
        _dashboard_port = port
        url = f"http://localhost:{port}"
        log.info("dashboard_launched", port=port, url=url)

        # Open browser after brief delay to let server start
        def open_browser():
            time.sleep(2)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

        return proc, port

    except Exception as e:
        log.warning("dashboard_launch_failed", error=str(e))
        return None


def _shutdown_dashboard() -> None:
    """Terminate the dashboard subprocess."""
    global _dashboard_process

    if _dashboard_process is not None:
        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(_dashboard_process.pid), signal.SIGTERM)
            _dashboard_process.wait(timeout=5)
            log.info("dashboard_stopped")
        except Exception as e:
            log.debug("dashboard_stop_error", error=str(e))
            try:
                _dashboard_process.kill()
            except Exception:
                pass
        finally:
            _dashboard_process = None


async def init_database():
    """Initialize authentication database on startup."""
    if config.auth_enabled:
        try:
            await create_db_and_tables()
            log.info("auth_database_ready")
        except Exception as e:
            log.warning("auth_database_init_failed", error=str(e))


async def startup_handler():
    """
    Startup handler for the server.

    - Probes backends to detect available models
    - Starts background save task for tracker
    - Pre-warms tiktoken encoder to avoid first-call delay
    - Clears expired sessions
    """
    # Probe all enabled backends to detect available models
    # This ensures we use actual models, not stale config from settings.json
    for backend in backend_manager.get_enabled_backends():
        try:
            probed = await backend_manager.probe_backend(backend.id)
            if probed:
                log.info("backend_probed_startup", id=backend.id, models=list(backend.models.keys()))
        except Exception as e:
            log.warning("backend_probe_failed_startup", id=backend.id, error=str(e))

    if config.tracking_enabled:
        await tracker.start_background_save()

    # Pre-warm tiktoken encoder in background to avoid 100-200ms delay on first request
    # Run in thread pool to avoid blocking startup
    from .tokens import prewarm_encoder
    await asyncio.to_thread(prewarm_encoder)

    # Clear expired sessions on startup
    sm = get_session_manager()
    cleared = sm.clear_expired_sessions()
    log.info("session_cleanup_startup", cleared=cleared)

    # Launch dashboard (non-blocking)
    _launch_dashboard()


async def shutdown_handler():
    """
    Cleanup handler for graceful server shutdown.

    - Closes all backend HTTP clients to prevent connection leaks
    - Saves tracker state to disk
    - Stops the dashboard subprocess
    This is called automatically on server shutdown.
    """
    from .backend_manager import shutdown_backends

    # Stop dashboard first
    _shutdown_dashboard()

    await shutdown_backends()

    # Save tracker state on shutdown
    if config.tracking_enabled:
        await tracker.shutdown()
