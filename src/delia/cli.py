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
Delia CLI - Setup wizard and client installation commands.

Provides easy setup and configuration for Delia MCP server:
- delia init: Interactive setup wizard
- delia install: Auto-configure MCP clients
- delia doctor: Diagnose configuration issues
- delia run: Start the MCP server (default)
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer

from .paths import SETTINGS_FILE, USER_DELIA_DIR

# Rich console for pretty output (optional, graceful fallback)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm

    console: Console | None = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None


# ============================================================
# CONSTANTS
# ============================================================

# MCP Client configuration locations by platform
MCP_CLIENTS: dict[str, dict[str, Any]] = {
    "claude": {
        "name": "Claude Code",
        "config_key": "mcpServers",
        "paths": {
            "Linux": Path.home() / ".claude" / "mcp.json",
            "Darwin": Path.home() / ".claude" / "mcp.json",
            "Windows": Path.home() / ".claude" / "mcp.json",
        },
        "executables": ["claude"],  # Claude Code CLI
    },
    "vscode": {
        "name": "VS Code / GitHub Copilot",
        "config_key": "servers",
        "paths": {
            "Linux": Path.home() / ".config" / "Code" / "User" / "mcp.json",
            "Darwin": Path.home() / "Library" / "Application Support" / "Code" / "User" / "mcp.json",
            "Windows": Path.home() / "AppData" / "Roaming" / "Code" / "User" / "mcp.json",
        },
        "executables": ["code"],  # VS Code CLI
    },
    "vscode-insiders": {
        "name": "VS Code Insiders",
        "config_key": "servers",
        "paths": {
            "Linux": Path.home() / ".config" / "Code - Insiders" / "User" / "mcp.json",
            "Darwin": Path.home() / "Library" / "Application Support" / "Code - Insiders" / "User" / "mcp.json",
            "Windows": Path.home() / "AppData" / "Roaming" / "Code - Insiders" / "User" / "mcp.json",
        },
        "executables": ["code-insiders"],  # VS Code Insiders CLI
    },
    "cursor": {
        "name": "Cursor",
        "config_key": "mcpServers",
        "paths": {
            "Linux": Path.home() / ".cursor" / "mcp.json",
            "Darwin": Path.home() / ".cursor" / "mcp.json",
            "Windows": Path.home() / ".cursor" / "mcp.json",
        },
        "executables": ["cursor"],  # Cursor CLI
    },
    "gemini": {
        "name": "Gemini CLI",
        "config_key": "mcpServers",
        "paths": {
            "Linux": Path.home() / ".gemini" / "settings.json",
            "Darwin": Path.home() / ".gemini" / "settings.json",
            "Windows": Path.home() / ".gemini" / "settings.json",
        },
        "executables": ["gemini"],  # Gemini CLI
    },
    "copilot-cli": {
        "name": "GitHub Copilot CLI",
        "config_key": "servers",
        "paths": {
            "Linux": Path.home() / ".copilot-cli" / "mcp.json",
            "Darwin": Path.home() / ".copilot-cli" / "mcp.json",
            "Windows": Path.home() / ".copilot-cli" / "mcp.json",
        },
        "executables": ["gh"],  # GitHub CLI (copilot is an extension)
    },
    "windsurf": {
        "name": "Windsurf",
        "config_key": "mcpServers",
        "paths": {
            "Linux": Path.home() / ".windsurf" / "mcp.json",
            "Darwin": Path.home() / ".windsurf" / "mcp.json",
            "Windows": Path.home() / ".windsurf" / "mcp.json",
        },
        "executables": ["windsurf"],  # Windsurf CLI
    },
}

# Backend detection endpoints
BACKEND_ENDPOINTS = [
    ("ollama", "http://localhost:11434", "/api/tags"),
    ("llamacpp", "http://localhost:8080", "/health"),
    ("lmstudio", "http://localhost:1234", "/v1/models"),
    ("vllm", "http://localhost:8000", "/v1/models"),
]


# ============================================================
# HELPER CLASSES
# ============================================================


@dataclass
class DetectedBackend:
    """Information about a detected LLM backend."""

    provider: str
    url: str
    models: list[str]
    healthy: bool


@dataclass
class DetectedClient:
    """Information about a detected MCP client."""

    id: str
    name: str
    config_path: Path
    installed: bool
    configured: bool  # Delia already in config


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def print_header(text: str) -> None:
    """Print a styled header."""
    if RICH_AVAILABLE and console:
        console.print(f"\n[bold blue]{text}[/bold blue]")
    else:
        print(f"\n{text}")
        print("=" * len(text))


def print_success(text: str) -> None:
    """Print success message."""
    if RICH_AVAILABLE and console:
        console.print(f"  [green][OK] {text}[/green]")
    else:
        print(f"  [OK] {text}")


def print_warning(text: str) -> None:
    """Print warning message."""
    if RICH_AVAILABLE and console:
        console.print(f"  [yellow][WARN] {text}[/yellow]")
    else:
        print(f"  [WARN] {text}")


def print_error(text: str) -> None:
    """Print error message."""
    if RICH_AVAILABLE and console:
        console.print(f"  [red][FAIL] {text}[/red]")
    else:
        print(f"  [FAIL] {text}")


def print_info(text: str) -> None:
    """Print info message."""
    if RICH_AVAILABLE and console:
        console.print(f"  [dim]{text}[/dim]")
    else:
        print(f"  {text}")


def prompt_confirm(text: str, default: bool = True) -> bool:
    """Prompt for yes/no confirmation."""
    if RICH_AVAILABLE:
        return Confirm.ask(text, default=default)
    else:
        suffix = " [Y/n]: " if default else " [y/N]: "
        response = input(text + suffix).strip().lower()
        if not response:
            return default
        return response in ("y", "yes")


def get_delia_root() -> Path:
    """Get the root directory of the Delia installation."""
    # Try to find the project root by looking for pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback: use the directory containing the src folder
    return Path(__file__).resolve().parent.parent.parent


def get_platform() -> str:
    """Get the current platform name."""
    return platform.system()


# ============================================================
# BACKEND DETECTION
# ============================================================


def detect_backends() -> list[DetectedBackend]:
    """Detect available LLM backends on localhost."""
    detected: list[DetectedBackend] = []

    for provider, base_url, health_path in BACKEND_ENDPOINTS:
        try:
            # Check health
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{base_url}{health_path}")
                if response.status_code != 200:
                    continue

                # Get models
                models: list[str] = []
                if provider == "ollama":
                    models_resp = client.get(f"{base_url}/api/tags")
                    if models_resp.status_code == 200:
                        data = models_resp.json()
                        models = [m.get("name", "") for m in data.get("models", [])]
                elif provider in ("llamacpp", "lmstudio", "vllm"):
                    models_resp = client.get(f"{base_url}/v1/models")
                    if models_resp.status_code == 200:
                        data = models_resp.json()
                        models = [m.get("id", "") for m in data.get("data", [])]

                detected.append(
                    DetectedBackend(
                        provider=provider,
                        url=base_url,
                        models=models,
                        healthy=True,
                    )
                )
        except Exception:  # noqa: S112  # Expected: skip unavailable backends
            continue

    return detected


# Import shared model tier assignment (canonical implementation)
from .model_detection import assign_models_to_tiers  # noqa: E402


# ============================================================
# CLIENT DETECTION
# ============================================================


def detect_clients() -> list[DetectedClient]:
    """Detect installed MCP clients by checking for their executables."""
    detected: list[DetectedClient] = []
    plat = get_platform()
    delia_root = get_delia_root()

    for client_id, client_info in MCP_CLIENTS.items():
        # Check if client executable exists in PATH
        installed = False
        for executable in client_info.get("executables", []):
            if shutil.which(executable):
                installed = True
                break

        if not installed:
            continue

        # Get config path for this platform
        config_path = client_info["paths"].get(plat)
        if not config_path:
            continue

        # Check if Delia is already configured
        configured = False
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                config_key = client_info["config_key"]
                if config_key in config and "delia" in config[config_key]:
                    # Treat mismatched configs as unconfigured so init can repair them
                    expected_config = generate_client_config(client_id, delia_root)
                    existing_config = config[config_key]["delia"]
                    configured = existing_config == expected_config
            except Exception:  # noqa: S110  # Expected: optional config check
                pass

        detected.append(
            DetectedClient(
                id=client_id,
                name=client_info["name"],
                config_path=config_path,
                installed=installed,
                configured=configured,
            )
        )

    return detected


def generate_client_config(client_id: str, delia_root: Path) -> dict[str, Any]:
    """Generate the MCP configuration for a specific client."""
    client_info = MCP_CLIENTS.get(client_id)
    if not client_info:
        raise ValueError(f"Unknown client: {client_id}")

    # Prioritize global 'delia' executable if available in PATH
    delia_path = shutil.which("delia")
    if delia_path:
        return {
            "command": delia_path,
            "args": ["serve"],
        }

    # Fallback: use uv to run from the source directory
    # This is more reliable for dev/editable installs where 'delia' might not be in PATH
    return {
        "command": "uv",
        "args": ["--directory", str(delia_root), "run", "delia", "serve"],
    }


def install_to_client(client_id: str, delia_root: Path, force: bool = False) -> bool:
    """
    Install Delia configuration to a specific MCP client.

    Returns True if successful, False otherwise.
    """
    client_info = MCP_CLIENTS.get(client_id)
    if not client_info:
        print_error(f"Unknown client: {client_id}")
        return False

    plat = get_platform()
    config_path = client_info["paths"].get(plat)
    if not config_path:
        print_error(f"Unsupported platform for {client_info['name']}")
        return False

    config_key = client_info["config_key"]

    # Load existing config or create new
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError:
            if not force:
                print_error(f"Invalid JSON in {config_path}")
                return False
            config = {}

    # Check if already configured
    if config_key not in config:
        config[config_key] = {}

    if "delia" in config[config_key] and not force:
        print_warning(f"Delia already configured in {client_info['name']}")
        return True

    # Generate and add Delia config
    server_config = generate_client_config(client_id, delia_root)
    config[config_key]["delia"] = server_config

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write config
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print_error(f"Failed to write config: {e}")
        return False


# ============================================================
# SETTINGS GENERATION
# ============================================================


def generate_settings(backends: list[DetectedBackend]) -> dict[str, Any]:
    """Generate settings.json content from detected backends."""
    settings: dict[str, Any] = {
        "version": "1.0",
        "system": {
            "gpu_memory_limit_gb": 24,
            "memory_buffer_gb": 2,
            "max_concurrent_requests_per_backend": 1,
        },
        "backends": [],
        "routing": {
            "prefer_local": True,
            "fallback_enabled": True,
        },
    }

    for i, backend in enumerate(backends):
        models = assign_models_to_tiers(backend.models)
        backend_config = {
            "id": f"{backend.provider}-local",
            "name": f"{backend.provider.title()} Local",
            "provider": backend.provider,
            "type": "local",
            "url": backend.url,
            "enabled": True,
            "priority": i + 1,
            "models": models,
        }
        settings["backends"].append(backend_config)

    return settings


# ============================================================
# CLI COMMANDS
# ============================================================

app = typer.Typer(
    name="delia",
    help="Delia - Local LLM chat and orchestration",
    no_args_is_help=False,
)


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration"),
) -> None:
    """
    Interactive setup wizard for first-time configuration.

    Detects available backends, configures model tiers, and optionally
    installs Delia to detected MCP clients.
    """
    # Ensure all data/cache directories exist
    from .paths import ensure_directories
    ensure_directories()

    print()
    if RICH_AVAILABLE and console:
        console.print(Panel.fit("[bold green]Welcome to Delia Setup![/bold green]", border_style="green"))
    else:
        print("🍈 Welcome to Delia Setup!")
        print("=" * 40)

    delia_root = get_delia_root()

    # Determine target path for settings
    # Default: Global user directory (~/.delia/settings.json)
    target_path = USER_DELIA_DIR / "settings.json"

    # Override: If local settings.json exists in CWD, use that instead
    cwd_settings = Path.cwd() / "settings.json"
    if cwd_settings.exists():
        target_path = cwd_settings
        print_info(f"Detected local configuration at {target_path}")
    else:
        # Ensure global directory exists
        USER_DELIA_DIR.mkdir(parents=True, exist_ok=True)

    # Check existing settings
    if target_path.exists() and not force:
        print_warning(f"settings.json already exists at {target_path}")
        if not prompt_confirm("Overwrite existing configuration?", default=False):
            print_info("Setup cancelled. Use --force to overwrite.")
            raise typer.Exit(0)

    # Step 1: Detect backends
    print_header("Checking for LLM backends...")

    backends = detect_backends()

    if not backends:
        print_error("No LLM backends found!")
        print_info("Please install and start one of:")
        print_info("  • Ollama: https://ollama.ai")
        print_info("  • LM Studio: https://lmstudio.ai")
        print_info("  • llama.cpp: https://github.com/ggerganov/llama.cpp")
        print()
        print_info("Then run 'delia init' again.")
        raise typer.Exit(1)

    for backend in backends:
        model_count = len(backend.models)
        print_success(f"{backend.provider.title()} found at {backend.url}")
        if backend.models:
            print_info(f"     Models: {', '.join(backend.models[:5])}")
            if model_count > 5:
                print_info(f"     ... and {model_count - 5} more")

    # Step 2: Generate settings
    print_header("Configuring model tiers...")

    settings = generate_settings(backends)

    # Show tier assignments
    if settings["backends"]:
        first_backend = settings["backends"][0]
        models = first_backend.get("models", {})
        for tier, model in models.items():
            print_info(f"  {tier:10} → {model}")

    # Step 3: Save settings
    try:
        with open(target_path, "w") as f:
            json.dump(settings, f, indent=2)
        print_success(f"Configuration saved to {target_path}")
    except Exception as e:
        print_error(f"Failed to save settings: {e}")
        raise typer.Exit(1) from None

    # Step 4: Detect and offer to configure clients
    print_header("Checking for MCP clients...")

    clients = detect_clients()

    if not clients:
        print_info("No MCP clients detected.")
        print_info("Supported clients: Claude Code, VS Code, Gemini CLI, Cursor, Windsurf")
    else:
        for client in clients:
            if client.configured:
                print_success(f"{client.name} (already configured)")
            else:
                print_info(f"{client.name} found")

        # Offer to install
        unconfigured = [c for c in clients if not c.configured]
        if unconfigured:
            print()
            if prompt_confirm(f"Install Delia to {len(unconfigured)} detected client(s)?", default=True):
                for client in unconfigured:
                    if install_to_client(client.id, delia_root, force=force):
                        print_success(f"{client.name} configured")
                    else:
                        print_error(f"Failed to configure {client.name}")

    # Done!
    print()
    if RICH_AVAILABLE and console:
        console.print(Panel.fit("[bold green]Setup complete![/bold green] Restart your AI assistant to use Delia.", border_style="green"))
    else:
        print("🍉 Setup complete! Restart your AI assistant to use Delia.")


@app.command()
def install(
    client: str = typer.Argument(None, help="Client to install (claude, vscode, gemini, cursor, etc.)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration"),
    list_clients: bool = typer.Option(False, "--list", "-l", help="List available clients"),
) -> None:
    """
    Install Delia to MCP client(s).

    Without arguments, auto-detects and installs to all found clients.
    Specify a client name to install to a specific client only.
    """
    delia_root = get_delia_root()

    # List available clients
    if list_clients:
        print_header("Available MCP Clients")
        for client_id, info in MCP_CLIENTS.items():
            print(f"  {client_id:15} - {info['name']}")
        raise typer.Exit(0)

    # Install to specific client
    if client:
        client_lower = client.lower()
        if client_lower not in MCP_CLIENTS:
            print_error(f"Unknown client: {client}")
            print_info("Use 'delia install --list' to see available clients")
            raise typer.Exit(1)

        if install_to_client(client_lower, delia_root, force=force):
            print_success(f"Delia installed to {MCP_CLIENTS[client_lower]['name']}")
            print_info("Restart your AI assistant to use Delia.")
        else:
            raise typer.Exit(1)
        return

    # Auto-detect and install to all
    print_header("Detecting MCP clients...")

    clients = detect_clients()

    if not clients:
        print_error("No MCP clients detected.")
        print_info("Supported clients:")
        for info in MCP_CLIENTS.values():
            print_info(f"  • {info['name']}")
        print()
        print_info("Install a client and run 'delia install' again,")
        print_info("or specify a client: 'delia install claude'")
        raise typer.Exit(1)

    installed_count = 0
    for detected_client in clients:
        if detected_client.configured and not force:
            print_success(f"{detected_client.name} (already configured)")
            continue

        if install_to_client(detected_client.id, delia_root, force=force):
            print_success(f"{detected_client.name} configured")
            installed_count += 1
        else:
            print_error(f"Failed to configure {detected_client.name}")

    if installed_count > 0:
        print()
        print_info(f"Installed to {installed_count} client(s). Restart your AI assistant to use Delia.")


@app.command()
def doctor() -> None:
    """
    Diagnose configuration and connectivity issues.

    Checks backends, settings, and MCP client configurations.
    """
    print()
    if RICH_AVAILABLE and console:
        console.print("[bold]Delia Health Check[/bold]")
    else:
        print("Delia Health Check")
        print("=" * 40)

    delia_root = get_delia_root()
    issues: list[str] = []

    # Check settings.json
    print_header("Configuration")

    # Use resolved settings file path
    if SETTINGS_FILE.exists():
        print_success(f"settings.json found at {SETTINGS_FILE}")
        try:
            with open(SETTINGS_FILE) as f:
                settings = json.load(f)
            backend_count = len(settings.get("backends", []))
            print_info(f"     {backend_count} backend(s) configured")
        except Exception as e:
            print_error(f"Invalid settings.json: {e}")
            issues.append("Fix settings.json syntax errors")
    else:
        print_error("settings.json not found")
        issues.append("Run 'delia init' to create configuration")

    # Check backends
    print_header("Backends")

    backends = detect_backends()
    if backends:
        for backend in backends:
            print_success(f"{backend.provider.title()} at {backend.url}")
            print_info(f"     {len(backend.models)} model(s) available")
    else:
        print_error("No backends responding")
        issues.append("Start Ollama or another LLM backend")

    # Check MCP clients
    print_header("MCP Integrations")

    clients = detect_clients()
    if clients:
        for client in clients:
            if client.configured:
                # Verify path in config is still valid
                try:
                    with open(client.config_path) as f:
                        config = json.load(f)
                    client_info = MCP_CLIENTS[client.id]
                    delia_config = config.get(client_info["config_key"], {}).get("delia", {})

                    # Check if path/command is valid
                    if "command" in delia_config:
                        cmd = delia_config["command"]
                        if shutil.which(cmd):
                            print_success(f"{client.name}: configured")
                        else:
                            print_warning(f"{client.name}: command '{cmd}' not found")
                            issues.append(f"Reinstall Delia to {client.name}: delia install {client.id}")
                    else:
                        print_success(f"{client.name}: configured")
                except Exception:
                    print_warning(f"{client.name}: config unreadable")
            else:
                print_info(f"{client.name}: detected but not configured")
                issues.append(f"Run 'delia install {client.id}' to configure")
    else:
        print_info("No MCP clients detected")

    # Summary
    print_header("Summary")

    if not issues:
        print_success("All checks passed!")
        print()
        print_info("Delia is ready to use. Start with:")
        print_info("  delia run              # STDIO mode (default)")
        print_info("  delia run -t sse       # HTTP/SSE mode")
    else:
        print_warning(f"{len(issues)} issue(s) found:")
        print()
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")


@app.command()
def run(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio, sse, http"),
    port: int = typer.Option(8200, "--port", "-p", help="Port for HTTP/SSE"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
) -> None:
    """
    Start the Delia MCP server.

    This is the default command when running 'delia' without arguments.
    """
    # Import and run the server
    from .mcp_server import run_server

    run_server(transport=transport, port=port, host=host)


@app.command()
def serve(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio, sse, http"),
    port: int = typer.Option(8200, "--port", "-p", help="Port for HTTP/SSE"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
) -> None:
    """
    Start the Delia MCP server (alias for 'run').

    Use this command in MCP client configurations.
    """
    run(transport=transport, port=port, host=host)


@app.command()
def api(
    port: int = typer.Option(None, "--port", "-p", help="Port for API server (default: auto-select)"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
) -> None:
    """
    Start the Delia HTTP API server for CLI frontend.

    This server provides SSE streaming endpoints for the TypeScript CLI.

    Endpoints:
        POST /api/agent/run   - Run agent with SSE streaming
        GET  /api/health      - Health check
        GET  /api/sessions    - List sessions

    Example:
        delia api --port 34589
    """
    from .api import run_api

    if port is None:
        port = _find_free_port()
        print_info(f"Using port {port}")

    run_api(host=host, port=port)


def _find_free_port(start: int = 34589, end: int = 34689) -> int:
    """Find a free port in the given range."""
    import socket
    import random

    # Try random ports in range
    ports = list(range(start, end))
    random.shuffle(ports)

    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port

    # Fallback: let OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@app.command()
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Model tier (quick/coder/moe) or specific model"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend type (local/remote)"),
    session: str = typer.Option(None, "--session", "-s", help="Resume existing session by ID"),
    port: int = typer.Option(None, "--port", "-p", help="API server port (default: auto-select)"),
) -> None:
    """
    Start an interactive chat session.

    This launches the chat interface, automatically starting the API server
    if needed. The chat provides a REPL-style interface for conversations.

    Examples:
        delia chat                    # Start chat with default settings
        delia chat --model coder      # Use coder model tier
        delia chat --session abc123   # Resume existing session
    """
    import atexit
    import signal
    import socket
    import sys
    import time

    # Auto-select port if not specified
    if port is None:
        port = _find_free_port()

    api_url = f"http://localhost:{port}"

    # Check if API server is already running
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    api_process = None

    if not is_port_in_use(port):
        print_info(f"Starting API server on port {port}...")

        # Pass settings file to subprocess via environment variable
        # This ensures the API uses the same settings as the CLI
        env = os.environ.copy()
        env["DELIA_SETTINGS_FILE"] = str(SETTINGS_FILE)

        # Start API server in background
        # Capture stderr for debugging startup failures
        import tempfile
        stderr_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
        api_process = subprocess.Popen(
            [sys.executable, "-m", "delia.api", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            env=env,
        )

        # Register cleanup
        def cleanup():
            if api_process and api_process.poll() is None:
                api_process.terminate()
                try:
                    api_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    api_process.kill()

        atexit.register(cleanup)

        # Also handle signals
        def signal_handler(sig, frame):
            cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Wait for server to be ready
        for _ in range(50):  # 5 second timeout (increased from 3s)
            if is_port_in_use(port):
                break
            # Check if process died
            if api_process.poll() is not None:
                stderr_file.close()
                with open(stderr_file.name) as f:
                    stderr_content = f.read()
                os.unlink(stderr_file.name)
                print_error("API server process exited unexpectedly")
                if stderr_content.strip():
                    print_error(f"Error output:\n{stderr_content[:500]}")
                raise typer.Exit(1)
            time.sleep(0.1)
        else:
            # Timeout - server didn't respond
            stderr_file.close()
            with open(stderr_file.name) as f:
                stderr_content = f.read()
            os.unlink(stderr_file.name)
            print_error("API server failed to start (timeout)")
            if stderr_content.strip():
                print_info(f"Server output:\n{stderr_content[:500]}")
            if api_process:
                api_process.terminate()
            raise typer.Exit(1)
        
        # Clean up temp file on success
        stderr_file.close()
        os.unlink(stderr_file.name)

        print_success("API server started")

    # Build delia-cli command
    cli_cmd = ["delia-cli", "chat", "--api-url", api_url]
    if model:
        cli_cmd.extend(["--model", model])
    if backend:
        cli_cmd.extend(["--backend", backend])
    if session:
        cli_cmd.extend(["--session", session])

    # Check if delia-cli is available
    if not shutil.which("delia-cli"):
        print_warning("delia-cli not found in PATH")
        
        # Check if we are in a dev environment (have packages/cli)
        cli_pkg_dir = delia_root / "packages" / "cli"
        if cli_pkg_dir.exists():
            print_info("Detected TypeScript CLI package. Attempting to build and link...")
            
            try:
                # 1. npm install
                print_info("  Running: npm install")
                subprocess.run(["npm", "install"], cwd=str(cli_pkg_dir), check=True, stdout=subprocess.DEVNULL)
                
                # 2. npm run build
                print_info("  Running: npm run build")
                subprocess.run(["npm", "run", "build"], cwd=str(cli_pkg_dir), check=True, stdout=subprocess.DEVNULL)
                
                # 3. npm link
                print_info("  Running: npm link")
                subprocess.run(["npm", "link"], cwd=str(cli_pkg_dir), check=True, stdout=subprocess.DEVNULL)
                
                print_success("TypeScript CLI built and linked successfully")
            except Exception as e:
                print_error(f"Failed to build/link TypeScript CLI: {e}")
                print_info("Please install it manually: cd packages/cli && npm install && npm run build && npm link")
                if api_process:
                    api_process.terminate()
                raise typer.Exit(1)
        else:
            print_info("Please install it manually: cd packages/cli && npm install && npm link")
            if api_process:
                api_process.terminate()
            raise typer.Exit(1)

    print()
    print_info("💡 Tip: Delia is agentic! Ask her to 'read files', 'run tests', or 'search the web'.")
    print()

    # Run the chat CLI
    try:
        result = subprocess.run(cli_cmd, check=False)
        raise typer.Exit(result.returncode)
    finally:
        if api_process and api_process.poll() is None:
            api_process.terminate()


@app.command()
def agent(
    task: str = typer.Argument(..., help="Task for the agent to complete"),
    model: str = typer.Option(None, "--model", "-m", help="Model tier (quick/coder/moe) or specific model"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Confine file operations to directory"),
    max_iterations: int = typer.Option(10, "--max-iterations", help="Maximum tool call iterations"),
    tools: str = typer.Option(None, "--tools", help="Comma-separated tools to enable (default: all)"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend type (local/remote)"),
    voting: bool = typer.Option(False, "--voting", help="Enable k-voting for reliability"),
    voting_k: int = typer.Option(None, "--voting-k", help="Override k value for voting"),
    # New features
    plan: bool = typer.Option(True, "--plan/--no-plan", help="Enable planning phase (default: True)"),
    reflect: bool = typer.Option(True, "--reflect/--no-reflect", help="Enable agentic reflection (default: True)"),
    reflection_confidence: str = typer.Option("normal", "--confidence", help="Reflection confidence (normal/high)"),
    tui: bool = typer.Option(True, "--tui/--no-tui", help="Use Rich TUI (default: True)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """
    Run an autonomous agent to complete a task.

    The agent can read files, search code, list directories, and fetch web content
    to accomplish the given task. It runs a loop of: think -> use tools -> respond.

    New in v3.2:
    - Planning: Generates a step-by-step plan before execution (--plan)
    - Reflection: Critiques its own work before finishing (--reflect)
    - TUI: Live dashboard of agent activity (--tui)

    Examples:
        delia agent "Refactor this file" --plan --reflect
        delia agent "Quick fix" --no-plan --no-reflect
    """
    from .agent_cli import AgentCLIConfig, run_agent_sync
    
    # Disable TUI if requested
    if not tui:
        import sys
        sys.modules["delia.agent_cli"].RICH_AVAILABLE = False

    # Parse tools
    tools_list = None
    if tools:
        tools_list = [t.strip() for t in tools.split(",") if t.strip()]

    config = AgentCLIConfig(
        model=model,
        workspace=workspace,
        max_iterations=max_iterations,
        tools=tools_list,
        backend_type=backend,
        verbose=verbose,
        voting_enabled=voting,
        voting_k=voting_k,
        planning_enabled=plan,
        reflection_enabled=reflect,
        reflection_confidence=reflection_confidence,
    )

    result = run_agent_sync(task, config)

    # Exit with error code if agent failed
    if not result.success:
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open configuration in editor"),
) -> None:
    """
    View or edit Delia configuration.
    """
    if not SETTINGS_FILE.exists():
        print_error("No configuration found. Run 'delia init' first.")
        raise typer.Exit(1)

    print_info(f"Using configuration at {SETTINGS_FILE}")

    if edit:
        # Try to open in default editor
        editor = os.environ.get("EDITOR", "nano")
        if get_platform() == "Windows":
            editor = os.environ.get("EDITOR", "notepad")

        subprocess.run([editor, str(SETTINGS_FILE)], check=False)
        return

    # Show configuration
    if show or not edit:
        try:
            with open(SETTINGS_FILE) as f:
                settings = json.load(f)

            if RICH_AVAILABLE and console:
                console.print_json(data=settings)
            else:
                print(json.dumps(settings, indent=2))
        except Exception as e:
            print_error(f"Failed to read configuration: {e}")
            raise typer.Exit(1) from None


@app.command()
def melons(
    task: str = typer.Option(None, "--task", "-t", help="Filter by task type (quick/coder/moe)"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """
    Display the melon leaderboard for Delia's model garden.

    Models earn melons for helpful responses:
    - 🍈 Regular melons for good work
    - 🏆 Golden melons (500 melons each) for top performers

    Golden melons influence routing - trusted models get more requests.

    Examples:
        delia melons              # Show full leaderboard
        delia melons --task coder # Show only coder task rankings
        delia melons --json       # Output as JSON
    """
    from .melons import get_melon_tracker

    tracker = get_melon_tracker()
    
    if json_output:
        leaderboard = tracker.get_leaderboard(task_type=task)
        data = {
            "leaderboard": [
                {
                    "model": s.model_id,
                    "task": s.task_type,
                    "melons": s.melons,
                    "golden_melons": s.golden_melons,
                    "total_value": s.total_melon_value,
                    "success_rate": round(s.success_rate, 3),
                    "total_responses": s.total_responses,
                }
                for s in leaderboard
            ]
        }
        print(json.dumps(data, indent=2))
        return

    # Pretty output
    leaderboard_text = tracker.get_leaderboard_text()
    
    if not leaderboard_text or "=" * 40 in leaderboard_text and len(leaderboard_text.split("\n")) < 5:
        print()
        print("🍈 DELIA'S MELON GARDEN")
        print("=" * 40)
        print()
        print("  No melons yet! The garden is empty.")
        print()
        print("  Models earn melons by being helpful.")
        print("  Use 'delia chat' to start growing the garden.")
        print()
        return

    print()
    print(leaderboard_text)


@app.command()
def setup() -> None:
    """
    Install delia globally to ~/.local/bin for easy access.

    This creates a wrapper script so you can run 'delia' from any terminal
    without manually activating a virtual environment.

    Example:
        delia setup
        # Now run 'delia chat' from anywhere
    """
    install_dir = Path.home() / ".local" / "bin"
    wrapper_path = install_dir / "delia"
    delia_root = get_delia_root()
    venv_path = delia_root / ".venv"

    # Check if venv exists
    if not venv_path.exists():
        print_error(f"Virtual environment not found at {venv_path}")
        print_info("Create it with: python -m venv .venv && source .venv/bin/activate && pip install -e .")
        raise typer.Exit(1)

    # Create install directory
    install_dir.mkdir(parents=True, exist_ok=True)

    # Create wrapper script
    wrapper_content = f"""#!/bin/bash
# Delia CLI wrapper - auto-activates virtual environment
source "{venv_path}/bin/activate"
exec delia "$@"
"""

    try:
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)
        wrapper_path.chmod(0o755)
        print_success(f"Installed to {wrapper_path}")
    except Exception as e:
        print_error(f"Failed to create wrapper: {e}")
        raise typer.Exit(1) from None

    # Check if ~/.local/bin is in PATH
    path_dirs = os.environ.get("PATH", "").split(":")
    local_bin = str(install_dir)
    if local_bin not in path_dirs and str(Path.home() / ".local/bin") not in path_dirs:
        print_warning("~/.local/bin is not in your PATH")
        print_info("Add to your shell config (~/.bashrc or ~/.zshrc):")
        print_info('  export PATH="$HOME/.local/bin:$PATH"')
        print()

    print()
    print_info("Restart your terminal or run: hash -r")
    print_info("Then use: delia chat")


@app.command()
def uninstall(
    client: str = typer.Argument(None, help="Client to uninstall from (or 'all' for all clients)"),
    full: bool = typer.Option(False, "--full", "-f", help="Also uninstall the delia package"),
) -> None:
    """
    Remove Delia from MCP client configuration(s).

    Examples:
        delia uninstall claude      # Remove from Claude Code only
        delia uninstall all         # Remove from all configured clients
        delia uninstall all --full  # Remove from all clients AND uninstall package
    """
    clients_to_uninstall: list[tuple[str, dict[str, Any]]] = []

    if client and client.lower() == "all":
        # Uninstall from all clients
        clients_to_uninstall = list(MCP_CLIENTS.items())
    elif client:
        # Uninstall from specific client
        client_lower = client.lower()
        if client_lower not in MCP_CLIENTS:
            print_error(f"Unknown client: {client}")
            print_info("Use 'delia install --list' to see available clients")
            raise typer.Exit(1)
        clients_to_uninstall = [(client_lower, MCP_CLIENTS[client_lower])]
    elif full:
        # --full without client means uninstall from all + package
        clients_to_uninstall = list(MCP_CLIENTS.items())
    else:
        print_error("Please specify a client or use 'all'")
        print_info("Examples:")
        print_info("  delia uninstall claude      # Remove from Claude Code")
        print_info("  delia uninstall all         # Remove from all clients")
        print_info("  delia uninstall all --full  # Full uninstall")
        raise typer.Exit(1)

    plat = get_platform()
    removed_count = 0

    for client_id, client_info in clients_to_uninstall:
        config_path = client_info["paths"].get(plat)

        if not config_path or not config_path.exists():
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)

            config_key = client_info["config_key"]
            if config_key in config and "delia" in config[config_key]:
                del config[config_key]["delia"]

                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                print_success(f"Removed from {client_info['name']}")
                removed_count += 1
        except Exception as e:
            print_error(f"Failed to update {client_info['name']}: {e}")

    if removed_count == 0 and not full:
        print_info("Delia was not configured in any clients")

    # Full uninstall: also remove the package
    if full:
        print_header("Uninstalling delia package...")

        try:
            result = subprocess.run(
                ["uv", "pip", "uninstall", "delia"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print_success("Delia package uninstalled")
            else:
                # Try pip as fallback
                result = subprocess.run(
                    ["pip", "uninstall", "-y", "delia"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    print_success("Delia package uninstalled")
                else:
                    print_warning("Package may not have been installed globally")
        except FileNotFoundError:
            print_warning("Could not find uv or pip to uninstall package")

        print()
        print_info("To completely remove Delia, also delete:")
        print_info(f"  • Data: ~/.delia/")
        print_info(f"  • Source: {get_delia_root()}")


# Default command: launch chat if no command specified
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """
    Delia - Local LLM orchestration and chat.

    Run without arguments to start an interactive chat session.
    Use 'delia serve' to start the MCP server.
    Use 'delia init' for first-time setup.
    """
    # Pre-flight check: ensure settings.json exists for commands that need it
    if ctx.invoked_subcommand not in ["init", "doctor", None]:
        if not SETTINGS_FILE.exists():
            print()
            print_warning("No configuration found!")
            print_info("Delia needs to detect your local LLMs (like Ollama) before starting.")
            print_info("Please run: [bold]delia init[/bold]")
            print()
            raise typer.Exit(1)

    if ctx.invoked_subcommand is None:
        # Check if initialized before default chat
        if not SETTINGS_FILE.exists():
            print()
            print_header("First Run Setup")
            print_info("Welcome! Let's get Delia configured for your system.")
            if prompt_confirm("Run interactive setup now?"):
                init(force=False)
            else:
                print_info("Setup skipped. Run 'delia init' later when ready.")
                raise typer.Exit(0)

        # Launch chat with defaults
        chat(model=None, backend=None, session=None, port=None)


if __name__ == "__main__":
    app()
