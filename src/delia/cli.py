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

# Suppress deprecation warnings from third-party libraries (torchao, swig)
# Must be before any imports that trigger these
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchao")
warnings.filterwarnings("ignore", message="builtin type Swig.*has no __module__")

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Annotated, Optional

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

                # For llama.cpp, do a more specific check to avoid false positives
                if provider == "llamacpp":
                    try:
                        props_resp = client.get(f"{base_url}/props")
                        if props_resp.status_code != 200:
                            continue
                    except Exception:
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
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model tier (quick/coder/moe) or specific model")] = None,
    backend: Annotated[Optional[str], typer.Option("--backend", "-b", help="Force backend type (local/remote)")] = None,
    session: Annotated[Optional[str], typer.Option("--session", "-s", help="Resume existing session by ID")] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging")] = False,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w", help="Confine file operations to directory")] = None,
    allow_write: Annotated[bool, typer.Option("--allow-write", help="Enable file write operations")] = False,
    allow_exec: Annotated[bool, typer.Option("--allow-exec", help="Enable shell command execution")] = False,
    yolo: Annotated[bool, typer.Option("--yolo", help="Skip all security prompts (dangerous!)")] = False,
) -> None:
    """
    Start an interactive chat session using the Premium Ink-based TUI.
    """
    import subprocess
    import sys
    import shutil
    from .paths import USER_DELIA_DIR
    
    # 1. Start the API server in the background if it is not already running
    # The TUI requires the API server for SSE streaming.
    # We use a simple health check to see if it is up.
    api_url = "http://localhost:34589/api/health"
    api_running = False
    try:
        import httpx
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(api_url)
            if resp.status_code == 200:
                api_running = True
    except Exception:
        api_running = False

    if not api_running:
        print_info("Starting Delia API server...")
        # Start the API server as a background process
        # We use the same python executable to ensure environment consistency
        api_cmd = [sys.executable, "-m", "delia.cli", "api", "--port", "34589"]
        subprocess.Popen(
            api_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        # Give it a moment to bind
        time.sleep(2)

    # 2. Find and run the TUI
    # Try to find delia-tui in PATH, or look in the project root
    tui_path = shutil.which("delia-tui")
    
    if not tui_path:
        # Fallback: Check in the project root/tui directory
        project_root = get_delia_root()
        tui_dir = project_root / "tui"
        if (tui_dir / "dist" / "cli.js").exists():
            tui_cmd = ["node", str(tui_dir / "dist" / "cli.js")]
        else:
            print_error("TUI not found or not built.")
            print_info("Run: cd tui && npm install && npm run build")
            return
    else:
        tui_cmd = [tui_path]

    # Build arguments
    if model: tui_cmd.extend(["--model", model])
    if backend: tui_cmd.extend(["--backend", backend])
    if session: tui_cmd.extend(["--session", session])
    if workspace: tui_cmd.extend(["--workspace", workspace])
    if allow_write: tui_cmd.append("--allow-write")
    if allow_exec: tui_cmd.append("--allow-exec")
    if yolo: tui_cmd.append("--yolo")

    try:
        subprocess.run(tui_cmd, check=False)
    except KeyboardInterrupt:
        pass



@app.command()
def agent(
    task: Annotated[str, typer.Argument(help="Task for the agent to complete")],
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model tier")] = None,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w", help="Confine to directory")] = None,
    backend: Annotated[Optional[str], typer.Option("--backend", "-b", help="Backend type")] = None,
    yolo: Annotated[bool, typer.Option("--yolo", help="Auto-approve all operations")] = False,
) -> None:
    """
    Run an autonomous agent to complete a task.

    Launches the TUI with the given task. The agent can read files, search code,
    and use tools to accomplish the task.

    Examples:
        delia agent "Refactor this file"
        delia agent "Find all TODO comments" --workspace ./src
    """
    import subprocess
    import shutil

    # Find TUI
    tui_path = shutil.which("delia-tui")
    if not tui_path:
        project_root = get_delia_root()
        tui_dir = project_root / "tui"
        if (tui_dir / "dist" / "cli.js").exists():
            tui_cmd = ["node", str(tui_dir / "dist" / "cli.js")]
        else:
            print_error("TUI not found. Run: cd tui && npm install && npm run build")
            raise typer.Exit(1)
    else:
        tui_cmd = [tui_path]

    # Add options
    if model: tui_cmd.extend(["--model", model])
    if backend: tui_cmd.extend(["--backend", backend])
    if workspace: tui_cmd.extend(["--workspace", workspace])
    if yolo: tui_cmd.append("--yolo")
    tui_cmd.extend(["--task", task])

    try:
        subprocess.run(tui_cmd, check=False)
    except KeyboardInterrupt:
        pass



@app.command()
def index(
    force: bool = typer.Option(False, "--force", "-f", help="Force a complete re-indexing of all files"),
    summarize: bool = typer.Option(False, "--summarize", "-s", help="Use LLM to generate architectural summaries (Deep Scan)"),
    parallel: int = typer.Option(4, "--parallel", "-p", help="Number of parallel summarization tasks"),
) -> None:
    """
    Manually index the project for high-fidelity orchestration.
    
    Scans the current directory to:
    1. Generate hierarchical file summaries (Project Map)
    2. Build a global dependency graph (Symbol Graph)
    3. Generate project-specific playbooks (with --summarize)
    
    The generated playbooks contain strategic bullets about:
    - Tech stack and frameworks detected
    - Coding patterns and conventions
    - Project structure and key directories
    - Testing and verification practices
    """
    import asyncio
    from .orchestration.summarizer import get_summarizer
    from .orchestration.graph import get_symbol_graph
    from .llm import init_llm_module
    from .queue import ModelQueue

    async def run_index():
        print_header("Initializing Delia Indexer...")
        
        # Initialize backends to resolve Ollama URL for embeddings
        from .backend_manager import backend_manager

        
        # 1. Setup environment
        if summarize:
            print_info("Initializing Inference Engine for Summaries...")
            model_queue = ModelQueue()
            init_llm_module(
                stats_callback=lambda *a, **k: None,
                save_stats_callback=lambda: None,
                model_queue=model_queue,
            )
        
        summarizer = get_summarizer()
        graph = get_symbol_graph()
        print_info("Building Semantic Vector Index (CPU-bound)...")
        
        # 2. Sync Symbol Graph (Static Analysis)
        print_info("Building Symbol Graph (Classes, Functions, Imports)...")
        graph_updated = await graph.sync(force=force)
        print_success(f"Symbol Graph updated ({graph_updated} files processed).")
        
        # 3. Sync Summaries (LLM Analysis) - uses olmo-3 for reliable JSON
        if summarize:
            print_info(f"Generating Project summaries (parallel={parallel}, model=olmo-3:7b-instruct)...")
        else:
            print_info("Generating embeddings...")
        summary_updated = await summarizer.sync_project(force=force, summarize=summarize, parallel=parallel)
        print_success(f"Project Map updated ({summary_updated} files processed).")
        
        # 4. Generate project playbook from codebase analysis
        if summarize:
            print_info("Generating project playbook from analysis...")
            from .playbook import generate_project_playbook
            playbook_count = await generate_project_playbook(summarizer)
            if playbook_count > 0:
                print_success(f"Project playbook generated ({playbook_count} strategic bullets).")
            else:
                print_info("No new playbook bullets generated (may already exist).")
        
        print()
        if RICH_AVAILABLE and console:
            console.print(Panel.fit("[bold green]Indexing Complete![/bold green] Delia now has full architectural awareness.", border_style="green"))
        else:
            print("Index updated successfully.")

    try:
        asyncio.run(run_index())
    except KeyboardInterrupt:
        print_warning("Indexing interrupted.")
    except Exception as e:
        print_error(f"Indexing failed: {e}")


@app.command("init-project")
def init_project(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing framework files"),
    skip_index: bool = typer.Option(False, "--skip-index", help="Skip indexing (use existing analysis)"),
    parallel: int = typer.Option(4, "--parallel", "-p", help="Number of parallel summarization tasks"),
) -> None:
    """
    Initialize a project with Delia's ACE Framework.

    This command:
    1. Indexes the codebase with LLM summarization
    2. Detects tech stack, patterns, and conventions
    3. Generates a customized CLAUDE.md framework file
    4. Creates .claude/, .gemini/, .github/ directories with synced configs

    Run this in your project root to enable Delia-powered AI assistance.
    """
    import asyncio
    from .playbook import detect_tech_stack, playbook_manager

    async def run_init_project():
        project_root = Path.cwd()
        project_name = project_root.name

        print_header(f"Initializing Delia Framework for '{project_name}'...")

        # Step 1: Index the project (unless skipped)
        if not skip_index:
            from .orchestration.summarizer import get_summarizer
            from .orchestration.graph import get_symbol_graph
            from .llm import init_llm_module
            from .queue import ModelQueue
            from .backend_manager import backend_manager

            print_info("Analyzing codebase structure...")

            model_queue = ModelQueue()
            init_llm_module(
                stats_callback=lambda *a, **k: None,
                save_stats_callback=lambda: None,
                model_queue=model_queue,
            )

            summarizer = get_summarizer()
            graph = get_symbol_graph()

            # Build symbol graph
            print_info("Building Symbol Graph...")
            await graph.sync(force=force)

            # Generate summaries
            print_info(f"Generating project summaries (parallel={parallel})...")
            await summarizer.sync_project(force=force, summarize=True, parallel=parallel)

            # Generate playbook
            print_info("Generating project playbook...")
            from .playbook import generate_project_playbook
            await generate_project_playbook(summarizer)

            print_success("Codebase analysis complete.")
        else:
            print_info("Skipping indexing (using existing analysis)...")

        # Step 2: Detect tech stack
        print_info("Detecting tech stack and patterns...")
        tech_stack = _detect_project_tech_stack(project_root)

        # Step 3: Generate framework files and sync to all detected agents
        print_info("Generating framework files...")
        claude_md_content = _generate_claude_md(project_name, tech_stack, project_root)

        # Check if CLAUDE.md exists and prompt for confirmation if not forcing
        claude_md_path = project_root / "CLAUDE.md"
        if claude_md_path.exists() and not force:
            if not prompt_confirm(f"CLAUDE.md already exists. Overwrite?", default=False):
                print_warning("Skipping framework file generation.")
                return

        # Sync to all detected AI agent instruction files
        print_info("Detecting and syncing AI assistant configs...")
        from .agent_sync import sync_agent_instruction_files, get_agent_summary
        files_written, detected_agents = sync_agent_instruction_files(
            project_root, claude_md_content, force=force
        )

        for file_path in files_written:
            print_success(f"  Synced {file_path}")

        # Final summary
        print()
        agent_summary = get_agent_summary(detected_agents)
        if RICH_AVAILABLE and console:
            from rich.panel import Panel
            console.print(Panel.fit(
                f"[bold green]Project initialized![/bold green]\n\n"
                f"Framework files synced:\n{agent_summary}\n\n"
                f"[dim]Delia will now provide dynamic playbook guidance for this project.[/dim]",
                border_style="green",
                title="Delia ACE Framework"
            ))
        else:
            print("\nProject initialized successfully!")
            print(f"Framework files synced:\n{agent_summary}")

    try:
        asyncio.run(run_init_project())
    except KeyboardInterrupt:
        print_warning("Initialization interrupted.")
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        raise


def _detect_project_tech_stack(project_root: Path) -> dict[str, Any]:
    """Detect tech stack from project files."""
    tech_stack = {
        "primary_language": None,
        "frameworks": [],
        "test_framework": None,
        "package_manager": None,
        "has_tests": False,
        "is_async": False,
    }

    # Check for Python
    if (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
        tech_stack["primary_language"] = "Python"
        tech_stack["package_manager"] = "uv" if (project_root / "uv.lock").exists() else "pip"

        # Check pyproject.toml for dependencies
        pyproject = project_root / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "fastapi" in content.lower():
                tech_stack["frameworks"].append("FastAPI")
            if "pydantic" in content.lower():
                tech_stack["frameworks"].append("Pydantic")
            if "structlog" in content.lower():
                tech_stack["frameworks"].append("structlog")
            if "asyncio" in content.lower() or "httpx" in content.lower():
                tech_stack["is_async"] = True
            if "pytest" in content.lower():
                tech_stack["test_framework"] = "pytest"
                tech_stack["has_tests"] = True

    # Check for TypeScript/JavaScript
    if (project_root / "package.json").exists():
        if tech_stack["primary_language"] is None:
            tech_stack["primary_language"] = "TypeScript"
        tech_stack["package_manager"] = "npm"

        try:
            pkg_json = json.loads((project_root / "package.json").read_text())
            deps = {**pkg_json.get("dependencies", {}), **pkg_json.get("devDependencies", {})}

            if "react" in deps or "react-native" in deps:
                tech_stack["frameworks"].append("React")
            if "react-native" in deps:
                tech_stack["frameworks"].append("React Native")
            if "expo" in deps:
                tech_stack["frameworks"].append("Expo")
            if "next" in deps:
                tech_stack["frameworks"].append("Next.js")
            if "tamagui" in deps:
                tech_stack["frameworks"].append("Tamagui")
            if "swr" in deps:
                tech_stack["frameworks"].append("SWR")
            if "zod" in deps:
                tech_stack["frameworks"].append("Zod")
            if "jest" in deps:
                tech_stack["test_framework"] = "Jest"
                tech_stack["has_tests"] = True
            if "vitest" in deps:
                tech_stack["test_framework"] = "Vitest"
                tech_stack["has_tests"] = True
            if "playwright" in deps:
                tech_stack["frameworks"].append("Playwright")
        except Exception:
            pass

    # Check for Supabase
    if (project_root / "supabase").is_dir():
        tech_stack["frameworks"].append("Supabase")

    # Check for tests directory
    if (project_root / "tests").is_dir() or (project_root / "test").is_dir():
        tech_stack["has_tests"] = True

    return tech_stack


def _generate_claude_md(project_name: str, tech_stack: dict[str, Any], project_root: Path) -> str:
    """Generate a customized CLAUDE.md based on detected tech stack."""

    # Determine language-specific content
    lang = tech_stack.get("primary_language", "Unknown")
    frameworks = tech_stack.get("frameworks", [])
    test_framework = tech_stack.get("test_framework", "pytest" if lang == "Python" else "Jest")
    pkg_manager = tech_stack.get("package_manager", "uv" if lang == "Python" else "npm")
    is_async = tech_stack.get("is_async", False)

    # Build framework list
    framework_str = ", ".join(frameworks) if frameworks else "Standard library"

    # Generate test command
    if lang == "Python":
        test_cmd = "uv run pytest" if pkg_manager == "uv" else "pytest"
        lint_cmd = "ruff check" if pkg_manager == "uv" else "ruff check"
    else:
        test_cmd = "npm run test" if test_framework else "npm test"
        lint_cmd = "npm run lint"

    # Build the CLAUDE.md content
    content = f'''# {project_name} Development Instructions

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## ACE Framework: Delia-Controlled Playbooks

Delia manages project-specific playbooks that provide learned strategies, patterns, and guidance.
Instead of reading static profile files, query Delia for dynamic, feedback-refined guidance.

### Getting Playbook Guidance

**Before starting a task**, query Delia for relevant playbook bullets:

```python
# Get task-specific guidance
get_playbook(task_type="coding")  # or: testing, architecture, debugging, project

# Get project context (tech stack, patterns, conventions)
get_project_context()
```

The playbooks contain strategic bullets with:
- **Learned lessons** from past tasks
- **Project-specific patterns** (detected from codebase analysis via `delia index --summarize`)
- **Utility scores** - bullets that helped more rank higher

### Applying Guidance

Use the bullet content to guide your work. Each bullet has an ID for feedback:

```json
{{
  "id": "strat-a1b2c3d4",
  "content": "This project uses async/await patterns. Prefer async def for I/O operations.",
  "utility_score": 0.85
}}
```

### Closing the Learning Loop

**After completing a task**, report whether the guidance helped:

```python
# If the bullet helped complete the task successfully
report_feedback(bullet_id="strat-a1b2c3d4", task_type="coding", helpful=True)

# If the bullet was misleading or irrelevant
report_feedback(bullet_id="strat-a1b2c3d4", task_type="coding", helpful=False)
```

This feedback updates bullet utility scores, improving future recommendations.

### Task Type Mapping

| Task Type | Keywords | Playbook |
|-----------|----------|----------|
| coding | implement, add, create, write | `coding` |
| testing | test, {test_framework.lower() if test_framework else "test"}, coverage, assert | `testing` |
| architecture | design, ADR, refactor, pattern | `architecture` |
| debugging | error, bug, fix, stack trace | `debugging` |
| project | general project context | `project` |

---

## Project Overview

**Project:** {project_name}
**Language:** {lang}
**Frameworks:** {framework_str}

---

## Build & Development Commands

```bash
# Install dependencies
{pkg_manager} {"sync" if pkg_manager == "uv" else "install"}

# Run tests
{test_cmd}

# Lint code
{lint_cmd}
```

---

## INLINED CRITICAL RULES

These rules are mandatory for all work in this project:

### From core.md - Universal Rules
```
ALWAYS:
- Search codebase for similar patterns before creating new code (DRY)
- Check existing utilities before creating new ones
- Run tests before committing
- Complete Integration: No placeholders, remove old code
- Type safety: {"No TypeScript 'any', use Zod validation" if lang == "TypeScript" else "Use type hints and Pydantic models"}
```

### From coding.md - Before Writing Code
```
PRE-IMPLEMENTATION CHECKLIST (Cannot skip):
[] Query Delia playbook for guidance
[] Search codebase for similar patterns (DRY)
[] Check existing utilities before creating new ones
[] Verify function signatures match project style
[] Run tests before committing
```

### From git.md - Git Operations
```
BRANCH DECISION:
- Multi-file changes? -> Create branch: feature/, fix/, refactor/
- Single-file trivial fix? -> Commit directly to main
- Experimental? -> Always branch

COMMIT FORMAT:
type(scope): description
Examples: feat(core):, fix(api):, refactor(utils):

NEVER:
- Force-push to main
- Skip tests before push
- Leave branches hanging
```

### From debugging.md - Fixing Issues
```
DEBUG ORDER:
1. Error stack trace -> exact file/line
2. Recent commits -> what changed?
3. Test output -> which test failing?
4. Logs -> structured log events

FIX STRATEGY:
- Small bug (<10 lines): Direct to main
- Complex bug (>10 lines): Create fix/ branch
- ALWAYS add regression test
```

---

## Anti-Patterns (NEVER DO)

```
Code:
- Placeholder delegations -> Actually extract and integrate
- Duplicate state (old + new module) -> Remove old, use new
- {"TypeScript 'any'" if lang == "TypeScript" else "Untyped functions"} -> Proper types{"" if lang == "TypeScript" else " and type hints"}
- Create abstractions for one-time use -> Keep it simple

Process:
- Commit without running tests -> NEVER
- Force-push to main -> NEVER
- Leave dead code "just in case" -> DELETE it
- Skip playbook guidance -> ALWAYS query Delia
```

---

## Validation Checklist (Before Marking Complete)

```
[] Playbook bullets queried and applied (get_playbook)
[] Feedback reported for useful bullets (report_feedback)
[] Code passes: {lint_cmd}
[] Tests pass: {test_cmd}
[] No placeholder delegations
[] Old code removed if extracting
[] Tests updated to match new implementation
```

---

**All projects use Delia for dynamic playbook guidance.**
'''

    # Add async-specific content if detected
    if is_async and lang == "Python":
        async_section = '''
## Async Patterns

This project uses async/await. Follow these patterns:

```python
# DO: Use async def for I/O operations
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# DO: Use asyncio.gather for parallel operations
results = await asyncio.gather(task1(), task2(), task3())
```
'''
        content = content.replace("---\n\n**All projects", f"---\n{async_section}\n---\n\n**All projects")

    return content


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
def compact(
    session_id: str = typer.Argument(..., help="Session ID to compact"),
    force: bool = typer.Option(False, "--force", "-f", help="Force compaction even if below threshold"),
) -> None:
    """
    Compact a session's conversation history.
    
    Summarizes older messages while preserving key artifacts to reduce
    context token usage.
    """
    import asyncio
    from .session_manager import get_session_manager

    async def run_compact():
        sm = get_session_manager()
        result = await sm.compact_session(session_id, force=force)
        
        if result["success"]:
            print_success(f"Session {session_id} compacted.")
            print_info(f"  Messages compacted: {result["messages_compacted"]}")
            print_info(f"  Tokens saved: {result["tokens_saved"]}")
            print_info(f"  Reduction: {result["compression_ratio"]:.1%}")
        else:
            print_error(f"Compaction failed: {result["error"]}")

    asyncio.run(run_compact())


@app.command()
def memory(
    reload: bool = typer.Option(False, "--reload", "-r", help="Force reload of all project memories"),
    show_content: bool = typer.Option(False, "--content", "-c", help="Show combined instructions content"),
) -> None:
    """
    List project memories (DELIA.md files) loaded into context.
    """
    from .project_memory import get_project_memory, reload_project_memories, list_project_memories

    if reload:
        reload_project_memories()
        print_success("Project memories reloaded.")

    memories = list_project_memories()
    pm = get_project_memory()

    if not memories:
        print_info("No project memories loaded.")
        return

    print_header("Loaded Project Memories")
    for m in memories:
        print_info(f"  • {m["name"]} ({m["source"]}) - {m["size_kb"]}KB")

    if show_content:
        print_header("Combined Content")
        print(pm._state.combined_content)


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


# =============================================================================
# Security Commands
# =============================================================================

@app.command()
def audit(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries to show"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    tool: str = typer.Option(None, "--tool", "-t", help="Filter by tool name"),
    result: str = typer.Option(None, "--result", "-r", help="Filter by result (success/error/denied)"),
) -> None:
    """
    View the security audit log.

    Shows recent tool executions with permission levels, approval methods,
    and results. Use filters to find specific operations.

    Examples:
        delia audit                    # Last 20 entries
        delia audit -n 50              # Last 50 entries
        delia audit -t shell_exec      # Only shell commands
        delia audit -r denied          # Only denied operations
        delia audit --json             # JSON output for scripting
    """
    from .security import get_security_manager

    sm = get_security_manager()
    entries = sm.get_audit_log(limit=limit * 2)  # Get more to allow filtering

    # Apply filters
    if tool:
        entries = [e for e in entries if tool.lower() in e.get("tool_name", "").lower()]
    if result:
        entries = [e for e in entries if e.get("result", "").lower() == result.lower()]

    # Limit after filtering
    entries = entries[-limit:]

    if not entries:
        print_info("No audit entries found.")
        return

    if json_output:
        import json as json_module
        print(json_module.dumps(entries, indent=2))
        return

    # Pretty print
    print()
    print_header(f"Audit Log (last {len(entries)} entries)")
    print()

    for entry in entries:
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        tool_name = entry.get("tool_name", "unknown")
        perm = entry.get("permission_level", "?")
        res = entry.get("result", "?")
        method = entry.get("approval_method", "?")
        duration = entry.get("duration_ms", 0)

        # Color-code result
        if res == "success":
            status = "[green]OK[/green]" if RICH_AVAILABLE else "OK"
        elif res == "denied":
            status = "[red]DENIED[/red]" if RICH_AVAILABLE else "DENIED"
        else:
            status = "[yellow]ERROR[/yellow]" if RICH_AVAILABLE else "ERROR"

        # Format line
        line = f"{ts}  {status:8}  {tool_name:20}  [{perm}]  via:{method}  {duration}ms"

        if RICH_AVAILABLE and console:
            console.print(line)
        else:
            print(line.replace("[green]", "").replace("[/green]", "")
                  .replace("[red]", "").replace("[/red]", "")
                  .replace("[yellow]", "").replace("[/yellow]", ""))

        # Show error if present
        if entry.get("error_message"):
            print_error(f"    {entry['error_message'][:80]}")

    print()
    print_info(f"Audit file: {sm.policy.audit_file}")


@app.command()
def undo(
    list_stack: bool = typer.Option(False, "--list", "-l", help="List undo stack without undoing"),
    session: str = typer.Option(None, "--session", "-s", help="Filter by session ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """
    Undo file operations.

    Restores files to their state before Delia modified them.
    Each file modification is backed up automatically.

    Examples:
        delia undo              # Undo last change (with confirmation)
        delia undo --list       # List undo stack
        delia undo --force      # Undo without confirmation
    """
    from .security import get_security_manager

    sm = get_security_manager()

    if not sm.policy.undo_enabled:
        print_error("Undo is disabled in security policy.")
        return

    stack = sm.get_undo_stack(session_id=session)

    if list_stack:
        if not stack:
            print_info("Undo stack is empty.")
            return

        print()
        print_header(f"Undo Stack ({len(stack)} entries)")
        print()

        for i, entry in enumerate(reversed(stack), 1):
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            op = entry.get("operation", "?")
            path = entry.get("path", "?")

            # Truncate long paths
            if len(path) > 60:
                path = "..." + path[-57:]

            line = f"{i:3}.  {ts}  {op:8}  {path}"
            print(line)

        print()
        print_info("Run 'delia undo' to restore the most recent change.")
        return

    # Actually undo
    if not stack:
        print_info("Nothing to undo.")
        return

    last = stack[-1]
    path = last.get("path", "unknown")
    op = last.get("operation", "unknown")

    print()
    print_info(f"Last operation: {op} on {path}")

    if not force:
        if not prompt_confirm("Undo this change?"):
            print_info("Cancelled.")
            return

    success, message = sm.undo_last(session_id=session)

    if success:
        print_success(message)
    else:
        print_error(message)


# Default command: launch chat if no command specified
@app.command()
def prewarm() -> None:
    """
    Predictively preload models into GPU memory based on usage patterns.
    
    Uses the Melon Economy and Prewarm Tracker to identify which
    models you are likely to use at this hour.
    """
    import asyncio
    from .config import get_prewarm_tracker, load_prewarm
    from .backend_manager import backend_manager
    from .llm import init_llm_module, get_llm_provider
    
    # Load learned patterns
    load_prewarm()
    tracker = get_prewarm_tracker()
    tiers = tracker.get_predicted_tiers()
    
    if not tiers:
        print_info("No usage patterns learned yet. Start chatting to enable pre-warming!")
        return

    print_header(f"Pre-warming {len(tiers)} predicted tiers...")
    
    async def run_prewarm():
        await init_llm_module()
        active_backend = backend_manager.get_active_backend()
        if not active_backend:
            print_error("No active backend found.")
            return

        provider = get_llm_provider("ollama") # Assume ollama for now
        
        for tier in tiers:
            model = active_backend.models.get(tier)
            if model:
                print_info(f"Pre-loading {tier:10} -> {model}...")
                # load_model uses keep_alive=-1 (Hot VRAM)
                res = await provider.load_model(model, backend_obj=active_backend)
                if res.success:
                    print_success(f"{model} is ready.")
                else:
                    print_error(f"Failed to load {model}: {res.error}")

    asyncio.run(run_prewarm())


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
        chat()


if __name__ == "__main__":
    app()
