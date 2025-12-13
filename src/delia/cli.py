# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
# ruff: noqa: T201  # CLI tool - print statements are intentional
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
        console.print(f"  [green]✅ {text}[/green]")
    else:
        print(f"  ✅ {text}")


def print_warning(text: str) -> None:
    """Print warning message."""
    if RICH_AVAILABLE and console:
        console.print(f"  [yellow]⚠️  {text}[/yellow]")
    else:
        print(f"  ⚠️  {text}")


def print_error(text: str) -> None:
    """Print error message."""
    if RICH_AVAILABLE and console:
        console.print(f"  [red]❌ {text}[/red]")
    else:
        print(f"  ❌ {text}")


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
                elif provider in ("llamacpp", "vllm"):
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


def assign_models_to_tiers(models: list[str]) -> dict[str, str]:
    """
    Assign detected models to tiers based on naming patterns.

    Returns a dict mapping tier names to model names.
    """
    if not models:
        return {}

    tiers: dict[str, str | None] = {
        "quick": None,
        "coder": None,
        "moe": None,
        "thinking": None,
    }

    # If only one model, use it for everything
    if len(models) == 1:
        return dict.fromkeys(tiers, models[0])

    # Classify models
    for model in models:
        model_lower = model.lower()

        # Thinking/reasoning models
        if any(kw in model_lower for kw in ["think", "reason", "r1", "o1", "deepseek-r"]):
            if not tiers["thinking"]:
                tiers["thinking"] = model

        # Coder models
        elif any(kw in model_lower for kw in ["code", "coder", "codestral", "starcoder"]):
            if not tiers["coder"]:
                tiers["coder"] = model

        # Large/MoE models (by size indicators)
        elif any(kw in model_lower for kw in ["30b", "32b", "70b", "72b", "moe", "mixtral", "qwen3:30"]):
            if not tiers["moe"]:
                tiers["moe"] = model

        # Small/quick models
        elif any(kw in model_lower for kw in ["7b", "8b", "3b", "1b", "small", "mini", "tiny"]) and not tiers["quick"]:
            tiers["quick"] = model

    # Fill in gaps with first available model
    first_model = models[0]
    for tier in tiers:
        if not tiers[tier]:
            tiers[tier] = first_model

    return {k: v for k, v in tiers.items() if v is not None}


# ============================================================
# CLIENT DETECTION
# ============================================================


def detect_clients() -> list[DetectedClient]:
    """Detect installed MCP clients by checking for their executables."""
    detected: list[DetectedClient] = []
    plat = get_platform()

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
                    configured = True
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

    # Determine if we should use the installed package or run from source
    # Check if 'delia' command is available (installed via pip/uv)
    delia_cmd = shutil.which("delia")

    if delia_cmd:
        # Use installed command
        server_config = {
            "command": "delia",
            "args": [],
        }
    else:
        # Run from source with uv
        server_config = {
            "command": "uv",
            "args": ["run", "--directory", str(delia_root), "delia"],
        }

    return server_config


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
    help="Delia MCP Server - Route prompts to local LLMs",
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
    print()
    if RICH_AVAILABLE and console:
        console.print(Panel.fit("[bold green]Welcome to Delia Setup![/bold green]", border_style="green"))
    else:
        print("🍈 Welcome to Delia Setup!")
        print("=" * 40)

    delia_root = get_delia_root()
    settings_path = delia_root / "settings.json"

    # Check existing settings
    if settings_path.exists() and not force:
        print_warning(f"settings.json already exists at {settings_path}")
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
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        print_success(f"Configuration saved to {settings_path}")
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
        console.print("[bold]🔍 Delia Health Check[/bold]")
    else:
        print("🔍 Delia Health Check")
        print("=" * 40)

    delia_root = get_delia_root()
    issues: list[str] = []

    # Check settings.json
    print_header("Configuration")

    settings_path = delia_root / "settings.json"
    if settings_path.exists():
        print_success(f"settings.json found at {settings_path}")
        try:
            with open(settings_path) as f:
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
                        if cmd == "delia" or shutil.which(cmd):
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
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open configuration in editor"),
) -> None:
    """
    View or edit Delia configuration.
    """
    delia_root = get_delia_root()
    settings_path = delia_root / "settings.json"

    if not settings_path.exists():
        print_error("No configuration found. Run 'delia init' first.")
        raise typer.Exit(1)

    if edit:
        # Try to open in default editor
        editor = os.environ.get("EDITOR", "nano")
        if get_platform() == "Windows":
            editor = os.environ.get("EDITOR", "notepad")

        subprocess.run([editor, str(settings_path)], check=False)
        return

    # Show configuration
    if show or not edit:
        try:
            with open(settings_path) as f:
                settings = json.load(f)

            if RICH_AVAILABLE and console:
                console.print_json(data=settings)
            else:
                print(json.dumps(settings, indent=2))
        except Exception as e:
            print_error(f"Failed to read configuration: {e}")
            raise typer.Exit(1) from None


@app.command()
def uninstall(
    client: str = typer.Argument(..., help="Client to uninstall from"),
) -> None:
    """
    Remove Delia from an MCP client's configuration.
    """
    client_lower = client.lower()
    if client_lower not in MCP_CLIENTS:
        print_error(f"Unknown client: {client}")
        raise typer.Exit(1)

    client_info = MCP_CLIENTS[client_lower]
    plat = get_platform()
    config_path = client_info["paths"].get(plat)

    if not config_path or not config_path.exists():
        print_error(f"No configuration found for {client_info['name']}")
        raise typer.Exit(1)

    try:
        with open(config_path) as f:
            config = json.load(f)

        config_key = client_info["config_key"]
        if config_key in config and "delia" in config[config_key]:
            del config[config_key]["delia"]

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            print_success(f"Delia removed from {client_info['name']}")
        else:
            print_info(f"Delia not configured in {client_info['name']}")
    except Exception as e:
        print_error(f"Failed to update configuration: {e}")
        raise typer.Exit(1) from None


# Default command: run the server if no command specified
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """
    Delia MCP Server - Route prompts to local LLMs.

    Run without arguments to start the server in STDIO mode.
    Use 'delia init' for first-time setup.
    """
    if ctx.invoked_subcommand is None:
        # No command specified, run the server
        run()


if __name__ == "__main__":
    app()
