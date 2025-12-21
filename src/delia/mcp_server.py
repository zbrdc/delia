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

import asyncio
import contextvars
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import httpx
import structlog
from structlog.types import Processor

from . import paths
from .container import get_container

# ============================================================
# EARLY LOGGING CONFIGURATION (must happen before other imports)
# ============================================================
# Configure structlog to suppress stdout output by default.
# This is critical for STDIO transport where stdout must be pure JSON-RPC.
# The configuration may be adjusted later in main() based on transport type.

# Use container to handle early logging setup
_container = get_container()
_container.logging_service.configure_structlog(use_stderr=True)


def _configure_structlog(use_stderr: bool = True) -> None:
    """Helper to reconfigure structlog (for transport changes)."""
    _container.logging_service.configure_structlog(use_stderr=use_stderr)


# ============================================================

import aiofiles
import humanize
from fastmcp import FastMCP
from pydantic import ValidationError
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Import unified backend manager (single source of truth from settings.json)
from .backend_manager import BackendConfig, backend_manager
from .routing import get_router

# Import configuration (all tunable values in config.py)
from .config import (
    STATS_FILE,
    config,
    detect_model_tier,
    get_affinity_tracker,
    get_backend_health,
    get_prewarm_tracker,
    load_affinity,
    load_backend_metrics,
    load_prewarm,
    save_affinity,
    save_backend_metrics,
    save_prewarm,
)

# Import voting system for batch consensus
from .voting import VotingConsensus
from .voting_stats import get_voting_stats_tracker
from .quality import ResponseQualityValidator
from .profiles import register_common_profiles

# Import status messages for logging and dashboard
from .messages import (
    StatusEvent,
    format_completion_stats,
    get_display_event,
    get_status_message,
    get_tier_message,
)

# Import validation functions
from .validation import (
    MAX_CONTENT_LENGTH,
    MAX_FILE_PATH_LENGTH,
    VALID_BACKENDS,
    VALID_MODELS,
    VALID_TASKS,
    validate_content,
    validate_file_path,
    validate_model_hint,
    validate_task,
)

# Import token counting utilities
from .tokens import count_tokens, estimate_tokens

# Import provider response models and provider classes
from .providers import (
    GeminiProvider,
    LlamaCppChoice,
    LlamaCppError,
    LlamaCppMessage,
    LlamaCppProvider,
    LlamaCppResponse,
    LlamaCppUsage,
    OllamaProvider,
    OllamaResponse,
    StreamChunk,
)

# Import LLM calling infrastructure (call_llm, call_llm_stream)
from .llm import call_llm, call_llm_stream, init_llm_module

# Import model queue system
from .queue import ModelQueue, QueuedRequest

# Import routing utilities (content detection, model override parsing, model selection)
from .routing import (
    CODE_INDICATORS,
    BackendScorer,
    detect_code_content,
    parse_model_override,
    select_model,
)

# Import language detection (LANGUAGE_CONFIGS, detect_language, get_system_prompt, optimize_prompt)
from .language import (
    LANGUAGE_CONFIGS,
    PYGMENTS_LANGUAGE_MAP,
    detect_language,
    get_system_prompt,
    optimize_prompt,
)

# Import file helpers (read_files, read_memory, read_file_safe, MEMORY_DIR)
from .file_helpers import MEMORY_DIR, read_file_safe
from .text_utils import strip_thinking_tags
from .types import Workspace

# Import orchestration service
from .orchestration.service import get_orchestration_service

# Import delegation helpers
from .delegation import (
    DelegateContext,
    delegate_impl,
    determine_task_type,
    get_delegate_signals,
    prepare_delegate_content,
    select_delegate_model as _select_delegate_model_impl,
    validate_delegate_request,
    _delegate_impl,
    _delegate_with_voting,
    _delegate_with_tot,
    _get_delegate_context,
    start_prewarm_task,
)

# Import session manager for multi-turn conversations
from .session_manager import get_session_manager

# Import task chain module for sequential task execution
from .task_chain import parse_chain_steps, execute_chain
from .task_workflow import parse_workflow_definition, execute_workflow

# Import stats service for usage tracking
from .stats import StatsService

# Import tools module for agentic capabilities
from .tools import (
    AgentConfig,
    AgentResult,
    get_default_tools,
    run_agent_loop,
    ToolRegistry,
    MCPClientManager,
)

# Conditional authentication imports (based on config.auth_enabled)
AUTH_ENABLED = config.auth_enabled
TRACKING_ENABLED = config.tracking_enabled

if AUTH_ENABLED:
    import jose.jwt
    from fastapi_users.authentication import JWTStrategy

    from .auth import (
        JWT_SECRET,
        User,
        UserCreate,
        auth_backend,
        create_db_and_tables,
        get_async_session_context,
        get_user_db_context,
        get_user_manager,
        get_user_manager_context,
    )

    def decode_jwt_token(token: str) -> dict[str, Any] | None:
        """
        Decode and verify a JWT token.

        Note: Audience verification is disabled because FastAPI-Users sets the 'aud'
        claim as a list (["fastapi-users:auth"]) but python-jose expects a string.
        This is a known compatibility issue. The token is still verified for:
        - Valid signature (using JWT_SECRET)
        - Expiration time
        - Algorithm (HS256)

        Args:
            token: The JWT token string

        Returns:
            The decoded payload dict, or None if decoding fails
        """
        try:
            return jose.jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        except jose.JWTError:
            return None
        except Exception:
            return None

else:
    # Stubs for when auth is disabled
    JWT_SECRET: str | None = None  # type: ignore[no-redef]

    async def create_db_and_tables() -> None:
        pass

    def decode_jwt_token(token: str) -> dict[str, Any] | None:
        return None


from .multi_user_tracking import tracker

# Background tasks set to prevent garbage collection of fire-and-forget tasks
_background_tasks: set[asyncio.Task[Any]] = set()


def _schedule_background_task(coro: Any) -> None:
    """Schedule a fire-and-forget background task, preventing garbage collection."""
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except RuntimeError:
        pass  # No running loop


# Pre-warming state
_prewarm_task_started = False


async def _prewarm_check_loop() -> None:
    """
    Background task that periodically checks and pre-warms predicted models.

    Runs until cancelled, checking every N minutes if models should be pre-loaded
    based on hourly usage patterns learned by PrewarmTracker.
    """
    global _prewarm_task_started
    _prewarm_task_started = True

    while True:
        try:
            # Get prewarm config
            prewarm_config = backend_manager.routing_config.get("prewarm", {})
            if not prewarm_config.get("enabled", False):
                # If disabled, wait a bit then check again (in case config changes)
                await asyncio.sleep(60)
                continue

            interval_minutes = prewarm_config.get("check_interval_minutes", 5)

            # Get predicted tiers for current hour
            tracker = get_prewarm_tracker()
            predicted_tiers = tracker.get_predicted_tiers()

            if predicted_tiers:
                # Get active backend to resolve tier -> model name
                active_backend = backend_manager.get_active_backend()
                if active_backend:
                    for tier in predicted_tiers:
                        model_name = active_backend.models.get(tier)
                        if model_name:
                            try:
                                # Trigger model loading via queue
                                await model_queue.acquire_model(
                                    model_name,
                                    task_type="prewarm",
                                    content_length=0,
                                    provider_name=active_backend.provider,
                                )
                                log.debug(
                                    "prewarm_model_acquired",
                                    tier=tier,
                                    model=model_name,
                                    backend=active_backend.id,
                                )
                            except Exception as e:
                                log.debug("prewarm_model_failed", tier=tier, error=str(e))

            # Wait for next check interval
            await asyncio.sleep(interval_minutes * 60)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("prewarm_loop_error", error=str(e))
            await asyncio.sleep(60)  # Wait before retrying


def start_prewarm_task() -> None:
    """Start the pre-warming background task if not already running."""
    global _prewarm_task_started
    if not _prewarm_task_started:
        _schedule_background_task(_prewarm_check_loop())


current_client_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_client_id", default=None)
current_username: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_username", default=None)


# Global service container (single source of truth)
container = get_container()

# Configure structlog with stderr by default
# (STDIO transport needs stdout reserved for JSON-RPC protocol messages)
container.initialize(use_stderr=True)
log = structlog.get_logger()

# Use services from container
model_queue = container.model_queue
mcp_client_manager = container.mcp_client_manager
stats_service = container.stats_service
logging_service = container.logging_service

# Circuit breaker stats file (for dashboard)
CIRCUIT_BREAKER_FILE = paths.CIRCUIT_BREAKER_FILE


def save_circuit_breaker_stats():
    """Save circuit breaker status to disk for dashboard."""
    try:
        active_backend = get_active_backend()
        data = {
            "ollama": get_backend_health("ollama").get_status(),
            "llamacpp": get_backend_health("llamacpp").get_status(),
            "active_backend": {
                "id": active_backend.id,
                "name": active_backend.name,
                "provider": active_backend.provider,
                "type": active_backend.type,
            }
            if active_backend
            else None,
            "timestamp": datetime.now().isoformat(),
        }
        temp_file = CIRCUIT_BREAKER_FILE.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(CIRCUIT_BREAKER_FILE)  # Atomic on POSIX
    except Exception as e:
        log.warning("circuit_breaker_save_failed", error=str(e))


def _update_stats_sync(
    model_tier: str,
    task_type: str,
    original_task: str,
    tokens: int,
    elapsed_ms: int,
    content_preview: str,
    enable_thinking: bool,
    backend: str = "ollama",
) -> None:
    """
    Thread-safe update of all in-memory stats via StatsService.

    This wrapper maintains the same interface for provider callbacks
    while delegating to the StatsService.
    """
    # Determine backend type from config
    backend_type = config.get_backend_type(backend)

    # Delegate to stats service
    stats_service.record_call(
        model_tier=model_tier,
        task_type=task_type,
        original_task=original_task,
        tokens=tokens,
        elapsed_ms=elapsed_ms,
        content_preview=content_preview,
        enable_thinking=enable_thinking,
        backend=backend,
        backend_type=backend_type,
    )  # deque auto-evicts oldest when maxlen reached


async def save_all_stats_async():
    """
    Save all stats asynchronously via StatsService.

    Saves:
    - Model usage and task stats via stats_service
    - Live logs and circuit breaker status
    - Backend performance metrics
    - Task-backend affinity scores
    """
    # Save model/task stats via service
    await stats_service.save_all()

    # Save other data (live logs, circuit breaker, backend metrics, affinity)
    await logging_service.save_live_logs_async()
    await asyncio.to_thread(save_circuit_breaker_stats)
    await asyncio.to_thread(save_backend_metrics)
    await asyncio.to_thread(save_affinity)
    await asyncio.to_thread(save_prewarm)


# Ensure all data directories exist
paths.ensure_directories()

# Load stats immediately at module import time
stats_service.load()
load_backend_metrics()
load_affinity()
load_prewarm()

# Load MCP server configurations (but don't start servers yet - that's async)
mcp_client_manager.load_config(backend_manager.get_mcp_servers())


# ============================================================
# BACKEND CLIENT MANAGEMENT
# HTTP clients are now managed by BackendManager (see backend_manager.py)
# The BackendManager reads from settings.json - the single source of truth
# ============================================================


def get_active_backend() -> BackendConfig | None:
    """Get the currently active backend configuration."""
    return backend_manager.get_active_backend()


def get_active_backend_id() -> str:
    """Get the ID of the currently active backend."""
    backend = backend_manager.get_active_backend()
    return backend.id if backend else "none"


def set_active_backend(backend_id: str) -> bool:
    """Set the active backend by ID."""
    return backend_manager.set_active_backend(backend_id)


async def get_loaded_models() -> list[str]:
    """Get list of currently loaded models from the active backend."""
    backend = backend_manager.get_active_backend()
    if not backend:
        return []

    client = backend.get_client()

    try:
        if backend.provider == "ollama":
            # Ollama-specific endpoint
            response = await client.get("/api/ps")
            if response.status_code == 200:
                try:
                    data = response.json()
                    return [m.get("name", "").replace(":latest", "") for m in data.get("models", [])]
                except json.JSONDecodeError:
                    log.warning("models_invalid_json", source="ollama_api_ps")
                    return []
        else:
            # OpenAI-compatible endpoint (llama.cpp, vLLM, etc.)
            response = await client.get(backend.models_endpoint)
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Handle OpenAI format: {"data": [{"id": "model-name", ...}]}
                    models = data.get("data", [])
                    return [m.get("id", "") for m in models if m.get("status", {}).get("value") == "loaded"]
                except json.JSONDecodeError:
                    log.warning("models_invalid_json", source=backend.models_endpoint)
                    return []
    except Exception as e:
        log.warning("models_load_failed", backend=backend.id, error=str(e))

    return []


def get_model_info(model_name: str) -> dict:
    """Get model information from configuration or estimate from name."""
    from .config import config

    # Check against active backend models
    backend = backend_manager.get_active_backend()
    if backend:
        models = backend.models
        if model_name == models.get("quick"):
            return {
                "vram_gb": config.model_quick.vram_gb,
                "context_tokens": config.model_quick.context_tokens,
                "tier": "quick",
            }
        elif model_name == models.get("coder"):
            return {
                "vram_gb": config.model_coder.vram_gb,
                "context_tokens": config.model_coder.context_tokens,
                "tier": "coder",
            }
        elif model_name == models.get("moe"):
            return {
                "vram_gb": config.model_moe.vram_gb,
                "context_tokens": config.model_moe.context_tokens,
                "tier": "moe",
            }
        elif model_name == models.get("thinking"):
            return {
                "vram_gb": config.model_thinking.vram_gb,
                "context_tokens": config.model_thinking.context_tokens,
                "tier": "thinking",
            }

    # Initialize defaults for estimation
    vram_gb: Any = "Unknown"
    context_tokens: Any = "Unknown"

    # Estimate from model name using regex patterns
    import re

    # Extract parameter count (e.g., "14B", "7b", "72B")
    param_match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:illion)?(?![a-zA-Z])", model_name, re.IGNORECASE)
    if param_match:
        params = float(param_match.group(1))
        # Rough VRAM estimation (very approximate)
        if params <= 7:
            vram_gb = 6
        elif params <= 14:
            vram_gb = 9
        elif params <= 30:
            vram_gb = 17
        else:
            vram_gb = 24

        # Context estimation based on model family
        if "qwen" in model_name.lower():
            context_tokens = 128_000 if params >= 14 else 40_000
        elif "llama" in model_name.lower():
            context_tokens = 128_000 if params >= 70 else 8_000
        else:
            context_tokens = 32_000  # Conservative default

    return {"vram_gb": vram_gb, "context_tokens": context_tokens, "tier": "unknown"}



# ============================================================
# LLM MODULE INITIALIZATION
# ============================================================


def _save_stats_background() -> None:
    """Schedule stats saving as a background task (non-blocking)."""
    _schedule_background_task(save_all_stats_async())


# Initialize the LLM module with callbacks and model queue
# This wires up the provider factory and queue lifecycle management
init_llm_module(
    stats_callback=_update_stats_sync,
    save_stats_callback=_save_stats_background,
    model_queue=model_queue,
)


# NOTE: call_llm and call_llm_stream are imported from .llm module


# ============================================================
# MCP SERVER SETUP
# ============================================================

# Load MCP instructions from external file
_MCP_INSTRUCTIONS_FILE = Path(__file__).parent / "mcp_instructions.md"
_MCP_INSTRUCTIONS_BASE = _MCP_INSTRUCTIONS_FILE.read_text() if _MCP_INSTRUCTIONS_FILE.exists() else ""


def _build_dynamic_instructions() -> str:
    """Build MCP instructions with dynamic playbook content."""
    from .playbook import playbook_manager
    
    parts = [_MCP_INSTRUCTIONS_BASE]
    
    # Load playbook bullets for common task types
    task_types = ["coding", "testing", "debugging", "architecture", "project"]
    playbook_sections = []
    
    for task_type in task_types:
        bullets = playbook_manager.get_top_bullets(task_type, limit=5)
        if bullets:
            bullet_lines = [f"- {b.content}" for b in bullets]
            playbook_sections.append(f"### {task_type.title()}\n" + "\n".join(bullet_lines))
    
    if playbook_sections:
        parts.append("\n\n## PROJECT PLAYBOOK (Auto-loaded)\n\n")
        parts.append("These are learned strategies from this project. Apply them to relevant tasks:\n\n")
        parts.append("\n\n".join(playbook_sections))
    
    return "".join(parts)


_MCP_INSTRUCTIONS = _build_dynamic_instructions()

mcp = FastMCP("delia", instructions=_MCP_INSTRUCTIONS)

def get_model_info(model_name: str) -> dict[str, Any]:
    """Get detailed information about a model (VRAM, context, tier)."""
    from .config import parse_model_name, detect_model_tier
    
    info = parse_model_name(model_name)
    tier = detect_model_tier(model_name)
    
    # Estimate VRAM based on parameter count and 4-bit quantization (standard)
    vram = info.params_b * 0.7 if info.params_b > 0 else "Unknown"
    
    # Estimate context window
    context = 32768 # Default
    if "32k" in model_name: context = 32768
    elif "128k" in model_name: context = 131072
    
    return {
        "model": model_name,
        "params_b": info.params_b,
        "vram_gb": vram,
        "context_tokens": context,
        "tier": tier,
        "is_coder": info.is_coder,
        "is_moe": info.is_moe
    }

# Register tool handlers from modular modules
from .tools.handlers import register_tool_handlers
from .tools.admin import register_admin_tools
from .tools.resources import register_resource_tools

register_tool_handlers(mcp)
register_admin_tools(mcp)
register_resource_tools(mcp)

# Legacy exports for backwards compatibility (functional only)
from .delegation import delegate_impl as _delegate_impl
from .tools.handlers import session_compact_impl as session_compact
from .tools.handlers import session_stats_impl as session_stats

# Register common specialist profiles on startup


# ============================================================
# MULTI-USER TRACKING MIDDLEWARE
# extracts user from JWT and tracks requests per user
# ============================================================
from .middleware import UserTrackingMiddleware
mcp.add_middleware(UserTrackingMiddleware())


# ============================================================
# AUTHENTICATION ROUTES (HTTP/SSE only)
# These routes are only active when AUTH_ENABLED=true
# ============================================================

if AUTH_ENABLED:
    from .auth_routes import register_auth_routes, register_oauth_routes

    register_auth_routes(mcp, tracker)
    register_oauth_routes(mcp)
else:
    log.info("auth_disabled", message="Authentication routes not registered. Set DELIA_AUTH_ENABLED=true to enable.")


# ============================================================
# MULTI-BACKEND INTELLIGENT ROUTING
# ============================================================


# ============================================================
# LSP TOOLS (Language Server Protocol code intelligence)
# ============================================================


@mcp.tool()
async def lsp_goto_definition(
    path: str,
    line: int,
    character: int,
) -> str:
    """
    Find the definition of a symbol at the given file position.

    Uses Language Server Protocol to provide semantic code navigation.
    Supports Python (pyright/pylsp), TypeScript, Rust, and Go.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        character: Character position (0-indexed)

    Returns:
        File path and line number where the symbol is defined

    Example:
        lsp_goto_definition(path="src/main.py", line=42, character=10)
    """
    from .tools.lsp import lsp_goto_definition as _lsp_goto_definition
    return await _lsp_goto_definition(path, line, character)


@mcp.tool()
async def lsp_find_references(
    path: str,
    line: int,
    character: int,
) -> str:
    """
    Find all references to a symbol at the given file position.

    Uses Language Server Protocol to find all usages of a symbol
    across the codebase. Supports Python, TypeScript, Rust, and Go.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        character: Character position (0-indexed)

    Returns:
        List of locations where the symbol is used

    Example:
        lsp_find_references(path="src/api.py", line=15, character=4)
    """
    from .tools.lsp import lsp_find_references as _lsp_find_references
    return await _lsp_find_references(path, line, character)


@mcp.tool()
async def lsp_hover(
    path: str,
    line: int,
    character: int,
) -> str:
    """
    Get documentation and type information for a symbol.

    Uses Language Server Protocol to retrieve docstrings, type signatures,
    and other documentation for the symbol at the given position.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        character: Character position (0-indexed)

    Returns:
        Documentation and type info in markdown format

    Example:
        lsp_hover(path="src/utils.py", line=20, character=8)
    """
    from .tools.lsp import lsp_hover as _lsp_hover
    return await _lsp_hover(path, line, character)


# ============================================================
# MCP SERVERS (External tool passthrough)
# ============================================================


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
    import json as json_mod
    import uuid

    if action == "status":
        status = mcp_client_manager.get_status()
        return json_mod.dumps(status, indent=2)

    elif action == "start_all":
        results = await mcp_client_manager.start_all()
        return json_mod.dumps({
            "action": "start_all",
            "results": results,
            "status": mcp_client_manager.get_status(),
        }, indent=2)

    elif action == "stop_all":
        await mcp_client_manager.stop_all()
        return json_mod.dumps({
            "action": "stop_all",
            "success": True,
            "status": mcp_client_manager.get_status(),
        }, indent=2)

    elif action == "start":
        if not server_id:
            return json_mod.dumps({"error": "server_id required for start action"})
        success = await mcp_client_manager.start_server(server_id)
        return json_mod.dumps({
            "action": "start",
            "server_id": server_id,
            "success": success,
            "status": mcp_client_manager.get_status(),
        }, indent=2)

    elif action == "stop":
        if not server_id:
            return json_mod.dumps({"error": "server_id required for stop action"})
        success = await mcp_client_manager.stop_server(server_id)
        return json_mod.dumps({
            "action": "stop",
            "server_id": server_id,
            "success": success,
            "status": mcp_client_manager.get_status(),
        }, indent=2)

    elif action == "add":
        if not command:
            return json_mod.dumps({"error": "command required for add action (JSON array)"})
        try:
            cmd_list = json_mod.loads(command)
        except json_mod.JSONDecodeError:
            return json_mod.dumps({"error": "Invalid command JSON - must be array"})

        if not isinstance(cmd_list, list):
            return json_mod.dumps({"error": "command must be a JSON array"})

        # Parse optional env
        env_dict = {}
        if env:
            try:
                env_dict = json_mod.loads(env)
            except json_mod.JSONDecodeError:
                return json_mod.dumps({"error": "Invalid env JSON - must be object"})

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

        return json_mod.dumps({
            "action": "add",
            "success": success,
            "server_id": new_id,
            "config": config if success else None,
        }, indent=2)

    elif action == "remove":
        if not server_id:
            return json_mod.dumps({"error": "server_id required for remove action"})

        # Stop if running
        await mcp_client_manager.stop_server(server_id)

        # Remove from config
        success = backend_manager.remove_mcp_server(server_id)
        if success:
            mcp_client_manager.load_config(backend_manager.get_mcp_servers())

        return json_mod.dumps({
            "action": "remove",
            "server_id": server_id,
            "success": success,
        }, indent=2)

    elif action == "tools":
        tools = mcp_client_manager.get_all_tools()
        return json_mod.dumps({
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
        return json_mod.dumps({
            "error": f"Unknown action: {action}",
            "valid_actions": ["status", "start", "stop", "start_all", "stop_all", "add", "remove", "tools"],
        })


# ============================================================
# MCP RESOURCES (Expose data for cross-server communication)
# ============================================================


@mcp.resource("delia://file/{path}")
async def resource_file(path: str) -> str:
    """
    Read a file from disk as an MCP resource.

    Enables other MCP servers/clients to read files through Delia.
    Useful for cross-server workflows where external MCP tools
    need to pass file content to Delia without serialization overhead.

    Args:
        path: File path (absolute or relative to cwd)

    Returns:
        File content as text, or error message if file not found/readable
    """
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    if not file_path.exists():
        return f"Error: File not found: {path}"

    if not file_path.is_file():
        return f"Error: Not a file: {path}"

    try:
        size = file_path.stat().st_size
        max_size = config.max_file_size  # 500KB default
        if size > max_size:
            return f"Error: File too large ({size // 1024}KB > {max_size // 1024}KB): {path}"

        content = file_path.read_text(encoding="utf-8")
        log.info("resource_file_read", path=path, size_kb=size // 1024)
        return content
    except Exception as e:
        log.warning("resource_file_failed", path=path, error=str(e))
        return f"Error reading file: {e}"


@mcp.resource("delia://stats", name="Usage Statistics", description="Current Delia usage statistics")
async def resource_stats() -> str:
    """
    Get current usage statistics as JSON.

    Returns token counts, call counts, and estimated cost savings
    across all model tiers.
    """
    model_usage, task_stats, _, recent_calls = stats_service.get_snapshot()
    stats = {
        "model_usage": model_usage,
        "task_stats": task_stats,
        "recent_calls_count": len(recent_calls),
    }
    return json.dumps(stats, indent=2)


@mcp.resource("delia://backends", name="Backend Status", description="Health and configuration of all backends")
async def resource_backends() -> str:
    """
    Get backend health status as JSON.

    Returns configuration and availability status for all configured
    backends, useful for monitoring and cross-server coordination.
    """
    status = await backend_manager.get_health_status()
    return json.dumps(status, indent=2)


@mcp.resource("delia://config", name="Configuration", description="Current Delia configuration")
async def resource_config() -> str:
    """
    Get current configuration as JSON.

    Returns routing settings, model tiers, and system configuration.
    Sensitive fields (API keys) are redacted.
    """
    config_data = {
        "routing": backend_manager.routing_config,
        "system": backend_manager.system_config,
        "backends": [
            {
                "id": b.id,
                "name": b.name,
                "provider": b.provider,
                "type": b.type,
                "url": b.url,
                "enabled": b.enabled,
                "models": b.models,
                # Redact API key for security
                "has_api_key": bool(b.api_key),
            }
            for b in backend_manager.backends.values()
        ],
    }
    return json.dumps(config_data, indent=2)


@mcp.resource("delia://memories", name="Available Memories", description="List of Delia's memory files")
async def resource_memories() -> str:
    """
    List available Delia memory files.

    Returns a JSON list of memory names that can be loaded via
    the `context` parameter in delegate/plant tools.
    """
    memories = []
    if MEMORY_DIR.exists():
        for f in MEMORY_DIR.glob("*.md"):
            memories.append(f.stem)
    return json.dumps({"memories": sorted(memories)}, indent=2)


# ============================================================
# MAIN
# ============================================================


async def _init_database():
    """Initialize authentication database on startup."""
    try:
        await create_db_and_tables()
        log.info("auth_database_ready")
    except Exception as e:
        log.warning("auth_database_init_failed", error=str(e))


async def _startup_handler():
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

    if TRACKING_ENABLED:
        await tracker.start_background_save()

    # Pre-warm tiktoken encoder in background to avoid 100-200ms delay on first request
    # Run in thread pool to avoid blocking startup
    from .tokens import prewarm_encoder
    await asyncio.to_thread(prewarm_encoder)

    # Clear expired sessions on startup
    sm = get_session_manager()
    cleared = sm.clear_expired_sessions()
    log.info("session_cleanup_startup", cleared=cleared)


async def _shutdown_handler():
    """
    Cleanup handler for graceful server shutdown.

    - Closes all backend HTTP clients to prevent connection leaks
    - Saves tracker state to disk
    This is called automatically on server shutdown.
    """
    from .backend_manager import shutdown_backends

    await shutdown_backends()

    # Save tracker state on shutdown
    if TRACKING_ENABLED:
        await tracker.shutdown()


def run_server(
    transport: str = "stdio",
    port: int = 8200,
    host: str = "0.0.0.0",
) -> None:
    """
    Run the Delia MCP server.

    This is the main entry point for starting the server, callable from
    both the CLI module and directly.

    Args:
        transport: Transport protocol - "stdio", "sse", "http", or "streamable-http"
        port: Port for HTTP/SSE transports (default: 8200)
        host: Host to bind for HTTP/SSE transports (default: 0.0.0.0)
    """
    global log
    import atexit

    # Normalize transport string
    transport = transport.lower().strip()

    # Note: Stats and logs are already loaded at module import time

    # CRITICAL FIX: Reconfigure logging for STDIO transport
    # For STDIO, stdout MUST be reserved for JSON-RPC protocol messages only
    # Redirect logs to stderr to avoid polluting the protocol stream
    if transport == "stdio":
        _configure_structlog(use_stderr=True)
        # Get fresh logger reference after reconfiguration
        log = structlog.get_logger()
        log.info("stdio_logging_configured", destination="stderr")

        # Run startup handler (probes backends, starts background tasks)
        asyncio.run(_startup_handler())

        # Register graceful shutdown handler
        atexit.register(lambda: asyncio.run(_shutdown_handler()))

        log.info("server_starting", transport="stdio", auth_enabled=False)
        # CRITICAL: show_banner=False keeps stdout clean for JSON-RPC protocol
        mcp.run(show_banner=False)

    elif transport in ("http", "streamable-http"):
        # Reconfigure logging to stderr for HTTP transports
        _configure_structlog(use_stderr=True)
        log = structlog.get_logger()
        log.info("http_logging_configured", destination="stderr")

        # Initialize auth database for HTTP transport (if auth enabled)
        if AUTH_ENABLED:
            asyncio.run(_init_database())

        # Run startup handler and register shutdown via atexit
        asyncio.run(_startup_handler())
        atexit.register(lambda: asyncio.run(_shutdown_handler()))

        auth_endpoints = ["/auth/register", "/auth/jwt/login", "/auth/me"] if AUTH_ENABLED else []
        log.info(
            "server_starting",
            transport="http",
            host=host,
            port=port,
            auth_enabled=AUTH_ENABLED,
            endpoints=auth_endpoints,
        )
        mcp.run(transport="http", host=host, port=port)

    elif transport == "sse":
        # Reconfigure logging to stderr for SSE transport
        _configure_structlog(use_stderr=True)
        log = structlog.get_logger()
        log.info("sse_logging_configured", destination="stderr")

        # Initialize auth database for SSE transport (if auth enabled)
        if AUTH_ENABLED:
            asyncio.run(_init_database())

        # Run startup handler and register shutdown via atexit
        asyncio.run(_startup_handler())
        atexit.register(lambda: asyncio.run(_shutdown_handler()))

        auth_endpoints = ["/auth/register", "/auth/jwt/login", "/auth/me"] if AUTH_ENABLED else []
        log.info(
            "server_starting",
            transport="sse",
            host=host,
            port=port,
            auth_enabled=AUTH_ENABLED,
            endpoints=auth_endpoints,
        )
        mcp.run(transport="sse", host=host, port=port)

    else:
        raise ValueError(f"Unknown transport: {transport}. Use: stdio, sse, http, streamable-http")


def main() -> None:
    """
    Legacy entry point for the MCP server.

    This provides backwards compatibility for direct invocation.
    For the full CLI experience, use 'delia' command instead.
    """
    from enum import Enum

    import typer

    class Transport(str, Enum):
        stdio = "stdio"
        sse = "sse"
        http = "http"
        streamable_http = "streamable-http"

    app = typer.Typer(
        help="Delia MCP Server (use 'delia' for full CLI with init/install/doctor)",
        no_args_is_help=False,
        add_completion=False,
    )

    @app.command()
    def run(
        transport: Transport = typer.Option(
            Transport.stdio,
            "-t",
            "--transport",
            help="Transport protocol",
        ),
        port: int = typer.Option(
            8200,
            "-p",
            "--port",
            help="Port for HTTP/SSE transport",
        ),
        host: str = typer.Option(
            "0.0.0.0",
            "--host",
            help="Host to bind for HTTP/SSE",
        ),
    ) -> None:
        """Run the Delia MCP server."""
        run_server(transport=transport.value, port=port, host=host)

    app()


if __name__ == "__main__":
    main()
