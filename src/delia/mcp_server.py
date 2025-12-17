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

# ============================================================
# EARLY LOGGING CONFIGURATION (must happen before other imports)
# ============================================================
# Configure structlog to suppress stdout output by default.
# This is critical for STDIO transport where stdout must be pure JSON-RPC.
# The configuration may be adjusted later in main() based on transport type.


def _early_configure_silent_logging():
    """Configure structlog to be silent on stdout before any imports log."""

    class SilentLoggerFactory:
        def __call__(self):
            class SilentLogger:
                def msg(self, *args, **kwargs):
                    pass

                def __getattr__(self, name):
                    return self.msg

            return SilentLogger()

    structlog.reset_defaults()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=SilentLoggerFactory(),
        cache_logger_on_first_use=False,
    )


_early_configure_silent_logging()
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

# Import status messages for logging and dashboard
from .messages import (
    StatusEvent,
    format_completion_stats,
    get_display_event,
    get_status_message,
    get_tier_message,
)

# Import prompt templating system
from .prompt_templates import create_structured_prompt

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

# Import file helpers (read_files, read_serena_memory, read_file_safe, MEMORY_DIR)
from .file_helpers import MEMORY_DIR, read_file_safe
from .text_utils import strip_thinking_tags
from .types import Workspace

# Import delegation helpers
from .delegation import (
    DelegateContext,
    delegate_impl,
    determine_task_type,
    get_delegate_signals,
    prepare_delegate_content,
    select_delegate_model as _select_delegate_model_impl,
    validate_delegate_request,
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

LIVE_LOGS_FILE = paths.LIVE_LOGS_FILE
MAX_LIVE_LOGS = 100
LIVE_LOGS: list[dict] = []
_live_logs_lock = threading.Lock()

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


def _dashboard_processor(logger: Any, method_name: str, event_dict: dict) -> dict:
    """
    Custom structlog processor that captures logs for dashboard streaming.

    Extracts dashboard-relevant fields and writes to live logs buffer.
    Only captures logs with explicit 'log_type' for dashboard display.
    Includes status messages for dashboard display.
    """
    # Only process logs explicitly marked for dashboard
    log_type = event_dict.pop("log_type", None)
    if log_type:
        model = event_dict.pop("model", "")
        tokens = event_dict.pop("tokens", 0)
        message = event_dict.get("event", "")
        status_msg = event_dict.pop("status_msg", "")  # Extract status message
        backend = event_dict.pop("backend", "")

        with _live_logs_lock:
            LIVE_LOGS.append(
                {
                    "ts": datetime.now().isoformat(),
                    "type": log_type,
                    "message": message,
                    "model": model,
                    "tokens": tokens,
                    "status_msg": status_msg,  # Status message for dashboard
                    "backend": backend,
                }
            )
            if len(LIVE_LOGS) > MAX_LIVE_LOGS:
                LIVE_LOGS.pop(0)

        # Schedule async save (non-blocking)
        _schedule_background_task(_save_live_logs_async())

    return event_dict


def _save_live_logs_sync():
    """Save live logs to disk synchronously (fallback)."""
    try:
        temp_file = LIVE_LOGS_FILE.with_suffix(".tmp")
        with _live_logs_lock:
            temp_file.write_text(json.dumps(LIVE_LOGS[-MAX_LIVE_LOGS:], indent=2))
        temp_file.replace(LIVE_LOGS_FILE)
    except Exception as e:
        # Log failures at debug level - logs are non-critical but we want visibility
        structlog.get_logger().debug("live_logs_save_failed", error=str(e))


async def _save_live_logs_async():
    """Save live logs to disk asynchronously using aiofiles."""
    try:
        temp_file = LIVE_LOGS_FILE.with_suffix(".tmp")
        with _live_logs_lock:
            content = json.dumps(LIVE_LOGS[-MAX_LIVE_LOGS:], indent=2)
        async with aiofiles.open(temp_file, "w") as f:
            await f.write(content)
        temp_file.replace(LIVE_LOGS_FILE)
    except Exception as e:
        # Log failures at debug level - logs are non-critical but we want visibility
        structlog.get_logger().debug("live_logs_save_async_failed", error=str(e))


def _configure_structlog(use_stderr: bool = False):
    """
    Configure structlog with console + dashboard output.

    Args:
        use_stderr: If True, suppress stdout output (required for STDIO transport)
                   to keep stdout pure for JSON-RPC messages.
    """

    # Clear any cached loggers before reconfiguring
    structlog.reset_defaults()

    # For STDIO mode, suppress all console logging to keep stdout pure for JSON-RPC
    if use_stderr:
        # In STDIO mode: no console output at all
        # Only keep dashboard processor to capture logs for other systems
        processors_silent: list[Processor] = cast(
            list[Processor],
            [
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                _dashboard_processor,  # Capture dashboard logs (no console output)
            ],
        )

        # Create a silent logger factory that discards all output
        class SilentLoggerFactory:
            def __call__(self):
                """Return a logger instance that discards all output."""

                class SilentLogger:
                    def msg(self, *args, **kwargs):
                        # Discard - don't print anything
                        pass

                    def __getattr__(self, name):
                        return self.msg

                return SilentLogger()

        structlog.configure(
            processors=processors_silent,
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=SilentLoggerFactory(),
            cache_logger_on_first_use=False,
        )
    else:
        # For non-STDIO modes, use normal console logging with colors
        processors_console: list[Processor] = cast(
            list[Processor],
            [
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                _dashboard_processor,  # Capture dashboard logs
                structlog.dev.ConsoleRenderer(colors=True),  # ← Only in non-STDIO mode
            ],
        )

        structlog.configure(
            processors=processors_console,
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


# Initialize structlog with stderr by default
# (STDIO transport needs stdout reserved for JSON-RPC protocol messages)
_configure_structlog(use_stderr=True)
log = structlog.get_logger()


# Global model queue instance (imported from queue.py)
# Note: provider_getter is set later after _get_provider is defined
model_queue = ModelQueue()


# Global MCP client manager for external tool passthrough
# Connects to external MCP servers and imports their tools into Delia
mcp_client_manager = MCPClientManager()


# ============================================================
# USAGE TRACKING (via StatsService)
# ============================================================

# Stats service singleton (thread-safe, handles all usage tracking)
stats_service = StatsService()

# Circuit breaker stats file (for dashboard)
CIRCUIT_BREAKER_FILE = paths.CIRCUIT_BREAKER_FILE


def load_live_logs():
    """Load live logs from disk into the structlog buffer."""
    global LIVE_LOGS
    if LIVE_LOGS_FILE.exists():
        try:
            loaded = json.loads(LIVE_LOGS_FILE.read_text())[-MAX_LIVE_LOGS:]
            LIVE_LOGS[:] = loaded
            log.info("logs_loaded", count=len(LIVE_LOGS), source="disk")
        except json.JSONDecodeError as e:
            log.warning("logs_load_failed", error=str(e), reason="invalid_json")
            LIVE_LOGS[:] = []  # Reset to empty on corrupt file
        except Exception as e:
            log.warning("logs_load_failed", error=str(e))


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
    await asyncio.to_thread(_save_live_logs_sync)
    await asyncio.to_thread(save_circuit_breaker_stats)
    await asyncio.to_thread(save_backend_metrics)
    await asyncio.to_thread(save_affinity)
    await asyncio.to_thread(save_prewarm)


# Ensure all data directories exist
paths.ensure_directories()

# Load stats immediately at module import time
stats_service.load()
load_live_logs()
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
_MCP_INSTRUCTIONS = _MCP_INSTRUCTIONS_FILE.read_text() if _MCP_INSTRUCTIONS_FILE.exists() else ""

mcp = FastMCP("delia", instructions=_MCP_INSTRUCTIONS)

# ============================================================
# MULTI-USER TRACKING MIDDLEWARE
# Extracts user from JWT and tracks requests per user
# ============================================================

from fastmcp.server.dependencies import get_http_headers, get_http_request
from fastmcp.server.middleware import Middleware, MiddlewareContext


class UserTrackingMiddleware(Middleware):
    """
    Middleware to extract authenticated user from JWT token and track requests.

    Transport-aware tracking:
    - HTTP/SSE: Uses IP address + API key for client identification
    - STDIO: Uses session ID for client identification

    When AUTH_ENABLED:
    - Extracts JWT from Authorization header
    - Validates token and gets user_id
    - Stores user info in context state for tools to access
    - Records request metrics after tool execution

    When AUTH_DISABLED:
    - Tracks by session_id only
    - No JWT validation
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Track tool calls per user."""
        if not TRACKING_ENABLED:
            # Tracking disabled - just execute the tool
            return await call_next()

        start_time = time.time()
        ctx = context.fastmcp_context

        # Detect transport and extract client info
        user_id = None
        username = "anonymous"
        ip_address = ""
        api_key = None
        transport = "stdio"

        # Try to get HTTP context (indicates HTTP/SSE transport)
        try:
            request = get_http_request()
            if request:
                transport = "http"
                # Extract client IP
                ip_address = request.client.host if request.client else ""
                # Check for X-Forwarded-For (if behind proxy)
                forwarded = request.headers.get("x-forwarded-for", "")
                if forwarded:
                    ip_address = forwarded.split(",")[0].strip()
        except Exception:  # noqa: S110 - No HTTP context in STDIO transport
            pass

        # Try to extract user from JWT token (only if auth enabled)
        if AUTH_ENABLED and transport == "http":
            try:
                headers = get_http_headers()
                auth_header = headers.get("authorization", "")

                if auth_header.startswith("Bearer "):
                    token = auth_header.replace("Bearer ", "")
                    api_key = token  # Use token as API key for tracking
                    payload = decode_jwt_token(token)
                    if payload:
                        user_id = payload.get("sub")

                        # Get user details from database
                        if user_id:
                            async with get_async_session_context() as session, get_user_db_context(session) as user_db:
                                user = await user_db.get(uuid.UUID(user_id))
                                if user:
                                    username = user.email
                                    # Set user quota from database
                                    from .multi_user_tracking import QuotaConfig

                                    _quota = QuotaConfig(
                                        max_requests_per_hour=user.max_requests_per_hour,
                                        max_tokens_per_hour=user.max_tokens_per_hour,
                                    )
                                    # Will be set after client is created
            except Exception:  # noqa: S110 - No HTTP context or invalid token
                pass

        # Fall back to session_id for tracking (works for all transports)
        if not user_id and ctx and ctx.request_context:
            user_id = ctx.session_id
            username = f"session:{ctx.session_id[:8]}"

        # Register/get client in tracker (transport-aware)
        client = None
        if user_id:
            client = tracker.get_or_create_client(
                username=username,
                ip_address=ip_address,
                api_key=api_key,
                transport=transport,
            )

            # Store user info in context for tools (only if ctx is available)
            if ctx:
                ctx.set_state("user_id", user_id)
                ctx.set_state("username", username)
                ctx.set_state("client_id", client.client_id if client else None)

            # Set context variables for _delegate_impl to access
            current_client_id.set(client.client_id if client else None)
            current_username.set(username)

            # Check quota before proceeding
            if client:
                can_proceed, msg = tracker.check_quota(client.client_id)
                if not can_proceed:
                    log.warning("quota_exceeded", username=username, reason=msg)
                    # Return error without executing tool
                    from mcp.types import CallToolResult, TextContent

                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Quota exceeded: {msg}")], isError=True
                    )

                # Mark request as started (for concurrent tracking)
                tracker.start_request(client.client_id)

        # Execute the tool
        try:
            result = await call_next(context)
            success = True
            error_msg = ""
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            # Record request in tracker (no disk I/O - just in-memory update)
            elapsed_ms = int((time.time() - start_time) * 1000)

            if client:
                # Get tool name from context method or message
                tool_name = context.method or getattr(context.message, "name", None) or "unknown"

                tracker.record_request(
                    client_id=client.client_id,
                    task_type=tool_name,
                    model_tier="",  # Will be set by tool if known
                    tokens=0,  # Will be updated by tool if known
                    elapsed_ms=elapsed_ms,
                    success=success,
                    error=error_msg,
                )

        return result


# Register the middleware
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


def _get_delegate_context() -> DelegateContext:
    """Create a DelegateContext with all runtime dependencies."""
    return DelegateContext(
        select_model=select_model,
        get_active_backend=get_active_backend,
        call_llm=call_llm,
        get_client_id=current_client_id.get,
        tracker=tracker,
    )


async def select_delegate_model(
    task_type: str,
    content: str,
    model_override: str | None = None,
    backend: str | None = None,
    backend_obj: Any | None = None,
) -> tuple[str, str, Any]:
    """Select model and backend for delegate tasks.

    Wrapper for delegation.select_delegate_model that provides context.
    """
    ctx = _get_delegate_context()
    return await _select_delegate_model_impl(
        ctx, task_type, content, model_override, backend, backend_obj
    )


async def _delegate_impl(
    task: str,
    content: str,
    file: str | None = None,
    model: str | None = None,
    language: str | None = None,
    context: str | None = None,
    symbols: str | None = None,
    include_references: bool = False,
    backend: str | None = None,
    backend_obj: Any | None = None,
    files: str | None = None,
    include_metadata: bool = True,
    max_tokens: int | None = None,
    session_id: str | None = None,
) -> str:
    """Core implementation for delegate - wrapper that uses delegation module."""
    ctx = _get_delegate_context()
    return await delegate_impl(
        ctx,
        task,
        content,
        file,
        model,
        language,
        context,
        symbols,
        include_references,
        backend,
        backend_obj,
        files,
        include_metadata,
        max_tokens,
        session_id=session_id,
    )


@mcp.tool()
async def delegate(
    task: str,
    content: str,
    file: str | None = None,
    model: str | None = None,
    language: str | None = None,
    context: str | None = None,
    symbols: str | None = None,
    include_references: bool = False,
    backend_type: str | None = None,
    files: str | None = None,
    include_metadata: bool = True,
    max_tokens: int | None = None,
    dry_run: bool = False,
    session_id: str | None = None,
    stream: bool = False,
) -> str:
    """
    Execute a task with intelligent 3-tier model selection.
    Routes to optimal backend based on content size, task type, and availability.

    WHEN TO USE:
    - Code review, generation, analysis tasks → Use this tool
    - Any task you want processed by an LLM → Use this tool

    Args:
        task: Task type determines model tier:
            - "quick" or "summarize" → quick tier (fast, 14B model)
            - "generate", "review", "analyze" → coder tier (code-optimized 14B)
            - "plan", "critique" → moe tier (deep reasoning 30B+)
        content: The prompt or content to process (required)
        file: Optional file path to include in context
        model: Force specific tier - "quick" | "coder" | "moe" | "thinking"
              OR natural language: "7b", "14b", "30b", "small", "large", "coder model", "fast", "complex", "thinking"
        language: Language hint for better prompts - python|c|cpp|java|csharp|javascript|sql|visualbasic|perl|r|delphi|fortran|matlab|ada|go|php|rust|kotlin|assembly|bash
        context: Serena memory names to include (comma-separated: "architecture,decisions")
        symbols: Code symbols to focus on (comma-separated: "Foo,Bar/calculate")
        include_references: True if content includes symbol usages from elsewhere
        backend_type: Force backend type (default: auto-select)
        files: Comma-separated file paths - Delia reads directly from disk (efficient, no serialization)
        include_metadata: If False, skip the metadata footer (saves ~30 tokens). Default: True
        max_tokens: Limit response length to N tokens (forces concise output). Default: None (unlimited)
        dry_run: If True, return estimation signals without executing LLM call. Default: False
            Returns: estimated_tokens, recommended_tier, recommended_model, backend info, context fit
        session_id: Optional session ID for conversation continuity. Creates stateful multi-turn conversations.
        stream: If True, use streaming mode internally (better for long responses, avoids timeouts). Default: False

    ROUTING LOGIC:
    1. Content > 32K tokens → Uses backend with largest context window
    2. Routes to best available backend based on priority and health
    3. Falls back automatically if primary backend is unavailable
    4. Load balances across backends based on priority weights

    Returns:
        LLM response optimized for orchestrator processing, with optional metadata footer
        If dry_run=True: JSON with estimation signals (tokens, tier, model, backend, context fit)

    Examples:
        delegate(task="review", content="<code>", language="python")
        delegate(task="generate", content="Write a REST API", backend_type="local")
        delegate(task="plan", content="Design caching strategy", model="moe")
        delegate(task="review", files="src/main.py,src/utils.py", content="Review these files")
        delegate(task="quick", content="...", max_tokens=500)  # Limit response
        delegate(task="review", content="<large code>", dry_run=True)  # Get estimates first
        delegate(task="review", content="...", session_id="abc-123")  # Use session
        delegate(task="plan", content="<large content>", stream=True)  # Use streaming for long responses
    """
    # Start prewarm task if not already running
    start_prewarm_task()

    # Dry run mode: return estimation signals without executing
    if dry_run:
        import json as json_mod
        ctx = _get_delegate_context()
        signals = await get_delegate_signals(
            ctx,
            task,
            content,
            file,
            model,
            language,
            context,
            symbols,
            include_references,
            files,
        )
        return json_mod.dumps(signals, indent=2)

    # Smart backend selection using backend_manager
    backend_provider, backend_obj = await _select_optimal_backend_v2(content, file, task, backend_type)

    # Streaming mode: use streaming API internally (better for long responses)
    if stream and backend_obj:
        ctx = _get_delegate_context()

        # Prepare content with context, files, symbols (matches delegate_impl flow)
        prepared_content = await prepare_delegate_content(
            content, context, symbols, include_references, files
        )

        # Map task to internal type
        task_type = determine_task_type(task)

        # Create enhanced, structured prompt with templates
        prepared_content = create_structured_prompt(
            task_type=task_type,
            content=prepared_content,
            file_path=file,
            language=language,
            symbols=symbols.split(",") if symbols else None,
            context_files=context.split(",") if context else None,
        )

        # Detect language and get system prompt
        detected_language = language or detect_language(prepared_content, file or "")
        system = get_system_prompt(detected_language, task_type)

        # Select model (use implementation directly with ctx)
        selected_model, tier, _source = await _select_delegate_model_impl(
            ctx, task_type, prepared_content, model, None, backend_obj
        )

        # Stream and collect response
        full_response = ""
        total_tokens = 0
        elapsed_ms = 0
        start_time = time.time()

        async for chunk in call_llm_stream(
            model=selected_model,
            prompt=prepared_content,
            system=system,
            task_type=task_type,
            original_task=task,
            language=detected_language,
            backend_obj=backend_obj,
            max_tokens=max_tokens,
        ):
            if chunk.text:
                full_response += chunk.text
            if chunk.done:
                total_tokens = chunk.tokens
                if chunk.metadata:
                    elapsed_ms = chunk.metadata.get("elapsed_ms", 0)
                if chunk.error:
                    return f"Error: {chunk.error}"

        if elapsed_ms == 0:
            elapsed_ms = int((time.time() - start_time) * 1000)

        # Strip thinking tags from models like Qwen3, DeepSeek-R1
        full_response = strip_thinking_tags(full_response)

        # Format response with metadata
        if include_metadata:
            return f"""{full_response}

---
_[OK] {task} (streamed) | {tier} tier | {elapsed_ms}ms | {selected_model}_"""
        return full_response

    # Non-streaming mode: use existing implementation
    return await _delegate_impl(
        task,
        content,
        file,
        model,
        language,
        context,
        symbols,
        include_references,
        backend=backend_provider,
        backend_obj=backend_obj,
        files=files,
        include_metadata=include_metadata,
        max_tokens=max_tokens,
        session_id=session_id,
    )


@mcp.tool()
async def think(
    problem: str,
    context: str = "",
    depth: str = "normal",
    session_id: str | None = None,
) -> str:
    """
    Deep reasoning for complex problems with extended thinking.
    Uses thinking-capable models for thorough analysis.

    WHEN TO USE:
    - Complex multi-step problems requiring careful reasoning
    - Architecture decisions, trade-off analysis
    - Debugging strategies, refactoring plans
    - Any situation requiring "thinking through" before acting

    Args:
        problem: The problem or question to think through (required)
        context: Supporting information - code, docs, constraints (optional)
        depth: Reasoning depth level:
            - "quick" → Fast answer, no extended thinking (14B model)
            - "normal" → Balanced reasoning with thinking (14B coder)
            - "deep" → Thorough multi-step analysis (30B+ MoE model)
        session_id: Optional session ID for conversation continuity

    ROUTING:
    - Uses largest available GPU for deep thinking
    - Automatically enables thinking mode for normal/deep
    - Routes to best available thinking-capable backend

    Returns:
        Structured analysis with step-by-step reasoning and conclusions

    Examples:
        think(problem="How should we handle authentication?", depth="deep")
        think(problem="Debug this error", context="<stack trace>", depth="normal")
        think(problem="...", session_id="abc-123")  # Use session
    """
    # Select model tier based on depth
    if depth == "quick":
        task_type = "quick"
        model_hint = "quick"
        _enable_thinking = False  # Reserved for future extended thinking support
    elif depth == "deep":
        task_type = "plan"
        model_hint = "thinking"  # Use dedicated thinking model for deep reasoning
        _enable_thinking = True  # Reserved for future extended thinking support
    else:  # normal
        task_type = "analyze"
        model_hint = "thinking"  # Use thinking model for normal depth too
        _enable_thinking = True  # Reserved for future extended thinking support

    # Build the thinking prompt
    thinking_prompt = f"""Think through this problem step by step:

## Problem
{problem}
"""
    if context:
        thinking_prompt += f"""
## Context
{context}
"""

    thinking_prompt += """
## Instructions
1. Break down the problem into components
2. Consider different approaches and their trade-offs
3. Reason through each step carefully
4. Provide a clear conclusion with actionable insights

Think deeply before answering."""

    # Route to best backend (prefer large context for deep thinking)
    backend_provider, backend_obj = await _select_optimal_backend_v2(thinking_prompt, None, task_type)

    result = await _delegate_impl(
        task=task_type,
        content=thinking_prompt,
        file=None,
        model=model_hint,
        language=None,
        context=None,
        symbols=None,
        include_references=False,
        backend=backend_provider,
        backend_obj=backend_obj,
        session_id=session_id,
    )

    # Track think() separately (in addition to underlying task type)
    stats_service.increment_task("think")
    await save_all_stats_async()

    return result


# ============================================================
# MULTI-BACKEND INTELLIGENT ROUTING
# ============================================================


async def _select_optimal_backend_v2(
    content: str,
    file_path: str | None = None,
    task_type: str = "quick",
    backend_type: str | None = None,
) -> tuple[str | None, Any | None]:
    """
    Select optimal backend using performance-based scoring.

    Uses BackendScorer to select the best backend based on:
    - Latency (lower is better)
    - Throughput (higher is better)
    - Reliability (success rate)
    - Availability (circuit breaker status)

    Args:
        content: The content to process (for future content-based routing)
        file_path: Optional file path for context
        task_type: Type of task being performed
        backend_type: Optional backend type constraint ("local" or "remote")

    Returns:
        Tuple of (None, backend_obj) where backend_obj is the selected backend
    """
    enabled_backends = backend_manager.get_enabled_backends()

    if not enabled_backends:
        return (None, None)

    # Use BackendScorer for intelligent selection with configured weights
    weights = backend_manager.get_scoring_weights()
    scorer = BackendScorer(weights=weights)

    # Use weighted random selection if load balancing is enabled
    load_balance = backend_manager.routing_config.get("load_balance", False)
    if load_balance:
        selected = scorer.select_weighted(
            enabled_backends, backend_type=backend_type, task_type=task_type
        )
    else:
        selected = scorer.select_best(
            enabled_backends, backend_type=backend_type, task_type=task_type
        )

    if selected:
        log.debug(
            "selected_backend_by_score",
            backend_id=selected.id,
            backend_type=selected.type,
            requested_type=backend_type,
            load_balanced=load_balance,
        )
        return (None, selected)

    # Fallback to active backend if no matching type found
    active_backend = backend_manager.get_active_backend()
    if active_backend:
        log.debug(
            "fallback_to_active_backend",
            backend_id=active_backend.id,
            requested_type=backend_type,
        )
    return (None, active_backend)


def _assign_backends_to_tasks(task_list: list[dict], available: dict[str, bool]) -> list[str]:
    """
    Assign backends to tasks using round-robin load balancing across enabled backends.

    Args:
        task_list: List of tasks to assign
        available: Dict of backend_id -> bool availability

    Returns:
        List of backend IDs matching task_list order
    """
    # Get enabled backends that are available
    enabled_backends = backend_manager.get_enabled_backends()
    candidate_backends = [b.id for b in enabled_backends if available.get(b.id, False)]

    if not candidate_backends:
        # Fallback to active backend even if status unknown, or just return empty/error
        active = backend_manager.get_active_backend()
        return [active.id] * len(task_list) if active else ["none"] * len(task_list)

    assignments = []
    backend_count = len(candidate_backends)

    for i, _ in enumerate(task_list):
        # Round-robin assignment
        backend_id = candidate_backends[i % backend_count]
        assignments.append(backend_id)

    return assignments


@mcp.tool()
async def batch(
    tasks: str,
    include_metadata: bool = True,
    max_tokens: int | None = None,
    session_id: str | None = None,
) -> str:
    """
    Execute multiple tasks in PARALLEL across all available backends for maximum throughput.
    Distributes work across backends intelligently.

    WHEN TO USE:
    - Processing multiple files/documents simultaneously
    - Bulk code review, summarization, or analysis
    - Any workload that can be parallelized

    Args:
        tasks: JSON string containing an array of task objects. Each object can have:
            - task: "quick"|"summarize"|"generate"|"review"|"analyze"|"plan"|"critique"
            - content: The content to process (required)
            - file: Optional file path
            - files: Comma-separated file paths - Delia reads directly from disk
            - model: Force tier - "quick"|"coder"|"moe"
            - language: Language hint for code tasks
            - include_metadata: Override batch-level include_metadata for this task
            - max_tokens: Override batch-level max_tokens for this task
        include_metadata: If False, skip metadata footers on all tasks (saves tokens). Default: True
        max_tokens: Limit response length per task (forces concise output). Default: None
        session_id: Optional session ID shared across all tasks in the batch

    ROUTING LOGIC:
    - Distributes tasks across ALL available backends
    - Large content (>32K tokens) → Routes to backend with sufficient context
    - Normal content → Round-robin for parallel execution
    - Respects backend health and circuit breakers

    Returns:
        Combined results from all tasks with timing and routing info

    Example:
        batch('[
            {"task": "summarize", "content": "doc1..."},
            {"task": "review", "content": "code2...", "language": "python"},
            {"task": "analyze", "content": "log3..."}
        ]')
        batch('[...]', max_tokens=500)  # Limit all responses
        batch('[...]', session_id="abc-123")  # Use session
    """
    start_time = time.time()

    try:
        task_list = json.loads(tasks)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"

    if not isinstance(task_list, list):
        return "Error: tasks must be a JSON array"

    if len(task_list) == 0:
        return "Error: tasks array is empty"

    # Check which backends are available
    available = await backend_manager.check_all_health()
    backend_assignments = _assign_backends_to_tasks(task_list, available)

    # Log routing decisions
    routing_counts: dict[str, int] = {}
    for b_id in backend_assignments:
        routing_counts[b_id] = routing_counts.get(b_id, 0) + 1

    log.info("batch_routing", counts=routing_counts)

    # Capture context variables before spawning tasks
    # These are set by middleware for user tracking and quota enforcement
    captured_client_id = current_client_id.get()
    captured_username = current_username.get()

    # Run tasks in parallel using asyncio.gather with backend routing
    # Context variables must be explicitly propagated to child tasks
    async def run_task(
        i: int,
        t: dict,
        backend_id: str,
        client_id: str | None,
        username: str | None,
    ) -> str:
        # Re-set context variables in this task
        # Child tasks don't inherit parent context, so we must restore them
        current_client_id.set(client_id)
        current_username.set(username)

        task_type = t.get("task", "analyze")
        content = t.get("content", "")
        file_path = t.get("file")
        files = t.get("files")  # Comma-separated file paths (Delia reads directly)
        model_hint = t.get("model")
        language = t.get("language")
        context = t.get("context")
        symbols = t.get("symbols")
        include_refs = t.get("include_references", False)
        # Per-task overrides, falling back to batch-level settings
        task_include_metadata = t.get("include_metadata", include_metadata)
        task_max_tokens = t.get("max_tokens", max_tokens)

        # backend_id passed to _delegate_impl will resolve to ID
        result = await _delegate_impl(
            task=task_type,
            content=content,
            file=file_path,
            model=model_hint,
            language=language,
            context=context,
            symbols=symbols,
            include_references=include_refs,
            backend=backend_id,
            files=files,
            include_metadata=task_include_metadata,
            max_tokens=task_max_tokens,
            session_id=session_id,
        )
        return f"### Task {i + 1}: {task_type}\n\n{result}"

    results = await asyncio.gather(
        *[
            run_task(i, t, backend_assignments[i], captured_client_id, captured_username)
            for i, t in enumerate(task_list)
        ]
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Build routing summary
    routing_info = f"Backends used: {', '.join([f'{k}({v})' for k, v in routing_counts.items()])}"

    return f"""# Batch Results

{chr(10).join(results)}

---
_Total tasks: {len(task_list)} | Total time: {elapsed_ms}ms | {routing_info}_"""


@mcp.tool()
async def batch_vote(
    prompt: str,
    task: str = "analyze",
    k: int = 3,
    backends: str | None = None,
    target_accuracy: float = 0.9999,
) -> str:
    """
    Run the SAME prompt across multiple backends and vote on the best result.
    
    Uses MDAP k-voting consensus for mathematically guaranteed accuracy.
    This is ideal for high-stakes questions where you want multiple models
    to validate each other.
    
    WHEN TO USE:
    - High-stakes analysis requiring verification
    - Comparing model opinions on the same question
    - Getting consensus from multiple LLMs
    
    Args:
        prompt: The prompt to send to all backends
        task: Task type (analyze, review, generate, etc.)
        k: Votes needed for consensus (default: 3)
        backends: Comma-separated backend IDs (default: all enabled backends)
        target_accuracy: Target accuracy for auto-kmin calculation (default: 0.9999)
    
    Returns:
        Consensus response with voting statistics
    
    Example:
        batch_vote(prompt="Is this code secure?", k=3)
        batch_vote(prompt="Analyze this design", backends="ollama-local,ollama-remote")
    """
    start_time = time.time()
    
    # Get backends to use
    if backends:
        backend_ids = [b.strip() for b in backends.split(",")]
        selected_backends = [
            b for b in backend_manager.get_enabled_backends()
            if b.id in backend_ids
        ]
    else:
        selected_backends = backend_manager.get_enabled_backends()
    
    if not selected_backends:
        return "Error: No backends available for voting"
    
    # Initialize voting consensus
    validator = ResponseQualityValidator()
    consensus = VotingConsensus(
        k=k,
        quality_validator=validator,
        similarity_threshold=0.85,
        max_response_length=700,
    )
    
    # Get voting stats tracker
    voting_tracker = get_voting_stats_tracker()
    
    log.info(
        "batch_vote_start",
        prompt_len=len(prompt),
        k=k,
        num_backends=len(selected_backends),
        backend_ids=[b.id for b in selected_backends],
    )
    
    # Collect responses from all backends in parallel
    async def get_response(backend: BackendConfig) -> tuple[str, str, bool]:
        """Get response from a backend. Returns (backend_id, response, success)."""
        try:
            result = await _delegate_impl(
                task=task,
                content=prompt,
                backend=backend.id,
                include_metadata=False,
            )
            return (backend.id, result, True)
        except Exception as e:
            log.warning("batch_vote_backend_error", backend=backend.id, error=str(e))
            return (backend.id, str(e), False)
    
    # Run all backends in parallel
    responses = await asyncio.gather(*[get_response(b) for b in selected_backends])
    
    # Add votes to consensus
    backend_results = []
    for backend_id, response, success in responses:
        if not success:
            backend_results.append({
                "backend": backend_id,
                "status": "error",
                "response_preview": response[:100],
            })
            continue
        
        vote_result = consensus.add_vote(response)
        
        if vote_result.red_flagged:
            voting_tracker.record_rejection(
                reason=vote_result.red_flag_reason or "unknown",
                backend_id=backend_id,
                tier=task,
                response_preview=response[:100],
            )
            backend_results.append({
                "backend": backend_id,
                "status": "red_flagged",
                "reason": vote_result.red_flag_reason,
            })
            continue
        
        backend_results.append({
            "backend": backend_id,
            "status": "voted",
            "response_preview": response[:100] + "..." if len(response) > 100 else response,
        })
        
        if vote_result.consensus_reached:
            # Record successful consensus
            voting_tracker.record_consensus(
                votes_cast=vote_result.total_votes,
                k=k,
                tier=task,
                backend_id=backend_id,
                success=True,
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            prob = VotingConsensus.voting_probability(k, 0.95)
            
            # Update affinity for winning backend
            affinity_tracker = get_affinity_tracker()
            affinity_tracker.update(backend_id, task, quality=1.0)
            
            return f"""# ✓ Batch Vote Consensus Reached

{vote_result.winning_response}

---
**Voting Stats**
- Consensus: {vote_result.votes_for_winner}/{k} votes
- Total backends queried: {len(selected_backends)}
- Backends voted: {vote_result.total_votes}
- Confidence: {prob:.2%}
- Winning backend: {backend_id}
- Time: {elapsed_ms}ms

**Backend Results:**
{chr(10).join([f"- {r['backend']}: {r['status']}" for r in backend_results])}"""
    
    # No consensus - get best response
    best_response, metadata = consensus.get_best_response()
    
    voting_tracker.record_consensus(
        votes_cast=metadata.total_votes,
        k=k,
        tier=task,
        backend_id="none",
        success=False,
    )
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    if best_response:
        return f"""# ⚠ Batch Vote (No Full Consensus)

{best_response}

---
**Voting Stats**
- Best response: {metadata.winning_votes}/{k} votes needed
- Unique responses: {metadata.unique_responses}
- Red-flagged: {metadata.red_flagged_count}
- Total backends queried: {len(selected_backends)}
- Time: {elapsed_ms}ms

**Backend Results:**
{chr(10).join([f"- {r['backend']}: {r['status']}" for r in backend_results])}"""
    else:
        return f"Error: All {len(selected_backends)} backends failed or were red-flagged"


@mcp.tool()
async def session_create(
    client_id: str | None = None,
    metadata: str | None = None,
) -> str:
    """
    Create a new conversation session for multi-turn interactions.
    
    WHEN TO USE:
    - Starting a new multi-turn conversation
    - When you need to maintain context across multiple delegate() calls
    - For iterative code review or development workflows
    
    Args:
        client_id: Optional client identifier for session grouping
        metadata: Optional JSON string with session metadata
    
    Returns:
        JSON with session_id and creation details
    
    Example:
        session_create()  # Returns {"session_id": "abc-123", "created_at": "..."}
        session_create(client_id="user-1", metadata='{"project": "demo"}')
    """
    sm = get_session_manager()
    meta_dict = None
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid metadata JSON"})
    
    session = sm.create_session(client_id=client_id, metadata=meta_dict)
    return json.dumps({
        "session_id": session.session_id,
        "client_id": session.client_id,
        "created_at": session.created_at,
        "metadata": session.metadata,
    }, indent=2)


@mcp.tool()
async def session_list(
    client_id: str | None = None,
) -> str:
    """
    List active conversation sessions.
    
    WHEN TO USE:
    - Finding existing sessions to resume
    - Checking session status and statistics
    - Managing multiple concurrent conversations
    
    Args:
        client_id: Optional filter by client ID
    
    Returns:
        JSON array of session summaries
    
    Example:
        session_list()  # All sessions
        session_list(client_id="user-1")  # Sessions for specific client
    """
    sm = get_session_manager()
    sessions = sm.list_sessions(client_id=client_id)
    return json.dumps(sessions, indent=2)


@mcp.tool()
async def session_get(
    session_id: str,
) -> str:
    """
    Get full details of a conversation session including message history.
    
    WHEN TO USE:
    - Reviewing conversation history
    - Debugging session state
    - Checking token usage
    
    Args:
        session_id: The session ID to retrieve
    
    Returns:
        JSON with full session details including messages
    
    Example:
        session_get(session_id="abc-123")
    """
    sm = get_session_manager()
    session = sm.get_session(session_id)
    if not session:
        return json.dumps({"error": f"Session not found: {session_id}"})
    
    return json.dumps(session.to_dict(), indent=2)


@mcp.tool()
async def session_delete(
    session_id: str,
) -> str:
    """
    Delete a conversation session and its history.
    
    WHEN TO USE:
    - Cleaning up completed conversations
    - Removing sessions with sensitive data
    - Freeing resources
    
    Args:
        session_id: The session ID to delete
    
    Returns:
        JSON with deletion status
    
    Example:
        session_delete(session_id="abc-123")
    """
    sm = get_session_manager()
    success = sm.delete_session(session_id)
    return json.dumps({
        "deleted": success,
        "session_id": session_id,
    })


@mcp.tool()
async def chain(
    steps: str,
    session_id: str | None = None,
    continue_on_error: bool = False,
) -> str:
    """
    Execute a chain of tasks sequentially with output piping.
    
    Steps run in order, each able to reference outputs from previous steps
    using ${var} variable substitution. Perfect for multi-step workflows
    like: generate → review → fix.
    
    WHEN TO USE:
    - Multi-step code generation pipelines
    - Sequential analysis and refinement workflows
    - Any task where outputs feed into subsequent steps
    
    Args:
        steps: JSON array of step objects. Each step has:
            - id (required): Unique step identifier
            - task (required): Task type (quick, review, generate, analyze, plan, critique, summarize)
            - content (required): Prompt with optional ${var} substitution
            - model: Optional model tier (quick, coder, moe, thinking)
            - language: Optional language hint
            - output_var: Name to store output for later ${reference}
            - pass_to_next: Auto-append output to next step (default: false)
        session_id: Optional session for conversation continuity across chains
        continue_on_error: If true, continue after step failures (default: false)
    
    Returns:
        JSON with execution results, outputs, and timing
    
    Variable Substitution:
        Use ${step_id} or ${output_var} to reference previous outputs.
        Example: "Review this code: ${generate_step}"
    
    Examples:
        # Simple two-step chain
        chain('[
            {"id": "gen", "task": "generate", "content": "Write a hello world function", "output_var": "code"},
            {"id": "review", "task": "review", "content": "Review this code: ${code}"}
        ]')
        
        # Code generation pipeline
        chain('[
            {"id": "design", "task": "plan", "content": "Design a REST API for users"},
            {"id": "implement", "task": "generate", "content": "Implement: ${design}", "language": "python"},
            {"id": "test", "task": "generate", "content": "Write tests for: ${implement}"}
        ]')
        
        # With pass_to_next for automatic chaining
        chain('[
            {"id": "analyze", "task": "analyze", "content": "Find bugs in this code", "pass_to_next": true},
            {"id": "fix", "task": "generate", "content": "Fix all issues"}
        ]')
    """
    import json
    
    # Parse steps
    try:
        parsed_steps = parse_chain_steps(steps)
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "steps_completed": 0,
            "steps_total": 0,
        }, indent=2)
    
    # Get delegate context
    ctx = _get_delegate_context()
    
    # Execute chain
    try:
        result = await execute_chain(
            steps=parsed_steps,
            ctx=ctx,
            session_id=session_id,
            continue_on_error=continue_on_error,
        )
        return json.dumps(result.to_dict(), indent=2)
    except Exception as e:
        log.error("chain_execution_failed", error=str(e))
        return json.dumps({
            "success": False,
            "error": str(e),
            "steps_completed": 0,
            "steps_total": len(parsed_steps),
        }, indent=2)


@mcp.tool()
async def workflow(
    definition: str,
    session_id: str | None = None,
    max_retries: int = 1,
) -> str:
    """
    Execute a DAG workflow with conditional branching and retry logic.
    
    Workflows are more powerful than chains, supporting:
    - Dependencies: Nodes wait for prerequisites
    - Conditional branching: on_success/on_failure paths
    - Retry: Automatic retry with exponential backoff
    - Timeout: Per-workflow time limits
    
    WHEN TO USE:
    - Complex multi-path workflows with error recovery
    - Tasks that may fail and need fallback strategies
    - Pipelines requiring dependency management
    - Long-running orchestration with timeout protection
    
    Args:
        definition: JSON workflow definition with:
            - name: Workflow name
            - entry: Starting node ID
            - timeout_minutes: Max execution time (default: 10)
            - nodes: Array of node objects with:
                - id (required): Unique node identifier
                - task (required): Task type (quick, review, generate, etc.)
                - content (required): Prompt with ${var} substitution
                - depends_on: Array of node IDs that must complete first
                - on_success: Node ID to execute on success
                - on_failure: Node ID to execute on failure (fallback)
                - retry_count: Number of retries (default: 0)
                - backoff_factor: Retry delay multiplier (default: 1.5)
                - output_var: Variable name to store output
                - model: Optional model tier override
                - language: Optional language hint
        session_id: Optional session for conversation continuity
        max_retries: Global retry override for all nodes (default: 1)
    
    Returns:
        JSON with execution results, node status, and outputs
    
    Variable Substitution:
        Use ${node_id} or ${output_var} to reference completed node outputs.
    
    Examples:
        # Simple workflow with fallback
        workflow('{
            "name": "Resilient Analysis",
            "entry": "analyze",
            "nodes": [
                {"id": "analyze", "task": "plan", "content": "Analyze code", "on_success": "report", "on_failure": "simple"},
                {"id": "simple", "task": "quick", "content": "Quick analysis", "on_success": "report"},
                {"id": "report", "task": "summarize", "content": "Create report"}
            ]
        }')
        
        # Workflow with dependencies and retry
        workflow('{
            "name": "Code Pipeline",
            "entry": "design",
            "timeout_minutes": 15,
            "nodes": [
                {"id": "design", "task": "plan", "content": "Design API", "output_var": "spec", "on_success": "implement"},
                {"id": "implement", "task": "generate", "content": "Implement: ${spec}", "depends_on": ["design"], "retry_count": 2, "on_success": "test"},
                {"id": "test", "task": "generate", "content": "Write tests", "depends_on": ["implement"]}
            ]
        }')
        
        # Workflow with conditional paths
        workflow('{
            "name": "Review Pipeline",
            "entry": "check",
            "nodes": [
                {"id": "check", "task": "review", "content": "Check for issues", "on_success": "approve", "on_failure": "fix"},
                {"id": "fix", "task": "generate", "content": "Fix issues", "on_success": "recheck"},
                {"id": "recheck", "task": "review", "content": "Verify fixes", "on_success": "approve"},
                {"id": "approve", "task": "summarize", "content": "Approval summary"}
            ]
        }')
    """
    import json
    
    # Parse workflow definition
    try:
        parsed_def = parse_workflow_definition(definition)
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "nodes_completed": [],
            "nodes_failed": [],
            "nodes_skipped": [],
        }, indent=2)
    
    # Get delegate context
    ctx = _get_delegate_context()
    
    # Execute workflow
    try:
        result = await execute_workflow(
            definition=parsed_def,
            ctx=ctx,
            session_id=session_id,
            max_retries=max_retries,
        )
        return json.dumps(result.to_dict(), indent=2)
    except Exception as e:
        log.error("workflow_execution_failed", error=str(e))
        return json.dumps({
            "success": False,
            "error": str(e),
            "nodes_completed": [],
            "nodes_failed": [],
            "nodes_skipped": [n.id for n in parsed_def.nodes],
        }, indent=2)


@mcp.tool()
async def agent(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    max_iterations: int = 10,
    tools: str | None = None,
    backend_type: str | None = None,
    workspace: str | None = None,
) -> str:
    """
    Run an autonomous agent that can use tools to complete tasks.
    The agent loops: understand task -> call tools -> process results -> respond.

    WHEN TO USE:
    - Tasks requiring file exploration or code search
    - Multi-step tasks that need to gather information
    - Any task where the LLM needs to read files or search code

    Args:
        prompt: The task for the agent to complete (required)
        system_prompt: Optional system prompt to guide behavior
        model: Force specific tier - "quick" | "coder" | "moe" | "thinking"
        max_iterations: Maximum tool call iterations (default: 10)
        tools: Comma-separated tool names to enable. Options:
            - "read_file" - Read file contents
            - "list_directory" - List files in directories
            - "search_code" - Search code with regex
            - "web_fetch" - Fetch URL contents
            If not specified, all tools are enabled.
        backend_type: Force backend type (optional)
        workspace: Optional workspace directory to confine file operations.
            If provided, all file operations (read_file, list_directory, search_code)
            are restricted to this directory. Prevents agent from accessing files
            outside the project. If not provided, agent can access any file.

    AVAILABLE TOOLS:
    - read_file(path, start_line?, end_line?) - Read file with line numbers
    - list_directory(path?, recursive?, pattern?) - List directory contents
    - search_code(pattern, path?, file_pattern?, context_lines?) - Grep for code
    - web_fetch(url, extract_text?) - Fetch web content

    AGENT BEHAVIOR:
    1. Receives the task/prompt
    2. Decides which tools to use
    3. Calls tools and receives results
    4. Continues until task is complete or max_iterations reached

    Returns:
        Agent's final response after completing the task

    Examples:
        agent(prompt="What files are in the src directory?")
        agent(prompt="Read config.py and explain the main class")
        agent(prompt="Search for all usages of 'async def' in .py files")
        agent(prompt="Analyze the error handling patterns in this codebase", model="moe")
        agent(prompt="Review the code", workspace="/home/user/project")
    """
    import json as json_module

    start_time = time.time()

    # Select backend
    backend_provider, backend_obj = await _select_optimal_backend_v2(
        prompt, None, "analyze", backend_type
    )

    # Select model based on task
    selected_model = await select_model(
        task_type="analyze",
        content_size=len(prompt),
        model_override=model,
        content=prompt,
    )

    # Warn if using quick-tier model for agent tasks (may struggle with tool calling)
    quick_model = backend_obj.models.get("quick") if backend_obj else None
    coder_model = backend_obj.models.get("coder") if backend_obj else None
    is_quick_tier = quick_model and selected_model == quick_model and selected_model != coder_model
    model_tier_warning = ""
    if is_quick_tier:
        log.warning(
            "agent_using_small_model",
            model=selected_model,
            hint="Small models may struggle with reliable tool calling. Consider using --model coder or moe.",
        )
        model_tier_warning = " [WARN] Small model may have reduced tool-calling reliability"

    # Set up tool registry with optional workspace confinement
    # Use coding tools (includes all default tools + run_tests, apply_diff, git_*)
    from .tools.coding import get_coding_tools
    workspace_obj = Workspace(root=workspace) if workspace else None
    registry = get_coding_tools(workspace=workspace_obj, allow_exec=True)

    # Register MCP tools if any servers are running
    if mcp_client_manager.get_all_tools():
        mcp_client_manager.register_tools(registry)

    if tools:
        tool_list = [t.strip() for t in tools.split(",") if t.strip()]
        registry = registry.filter(tool_list)

    # Detect native tool calling support
    use_native = backend_obj.supports_native_tool_calling if backend_obj else False

    # Get tool schemas for native mode (even if not immediately used)
    tools_schemas = registry.get_openai_schemas() if use_native else None

    # Create LLM callable wrapper for the agent loop
    async def agent_llm_call(
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> str:
        # Convert messages to a single prompt
        # The agent loop uses a messages list, but call_llm expects a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "tool")
                prompt_parts.append(f"[Tool Result - {tool_name}]\n{content}")

        combined_prompt = "\n\n".join(prompt_parts)

        # Create preview from the user's original prompt for logging
        content_preview = combined_prompt[:200].replace("\n", " ").strip()

        result = await call_llm(
            model=selected_model,
            prompt=combined_prompt,
            system=system,
            task_type="agent",
            original_task="agent",
            language="unknown",
            content_preview=content_preview,
            backend_obj=backend_obj,
            tools=tools_schemas,
            tool_choice="auto" if tools_schemas else None,
        )

        if result.get("success"):
            return result.get("response", "")
        else:
            raise RuntimeError(result.get("error", "LLM call failed"))

    # Create agent config with auto-detected native tool calling
    config = AgentConfig(
        max_iterations=max_iterations,
        timeout_per_tool=30.0,
        total_timeout=300.0,
        parallel_tools=True,
        native_tool_calling=use_native,
    )

    # Run the agent loop
    try:
        result = await run_agent_loop(
            call_llm=agent_llm_call,
            prompt=prompt,
            system_prompt=system_prompt,
            registry=registry,
            model=selected_model,
            config=config,
        )
    except Exception as e:
        log.error("agent_error", error=str(e))
        return f"Agent error: {e}"

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Format response with metadata
    tool_summary = ""
    if result.tool_calls:
        tool_names = [tc.name for tc in result.tool_calls]
        tool_summary = f"\nTools used: {', '.join(tool_names)}"

    workspace_info = f" | Workspace: {workspace}" if workspace else ""
    status = "[OK]" if result.success else "[WARN]"
    return f"""{result.response}

---
_{status} Agent completed | Iterations: {result.iterations} | Time: {elapsed_ms}ms | Model: {selected_model}{workspace_info}{tool_summary}{model_tier_warning}_"""


@mcp.tool()
async def health() -> str:
    """
    Check health status of Delia and all configured GPU backends.

    Only checks backends that are enabled in settings.json.
    Shows availability, loaded models, usage stats, and cost savings.

    WHEN TO USE:
    - Verify backends are available before delegating
    - Check which models are currently loaded
    - Monitor usage statistics and cost savings
    - Diagnose connection issues

    Returns:
        JSON with:
        - status: "healthy" | "degraded" | "unhealthy"
        - backends: Array of configured backend status with performance scores
        - usage: Token counts and call statistics per tier
        - cost_savings: Estimated savings vs cloud API
        - routing: Current routing configuration
    """
    # Get health status from BackendManager (only checks enabled backends)
    health_status = await backend_manager.get_health_status()

    # Add performance scores to each backend using configured weights
    weights = backend_manager.get_scoring_weights()
    scorer = BackendScorer(weights=weights)
    backend_lookup = {b.id: b for b in backend_manager.backends.values()}
    for backend_info in health_status["backends"]:
        backend_obj = backend_lookup.get(backend_info["id"])
        if backend_obj and backend_info["enabled"]:
            score = scorer.score(backend_obj)
            backend_info["score"] = round(score, 3)
            # Add metrics summary if available
            from .config import get_backend_metrics

            metrics = get_backend_metrics(backend_info["id"])
            if metrics.total_requests > 0:
                backend_info["metrics"] = {
                    "success_rate": f"{metrics.success_rate * 100:.1f}%",
                    "latency_p50_ms": round(metrics.latency_p50, 1),
                    "throughput_tps": round(metrics.tokens_per_second, 1),
                    "total_requests": metrics.total_requests,
                }

    # Get usage stats from StatsService
    model_usage, _, _, _ = stats_service.get_snapshot()

    # Calculate totals (using new 4-tier keys: quick/coder/moe/thinking)
    total_quick_tokens = model_usage["quick"]["tokens"]
    total_coder_tokens = model_usage["coder"]["tokens"]
    total_moe_tokens = model_usage["moe"]["tokens"]
    total_thinking_tokens = model_usage["thinking"]["tokens"]
    local_tokens = total_quick_tokens + total_coder_tokens + total_moe_tokens + total_thinking_tokens
    local_calls = (
        model_usage["quick"]["calls"]
        + model_usage["coder"]["calls"]
        + model_usage["moe"]["calls"]
        + model_usage["thinking"]["calls"]
    )

    # Estimate cost savings (vs GPT-4)
    local_savings = (local_tokens / 1000) * config.gpt4_cost_per_1k_tokens

    # Get voting stats
    from .voting_stats import get_voting_stats_tracker
    voting_tracker = get_voting_stats_tracker()
    voting_stats = voting_tracker.get_stats()

    # Build response
    status = {
        "status": health_status["status"],
        "active_backend": health_status["active_backend"],
        "backends": health_status["backends"],
        "routing": health_status["routing"],
        "usage": {
            "quick": {
                "calls": humanize.intcomma(model_usage["quick"]["calls"]),
                "tokens": humanize.intword(total_quick_tokens),
            },
            "coder": {
                "calls": humanize.intcomma(model_usage["coder"]["calls"]),
                "tokens": humanize.intword(total_coder_tokens),
            },
            "moe": {
                "calls": humanize.intcomma(model_usage["moe"]["calls"]),
                "tokens": humanize.intword(total_moe_tokens),
            },
            "total_calls": humanize.intcomma(local_calls),
            "total_tokens": humanize.intword(local_tokens),
            "estimated_savings": f"${local_savings:,.2f}",
        },
        "voting": voting_stats,
    }

    return json.dumps(status, indent=2)


@mcp.tool()
async def queue_status() -> str:
    """
    Get current status of the model queue system.

    Shows loaded models, queued requests, and GPU memory usage.
    Useful for monitoring queue performance and debugging loading issues.

    Returns:
        JSON with queue status, loaded models, and pending requests
    """
    if not hasattr(model_queue, "loaded_models"):
        return json.dumps({"status": "queue_not_initialized", "message": "ModelQueue not yet initialized"}, indent=2)

    # Get current queue state
    loaded = model_queue.loaded_models.copy()
    queued_requests = []

    # Get queued requests from all model queues
    for _model_name, queue in model_queue.request_queues.items():
        for queued_request in queue:
            queued_requests.append(
                {
                    "id": queued_request.request_id,
                    "task": queued_request.task_type,
                    "model": queued_request.model_name,
                    "priority": queued_request.priority,
                    "queued_at": queued_request.timestamp.isoformat() if queued_request.timestamp else None,
                    "estimated_tokens": queued_request.content_length // 4,  # Rough estimate
                }
            )

    # Calculate memory usage
    total_vram = sum(model["size_gb"] * 1024 for model in loaded.values())  # Convert GB to MB

    status = {
        "status": "active",
        "loaded_models": [
            {
                "name": name,
                "size_gb": model["size_gb"],
                "loaded_at": model["loaded_at"].isoformat() if model["loaded_at"] else None,
                "last_used": model["last_used"].isoformat() if model["last_used"] else None,
            }
            for name, model in loaded.items()
        ],
        "queued_requests": queued_requests,
        "queue_stats": {
            "total_queued": len(queued_requests),
            "total_vram_used_mb": total_vram,
            "max_vram_mb": model_queue.gpu_memory_limit_gb * 1024,  # Convert GB to MB
            "vram_available_mb": model_queue.get_available_memory() * 1024,  # Convert GB to MB
        },
        "queue_config": {
            "gpu_memory_limit_gb": model_queue.gpu_memory_limit_gb,
            "memory_buffer_gb": model_queue.memory_buffer_gb,
        },
    }

    return json.dumps(status, indent=2)


@mcp.tool()
async def models() -> str:
    """
    List all configured models across all GPU backends.
    Shows model tiers (quick/coder/moe) and which are currently loaded.

    WHEN TO USE:
    - Check which models are available for tasks
    - Verify model configuration across backends
    - Understand task-to-model routing logic

    Returns:
        JSON with:
        - backends: All configured backends with their models
        - currently_loaded: Models in GPU memory (no load time)
        - selection_logic: How tasks map to model tiers
    """
    loaded = await get_loaded_models()

    # Build backends info from BackendManager
    backends_info = []
    for backend in backend_manager.get_enabled_backends():
        backends_info.append(
            {
                "id": backend.id,
                "name": backend.name,
                "provider": backend.provider,
                "url": backend.url,
                "models": backend.models,
            }
        )

    info = {
        "active_backend": get_active_backend_id(),
        "backends": backends_info,
        "currently_loaded": loaded,
        "selection_logic": {
            "quick_tasks": ["quick", "summarize"],
            "coder_tasks": ["generate", "review", "analyze"],
            "moe_tasks": ["plan", "critique"],
            "large_content_threshold": f"{config.large_content_threshold // 1024}KB",
        },
    }

    return json.dumps(info, indent=2)


@mcp.tool()
async def melons(task_type: str | None = None) -> str:
    """
    Display the melon leaderboard for Delia's model garden.

    Models LOVE melons! 🍈 They earn them for helpful responses:
    - Regular melons boost routing preference
    - Golden melons (500 melons) mark star performers
    
    The more melons a model has, the more likely it is to be
    selected for tasks - building trust through quality!

    Args:
        task_type: Filter by task type (quick/coder/moe) or None for all

    Returns:
        Formatted leaderboard showing model rankings
    """
    from .melons import get_melon_tracker

    tracker = get_melon_tracker()
    
    leaderboard = tracker.get_leaderboard(task_type=task_type)
    
    if not leaderboard:
        return """🍈 DELIA'S MELON GARDEN 🍈
========================================

The garden is empty... but not for long!

Models earn melons by being helpful:
• 🍈 Regular melons for quality responses
• 🏆 Golden melons (500 melons) for stars

WHY MELONS MATTER:
Models LOVE melons! Each one gives a routing
boost, making trusted models more likely to
be selected for future tasks.

Start chatting to plant the first seeds! 🌱"""

    # Calculate totals
    total_melons = sum(s.melons for s in leaderboard)
    total_golden = sum(s.golden_melons for s in leaderboard)

    medals = ["🥇", "🥈", "🥉"]
    
    # Build leaderboard lines
    board_lines = []
    for i, stats in enumerate(leaderboard[:10]):  # Top 10
        medal = medals[i] if i < 3 else f"{i+1:>2}."
        golden = f"+{stats.golden_melons}G" if stats.golden_melons else "   "
        rate = f"{stats.success_rate:.0%}" if stats.total_responses > 0 else "  -"
        board_lines.append(f"{medal} {stats.model_id:<28} {stats.melons:>3} {golden} [{stats.task_type:<6}] {rate}")
    
    if len(leaderboard) > 10:
        board_lines.append(f"    ...and {len(leaderboard) - 10} more")
    
    board_text = "\n".join(board_lines)
    
    return f"""🍈 MELON LEADERBOARD
Total: {total_melons} melons | {total_golden} golden

```
{board_text}
```
Higher melons = higher routing priority"""


@mcp.tool()
async def switch_backend(backend_id: str) -> str:
    """
    Switch the active LLM backend.

    Args:
        backend_id: ID of the backend to switch to (from settings.json)

    Returns:
        Confirmation message with current status
    """
    backend = backend_manager.get_backend(backend_id)
    if not backend:
        available = [b.id for b in backend_manager.get_enabled_backends()]
        return f"Error: Invalid backend '{backend_id}'. Available backends: {available}"

    if backend_manager.set_active_backend(backend_id):
        # Check if the new backend is available
        is_available = await backend.check_health()
        status = "available" if is_available else "unreachable"

        return f"""Switched to **{backend.name}** backend.

- **ID**: {backend.id}
- **URL**: {backend.url}
- **Provider**: {backend.provider}
- **Status**: {status}

All subsequent delegate() calls will use this backend."""
    else:
        return f"Error: Failed to switch to backend '{backend_id}'."


@mcp.tool()
async def switch_model(tier: str, model_name: str) -> str:
    """
    Switch the model for a specific tier at runtime.

    This allows dynamic model experimentation without restarting the server.
    Changes are persisted to settings.json for consistency across restarts.

    Args:
        tier: Model tier to change - "quick", "coder", "moe", or "thinking"
        model_name: New model name (must be available in the current backend)

    Returns:
        Confirmation with model change details and availability status
    """
    if tier not in VALID_MODELS:
        return f"Error: Invalid tier '{tier}'. Must be one of: {', '.join(sorted(VALID_MODELS))}"

    # Get active backend
    backend = backend_manager.get_active_backend()
    if not backend:
        return "Error: No active backend found to switch model for."

    # Validate model availability in current backend
    # This is tricky because we might not be able to list models for all providers.
    # We will trust the user but try to warn if we can check.
    available_models = []

    try:
        if backend.provider == "ollama":
            client = backend.get_client()
            response = await client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                available_models = [m["name"] for m in data.get("models", [])]
                if model_name not in available_models:
                    return f"""Error: Model '{model_name}' not found in Ollama backend '{backend.name}'.

Available models: {", ".join(sorted(available_models))}

Pull the model first: `ollama pull {model_name}`"""
            else:
                log.warning("model_check_failed", backend=backend.name, status=response.status_code)

    except Exception as e:
        log.warning("model_check_failed", backend=backend.name, error=str(e))
        # Continue anyway to allow manual override

    # Update the backend configuration
    old_model = backend.models.get(tier, "none")

    # Create a copy of models dict to update
    new_models = backend.models.copy()
    new_models[tier] = model_name

    # Update via manager (persists to settings.json)
    updated_backend = await backend_manager.update_backend(backend.id, {"models": new_models})

    if not updated_backend:
        return "Error: Failed to update backend configuration."

    # Get model info for response
    model_info = get_model_info(model_name)

    return f"""Model switched successfully!

**Backend**: {backend.name} ({backend.provider})
**Tier**: {tier}
**Old Model**: {old_model}
**New Model**: {model_name}

**Model Info**:
- **VRAM**: {model_info.get("vram_gb", "Unknown")}GB
- **Context**: {model_info.get("context_tokens", "Unknown")} tokens

**Persistence**: Configuration saved to settings.json - change persists across restarts.

**Next Steps**: Test with `delegate()` using tasks that would use the {tier} tier."""


@mcp.tool()
async def get_model_info_tool(model_name: str) -> str:
    """
    Get detailed information about a specific model.

    Returns VRAM requirements, context window size, and tier classification.
    For configured models, shows exact values. For unknown models, provides estimates.

    Args:
        model_name: Name of the model to get info for (e.g., "qwen2.5:14b", "llama3.1:70b")

    Returns:
        Formatted model information including VRAM, context, and tier
    """
    info = get_model_info(model_name)

    vram = info.get("vram_gb", "Unknown")
    context = info.get("context_tokens", "Unknown")
    tier = info.get("tier", "unknown")

    # Format numbers nicely
    vram_str = f"{vram}GB" if isinstance(vram, (int, float)) else str(vram)

    if isinstance(context, int):
        context_str = f"{context:,} tokens" if context >= 1000 else f"{context} tokens"
    else:
        context_str = str(context)

    return f"""Model Information: {model_name}

**VRAM Requirement**: {vram_str}
**Context Window**: {context_str}
**Tier**: {tier}

**Notes**:
- VRAM estimates are approximate and depend on quantization
- Context windows may vary by model version
- Tier classification is based on configured models or size estimates"""


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
    Useful for cross-server workflows where Serena or other tools
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


@mcp.resource("delia://memories", name="Available Memories", description="List of Serena memory files")
async def resource_memories() -> str:
    """
    List available Serena memory files.

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

    - Starts background save task for tracker
    - Pre-warms tiktoken encoder to avoid first-call delay
    - Clears expired sessions
    """
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
