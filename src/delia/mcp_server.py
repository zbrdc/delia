#!/usr/bin/env python3
# Copyright (C) 2023 the project owner
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
Delia — Local LLM Delegation Server

A pure MCP server that intelligently routes tasks to optimal local models.
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
from .config import STATS_FILE, config, detect_model_tier, get_backend_health

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
)

# Import model queue system
from .queue import ModelQueue, QueuedRequest

# Import routing utilities (content detection, model override parsing)
from .routing import CODE_INDICATORS, detect_code_content, parse_model_override

# Import language detection (LANGUAGE_CONFIGS, detect_language, get_system_prompt, optimize_prompt)
from .language import (
    LANGUAGE_CONFIGS,
    PYGMENTS_LANGUAGE_MAP,
    detect_language,
    get_system_prompt,
    optimize_prompt,
)

# Import file helpers (read_files, read_serena_memory, read_file_safe, MEMORY_DIR)
from .file_helpers import MEMORY_DIR, read_file_safe, read_files, read_serena_memory

# Import stats service for usage tracking
from .stats import StatsService

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
model_queue = ModelQueue()


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
    """
    # Save model/task stats via service
    await stats_service.save_all()

    # Save other data (live logs, circuit breaker)
    await asyncio.to_thread(_save_live_logs_sync)
    await asyncio.to_thread(save_circuit_breaker_stats)


# Ensure all data directories exist
paths.ensure_directories()

# Load stats immediately at module import time
stats_service.load()
load_live_logs()

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





async def select_model(
    task_type: str, content_size: int = 0, model_override: str | None = None, content: str = ""
) -> str:
    """
    Select the best model for the task with intelligent code-aware routing.

    Tiers (configured in settings.json):
    - quick: Fast general tasks, text analysis, summarize
    - coder: Code generation, review, analysis
    - moe: Complex reasoning - plan, critique, large text

    Strategy:
    1. Honor explicit overrides
    2. MoE tasks (plan, critique) always use MoE model
    3. Detect if content is CODE or TEXT:
       - Large CODE → coder (specialized for programming)
       - Large TEXT → moe (better reasoning for prose)
    4. Code-focused tasks on code content → coder
    5. Default to quick for everything else
    """
    # Get models from active backend
    backend = backend_manager.get_active_backend()
    if backend:
        model_quick = backend.models.get("quick", config.model_quick.ollama_model)
        model_coder = backend.models.get("coder", config.model_coder.ollama_model)
        model_moe = backend.models.get("moe", config.model_moe.ollama_model)
        model_thinking = backend.models.get("thinking", config.model_thinking.ollama_model)
    else:
        # Fallback to config defaults
        model_quick = config.model_quick.ollama_model
        model_coder = config.model_coder.ollama_model
        model_moe = config.model_moe.ollama_model
        model_thinking = config.model_thinking.ollama_model

    # Helper to resolve tier name to model name
    def resolve_tier(tier_name):
        if tier_name == "quick":
            return model_quick
        if tier_name == "coder":
            return model_coder
        if tier_name == "moe":
            return model_moe
        if tier_name == "thinking":
            return model_thinking
        return tier_name  # Assume it's a model name if not a tier

    # Priority 1: Explicit override
    if model_override:
        resolved = resolve_tier(model_override)
        log.info("model_selected", source="override", tier=model_override, model=resolved)
        return resolved

    # Priority 2: Tasks that REQUIRE MoE (complex multi-step reasoning)
    if task_type in config.moe_tasks:
        log.info("model_selected", source="moe_task", task=task_type, tier="moe")
        return model_moe

    # Detect code content once (cache result for reuse below)
    code_detection = None
    if content and (content_size > config.large_content_threshold or task_type in config.coder_tasks):
        code_detection = detect_code_content(content)

    # Priority 3: Large content - detect if code or text
    if content_size > config.large_content_threshold and code_detection:
        is_code, confidence, reasoning = code_detection
        if is_code and confidence > 0.5:
            log.info(
                "model_selected",
                source="large_code",
                content_kb=content_size // 1000,
                confidence=f"{confidence:.0%}",
                tier="coder",
                reasoning=reasoning,
            )
            return model_coder
        else:
            # Large text content benefits from MoE's reasoning
            log.info(
                "model_selected",
                source="large_text",
                content_kb=content_size // 1000,
                confidence=f"{1 - confidence:.0%}",
                tier="moe",
                reasoning=reasoning,
            )
            return model_moe

    # Priority 4: Code-focused tasks - check if content is actually code (reuse cached detection)
    if task_type in config.coder_tasks and code_detection:
        is_code, confidence, reasoning = code_detection
        if is_code or confidence > 0.3:
            log.info("model_selected", source="coder_task_code", task=task_type, tier="coder", reasoning=reasoning)
            return model_coder
        else:
            # Task like "analyze" on text should use quick/moe, not coder
            log.info("model_selected", source="coder_task_text", task=task_type, tier="quick", reasoning=reasoning)
            # Fall through to default

    # Priority 5: Default to quick (fastest model)
    log.info("model_selected", source="default", task=task_type, tier="quick")
    return model_quick


# ============================================================
# PROVIDER FACTORY (lazy initialization)
# ============================================================

# Cache for provider instances (created lazily on first use)
_provider_cache: dict[str, OllamaProvider | LlamaCppProvider | GeminiProvider] = {}

# Provider name to class mapping (includes aliases)
_PROVIDER_CLASS_MAP: dict[str, type[OllamaProvider | LlamaCppProvider | GeminiProvider]] = {
    "ollama": OllamaProvider,
    "llamacpp": LlamaCppProvider,
    "llama.cpp": LlamaCppProvider,
    "lmstudio": LlamaCppProvider,
    "vllm": LlamaCppProvider,
    "openai": LlamaCppProvider,
    "custom": LlamaCppProvider,
    "gemini": GeminiProvider,
}


def _save_stats_background() -> None:
    """Schedule stats saving as a background task (non-blocking)."""
    _schedule_background_task(save_all_stats_async())


def _get_provider(provider_name: str) -> OllamaProvider | LlamaCppProvider | GeminiProvider | None:
    """Get or create a provider instance for the given provider name.

    Uses lazy initialization - providers are created on first use and cached.
    This avoids circular dependency issues with _update_stats_sync.

    Args:
        provider_name: Provider type (ollama, llamacpp, gemini, etc.)

    Returns:
        Provider instance or None if provider not found
    """
    if provider_name not in _PROVIDER_CLASS_MAP:
        return None

    # Return cached instance if available
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    # Create new instance with dependencies
    # Note: save_stats_callback is wrapped to schedule async task properly
    provider_class = _PROVIDER_CLASS_MAP[provider_name]
    provider = provider_class(
        config=config,
        backend_manager=backend_manager,
        stats_callback=_update_stats_sync,
        save_stats_callback=_save_stats_background,
    )
    _provider_cache[provider_name] = provider
    return provider



async def call_llm(
    model: str,
    prompt: str,
    system: str | None = None,
    enable_thinking: bool = False,
    task_type: str = "unknown",
    original_task: str = "unknown",
    language: str = "unknown",
    content_preview: str = "",
    backend: str | BackendConfig | None = None,
    backend_obj: BackendConfig | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Unified LLM call dispatcher that routes to the appropriate backend.

    Args:
        model: Model name/tier
        prompt: The prompt to send
        system: Optional system prompt
        enable_thinking: Enable thinking mode for supported models
        task_type: Type of task for tracking
        language: Detected language for tracking
        content_preview: Preview for logging
        backend: Override backend ID or provider name
        backend_obj: Optional BackendConfig object to use directly
        max_tokens: Maximum response tokens (limits verbosity)
    """
    # Acquire model from queue (prevents concurrent loading)
    # Note: Gemini is cloud-based so we might skip local queue for it,
    # but the queue also manages concurrency limits which is good.
    content_length = len(prompt) + len(system or "")
    is_available, queue_future = await model_queue.acquire_model(model, task_type, content_length)

    # If model is not immediately available, wait for it to be queued and processed
    if not is_available and queue_future:
        try:
            await asyncio.wait_for(queue_future, timeout=300)  # Wait up to 5 minutes for model to load
        except TimeoutError:
            model_queue.queue_timeouts += 1
            await model_queue.release_model(model, success=False)
            log.warning(
                "queue_timeout",
                model=model,
                wait_seconds=300,
                total_timeouts=model_queue.queue_timeouts,
                log_type="QUEUE",
            )
            return {"success": False, "error": f"Timeout waiting for model {model} to load (waited 5 minutes)"}
        except Exception as e:
            await model_queue.release_model(model, success=False)
            log.error("queue_error", model=model, error=str(e), log_type="QUEUE")
            return {"success": False, "error": f"Error waiting for model {model}: {e!s}"}

    try:
        # Determine backend to use
        active_backend = None

        # 1. Use passed object if available
        if backend_obj:
            active_backend = backend_obj
        # 2. Resolve by ID or provider name
        elif backend:
            # If backend is already a BackendConfig, use it directly
            if isinstance(backend, BackendConfig):
                active_backend = backend
            else:
                # Try as ID
                active_backend = backend_manager.get_backend(backend)
                if not active_backend:
                    # Try as provider name
                    for b in backend_manager.get_enabled_backends():
                        if b.provider == backend:
                            active_backend = b
                            break
        # 3. Use default active backend
        else:
            active_backend = backend_manager.get_active_backend()

        if not active_backend:
            await model_queue.release_model(model, success=False)
            return {"success": False, "error": "No active backend found"}

        # Dispatch using provider factory (lazy initialization, cached)
        provider = _get_provider(active_backend.provider)
        if not provider:
            await model_queue.release_model(model, success=False)
            return {"success": False, "error": f"Unsupported provider: {active_backend.provider}"}

        response = await provider.call(
            model=model,
            prompt=prompt,
            system=system,
            enable_thinking=enable_thinking,
            task_type=task_type,
            original_task=original_task,
            language=language,
            content_preview=content_preview,
            backend_obj=active_backend,
            max_tokens=max_tokens,
        )
        result = response.to_dict()

        # Release model on success
        await model_queue.release_model(model, success=result.get("success", False))
        return result
    except Exception:
        # Release model on failure
        await model_queue.release_model(model, success=False)
        raise


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


async def validate_delegate_request(
    task: str,
    content: str,
    file: str | None = None,
    model: str | None = None,
) -> tuple[bool, str]:
    """Validate all inputs for delegate request."""
    # Input validation
    valid, error = validate_task(task)
    if not valid:
        return False, f"Error: {error}"

    valid, error = validate_content(content)
    if not valid:
        return False, f"Error: {error}"

    valid, error = validate_file_path(file)
    if not valid:
        return False, f"Error: {error}"

    valid, error = validate_model_hint(model)
    if not valid:
        return False, f"Error: {error}"

    return True, ""


async def prepare_delegate_content(
    content: str,
    context: str | None = None,
    symbols: str | None = None,
    include_references: bool = False,
    files: str | None = None,
) -> str:
    """Prepare content with context, files, and symbol focus.

    Args:
        content: The main task content/prompt
        context: Comma-separated Serena memory names to include
        symbols: Comma-separated symbol names to focus on
        include_references: Whether references to symbols are included
        files: Comma-separated file paths to read and include (Delia reads directly)
    """
    parts = []

    # Load files directly from disk (efficient - no Claude serialization)
    if files:
        file_contents = read_files(files)
        if file_contents:
            for path, file_content in file_contents:
                # Detect language from extension for syntax highlighting hint
                ext = Path(path).suffix.lstrip(".")
                lang_hint = ext if ext else ""
                parts.append(f"### File: `{path}`\n```{lang_hint}\n{file_content}\n```")
            log.info("files_loaded", count=len(file_contents), paths=[p for p, _ in file_contents])

    # Load Serena memory context if specified
    if context:
        memory_names = [m.strip() for m in context.split(",")]
        for mem_name in memory_names:
            mem_content = read_serena_memory(mem_name)
            if mem_content:
                parts.append(f"### Context from '{mem_name}':\n{mem_content}")
                log.info("context_memory_loaded", memory=mem_name)

    # Add symbol focus hint if symbols specified
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        symbol_hint = f"### Focus Symbols: {', '.join(symbol_list)}"
        if include_references:
            symbol_hint += "\n_References to these symbols are included below._"
        parts.append(symbol_hint)
        log.info("context_symbol_focus", symbols=symbol_list, include_references=include_references)

    # Add task content
    if parts:
        parts.append(f"---\n\n### Task:\n{content}")
        return "\n\n".join(parts)
    else:
        return content


def determine_task_type(task: str) -> str:
    """Map user task to internal task type."""
    task_map = {
        "review": "review",
        "analyze": "analyze",
        "generate": "generate",
        "summarize": "summarize",
        "critique": "critique",
        "quick": "quick",
        "plan": "plan",
        "think": "analyze",  # Treat direct think tasks as analyze-tier by default
    }
    return task_map.get(task, "analyze")


async def select_delegate_model(
    task_type: str,
    content: str,
    model_override: str | None = None,
    backend: str | None = None,
    backend_obj: Any | None = None,
) -> tuple[str, str, str | BackendConfig | None]:
    """Select appropriate model and backend for the task."""
    # Determine model tier from task type
    if task_type in config.moe_tasks:
        tier = "moe"
    elif task_type in config.coder_tasks:
        tier = "coder"
    elif task_type == "quick" or task_type == "summarize":
        tier = "quick"
    else:
        tier = "quick"  # Default

    # Override tier with model hint if provided
    if model_override and model_override in VALID_MODELS:
        tier = model_override

    # Get the actual model name from backend manager or fall back to config.py
    selected_model = None

    # First priority: use backend_obj if provided (passed from delegate())
    if backend_obj:
        selected_model = backend_obj.models.get(tier)
        if selected_model:
            log.info("model_from_backend_obj", backend=backend_obj.id, tier=tier, model=selected_model)

    # Try simplified backend selection (replaces complex backend_manager)
    if not selected_model:
        selected_model = await select_model(task_type, len(content), model_override, content)
        log.info("model_from_simplified_selection", tier=tier, model=selected_model)

    # Use provided backend or fall back to active backend
    target_backend = backend or get_active_backend()

    return selected_model, tier, target_backend


async def execute_delegate_call(
    selected_model: str,
    content: str,
    system: str,
    task_type: str,
    original_task: str,
    detected_language: str,
    target_backend: str | BackendConfig | None,
    backend_obj: Any | None = None,
    max_tokens: int | None = None,
) -> tuple[str, int]:
    """Execute the LLM call and return response with metadata."""
    # Call LLM
    enable_thinking = task_type in config.thinking_tasks
    # Create a preview for the recent calls log
    content_preview = content[:200].replace("\n", " ").strip()

    result = await call_llm(
        selected_model,
        content,
        system,
        enable_thinking,
        task_type=task_type,
        original_task=original_task,
        language=detected_language,
        content_preview=content_preview,
        backend=target_backend,
        backend_obj=backend_obj,
        max_tokens=max_tokens,
    )

    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        raise Exception(f"LLM call failed: {error_msg}")

    response_text = result.get("response", "")
    tokens = result.get("tokens", 0)

    # Strip thinking tags
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1].strip()

    return response_text, tokens


def finalize_delegate_response(
    response_text: str,
    selected_model: str,
    tokens: int,
    elapsed_ms: int,
    detected_language: str,
    target_backend: Any,
    tier: str,
    include_metadata: bool = True,
) -> str:
    """Add metadata footer and update tracking.

    Args:
        include_metadata: If False, return response without footer (saves ~30 tokens)
    """
    # Update tracker with actual token count and model tier
    client_id = current_client_id.get()
    if client_id:
        tracker.update_last_request(client_id, tokens=tokens, model_tier=tier)

    # Return without metadata if requested (saves Claude tokens)
    if not include_metadata:
        return response_text

    # Extract backend name (handle both string and BackendConfig)
    backend_name = target_backend.name if hasattr(target_backend, "name") else str(target_backend)

    # Add concise metadata footer
    return f"""{response_text}

---
_Model: {selected_model} | Tokens: {tokens} | Time: {elapsed_ms}ms | Backend: {backend_name}_"""


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
    backend_obj: Any | None = None,  # Backend object from backend_manager
    files: str | None = None,  # Comma-separated file paths (Delia reads directly)
    include_metadata: bool = True,  # Include footer with model/tokens/time info
    max_tokens: int | None = None,  # Limit response length (reduces verbosity)
) -> str:
    """
    Core implementation for delegate - can be called directly by batch().

    Enhanced context parameters:
        symbols: Comma-separated symbol names to focus on (e.g., "Foo,Foo/calculate")
        include_references: If True, indicates that references to symbols are included in content
        backend: Override backend ("ollama" or "llamacpp"), defaults to active_backend
        files: Comma-separated file paths - Delia reads directly from disk (efficient)
        include_metadata: If False, skip the metadata footer (saves ~30 Claude tokens)
        max_tokens: Limit response tokens (forces concise output, saves Claude tokens)
    """
    start_time = time.time()

    # Validate request
    valid, error = await validate_delegate_request(task, content, file, model)
    if not valid:
        return error

    # Prepare content with context, files, and symbols
    prepared_content = await prepare_delegate_content(content, context, symbols, include_references, files)

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

    # Select model and backend
    selected_model, tier, target_backend = await select_delegate_model(
        task_type, prepared_content, model, backend, backend_obj
    )

    # Execute the LLM call
    try:
        response_text, tokens = await execute_delegate_call(
            selected_model,
            prepared_content,
            system,
            task_type,
            task,
            detected_language,
            target_backend,
            backend_obj,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return f"Error: {e!s}"

    # Calculate timing
    elapsed_ms = int((time.time() - start_time) * 1000)

    # Finalize response with metadata
    return finalize_delegate_response(
        response_text,
        selected_model,
        tokens,
        elapsed_ms,
        detected_language,
        target_backend,
        tier,
        include_metadata=include_metadata,
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
) -> str:
    """
    Execute a task on local/remote GPU with intelligent 3-tier model selection.
    Routes to optimal backend based on content size, task type, and GPU availability.

    WHEN TO USE:
    - "locally", "on my GPU", "without API", "privately" → Use this tool
    - Code review, generation, analysis tasks → Use this tool
    - Any task you want processed on local hardware → Use this tool

    Args:
        task: Task type determines model tier:
            - "quick" or "summarize" → quick tier (fast, 14B model)
            - "generate", "review", "analyze" → coder tier (code-optimized 14B)
            - "plan", "critique" → moe tier (deep reasoning 30B+)
        content: The prompt or content to process (required)
        file: Optional file path to include in context
        model: Force specific tier - "quick" | "coder" | "moe" | "thinking"
              OR natural language: "7b", "14b", "30b", "small", "large", "coder model", "fast", "complex", "thinking"
        language: Language hint for better prompts - python|typescript|react|nextjs|rust|go
        context: Serena memory names to include (comma-separated: "architecture,decisions")
        symbols: Code symbols to focus on (comma-separated: "Foo,Bar/calculate")
        include_references: True if content includes symbol usages from elsewhere
        backend_type: Force backend type - "local" | "remote" (default: auto-select)
        files: Comma-separated file paths - Delia reads directly from disk (efficient, no serialization)
        include_metadata: If False, skip the metadata footer (saves ~30 tokens). Default: True
        max_tokens: Limit response length to N tokens (forces concise output). Default: None (unlimited)

    ROUTING LOGIC:
    1. Content > 32K tokens → Uses backend with largest context window
    2. Prefer local GPUs (lower latency) unless unavailable
    3. Falls back to remote if local circuit breaker is open
    4. Load balances across available backends based on priority weights

    Returns:
        LLM response optimized for orchestrator processing, with optional metadata footer

    Examples:
        delegate(task="review", content="<code>", language="python")
        delegate(task="generate", content="Write a REST API", backend_type="local")
        delegate(task="plan", content="Design caching strategy", model="moe")
        delegate(task="review", files="src/main.py,src/utils.py", content="Review these files")
        delegate(task="quick", content="...", max_tokens=500)  # Limit response
    """
    # Smart backend selection using backend_manager
    backend_provider, backend_obj = await _select_optimal_backend_v2(content, file, task, backend_type)
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
    )


@mcp.tool()
async def think(
    problem: str,
    context: str = "",
    depth: str = "normal",
) -> str:
    """
    Deep reasoning for complex problems using local GPU with extended thinking.
    Offloads complex analysis to local LLM - zero API costs.

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

    ROUTING:
    - Uses largest available GPU for deep thinking
    - Automatically enables thinking mode for normal/deep
    - Prefers local GPU, falls back to remote if needed

    Returns:
        Structured analysis with step-by-step reasoning and conclusions

    Examples:
        think(problem="How should we handle authentication?", depth="deep")
        think(problem="Debug this error", context="<stack trace>", depth="normal")
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
    Select optimal backend.

    If backend_type is specified ("local" or "remote"), tries to find a matching backend.
    Otherwise uses the active backend.
    """
    if backend_type:
        # Try to find a backend of the requested type
        for b in backend_manager.get_enabled_backends():
            if b.type == backend_type:
                return (None, b)

    # Default to active backend
    active_backend = backend_manager.get_active_backend()
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
) -> str:
    """
    Execute multiple tasks in PARALLEL across all available GPUs for maximum throughput.
    Distributes work across local and remote backends intelligently.

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

    ROUTING LOGIC:
    - Distributes tasks across ALL available GPUs (local + remote)
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
        - backends: Array of configured backend status
        - usage: Token counts and call statistics per tier
        - cost_savings: Estimated savings vs cloud API
        - routing: Current routing configuration
    """
    # Get health status from BackendManager (only checks enabled backends)
    health_status = await backend_manager.get_health_status()

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
# STRUCTURED TOOLS (LLM-to-LLM optimized JSON interfaces)
# ============================================================

# Import structured tools module - this registers additional MCP tools
# with typed JSON input/output for programmatic use by AI assistants.
# The import must happen after mcp and all helper functions are defined.
from . import structured_tools  # noqa: F401


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
    """
    if TRACKING_ENABLED:
        await tracker.start_background_save()


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
