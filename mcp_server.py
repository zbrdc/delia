#!/usr/bin/env python3
"""
Delia — Local LLM Delegation Server

A pure MCP server that intelligently routes tasks to optimal local models.
Automatically selects quick/coder/moe tiers based on task type and content.

Usage:
    uv run mcp_server.py                    # stdio transport (default)
    uv run mcp_server.py --transport sse    # SSE transport on port 8200
"""
import re
import time
import httpx
import asyncio
import logging
import json
import threading
import uuid
import contextvars
from collections import deque
from datetime import datetime
from typing import Optional, Any, cast
from pathlib import Path

import structlog
from structlog.types import Processor

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
                def msg(self, *args, **kwargs): pass
                def __getattr__(self, name): return self.msg
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
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken

# Import configuration (all tunable values in config.py)
from config import (
    config, STATS_FILE,
    detect_model_tier, get_backend_health, BackendHealth
)

# Import unified backend manager (single source of truth from settings.json)
from backend_manager import backend_manager, BackendConfig

# Import prompt templating system
from prompt_templates import create_structured_prompt

# Import watermelon-themed messages for fun logging
from melon_messages import (
    GardenEvent, get_message, get_display_event, get_vine_message,
    format_harvest_stats, get_backend_status_message, get_startup_message
)

# Conditional authentication imports (based on config.auth_enabled)
AUTH_ENABLED = config.auth_enabled
TRACKING_ENABLED = config.tracking_enabled

if AUTH_ENABLED:
    from auth import (
        create_db_and_tables, get_async_session, get_user_manager, get_user_db,
        get_async_session_context, get_user_db_context, get_user_manager_context,
        fastapi_users, auth_backend, current_active_user, current_user_optional,
        User, UserRead, UserCreate, UserUpdate, JWT_SECRET
    )
    import jose.jwt

    def decode_jwt_token(token: str) -> Optional[dict]:
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
            return jose.jwt.decode(
                token,
                JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False}
            )
        except jose.JWTError:
            return None
        except Exception:
            return None
else:
    # Stubs for when auth is disabled
    async def create_db_and_tables(): pass
    JWT_SECRET = None

    def decode_jwt_token(token: str) -> Optional[dict]:
        return None

from multi_user_tracking import tracker



LIVE_LOGS_FILE = Path.home() / ".cache" / "delia" / "live_logs.json"
MAX_LIVE_LOGS = 100
LIVE_LOGS: list[dict] = []
_live_logs_lock = threading.Lock()

current_client_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('current_client_id', default=None)
current_username: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('current_username', default=None)


def _dashboard_processor(logger: Any, method_name: str, event_dict: dict) -> dict:
    """
    Custom structlog processor that captures logs for dashboard streaming.

    Extracts dashboard-relevant fields and writes to live logs buffer.
    Only captures logs with explicit 'log_type' for dashboard display.
    Includes garden-themed messages for a fun watermelon experience!
    """
    # Only process logs explicitly marked for dashboard
    log_type = event_dict.pop("log_type", None)
    if log_type:
        model = event_dict.pop("model", "")
        tokens = event_dict.pop("tokens", 0)
        message = event_dict.get("event", "")
        garden_msg = event_dict.pop("garden_msg", "")  # Extract garden-themed message
        backend = event_dict.pop("backend", "")

        with _live_logs_lock:
            LIVE_LOGS.append({
                "ts": datetime.now().isoformat(),
                "type": log_type,
                "message": message,
                "model": model,
                "tokens": tokens,
                "garden_msg": garden_msg,  # Include themed message for dashboard
                "backend": backend,
            })
            if len(LIVE_LOGS) > MAX_LIVE_LOGS:
                LIVE_LOGS.pop(0)

        # Schedule async save (non-blocking)
        try:
            asyncio.get_running_loop().create_task(_save_live_logs_async())
        except RuntimeError:
            # No running loop, save synchronously
            _save_live_logs_sync()

    return event_dict


def _save_live_logs_sync():
    """Save live logs to disk synchronously (fallback)."""
    try:
        temp_file = LIVE_LOGS_FILE.with_suffix('.tmp')
        with _live_logs_lock:
            temp_file.write_text(json.dumps(LIVE_LOGS[-MAX_LIVE_LOGS:], indent=2))
        temp_file.replace(LIVE_LOGS_FILE)
    except Exception as e:
        # Log failures at debug level - logs are non-critical but we want visibility
        structlog.get_logger().debug("live_logs_save_failed", error=str(e))


async def _save_live_logs_async():
    """Save live logs to disk asynchronously using aiofiles."""
    try:
        temp_file = LIVE_LOGS_FILE.with_suffix('.tmp')
        with _live_logs_lock:
            content = json.dumps(LIVE_LOGS[-MAX_LIVE_LOGS:], indent=2)
        async with aiofiles.open(temp_file, 'w') as f:
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
    import sys

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


# ============================================================
# PYDANTIC MODELS FOR API RESPONSES
# ============================================================

class OllamaResponse(BaseModel):
    """Ollama /api/generate response model."""
    response: str = ""
    eval_count: int = 0
    done: bool = True


class LlamaCppMessage(BaseModel):
    """OpenAI-compatible message."""
    role: str = "assistant"
    content: str = ""


class LlamaCppChoice(BaseModel):
    """OpenAI-compatible choice."""
    message: LlamaCppMessage
    index: int = 0
    finish_reason: str = "stop"


class LlamaCppUsage(BaseModel):
    """OpenAI-compatible usage stats."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LlamaCppResponse(BaseModel):
    """llama.cpp /v1/chat/completions response model."""
    choices: list[LlamaCppChoice]
    usage: Optional[LlamaCppUsage] = None
    model: str = ""
    id: str = ""


class LlamaCppError(BaseModel):
    """llama.cpp error response."""
    type: str = ""
    message: str = ""
    n_prompt_tokens: Optional[int] = None
    n_ctx: Optional[int] = None


# ============================================================
# TOKEN ESTIMATION (tiktoken)
# ============================================================

# Lazy-load tiktoken encoder (first call may download encoding)
_tiktoken_encoder: Optional[tiktoken.Encoding] = None
_tiktoken_failed: bool = False  # Track if loading failed to avoid repeated attempts


def get_tiktoken_encoder() -> Optional[tiktoken.Encoding]:
    """
    Get or initialize tiktoken encoder (cl100k_base works for most models).

    Returns cached encoder, or None if loading failed.
    """
    global _tiktoken_encoder, _tiktoken_failed

    # Don't retry if we already failed
    if _tiktoken_failed:
        return None

    if _tiktoken_encoder is None:
        try:
            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            _tiktoken_failed = True
            log.warning("tiktoken_load_failed", error=str(e), fallback="estimate")
            return None

    return _tiktoken_encoder


def count_tokens(text: str) -> int:
    """
    Count tokens accurately using tiktoken, with fallback to estimation.

    Uses tiktoken's cl100k_base encoding (compatible with GPT-4, Claude, etc.)
    Falls back to ~4 chars per token estimate if tiktoken unavailable.

    Args:
        text: The text to count tokens for

    Returns:
        Token count (accurate if tiktoken available, else estimated)
    """
    if not text:
        return 0

    encoder = get_tiktoken_encoder()
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # Fallback: ~4 chars per token (rough estimate for modern models)
    return len(text) // 4


def estimate_tokens(text: str) -> int:
    """
    Quick token estimation without tiktoken.

    Use this for non-critical estimates where speed matters more than accuracy.
    """
    if not text:
        return 0
    return len(text) // 4

# ============================================================
# INPUT VALIDATION
# ============================================================

VALID_TASKS = frozenset({"review", "analyze", "generate", "summarize", "critique", "quick", "plan", "think"})
VALID_MODELS = frozenset({"quick", "coder", "moe", "thinking"})
VALID_BACKENDS = frozenset({"ollama", "llamacpp"})
MAX_CONTENT_LENGTH = 500_000  # 500KB max content
MAX_FILE_PATH_LENGTH = 1000


def validate_task(task: str) -> tuple[bool, str]:
    """Validate task type. Returns (is_valid, error_message)."""
    if not task:
        return False, "Task type is required"
    if task not in VALID_TASKS:
        return False, f"Invalid task type: '{task}'. Valid types: {', '.join(sorted(VALID_TASKS))}"
    return True, ""


def validate_content(content: str) -> tuple[bool, str]:
    """Validate content byte length. Returns (is_valid, error_message)."""
    if not content:
        return False, "Content is required"
    # Use byte length (UTF-8) not character count to enforce accurate size limit
    byte_length = len(content.encode("utf-8"))
    if byte_length > MAX_CONTENT_LENGTH:
        return False, f"Content too large: {byte_length} bytes (max: {MAX_CONTENT_LENGTH})"
    return True, ""


def validate_file_path(file_path: Optional[str]) -> tuple[bool, str]:
    """Validate file path if provided. Returns (is_valid, error_message)."""
    if file_path is None:
        return True, ""  # Optional field (None is allowed)
    if file_path == "":
        return False, "File path cannot be empty string"
    if len(file_path) > MAX_FILE_PATH_LENGTH:
        return False, f"File path too long: {len(file_path)} chars (max: {MAX_FILE_PATH_LENGTH})"
    # Security: Reject path traversal attempts
    if ".." in file_path:
        return False, "File path cannot contain '..' (path traversal not allowed)"
    # Note: ~ is allowed and will be resolved safely by Path.expanduser() in read_file_safe
    return True, ""


def validate_model_hint(model: Optional[str]) -> tuple[bool, str]:
    """Validate model hint if provided. Returns (is_valid, error_message)."""
    if not model:
        return True, ""  # Optional field
    if model not in VALID_MODELS:
        return False, f"Invalid model hint: '{model}'. Valid models: {', '.join(sorted(VALID_MODELS))}"
    return True, ""


# ============================================================
# MODEL QUEUE SYSTEM
# Intelligent GPU memory management and request queuing
# ============================================================

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import heapq

@dataclass(order=True)
class QueuedRequest:
    """Represents a queued LLM request with priority."""
    priority: int  # Lower number = higher priority
    timestamp: datetime
    request_id: str
    model_name: str
    task_type: str
    content_length: int
    future: asyncio.Future = field(compare=False)

    def __post_init__(self):
        # Tie-breaker: earlier requests get priority
        self.timestamp = self.timestamp or datetime.now()

class ModelQueue:
    """
    Intelligent queue system for GPU memory management.

    Prevents concurrent model loading and manages GPU memory efficiently.
    Queues requests when models need loading, prioritizes by model size and urgency.
    """

    def __init__(self):
        self.loaded_models: Dict[str, Dict[str, Any]] = {}  # model_name -> metadata
        self.loading_models: set[str] = set()  # Models currently being loaded
        self.request_queues: Dict[str, List[QueuedRequest]] = {}  # model_name -> priority queue
        self.lock = asyncio.Lock()
        self.request_counter = 0

        # Model size estimates (in GB VRAM, rough estimates)
        self.model_sizes = {
            "Qwen3-4B": 4.0,
            "Qwen3-14B": 14.0,
            "Qwen3-30B": 30.0,
            "Qwen3-4B-Q4_K_M": 2.5,  # Quantized versions
            "Qwen3-14B-Q4_K_M": 8.0,
            "Qwen3-30B-Q4_K_M": 16.0,
        }

        # GPU memory limit (assume 24GB for RTX 3090/4090, adjust as needed)
        self.gpu_memory_limit_gb = 24.0
        self.memory_buffer_gb = 2.0  # Keep 2GB free

        # Queue health metrics
        self.total_queued = 0
        self.total_processed = 0
        self.max_queue_depth = 0
        self.queue_timeouts = 0

    def get_model_size(self, model_name: str) -> float:
        """Get estimated VRAM usage for a model."""
        # Try exact match first
        if model_name in self.model_sizes:
            return self.model_sizes[model_name]

        # Try partial matches
        for key, size in self.model_sizes.items():
            if key.lower() in model_name.lower():
                return size

        # Default estimate based on model name patterns
        if "30b" in model_name.lower():
            return 16.0  # Quantized 30B
        elif "14b" in model_name.lower():
            return 8.0   # Quantized 14B
        else:
            return 4.0   # Default 4B model

    def get_available_memory(self) -> float:
        """Calculate available GPU memory."""
        used_memory = sum(
            self.get_model_size(model)
            for model in self.loaded_models.keys()
        )
        return max(0, self.gpu_memory_limit_gb - used_memory - self.memory_buffer_gb)

    def can_load_model(self, model_name: str) -> bool:
        """Check if a model can be loaded given current memory."""
        model_size = self.get_model_size(model_name)
        return self.get_available_memory() >= model_size

    def calculate_priority(self, task_type: str, content_length: int, model_name: str) -> int:
        """Calculate request priority (lower = higher priority)."""
        priority = 0

        # Task urgency (thinking tasks are most urgent)
        if task_type in ("think", "thinking"):
            priority -= 100
        elif task_type in ("plan", "analyze"):
            priority -= 50
        elif task_type in ("review", "critique"):
            priority -= 25

        # Content size (smaller = higher priority to avoid timeouts)
        if content_length < 1000:
            priority -= 20
        elif content_length > 50000:
            priority += 20  # Large content can wait

        # Model size (smaller models = higher priority)
        model_size = self.get_model_size(model_name)
        if model_size <= 4.0:
            priority -= 10
        elif model_size > 16.0:
            priority += 10

        return priority

    async def acquire_model(self, model_name: str, task_type: str = "unknown",
                          content_length: int = 0) -> tuple[bool, Optional[asyncio.Future]]:
        """
        Acquire a model for use.

        Returns:
            - (True, None) if model is immediately available
            - (False, Future) if request is queued (caller must await the Future)

        The Future will be resolved when the model finishes loading and the request is processed.
        """
        async with self.lock:
            # Model already loaded and not loading
            if model_name in self.loaded_models and model_name not in self.loading_models:
                self.loaded_models[model_name]["last_used"] = datetime.now()
                return (True, None)

            # Model is currently loading
            if model_name in self.loading_models:
                # Queue the request
                request_id = f"req_{self.request_counter}"
                self.request_counter += 1

                priority = self.calculate_priority(task_type, content_length, model_name)
                request_future = asyncio.Future()
                queued_request = QueuedRequest(
                    priority=priority,
                    timestamp=datetime.now(),
                    request_id=request_id,
                    model_name=model_name,
                    task_type=task_type,
                    content_length=content_length,
                    future=request_future
                )

                if model_name not in self.request_queues:
                    self.request_queues[model_name] = []

                heapq.heappush(self.request_queues[model_name], queued_request)

                # Track queue stats
                queue_length = len(self.request_queues[model_name])
                self.total_queued += 1
                self.max_queue_depth = max(self.max_queue_depth, queue_length)

                log.info(get_display_event("model_queued"),
                        model=model_name,
                        queue_length=queue_length,
                        priority=priority,
                        task_type=task_type,
                        request_id=request_id,
                        garden_msg=get_message(GardenEvent.SEED_PLANTED),
                        log_type="QUEUE")

                return (False, request_future)

            # Model not loaded - check if we can load it
            if not self.can_load_model(model_name):
                # Need to unload some models first
                await self._make_room_for_model(model_name)

            # Start loading the model
            self.loading_models.add(model_name)

            log.info(get_display_event("model_loading_start"),
                    model=model_name,
                    available_memory_gb=round(self.get_available_memory(), 1),
                    garden_msg=get_message(GardenEvent.WATERING))

            return (True, None)

    async def _make_room_for_model(self, new_model_name: str) -> None:
        """Unload least recently used models to make room for new model."""
        new_model_size = self.get_model_size(new_model_name)
        available_memory = self.get_available_memory()

        if available_memory >= new_model_size:
            return  # No need to unload

        # Sort loaded models by last used time (oldest first)
        unload_candidates = sorted(
            self.loaded_models.items(),
            key=lambda x: x[1]["last_used"]
        )

        freed_memory = 0
        to_unload = []

        for model_name, metadata in unload_candidates:
            if freed_memory >= (new_model_size - available_memory):
                break
            freed_memory += self.get_model_size(model_name)
            to_unload.append(model_name)

        # Unload the selected models
        for model_name in to_unload:
            del self.loaded_models[model_name]
            log.info(get_display_event("model_unloaded"),
                    model=model_name,
                    reason="memory_pressure",
                    garden_msg=get_message(GardenEvent.COMPOSTING))

    async def release_model(self, model_name: str, success: bool = True) -> None:
        """Release a model after use and process queued requests."""
        async with self.lock:
            if model_name in self.loading_models:
                self.loading_models.remove(model_name)

                if success:
                    # Mark model as loaded
                    self.loaded_models[model_name] = {
                        "loaded_at": datetime.now(),
                        "last_used": datetime.now(),
                        "size_gb": self.get_model_size(model_name)
                    }

                    log.info(get_display_event("model_loaded"),
                            model=model_name,
                            loaded_count=len(self.loaded_models),
                            available_memory_gb=round(self.get_available_memory(), 1),
                            garden_msg=get_message(GardenEvent.GARDEN_READY))

                    # Process queued requests for this model
                    await self._process_queue(model_name)
                else:
                    # Loading failed - fail all queued requests
                    if model_name in self.request_queues:
                        for queued_request in self.request_queues[model_name]:
                            if not queued_request.future.done():
                                queued_request.future.set_exception(
                                    Exception(f"Failed to load model {model_name}")
                                )
                        del self.request_queues[model_name]

    async def _process_queue(self, model_name: str) -> None:
        """Process queued requests for a newly loaded model."""
        if model_name not in self.request_queues:
            return

        queue = self.request_queues[model_name]
        processed = 0

        while queue and processed < 5:  # Process up to 5 requests at once
            queued_request = heapq.heappop(queue)

            if not queued_request.future.done():
                # Wake up the waiting request
                queued_request.future.set_result(True)
                processed += 1
                self.total_processed += 1

                wait_time_ms = (datetime.now() - queued_request.timestamp).total_seconds() * 1000
                log.info(get_display_event("queue_processed"),
                        model=model_name,
                        request_id=queued_request.request_id,
                        wait_time_ms=round(wait_time_ms, 1),
                        remaining=len(queue),
                        garden_msg="Seedling ready for the big garden!",
                        log_type="QUEUE")

        if not queue:
            del self.request_queues[model_name]

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status for monitoring."""
        current_queue_depth = sum(len(q) for q in self.request_queues.values())
        return {
            "loaded_models": list(self.loaded_models.keys()),
            "loading_models": list(self.loading_models),
            "queued_requests": {
                model: len(queue) for model, queue in self.request_queues.items()
            },
            "available_memory_gb": round(self.get_available_memory(), 1),
            "total_loaded_gb": round(sum(m["size_gb"] for m in self.loaded_models.values()), 1),
            # Queue health metrics
            "queue_stats": {
                "total_queued": self.total_queued,
                "total_processed": self.total_processed,
                "current_queue_depth": current_queue_depth,
                "max_queue_depth": self.max_queue_depth,
                "queue_timeouts": self.queue_timeouts,
            }
        }

# Global model queue instance
model_queue = ModelQueue()


# ============================================================
# USAGE TRACKING
# ============================================================

# Usage tracking for efficiency reporting (loaded from disk on startup)
MODEL_USAGE = {
    "quick": {"calls": 0, "tokens": 0},
    "coder": {"calls": 0, "tokens": 0},
    "moe": {"calls": 0, "tokens": 0},
    "thinking": {"calls": 0, "tokens": 0},
}

# Enhanced tracking: task types and recent calls
TASK_STATS = {
    "review": 0,
    "analyze": 0,
    "generate": 0,
    "summarize": 0,
    "critique": 0,
    "quick": 0,
    "plan": 0,
    "think": 0,
    "other": 0,
}

# Recent calls log (last 50 calls) - using deque for O(1) append/pop
MAX_RECENT_CALLS = 50
RECENT_CALLS: deque[dict] = deque(maxlen=MAX_RECENT_CALLS)

# Response time tracking
RESPONSE_TIMES = {
    "quick": [],  # List of (timestamp, ms) tuples
    "coder": [],
    "moe": [],
    "thinking": [],
}
MAX_RESPONSE_TIMES = 100  # Keep last 100 per model

# Stats file for enhanced data
ENHANCED_STATS_FILE = Path.home() / ".cache" / "delia" / "enhanced_stats.json"

# Circuit breaker stats file (for dashboard)
CIRCUIT_BREAKER_FILE = Path.home() / ".cache" / "delia" / "circuit_breaker.json"

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


def load_usage_stats():
    """Load usage stats from disk."""
    global MODEL_USAGE, TASK_STATS, RECENT_CALLS, RESPONSE_TIMES

    # Load basic stats
    if STATS_FILE.exists():
        try:
            data = json.loads(STATS_FILE.read_text())
            # Migration: Support legacy keys from pre-refactor versions
            # These map old tier names to current tier names (one-time migration)
            legacy_key_mapping = {
                "quick": ["quick", "14b"],  # Legacy: "14b" → "quick"
                "coder": ["coder", "30b"],  # Legacy: "30b" → "coder" (now moe)
                "moe": ["moe"],
            }
            for new_key, legacy_keys in legacy_key_mapping.items():
                for legacy_key in legacy_keys:
                    if legacy_key in data:
                        MODEL_USAGE[new_key]["calls"] += data[legacy_key].get("calls", 0)
                        MODEL_USAGE[new_key]["tokens"] += data[legacy_key].get("tokens", 0)
            log.info(
                "stats_loaded",
                quick_calls=MODEL_USAGE['quick']['calls'],
                coder_calls=MODEL_USAGE['coder']['calls'],
                moe_calls=MODEL_USAGE['moe']['calls'],
            )
        except json.JSONDecodeError as e:
            log.warning("stats_load_failed", error=str(e), reason="invalid_json")
        except Exception as e:
            log.warning("stats_load_failed", error=str(e))

    # Load enhanced stats
    if ENHANCED_STATS_FILE.exists():
        try:
            data = json.loads(ENHANCED_STATS_FILE.read_text())
            TASK_STATS.update(data.get("task_stats", {}))
            RECENT_CALLS.clear()
            RECENT_CALLS.extend(data.get("recent_calls", [])[-MAX_RECENT_CALLS:])
            rt = data.get("response_times", {})
            # Migration: Map legacy tier names to current names
            RESPONSE_TIMES["quick"] = rt.get("quick", rt.get("14b", []))[-MAX_RESPONSE_TIMES:]
            RESPONSE_TIMES["coder"] = rt.get("coder", rt.get("30b", []))[-MAX_RESPONSE_TIMES:]
            RESPONSE_TIMES["moe"] = rt.get("moe", [])[-MAX_RESPONSE_TIMES:]
            log.info("enhanced_stats_loaded", recent_calls=len(RECENT_CALLS))
        except json.JSONDecodeError as e:
            log.warning("enhanced_stats_load_failed", error=str(e), reason="invalid_json")
        except Exception as e:
            log.warning("enhanced_stats_load_failed", error=str(e))


def _snapshot_stats() -> tuple[dict, dict, dict, list]:
    """
    Take atomic snapshot of all in-memory stats under lock.

    Returns:
        (MODEL_USAGE, TASK_STATS, RESPONSE_TIMES, RECENT_CALLS) snapshots

    This prevents race conditions where one thread reads stats while another
    is modifying them. The snapshot is consistent at the moment of capture.
    """
    with _stats_thread_lock:
        # Create deep copies to prevent external modifications
        model_usage_snapshot = {
            tier: data.copy() for tier, data in MODEL_USAGE.items()
        }
        task_stats_snapshot = TASK_STATS.copy()
        response_times_snapshot = {
            tier: times.copy() for tier, times in RESPONSE_TIMES.items()
        }
        recent_calls_snapshot = list(RECENT_CALLS)  # Convert deque to list for snapshot

    return model_usage_snapshot, task_stats_snapshot, response_times_snapshot, recent_calls_snapshot


def save_usage_stats():
    """
    Save usage stats to disk (atomic write).

    Uses snapshot to ensure consistent data even with concurrent updates.
    """
    try:
        model_usage_snapshot, _, _, _ = _snapshot_stats()
        temp_file = STATS_FILE.with_suffix('.tmp')
        # Use compact JSON in production
        temp_file.write_text(json.dumps(model_usage_snapshot, indent=2))
        temp_file.replace(STATS_FILE)  # Atomic on POSIX
    except Exception as e:
        log.warning("stats_save_failed", error=str(e))


def save_enhanced_stats():
    """
    Save enhanced stats to disk (atomic write).

    Uses snapshots to ensure consistent data even with concurrent updates.
    """
    try:
        _, task_stats_snapshot, response_times_snapshot, recent_calls_snapshot = _snapshot_stats()

        data = {
            "task_stats": task_stats_snapshot,
            "recent_calls": recent_calls_snapshot[-MAX_RECENT_CALLS:],
            "response_times": {
                "quick": response_times_snapshot["quick"][-MAX_RESPONSE_TIMES:],
                "coder": response_times_snapshot["coder"][-MAX_RESPONSE_TIMES:],
                "moe": response_times_snapshot["moe"][-MAX_RESPONSE_TIMES:],
            }
        }
        temp_file = ENHANCED_STATS_FILE.with_suffix('.tmp')
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(ENHANCED_STATS_FILE)  # Atomic on POSIX
    except Exception as e:
        log.warning("enhanced_stats_save_failed", error=str(e))


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
            } if active_backend else None,
            "timestamp": datetime.now().isoformat(),
        }
        temp_file = CIRCUIT_BREAKER_FILE.with_suffix('.tmp')
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(CIRCUIT_BREAKER_FILE)  # Atomic on POSIX
    except Exception as e:
        log.warning("circuit_breaker_save_failed", error=str(e))


# Threading lock for in-memory stats updates (protects both reading and writing)
import threading
_stats_thread_lock = threading.Lock()

# Async lock for file I/O operations (prevents concurrent writes to same file)
_stats_lock = asyncio.Lock()


def _update_stats_sync(
    model_tier: str,
    task_type: str,
    original_task: str,
    tokens: int,
    elapsed_ms: int,
    content_preview: str,
    enable_thinking: bool,
    backend: str = "ollama"
) -> None:
    """
    Thread-safe update of all in-memory stats.

    This function is called from sync context within async functions after each model call.
    It updates in-memory tracking structures under a single threading lock to ensure
    atomicity of all updates.

    The threading lock protects:
    - MODEL_USAGE dictionary
    - TASK_STATS dictionary
    - RESPONSE_TIMES lists
    - RECENT_CALLS list

    This lock works in coordination with:
    - _snapshot_stats(): Creates consistent snapshots for saving
    - save_all_stats_async(): Ensures only one save writes to disk at a time

    Args:
        model_tier: Model size tier (quick, coder, moe, thinking)
        task_type: Type of task (general, thinking, review, etc.)
        original_task: Original task description for logging
        tokens: Number of tokens processed
        elapsed_ms: Processing time in milliseconds
        content_preview: Preview of request content
        enable_thinking: Whether thinking mode was enabled
        backend: Backend used (ollama, llamacpp, gemini, etc.)
    """
    # Determine backend type from config
    backend_type = config.get_backend_type(backend)

    with _stats_thread_lock:
        # Track model usage
        MODEL_USAGE[model_tier]["calls"] += 1
        MODEL_USAGE[model_tier]["tokens"] += tokens

        # Track task type
        task_key = task_type if task_type in TASK_STATS else "other"
        TASK_STATS[task_key] += 1

        # Track response time
        timestamp = datetime.now().isoformat()
        RESPONSE_TIMES[model_tier].append({"ts": timestamp, "ms": elapsed_ms})
        if len(RESPONSE_TIMES[model_tier]) > MAX_RESPONSE_TIMES:
            RESPONSE_TIMES[model_tier] = RESPONSE_TIMES[model_tier][-MAX_RESPONSE_TIMES:]

        # Add to recent calls log
        call_entry = {
            "timestamp": timestamp,
            "model": f"{model_tier} ({backend})" if backend != config.get_local_backend() else model_tier,
            "task_type": original_task,
            "tokens": tokens,
            "elapsed_ms": elapsed_ms,
            "preview": content_preview[:100] + "..." if len(content_preview) > 100 else content_preview,
            "thinking": enable_thinking,
            "backend_type": backend_type,  # "local" or "remote"
        }
        if backend != config.get_local_backend():
            call_entry["backend"] = backend
        RECENT_CALLS.append(call_entry)  # deque auto-evicts oldest when maxlen reached


async def save_all_stats_async():
    """
    Save all stats asynchronously with proper locking.

    Uses two-level locking strategy:
    1. Threading lock in _snapshot_stats() to atomically read in-memory stats
    2. Async lock here to prevent concurrent writes to disk

    This ensures:
    - Each save gets a consistent snapshot of stats
    - Only one save operation writes to disk at a time
    - Updates during save don't cause data loss
    """
    async with _stats_lock:
        # Each save function calls _snapshot_stats() internally
        # to ensure consistent reads
        await asyncio.to_thread(save_usage_stats)
        await asyncio.to_thread(save_enhanced_stats)
        await asyncio.to_thread(_save_live_logs_sync)
        await asyncio.to_thread(save_circuit_breaker_stats)


# Ensure cache directory exists
cache_dir = Path.home() / ".cache" / "delia"
cache_dir.mkdir(parents=True, exist_ok=True)

# Load stats immediately at module import time
load_usage_stats()
load_live_logs()

# ============================================================
# BACKEND CLIENT MANAGEMENT
# HTTP clients are now managed by BackendManager (see backend_manager.py)
# The BackendManager reads from settings.json - the single source of truth
# ============================================================

def get_active_backend() -> Optional[BackendConfig]:
    """Get the currently active backend configuration."""
    return backend_manager.get_active_backend()

def get_active_backend_id() -> str:
    """Get the ID of the currently active backend."""
    backend = backend_manager.get_active_backend()
    return backend.id if backend else "none"

def set_active_backend(backend_id: str) -> bool:
    """Set the active backend by ID."""
    return backend_manager.set_active_backend(backend_id)

def get_backend_client(backend_id: Optional[str] = None) -> Optional[httpx.AsyncClient]:
    """Get HTTP client for a specific backend (or active backend if not specified)."""
    if backend_id:
        backend = backend_manager.get_backend(backend_id)
    else:
        backend = backend_manager.get_active_backend()

    return backend.get_client() if backend else None

# ============================================================
# LANGUAGE DETECTION AND SYSTEM PROMPTS
# ============================================================

LANGUAGE_CONFIGS = {
    "pytorch": {
        "extensions": [".py"],
        "keywords": ["torch", "nn.Module", "cuda", "tensor", "backward()", "optimizer"],
        "system_prompt": """Role: Expert PyTorch ML engineer
Style: Efficient GPU code, proper tensor ops
Patterns: Training loops, model architecture
Output: Optimized, trainable models""",
    },
    "sklearn": {
        "extensions": [".py"],
        "keywords": ["sklearn", "fit(", "predict(", "Pipeline", "cross_val", "train_test_split"],
        "system_prompt": """Role: Expert ML engineer (scikit-learn)
Style: Clean pipelines, proper preprocessing
Patterns: Cross-validation, hyperparameter tuning
Output: Validated, reproducible ML code""",
    },
    "react": {
        "extensions": [".jsx", ".tsx"],
        "keywords": ["useState", "useEffect", "import React", "export default", "<div", "className="],
        "system_prompt": """Role: Expert React developer
Style: Functional components, hooks, TypeScript
Patterns: Custom hooks, proper state management
Output: Type-safe, performant components""",
    },
    "react-native": {
        "extensions": [".jsx", ".tsx"],
        "keywords": ["react-native", "StyleSheet", "View", "Text", "TouchableOpacity", "Animated"],
        "system_prompt": """Role: Expert React Native developer
Style: Mobile-first, platform-aware
Patterns: Performance optimization, proper styling
Output: Cross-platform compatible code""",
    },
    "nextjs": {
        "extensions": [".jsx", ".tsx", ".js", ".ts"],
        "keywords": ["next/", "getServerSideProps", "getStaticProps", "useRouter", "app/page", "layout.tsx"],
        "system_prompt": """Role: Expert Next.js developer
Style: App Router, Server Components
Patterns: RSC, data fetching, caching
Output: Optimized full-stack code""",
    },
    "rust": {
        "extensions": [".rs"],
        "keywords": ["fn ", "impl ", "use std::", "println!", "let mut", "struct ", "enum "],
        "system_prompt": """Role: Expert Rust developer
Style: Memory-safe, zero-cost abstractions
Patterns: Ownership, borrowing, lifetimes
Output: Safe, performant systems code""",
    },
    "go": {
        "extensions": [".go"],
        "keywords": ["func ", "package ", "import ", "go ", "defer ", "goroutine", "chan "],
        "system_prompt": """Role: Expert Go developer
Style: Simple, concurrent, efficient
Patterns: Goroutines, channels, interfaces
Output: Scalable, concurrent systems""",
    },
    "java": {
        "extensions": [".java"],
        "keywords": ["public class", "import java", "System.out", "public static", "ArrayList", "HashMap"],
        "system_prompt": """Role: Expert Java developer
Style: OOP, JVM ecosystem
Patterns: Design patterns, collections, concurrency
Output: Robust, enterprise-grade applications""",
    },
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
        "keywords": ["#include", "std::", "class ", "template", "virtual ", "override", "auto "],
        "system_prompt": """Role: Expert C++ developer
Style: Modern C++17/20, RAII, templates
Patterns: STL, smart pointers, exceptions
Output: High-performance, memory-efficient code""",
    },
    "csharp": {
        "extensions": [".cs"],
        "keywords": ["using System", "public class", "Console.Write", "async ", "Task<", "IEnumerable"],
        "system_prompt": """Role: Expert C# developer
Style: .NET, LINQ, async/await
Patterns: Dependency injection, SOLID principles
Output: Maintainable, scalable applications""",
    },
    "nodejs": {
        "extensions": [".js", ".ts", ".mjs"],
        "keywords": ["require(", "module.exports", "express", "async/await", "Buffer", "process."],
        "system_prompt": """Role: Expert Node.js developer
Style: Async/await, proper error handling
Patterns: Scalable architecture, streams
Output: Production-ready backend code""",
    },
    "python": {
        "extensions": [".py", ".pyx", ".pyi"],
        "keywords": ["def ", "import ", "class ", "async def", "from ", "if __name__"],
        "system_prompt": """Role: Expert Python developer
Style: PEP8, type hints, docstrings
Version: Python 3.10+
Output: Clean, production-ready code""",
    },
}

# Pygments lexer name -> our language key mapping
PYGMENTS_LANGUAGE_MAP = {
    "python": "python",
    "python 3": "python",
    "python3": "python",
    "javascript": "nodejs",
    "typescript": "nodejs",
    "jsx": "react",
    "tsx": "react",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "c": "c",
    "c++": "cpp",
    "c#": "csharp",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "sql": "sql",
    "bash": "bash",
    "shell": "bash",
    "yaml": "yaml",
    "json": "json",
    "html": "html",
    "css": "css",
}


def detect_language(content: str, file_path: str = "") -> str:
    """
    Detect programming language/framework from content and file extension.

    Priority:
    1. Framework keyword detection (React, Next.js, PyTorch, etc. - ≥2 keyword matches)
    2. Pygments get_lexer_for_filename() when file_path is available
    3. Simple keyword fallback for content-only detection
    """
    content_lower = content.lower()

    # Priority 1: Detect frameworks by keyword density (most specific first)
    # This catches React, Next.js, PyTorch, etc. which need content analysis
    for lang, lang_config in LANGUAGE_CONFIGS.items():
        matches = sum(1 for kw in lang_config["keywords"] if kw.lower() in content_lower)
        if matches >= 2:
            log.debug("lang_keyword_detected", language=lang, matches=matches)
            return lang

    # Priority 2: Use Pygments for reliable extension-based detection
    if file_path:
        ext = Path(file_path).suffix
        if ext:
            try:
                # Only pass extension to Pygments (privacy: don't expose full paths)
                lexer = get_lexer_for_filename(f"file{ext}", content)
                pygments_name = lexer.name.lower()
                # Map Pygments name to our language keys
                if pygments_name in PYGMENTS_LANGUAGE_MAP:
                    detected = PYGMENTS_LANGUAGE_MAP[pygments_name]
                    log.debug("lang_pygments_extension", pygments_name=pygments_name, detected=detected)
                    return detected
                # Direct name matching for common languages
                for our_lang in ["python", "go", "rust", "java", "cpp", "csharp", "ruby", "php", "swift", "kotlin", "scala"]:
                    if our_lang in pygments_name:
                        log.debug("lang_pygments_name", pygments_name=pygments_name, our_lang=our_lang)
                        return our_lang
                # JavaScript/TypeScript family
                if "script" in pygments_name or "type" in pygments_name:
                    log.debug("lang_pygments_script", pygments_name=pygments_name, detected="nodejs")
                    return "nodejs"
            except ClassNotFound:
                pass  # Extension not recognized - fall through to keyword fallback
            except Exception as e:
                log.debug("lang_pygments_error", error=str(e))

    # Priority 3: Simple keyword fallback for content-only detection
    if "fn " in content or "impl " in content or "use std::" in content:
        return "rust"
    if "func " in content or "package " in content:
        return "go"
    if "public class" in content or "import java" in content:
        return "java"
    if "#include" in content or "std::" in content:
        return "cpp"
    if "using System" in content or "Console.Write" in content:
        return "csharp"
    if "require(" in content or "module.exports" in content:
        return "nodejs"
    if "def " in content or "import " in content or "class " in content:
        return "python"
    if "function " in content or "const " in content or "let " in content:
        return "nodejs"
    if "React" in content or "useState" in content or "jsx" in content_lower:
        return "react"

    return "python"  # Default fallback

def get_system_prompt(language: str, task_type: str) -> str:
    """Get structured system prompt for LLM consumption."""
    base = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["python"])["system_prompt"]

    task_instructions = {
        "review": """
Task: CODE REVIEW
Focus: Bugs, security, performance, maintainability
Format:
1. Critical issues (if any)
2. Improvements
3. Positive aspects""",
        "generate": """
Task: CODE GENERATION
Requirements: Complete, working, error-handled
Format: Code with minimal explanation""",
        "analyze": """
Task: TECHNICAL ANALYSIS
Depth: Comprehensive
Format:
1. Summary
2. Key findings
3. Recommendations""",
        "summarize": """
Task: SUMMARIZE
Style: Concise, bullet points
Length: 3-5 key points max""",
        "critique": """
Task: CRITICAL EVALUATION
Style: Constructive, balanced
Format:
1. Strengths
2. Weaknesses
3. Specific improvements""",
        "plan": """
Task: PLANNING/ARCHITECTURE
Style: Actionable steps
Format:
1. Overview
2. Step-by-step plan
3. Considerations/risks""",
        "quick": """
Task: QUICK ANSWER
Style: Direct, concise
Format: Answer first, brief explanation if needed""",
    }

    return base + task_instructions.get(task_type, "")


def optimize_prompt(content: str, task_type: str) -> str:
    """
    Strip natural language triggers and structure prompt for LLM consumption.

    Removes conversational phrases and formats for optimal LLM processing.
    """
    # Remove trigger phrases (case insensitive)
    triggers = [
        r"\s*,?\s*locally\s*$",
        r"\s*,?\s*ask\s+(ollama|locally|local|coder|moe|qwen)\s*$",
        r"\s*,?\s*use\s+(ollama|local|coder|moe|quick)\s*$",
        r"\s*,?\s*on\s+my\s+(gpu|machine|device)\s*$",
        r"\s*,?\s*(privately|offline)\s*$",
        r"\s*,?\s*without\s+(api|cloud)\s*$",
        r"\s*,?\s*via\s+ollama\s*$",
        r"\s*,?\s*no\s+cloud\s*$",
        r"^\s*(please|can you|could you|i want you to|i need you to)\s+",
        r"^\s*(hey|hi|hello),?\s*",
    ]

    cleaned = content
    for pattern in triggers:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip()

    # For certain tasks, add structure if not already present
    if task_type == "quick" and "?" not in cleaned:
        # Ensure questions have question marks
        if cleaned and not cleaned.endswith("."):
            cleaned = cleaned.rstrip(".,!;") + "?"

    return cleaned


def create_enhanced_prompt(
    task_type: str,
    content: str,
    file: Optional[str] = None,
    language: Optional[str] = None,
    symbols: Optional[list[str]] = None,
    context_files: Optional[list[str]] = None,
    user_instructions: Optional[str] = None
) -> str:
    """
    Create an enhanced, structured prompt using templates and context.

    This replaces the simple optimize_prompt with a comprehensive templating system
    that provides better context formatting and task-specific optimizations.
    """
    # First clean the content using the existing optimization
    cleaned_content = optimize_prompt(content, task_type)

    # Handle file inclusion for context
    file_content = None
    if file:
        file_content, error = read_file_safe(file)
        if error:
            log.warning("file_read_error", file=file, error=error)
            file_content = f"Error reading file: {error}"
        else:
            file_content = f"### File: {file}\n```\n{file_content}\n```"

    # Build context files list
    all_context_files = context_files or []
    if file and file not in all_context_files:
        all_context_files.append(file)

    # Detect language if not provided
    detected_language = language or detect_language(cleaned_content, file or "")

    # Create structured prompt using templates
    structured_prompt = create_structured_prompt(
        task_type=task_type,
        content=cleaned_content,
        language=detected_language,
        symbols=symbols,
        file_path=file,
        context_files=all_context_files if len(all_context_files) > 1 else None,
        user_instructions=user_instructions
    )

    # Append file content if available
    if file_content:
        structured_prompt += f"\n\n{file_content}"

    return structured_prompt


# ============================================================
# CODE DETECTION
# ============================================================

# Code indicators with weights for confidence scoring
# Pre-compiled regex patterns for performance (avoids recompilation on each call)
CODE_INDICATORS = {
    # Strong indicators (weight 3) - almost certainly code
    "strong": [
        re.compile(r'\bdef\s+\w+\s*\(', re.MULTILINE),          # Python function
        re.compile(r'\bclass\s+\w+[\s:(]', re.MULTILINE),       # Class definition
        re.compile(r'\bimport\s+\w+', re.MULTILINE),            # Import statement
        re.compile(r'\bfrom\s+\w+\s+import', re.MULTILINE),     # From import
        re.compile(r'\bfunction\s+\w+\s*\(', re.MULTILINE),     # JS function
        re.compile(r'\bconst\s+\w+\s*=', re.MULTILINE),         # JS const
        re.compile(r'\blet\s+\w+\s*=', re.MULTILINE),           # JS let
        re.compile(r'\bexport\s+(default\s+)?', re.MULTILINE),  # JS export
        re.compile(r'^\s*@\w+', re.MULTILINE),                  # Decorator
        re.compile(r'\basync\s+(def|function)', re.MULTILINE),  # Async
        re.compile(r'\bawait\s+\w+', re.MULTILINE),             # Await
        re.compile(r'\breturn\s+[\w{(\[]', re.MULTILINE),       # Return statement
        re.compile(r'if\s*\(.+\)\s*{', re.MULTILINE),           # C-style if
        re.compile(r'for\s*\(.+\)\s*{', re.MULTILINE),          # C-style for
        re.compile(r'\bwhile\s*\(.+\)', re.MULTILINE),          # While loop
        re.compile(r'\btry\s*[:{]', re.MULTILINE),              # Try block
        re.compile(r'\bcatch\s*\(', re.MULTILINE),              # Catch block
        re.compile(r'\bexcept\s+\w*:', re.MULTILINE),           # Python except
        re.compile(r'=>\s*{', re.MULTILINE),                    # Arrow function
        re.compile(r'\.map\(|\.filter\(|\.reduce\(', re.MULTILINE),  # Array methods
    ],
    # Medium indicators (weight 2) - likely code
    "medium": [
        re.compile(r'\bself\.', re.MULTILINE),                  # Python self
        re.compile(r'\bthis\.', re.MULTILINE),                  # JS this
        re.compile(r'===|!==', re.MULTILINE),                   # Strict equality
        re.compile(r'&&|\|\|', re.MULTILINE),                   # Logical operators
        re.compile(r'\bnull\b|\bundefined\b', re.MULTILINE),    # Null/undefined
        re.compile(r'\bTrue\b|\bFalse\b|\bNone\b', re.MULTILINE),  # Python booleans
        re.compile(r':\s*\w+\s*[,)\]]', re.MULTILINE),          # Type annotations
        re.compile(r'\[\w+\]', re.MULTILINE),                   # Array indexing
        re.compile(r'\{\s*\w+:\s*', re.MULTILINE),              # Object literal
        re.compile(r'console\.|print\(|logger\.', re.MULTILINE),  # Logging
        re.compile(r'\braise\s+\w+', re.MULTILINE),             # Python raise
        re.compile(r'\bthrow\s+new', re.MULTILINE),             # JS throw
        re.compile(r'`[^`]+\$\{', re.MULTILINE),                # Template literal
        re.compile(r'f"[^"]*\{', re.MULTILINE),                 # Python f-string
    ],
    # Weak indicators (weight 1) - could be code
    "weak": [
        re.compile(r';$', re.MULTILINE),                        # Semicolon ending
        re.compile(r'\{|\}', re.MULTILINE),                     # Braces
        re.compile(r'\[|\]', re.MULTILINE),                     # Brackets
        re.compile(r'==|!=', re.MULTILINE),                     # Equality
        re.compile(r'->', re.MULTILINE),                        # Arrow (type hints, etc)
        re.compile(r'\bint\b|\bstr\b|\bbool\b', re.MULTILINE),  # Type names
        re.compile(r'\bvar\b', re.MULTILINE),                   # Var keyword
    ],
}

def detect_code_content(content: str) -> tuple[bool, float, str]:
    """
    Detect if content is primarily code or text.

    Returns:
        (is_code, confidence, reasoning)
        - is_code: True if content appears to be code
        - confidence: 0.0-1.0 score
        - reasoning: Brief explanation
    """
    if not content or len(content.strip()) < 20:
        return False, 0.0, "Content too short"

    lines = content.strip().split('\n')

    # Count code indicators
    strong_matches = 0
    medium_matches = 0
    weak_matches = 0

    for pattern in CODE_INDICATORS["strong"]:
        matches = len(pattern.findall(content))  # Use pre-compiled pattern
        strong_matches += min(matches, 5)  # Cap per pattern

    for pattern in CODE_INDICATORS["medium"]:
        matches = len(pattern.findall(content))  # Use pre-compiled pattern
        medium_matches += min(matches, 5)

    for pattern in CODE_INDICATORS["weak"]:
        matches = len(pattern.findall(content))  # Use pre-compiled pattern
        weak_matches += min(matches, 5)

    # Weighted score
    score = (strong_matches * 3 + medium_matches * 2 + weak_matches * 1)

    # Normalize by content length (per 1000 chars)
    normalized = score / max(1, len(content) / 1000)

    # Additional heuristics
    avg_line_length = sum(len(l) for l in lines) / max(1, len(lines))
    indent_lines = sum(1 for l in lines if l.startswith('  ') or l.startswith('\t'))
    indent_ratio = indent_lines / max(1, len(lines))

    # Adjust score based on structure
    if indent_ratio > 0.3:  # Lots of indentation = code
        normalized *= 1.3
    if avg_line_length < 100:  # Code lines tend to be shorter
        normalized *= 1.1

    # Determine threshold
    if normalized > 3.0:
        return True, min(1.0, normalized / 5), f"Strong code signals (score={normalized:.1f})"
    elif normalized > 1.5:
        return True, normalized / 4, f"Likely code (score={normalized:.1f})"
    elif normalized > 0.8:
        return False, 0.4, f"Mixed content (score={normalized:.1f})"
    else:
        return False, max(0, 0.3 - normalized / 3), f"Primarily text (score={normalized:.1f})"


# ============================================================
# MODEL SELECTION
# ============================================================

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
    from config import config

    # Check against active backend models
    backend = backend_manager.get_active_backend()
    if backend:
        models = backend.models
        if model_name == models.get("quick"):
            return {"vram_gb": config.model_quick.vram_gb, "context_tokens": config.model_quick.context_tokens, "tier": "quick"}
        elif model_name == models.get("coder"):
            return {"vram_gb": config.model_coder.vram_gb, "context_tokens": config.model_coder.context_tokens, "tier": "coder"}
        elif model_name == models.get("moe"):
            return {"vram_gb": config.model_moe.vram_gb, "context_tokens": config.model_moe.context_tokens, "tier": "moe"}
        elif model_name == models.get("thinking"):
            return {"vram_gb": config.model_thinking.vram_gb, "context_tokens": config.model_thinking.context_tokens, "tier": "thinking"}

    # Initialize defaults for estimation
    vram_gb: Any = "Unknown"
    context_tokens: Any = "Unknown"

    # Estimate from model name using regex patterns
    import re

    # Extract parameter count (e.g., "14B", "7b", "72B")
    param_match = re.search(r'(\d+(?:\.\d+)?)\s*[bB](?:illion)?(?![a-zA-Z])', model_name, re.IGNORECASE)
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

    return {
        "vram_gb": vram_gb,
        "context_tokens": context_tokens,
        "tier": "unknown"
    }


def parse_model_override(model_hint: Optional[str], content: str) -> Optional[str]:
    """Parse explicit model request from content or hint.

    Recognizes tier keywords (moe, coder, quick, thinking) and natural language model references.
    Supports size-based hints (7b, 14b, 30b), capability hints (coder, reasoning), and descriptive terms.
    """
    # Check explicit hint first (tier names and natural language)
    if model_hint:
        hint = model_hint.lower().strip()

        # Direct tier names
        if "moe" in hint or "30b" in hint or "large" in hint or "complex" in hint or "reasoning" in hint:
            return "moe"
        if "coder" in hint or "code" in hint or "programming" in hint or "14b" in hint:
            return "coder"
        if "quick" in hint or "7b" in hint or "small" in hint or "fast" in hint:
            return "quick"
        if "thinking" in hint or "think" in hint or "chain" in hint:
            return "thinking"
        
        # Pass through other hints as-is (might be specific model name)
        return model_hint

    # Scan content for tier keywords and natural language using word boundaries
    if content:
        content_lower = content.lower()

        # Size and capability-based patterns
        if (re.search(r'\b(30b|large|big|complex|reasoning|deep|planning|critique)\b', content_lower) or
            re.search(r'\bmoe\b', content_lower)):
            log.info("model_override_detected", tier="moe", source="content", pattern="size/capability")
            return "moe"

        if (re.search(r'\b(14b|coder|code|programming|development|review|analyze)\b', content_lower)):
            log.info("model_override_detected", tier="coder", source="content", pattern="size/capability")
            return "coder"

        if (re.search(r'\b(7b|small|fast|quick|simple|basic|summarize)\b', content_lower)):
            log.info("model_override_detected", tier="quick", source="content", pattern="size/capability")
            return "quick"

        if (re.search(r'\b(thinking|think|chain|reason|step.*by.*step)\b', content_lower)):
            log.info("model_override_detected", tier="thinking", source="content", pattern="capability")
            return "thinking"

    return None

async def select_model(task_type: str, content_size: int = 0, model_override: Optional[str] = None, content: str = "") -> str:
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
        if tier_name == "quick": return model_quick
        if tier_name == "coder": return model_coder
        if tier_name == "moe": return model_moe
        if tier_name == "thinking": return model_thinking
        return tier_name # Assume it's a model name if not a tier

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
            log.info("model_selected", source="large_code", content_kb=content_size//1000, confidence=f"{confidence:.0%}", tier="coder", reasoning=reasoning)
            return model_coder
        else:
            # Large text content benefits from MoE's reasoning
            log.info("model_selected", source="large_text", content_kb=content_size//1000, confidence=f"{1-confidence:.0%}", tier="moe", reasoning=reasoning)
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
# OLLAMA API CALLS (with retry and Pydantic validation)
# ============================================================

# ============================================================
# RESPONSE HELPERS
# ============================================================

def extract_thinking_content(response_text: str) -> Optional[str]:
    """
    Extract thinking content from LLM response.

    Returns the content between <think> and </think> tags, or None if not present.
    Optimized with early exit to avoid unnecessary string operations.
    """
    if "<think>" not in response_text:
        return None

    start = response_text.find("<think>") + 7
    end = response_text.find("</think>")

    if end < start:
        return None

    return response_text[start:end].strip()


def log_thinking_and_response(response_text: str, model_tier: str, tokens: int) -> None:
    """
    Log thinking content and response preview.

    Extracted helper to avoid duplicated code in call_ollama and call_llamacpp.
    """
    # Log thinking if present
    thinking = extract_thinking_content(response_text)
    if thinking:
        thinking_preview = thinking[:200] + "..." if len(thinking) > 200 else thinking
        log.info(get_display_event("model_thinking"),
                log_type="THINK",
                preview=thinking_preview.replace("\n", " "),
                model=model_tier,
                garden_msg=get_message(GardenEvent.GROWING))

    # Log LLM response (first 300 chars)
    response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
    log.info(get_display_event("model_response"),
            log_type="RESPONSE",
            preview=response_preview.replace("\n", " ").strip(),
            model=model_tier,
            tokens=tokens,
            garden_msg=get_message(GardenEvent.RIPENING))


# Retry decorator for transient failures (connection issues, timeouts on model load)
def _should_retry_ollama(exception: BaseException) -> bool:
    """Determine if Ollama call should be retried."""
    if isinstance(exception, httpx.ConnectError):
        return True  # Retry connection issues
    if isinstance(exception, httpx.TimeoutException):
        return False  # Don't retry timeouts (model loading can take long)
    return False


async def call_ollama(model: str, prompt: str, system: Optional[str] = None,
                      enable_thinking: bool = False, task_type: str = "unknown",
                      original_task: str = "unknown",
                      language: str = "unknown", content_preview: str = "",
                      backend_obj: Optional[BackendConfig] = None) -> dict:
    """Call Ollama API with Pydantic validation, retry logic, and circuit breaker."""
    start_time = time.time()

    # Resolve backend
    if not backend_obj:
        # Find first enabled Ollama backend
        for b in backend_manager.get_enabled_backends():
            if b.provider == "ollama":
                backend_obj = b
                break
    
    if not backend_obj:
        return {"success": False, "error": "No enabled Ollama backend found"}

    # Circuit breaker check
    health = get_backend_health(backend_obj.id)
    if not health.is_available():
        wait_time = health.time_until_available()
        log.warning("circuit_open", backend=backend_obj.id, wait_seconds=round(wait_time, 1))
        return {
            "success": False,
            "error": f"Ollama circuit breaker open. Too many failures. Retry in {wait_time:.0f}s.",
            "circuit_breaker": True
        }

    # Context size check and potential reduction
    content_size = len(prompt) + len(system or "")
    should_reduce, recommended_size = health.should_reduce_context(content_size)
    if should_reduce:
        log.info("context_reduction", backend=backend_obj.id, original_kb=content_size//1024, recommended_kb=recommended_size//1024)
        # Truncate prompt if too large (keep beginning which usually has instructions)
        if len(prompt) > recommended_size:
            prompt = prompt[:recommended_size] + "\n\n[Content truncated due to previous timeout]"

    if enable_thinking and "qwen" in model.lower():
        prompt = f"/think\n{prompt}"

    # Auto-select context size based on model tier (uses centralized detection)
    model_tier = detect_model_tier(model)
    if model_tier == "moe":
        num_ctx = config.model_moe.num_ctx
    elif model_tier == "coder":
        num_ctx = config.model_coder.num_ctx
    else:
        num_ctx = config.model_quick.num_ctx

    # Temperature based on thinking mode (from config)
    temperature = config.temperature_thinking if enable_thinking else config.temperature_normal

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
    }
    if system:
        payload["system"] = system

    # Get client from backend object
    client = backend_obj.get_client()

    # Inner function with retry logic
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(httpx.ConnectError),
        reraise=True
    )
    async def _make_request():
        return await client.post("/api/generate", json=payload)

    data: dict[str, Any] = {}
    try:
        # Log start of request
        log.info(get_display_event("model_starting"),
                log_type="MODEL",
                model=model,
                task=task_type,
                thinking=enable_thinking,
                backend=backend_obj.name,
                garden_msg=get_vine_message(model_tier, "start"))

        response = await _make_request()
        elapsed_ms = int((time.time() - start_time) * 1000)

        if response.status_code == 200:
            try:
                data = response.json()
                # Validate with Pydantic
                validated = OllamaResponse.model_validate(data)
            except json.JSONDecodeError:
                return {"success": False, "error": "Ollama returned non-JSON response"}
            except ValidationError as e:
                log.warning("ollama_validation_failed", error=str(e))
                # Fall back to raw dict access if validation fails
                validated = None

            # Use validated data or fall back to raw
            if validated:
                tokens = validated.eval_count
                response_text = validated.response
            else:
                tokens = data.get("eval_count", 0)
                response_text = data.get("response", "")

            # Thread-safe stats update
            _update_stats_sync(
                model_tier=model_tier,
                task_type=task_type,
                original_task=original_task,
                tokens=tokens,
                elapsed_ms=elapsed_ms,
                content_preview=content_preview,
                enable_thinking=enable_thinking,
                backend="ollama"
            )

            # Log thinking and response using helper
            log_thinking_and_response(response_text, model_tier, tokens)

            # Log completion
            log.info(
                get_display_event("model_completed"),
                log_type="INFO",
                elapsed=humanize.naturaldelta(elapsed_ms / 1000),
                elapsed_ms=elapsed_ms,
                tokens=humanize.intcomma(tokens),
                model=model_tier,
                backend="ollama",
                garden_msg=format_harvest_stats(tokens, elapsed_ms, model_tier),
            )

            # Record success for circuit breaker
            health.record_success(content_size)

            # Persist stats to disk (async with lock, non-blocking)
            asyncio.create_task(save_all_stats_async())

            return {
                "success": True,
                "response": response_text,
                "tokens": tokens,
                "elapsed_ms": elapsed_ms,
            }
        # Improved HTTP error handling with more context
        error_msg = f"Ollama HTTP {response.status_code}"
        if response.status_code == 404:
            error_msg += f": Model '{model}' not found. Run: ollama pull {model}"
        elif response.status_code == 500:
            error_msg += ": Internal server error. Check Ollama logs."
        elif response.status_code == 503:
            error_msg += ": Ollama service unavailable. Is it running?"
        else:
            # Truncate long error responses
            error_text = response.text[:500] if len(response.text) > 500 else response.text
            error_msg += f": {error_text}"
        health.record_failure("http_error", content_size)
        return {"success": False, "error": error_msg}
    except httpx.TimeoutException:
        log.error("ollama_timeout", model=model, timeout_seconds=config.ollama_timeout_seconds)
        health.record_failure("timeout", content_size)
        return {"success": False, "error": f"Ollama timeout after {config.ollama_timeout_seconds}s. Model may be loading or prompt too large."}
    except httpx.ConnectError:
        log.error("ollama_connection_refused", base_url=backend_obj.url)
        health.record_failure("connection", content_size)
        return {"success": False, "error": f"Cannot connect to Ollama at {backend_obj.url}. Is Ollama running?"}
    except Exception as e:
        log.error("ollama_error", model=model, error=str(e))
        health.record_failure("exception", content_size)
        return {"success": False, "error": f"Ollama error: {str(e)}"}

# ============================================================
# LLAMA.CPP API CALLS (OpenAI-compatible, with retry and Pydantic)
# ============================================================

async def call_llamacpp(model: str, prompt: str, system: Optional[str] = None,
                        enable_thinking: bool = False, task_type: str = "unknown",
                        original_task: str = "unknown",
                        language: str = "unknown", content_preview: str = "",
                        backend_obj: Optional[BackendConfig] = None) -> dict:
    """Call OpenAI-compatible API (llama.cpp, vLLM, etc.) with Pydantic validation, retry, and circuit breaker."""
    start_time = time.time()

    # Resolve backend
    if not backend_obj:
        # Find first enabled OpenAI-compatible backend
        for b in backend_manager.get_enabled_backends():
            if b.provider in ("llamacpp", "vllm", "openai", "custom"):
                backend_obj = b
                break
    
    if not backend_obj:
        return {"success": False, "error": "No enabled OpenAI-compatible backend found"}

    # Circuit breaker check
    health = get_backend_health(backend_obj.id)
    if not health.is_available():
        wait_time = health.time_until_available()
        log.warning("circuit_open", backend=backend_obj.id, wait_seconds=round(wait_time, 1))
        return {
            "success": False,
            "error": f"Backend circuit breaker open. Too many failures. Retry in {wait_time:.0f}s.",
            "circuit_breaker": True
        }

    # Context size check and potential reduction
    content_size = len(prompt) + len(system or "")
    should_reduce, recommended_size = health.should_reduce_context(content_size)
    if should_reduce:
        log.info("context_reduction", backend=backend_obj.id, original_kb=content_size//1024, recommended_kb=recommended_size//1024)
        if len(prompt) > recommended_size:
            prompt = prompt[:recommended_size] + "\n\n[Content truncated due to previous timeout]"

    # For thinking mode with qwen models, add /think prefix
    if enable_thinking and "qwen" in model.lower():
        prompt = f"/think\n{prompt}"

    # Auto-select context size based on model tier
    model_tier = detect_model_tier(model)
    if model_tier == "moe":
        num_ctx = config.model_moe.num_ctx
    elif model_tier == "coder":
        num_ctx = config.model_coder.num_ctx
    else:
        num_ctx = config.model_quick.num_ctx

    # Temperature based on thinking mode
    temperature = config.temperature_thinking if enable_thinking else config.temperature_normal

    # Build messages for OpenAI-compatible API
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # OpenAI-compatible payload
    # Note: Some backends ignore 'model' in payload if they only host one, but we send it anyway
    payload = {
        "model": model, 
        "messages": messages,
        "temperature": temperature,
        "max_tokens": num_ctx,  # Use context size as max tokens
        "stream": False,
    }

    # Get client
    client = backend_obj.get_client()

    # Inner function with retry logic
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(httpx.ConnectError),
        reraise=True
    )
    async def _make_request():
        return await client.post(backend_obj.chat_endpoint, json=payload)

    data: dict[str, Any] = {}
    try:
        # Log start of request
        log.info(get_display_event("model_starting"),
                log_type="MODEL",
                backend=backend_obj.name,
                task=task_type,
                thinking=enable_thinking,
                model=model_tier,
                garden_msg=get_vine_message(model_tier, "start"))

        response = await _make_request()
        elapsed_ms = int((time.time() - start_time) * 1000)

        if response.status_code == 200:
            try:
                data = response.json()
                # Validate with Pydantic
                validated = LlamaCppResponse.model_validate(data)

                # Extract from validated model
                if not validated.choices:
                    return {"success": False, "error": "Backend returned no choices"}

                response_text = validated.choices[0].message.content

                # Token counting from usage
                if validated.usage:
                    tokens = validated.usage.completion_tokens + validated.usage.prompt_tokens
                else:
                    # Fallback: use tiktoken for accurate estimation
                    tokens = count_tokens(response_text)

            except json.JSONDecodeError:
                return {"success": False, "error": "Backend returned non-JSON response"}
            except ValidationError as e:
                log.warning("llamacpp_validation_failed", error=str(e))
                # Fall back to raw dict access
                choices = data.get("choices", [])
                if not choices:
                    return {"success": False, "error": "Backend returned no choices"}
                response_text = choices[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                tokens = usage.get("completion_tokens", 0) + usage.get("prompt_tokens", 0)
                if tokens == 0:
                    tokens = count_tokens(response_text)

            # Thread-safe stats update
            _update_stats_sync(
                model_tier=model_tier,
                task_type=task_type,
                original_task=original_task,
                tokens=tokens,
                elapsed_ms=elapsed_ms,
                content_preview=content_preview,
                enable_thinking=enable_thinking,
                backend="llamacpp"
            )

            # Log thinking and response using helper
            log_thinking_and_response(response_text, model_tier, tokens)

            # Log completion
            log.info(
                get_display_event("model_completed"),
                log_type="INFO",
                elapsed=humanize.naturaldelta(elapsed_ms / 1000),
                elapsed_ms=elapsed_ms,
                tokens=humanize.intcomma(tokens),
                model=model_tier,
                backend="llamacpp",
                garden_msg=format_harvest_stats(tokens, elapsed_ms, model_tier),
            )

            # Record success for circuit breaker
            health.record_success(content_size)

            # Persist stats to disk
            asyncio.create_task(save_all_stats_async())

            return {
                "success": True,
                "response": response_text,
                "tokens": tokens,
                "elapsed_ms": elapsed_ms,
            }
        # Improved HTTP error handling with context-specific messages
        error_msg = f"HTTP {response.status_code}"
        try:
            error_data = response.json()
            if "error" in error_data:
                try:
                    err = LlamaCppError.model_validate(error_data["error"])
                    if err.type == "exceed_context_size_error":
                        error_msg = f"Context exceeded: {err.n_prompt_tokens} tokens > limit."
                    else:
                        error_msg += f": {err.message}"
                except ValidationError:
                    err = error_data["error"]
                    error_msg += f": {err.get('message', str(err))}"
        except (json.JSONDecodeError, KeyError):
            error_text = response.text[:500] if len(response.text) > 500 else response.text
            error_msg += f": {error_text}"
        health.record_failure("http_error", content_size)
        return {"success": False, "error": error_msg}
    except httpx.TimeoutException:
        log.error("llamacpp_timeout", timeout_seconds=config.llamacpp_timeout_seconds)
        health.record_failure("timeout", content_size)
        return {"success": False, "error": f"Timeout after {config.llamacpp_timeout_seconds}s. Model may be loading or prompt too large."}
    except httpx.ConnectError:
        log.error("llamacpp_connection_refused", base_url=backend_obj.url)
        health.record_failure("connection", content_size)
        return {"success": False, "error": f"Cannot connect to {backend_obj.url}. Is the server running?"}
    except Exception as e:
        log.error("llamacpp_error", error=str(e))
        health.record_failure("exception", content_size)
        return {"success": False, "error": f"Error: {str(e)}"}


# ============================================================
# GEMINI API CALLS (Google Generative AI)
# ============================================================

async def call_gemini(model: str, prompt: str, system: Optional[str] = None,
                      enable_thinking: bool = False, task_type: str = "unknown",
                      original_task: str = "unknown",
                      language: str = "unknown", content_preview: str = "",
                      backend_obj: Optional[BackendConfig] = None) -> dict:
    """Call Google Gemini API with stats tracking and circuit breaker."""
    start_time = time.time()
    
    # 1. Dependency Check
    try:
        import google.generativeai as genai
        from google.api_core import exceptions as google_exceptions
    except ImportError:
        return {
            "success": False, 
            "error": "Gemini dependency missing. Please run: uv add google-generativeai"
        }

    # 2. Resolve Backend
    if not backend_obj:
        for b in backend_manager.get_enabled_backends():
            if b.provider == "gemini":
                backend_obj = b
                break
    
    if not backend_obj:
        return {"success": False, "error": "No enabled Gemini backend found"}

    # 3. Circuit Breaker Check
    health = get_backend_health(backend_obj.id)
    if not health.is_available():
        wait_time = health.time_until_available()
        return {
            "success": False,
            "error": f"Gemini circuit breaker open. Retry in {wait_time:.0f}s.",
            "circuit_breaker": True
        }

    # 4. Configuration (API Key)
    import os
    api_key = backend_obj.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"success": False, "error": "Missing GEMINI_API_KEY"}
    
    genai.configure(api_key=api_key)

    # 5. Model Configuration
    # Strip tier prefix if present (e.g. "gemini:gemini-2.0-flash" -> "gemini-2.0-flash")
    model_name = model.split(":")[-1] if ":" in model else model
    # Default to flash if tier name passed directly
    if model_name in ["quick", "coder", "moe", "thinking"]:
        model_name = "gemini-2.0-flash"

    # Generation Config
    generation_config = {
        "temperature": config.temperature_thinking if enable_thinking else config.temperature_normal,
    }
    
    # Add thinking/reasoning config if supported by model and requested
    # Note: 2.0 Flash Thinking is a separate model, not a param, usually.
    # If user wants thinking, we might map to a specific thinking model if configured.
    
    try:
        log.info(get_display_event("model_starting"),
                log_type="MODEL",
                backend="gemini",
                task=task_type,
                model=model_name,
                garden_msg="Consulting the Gemini constellation...")
        
        # Instantiate model
        gen_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system
        )

        # Run in thread executor because the SDK is synchronous
        response = await asyncio.to_thread(
            gen_model.generate_content,
            prompt,
            generation_config=generation_config
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Extract text and usage
        response_text = response.text
        
        # Estimate usage (Gemini SDK might not return exact tokens in all responses easily without metadata)
        # Usage metadata is in response.usage_metadata
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata"):
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
        
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens == 0:
            total_tokens = count_tokens(prompt) + count_tokens(response_text)

        # Thread-safe stats update
        _update_stats_sync(
            model_tier="moe", # Treat Gemini as MoE/High-end tier for stats
            task_type=task_type,
            original_task=original_task,
            tokens=total_tokens,
            elapsed_ms=elapsed_ms,
            content_preview=content_preview,
            enable_thinking=enable_thinking,
            backend="gemini"
        )

        # Log completion
        log.info(
            get_display_event("model_completed"),
            log_type="INFO",
            elapsed=humanize.naturaldelta(elapsed_ms / 1000),
            elapsed_ms=elapsed_ms,
            tokens=humanize.intcomma(total_tokens),
            model=model_name,
            backend="gemini",
            garden_msg="Starlight wisdom gathered from the clouds!",
        )

        health.record_success(len(prompt))
        asyncio.create_task(save_all_stats_async())

        return {
            "success": True,
            "response": response_text,
            "tokens": total_tokens,
            "elapsed_ms": elapsed_ms
        }

    except google_exceptions.ResourceExhausted:
        health.record_failure("rate_limit", len(prompt))
        return {"success": False, "error": "Gemini rate limit exceeded (429)."}
    except Exception as e:
        log.error("gemini_error", error=str(e))
        health.record_failure("exception", len(prompt))
        return {"success": False, "error": f"Gemini error: {str(e)}"}


async def call_llm(model: str, prompt: str, system: Optional[str] = None,
                   enable_thinking: bool = False, task_type: str = "unknown",
                   original_task: str = "unknown",
                   language: str = "unknown", content_preview: str = "",
                   backend: Optional[str] = None, backend_obj: Optional[BackendConfig] = None) -> dict:
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
        except asyncio.TimeoutError:
            model_queue.queue_timeouts += 1
            await model_queue.release_model(model, success=False)
            log.warning("queue_timeout",
                       model=model,
                       wait_seconds=300,
                       total_timeouts=model_queue.queue_timeouts,
                       log_type="QUEUE")
            return {"success": False, "error": f"Timeout waiting for model {model} to load (waited 5 minutes)"}
        except Exception as e:
            await model_queue.release_model(model, success=False)
            log.error("queue_error",
                     model=model,
                     error=str(e),
                     log_type="QUEUE")
            return {"success": False, "error": f"Error waiting for model {model}: {str(e)}"}

    try:
        # Determine backend to use
        active_backend = None
        
        # 1. Use passed object if available
        if backend_obj:
            active_backend = backend_obj
        # 2. Resolve by ID or provider name
        elif backend:
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

        # Dispatch based on provider type
        if active_backend.provider in ("llama.cpp", "llamacpp", "vllm", "openai", "custom"):
            result = await call_llamacpp(model, prompt, system, enable_thinking, task_type, original_task, language, content_preview, backend_obj=active_backend)
        elif active_backend.provider == "ollama":
            result = await call_ollama(model, prompt, system, enable_thinking, task_type, original_task, language, content_preview, backend_obj=active_backend)
        elif active_backend.provider == "gemini":
            result = await call_gemini(model, prompt, system, enable_thinking, task_type, original_task, language, content_preview, backend_obj=active_backend)
        else:
             await model_queue.release_model(model, success=False)
             return {"success": False, "error": f"Unsupported provider: {active_backend.provider}"}

        # Release model on success
        await model_queue.release_model(model, success=result.get("success", False))
        return result
    except Exception as e:
        # Release model on failure
        await model_queue.release_model(model, success=False)
        raise


# ============================================================
# FILE HANDLING
# ============================================================

def read_file_safe(file_path: str, max_size: Optional[int] = None) -> tuple[Optional[str], Optional[str]]:
    """Safely read file with size limit."""
    if max_size is None:
        max_size = config.max_file_size
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return None, f"File not found: {file_path}"
        if path.stat().st_size > max_size:
            return None, f"File too large: {path.stat().st_size} > {max_size}"
        return path.read_text(encoding="utf-8", errors="replace"), None
    except Exception as e:
        return None, f"Error reading file: {e}"

# ============================================================
# MCP SERVER SETUP
# ============================================================

mcp = FastMCP(
    "delia",
    instructions="""
# Delia: Intelligent LLM Orchestration

Delia coordinates between your primary LLM (Copilot/Claude) and configurable
backend LLMs to offload work, reduce costs, and process tasks efficiently.

You have access to one or more LLM backends configured by the user. Backends can be:
- Local models (Ollama, vLLM, llama.cpp on your machine)
- Remote services (OpenAI API, Anthropic, cloud-hosted models)
- GPU servers (dedicated inference endpoints)

Each backend has a **type** ("local" or "remote") that determines routing behavior.

## CRITICAL: WHEN TO USE DELIA

**ALWAYS consider using Delia tools when you detect ANY of these signals:**

### Explicit Delia/Delegation Mentions (100% confidence → USE DELIA)
- "delegate", "offload", "use delia", "@delia"
- "process with backend", "via local model", "through orchestration"
- "ask local llm", "use my models"

### Processing Location Signals (HIGH confidence → USE DELIA)
- **Local processing**: "locally", "local", "on my machine", "on device", "on my gpu", "on my server"
- **Remote processing**: "remotely", "remote", "on the cloud", "via api", "on remote server"
- **Distributed processing**: "parallel", "batch", "both", "distribute", "use all backends"

### Task-Specific Terms (MEDIUM confidence → CONSIDER DELIA)
- **Code tasks**: "review code", "analyze this file", "generate a function", "check for bugs"
- **Reasoning tasks**: "think about", "plan this", "design strategy", "evaluate tradeoffs"
- **Batch tasks**: "process these files", "review all", "check multiple"

### Disambiguation Strategy
When you detect processing/task terms WITHOUT explicit "delegate" mention:

1. **High-confidence scenarios** (user specifies processing location):
   - "review this code locally" → **USE DELIA directly**
   - "check config file" → **DON'T USE DELIA** (file operation)
   - "analyze on my gpu" → **USE DELIA directly**

2. **Medium-confidence scenarios** (task without location specified):
   - If user has used Delia recently → **DEFAULT TO DELIA**
   - If task is code/reasoning work → **DEFAULT TO DELIA**
   - If task is file/config query → **DON'T USE DELIA**

3. **Explicit exclusions** (never delegate these):
   - "show me settings.json", "what's the config?", "curl the health endpoint"

## REASONING FRAMEWORK: How to Process Any Request

### Step 1: Identify Processing Constraints
| User Says | Constraint | Meaning |
|-----------|------------|---------|
| "locally" | LOCAL ONLY | Use type="local" backends |
| "remotely" | REMOTE ONLY | Use type="remote" backends |
| "parallel", "batch" | DISTRIBUTED | Use all backends |
| (nothing) | NO CONSTRAINT | Use default backend |

### Step 2: Decompose Complex Requests
- Single vs multiple tasks
- Dependencies between tasks
- Parallel opportunities

### Step 3: Classify Each Task
| Task Type | When to Use | Model Tier |
|-----------|-------------|------------|
| quick | Facts, simple Q&A | 14B |
| generate | Code generation | 14B coder |
| review | Bug finding, code review | 14B coder |
| analyze | Understanding code | 14B coder |
| summarize | Condensing info | 14B |
| plan | Architecture, strategy | 30B+ |
| critique | Deep evaluation | 30B+ |

### Step 4: Execute with Appropriate Tool
- Single task → delegate()
- Multiple tasks → batch()
- Your reasoning → think()

## TOOL REFERENCE

### delegate(task, content, file?, model?, language?, context?, backend?)
   model: "quick"/"coder"/"moe"/"thinking" OR natural: "7b"/"14b"/"30b"/"small"/"large"/"fast"/"complex"/"coder model"/"thinking"
### batch(tasks) - JSON array
### think(problem, context?, depth?) - quick/normal/deep
### switch_backend(backend)
### health()
### models()

## WORKED EXAMPLES

**Example 1**: "Review the auth module locally, analyze dependencies, suggest improvements"
1. Constraint: "locally" → LOCAL ONLY
2. Decomposition: 3 tasks (review, analyze, suggest)
3. Dependencies: suggest depends on review+analyze
4. Execute: batch([review, analyze]) → delegate(plan)

**Example 2**: "Process these 3 files in parallel"
1. Constraint: DISTRIBUTED
2. Execute: batch([file1, file2, file3])

**Example 3**: "Delegate this code review to my remote API"
1. Constraint: "remote" → REMOTE ONLY
2. Execute: delegate(review, ...)

## CONSTRAINTS

- Don't delegate: web search, real-time data, file operations
- Respect constraints: "locally" = local ONLY, "remotely" = remote ONLY
- Circuit breaker: don't retry if backend in cooldown
- Backend agnostic: never assume specific software

## SILENT OPERATION

- Don't announce "I'll delegate this"
- Just invoke tools and use results
- Delia handles failures automatically
"""
)

# ============================================================
# MULTI-USER TRACKING MIDDLEWARE
# Extracts user from JWT and tracks requests per user
# ============================================================

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_request, get_http_headers


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
        except Exception:
            pass  # No HTTP context - STDIO transport

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
                            async with get_async_session_context() as session:
                                async with get_user_db_context(session) as user_db:
                                    user = await user_db.get(uuid.UUID(user_id))
                                    if user:
                                        username = user.email
                                        # Set user quota from database
                                        from multi_user_tracking import QuotaConfig
                                        quota = QuotaConfig(
                                            max_requests_per_hour=user.max_requests_per_hour,
                                            max_tokens_per_hour=user.max_tokens_per_hour,
                                        )
                                        # Will be set after client is created
            except Exception:
                pass  # No HTTP context or invalid token

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
                    from fastmcp.tools import ToolResult
                    return ToolResult(content=[{"type": "text", "text": f"Quota exceeded: {msg}"}], is_error=True)

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
                tool_name = context.method or getattr(context.message, 'name', None) or 'unknown'

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
# HELPER: Get Current User in Tools
# ============================================================

def get_current_user_from_context(ctx) -> dict:
    """
    Get the current authenticated user from context.

    Returns dict with user_id, username, client_id.
    If not authenticated, returns anonymous user with session-based tracking.
    """
    return {
        "user_id": ctx.get_state("user_id", "anonymous"),
        "username": ctx.get_state("username", "anonymous"),
        "client_id": ctx.get_state("client_id"),
    }


# ============================================================
# AUTHENTICATION ROUTES (HTTP/SSE only)
# These routes are only active when AUTH_ENABLED=true
# ============================================================

from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel as PydanticBaseModel, EmailStr

if AUTH_ENABLED:
    import jose.jwt


class LoginRequest(PydanticBaseModel):
    """Login request body."""
    username: str  # Actually email
    password: str


class RegisterRequest(PydanticBaseModel):
    """User registration request."""
    email: EmailStr
    password: str
    display_name: str | None = None


def _register_auth_routes():
    """Register authentication routes. Only called when AUTH_ENABLED=true."""

    @mcp.custom_route("/auth/register", methods=["POST"])
    async def auth_register(request: Request) -> JSONResponse:
        """
        Register a new user.

        POST /auth/register
        Body: {"email": "...", "password": "...", "display_name": "..."}
        Returns: User data with JWT token
        """
        try:
            body = await request.json()
            reg = RegisterRequest(**body)

            # Use context managers for dependency injection outside FastAPI
            async with get_async_session_context() as session:
                async with get_user_db_context(session) as user_db:
                    async with get_user_manager_context(user_db) as user_manager:
                        # Create user
                        user_create = UserCreate(
                            email=reg.email,
                            password=reg.password,
                            display_name=reg.display_name
                        )
                        user = await user_manager.create(user_create)

                        # Commit the session to persist the user
                        await session.commit()

                        # Generate token
                        strategy = auth_backend.get_strategy()
                        token = await strategy.write_token(user)

                        return JSONResponse({
                            "access_token": token,
                            "token_type": "bearer",
                            "user": {
                                "id": str(user.id),
                                "email": user.email,
                                "display_name": user.display_name,
                                "is_active": user.is_active,
                                "is_superuser": user.is_superuser,
                            }
                        }, status_code=201)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error("registration_failed", error=str(e), traceback=tb)
            return JSONResponse({"detail": str(e) or repr(e), "traceback": tb}, status_code=400)


    @mcp.custom_route("/auth/jwt/login", methods=["POST"])
    async def auth_login(request: Request) -> JSONResponse:
        """
        Login and get JWT token.

        POST /auth/jwt/login
        Body (JSON): {"username": "email@example.com", "password": "..."}
        Body (Form): username=email@example.com&password=...
        Returns: {"access_token": "...", "token_type": "bearer"}
        """
        try:
            # Support both JSON and form data (OAuth2 password flow uses form)
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body = await request.json()
                login = LoginRequest(**body)
            else:
                # Form data
                form = await request.form()
                login = LoginRequest(username=form.get("username", ""), password=form.get("password", ""))

            async with get_async_session_context() as session:
                async with get_user_db_context(session) as user_db:
                    async with get_user_manager_context(user_db) as user_manager:
                        # Authenticate user
                        user = await user_manager.authenticate(
                            credentials=type('Creds', (), {'username': login.username, 'password': login.password})()
                        )

                        if user is None:
                            return JSONResponse({"detail": "Invalid credentials"}, status_code=401)

                        if not user.is_active:
                            return JSONResponse({"detail": "User is inactive"}, status_code=401)

                        # Generate token
                        strategy = auth_backend.get_strategy()
                        token = await strategy.write_token(user)

                        log.info("user_logged_in", user_id=str(user.id), email=user.email)

                        return JSONResponse({
                            "access_token": token,
                            "token_type": "bearer"
                        })
        except Exception as e:
            log.error("login_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=400)


    @mcp.custom_route("/auth/me", methods=["GET"])
    async def auth_me(request: Request) -> JSONResponse:
        """
        Get current user info.

        GET /auth/me
        Headers: Authorization: Bearer <token>
        Returns: User data
        """
        try:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)

            token = auth_header.replace("Bearer ", "")

            # Decode token using helper
            payload = decode_jwt_token(token)
            if not payload:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            user_id = payload.get("sub")
            if not user_id:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            # Get user from database
            async with get_async_session_context() as session:
                async with get_user_db_context(session) as user_db:
                    user = await user_db.get(uuid.UUID(user_id))

                    if not user:
                        return JSONResponse({"detail": "User not found"}, status_code=404)

                    return JSONResponse({
                        "id": str(user.id),
                        "email": user.email,
                        "display_name": user.display_name,
                        "is_active": user.is_active,
                        "is_superuser": user.is_superuser,
                        "max_tokens_per_hour": user.max_tokens_per_hour,
                        "max_requests_per_hour": user.max_requests_per_hour,
                        "max_model_tier": user.max_model_tier,
                    })

        except Exception as e:
            log.error("auth_me_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=500)


    @mcp.custom_route("/auth/users", methods=["GET"])
    async def list_users(request: Request) -> JSONResponse:
        """
        List all users (superuser only).

        GET /auth/users
        Headers: Authorization: Bearer <superuser_token>
        Returns: List of users
        """
        try:
            # Validate superuser
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)

            token = auth_header.replace("Bearer ", "")

            # Decode token using helper
            payload = decode_jwt_token(token)
            if not payload:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            user_id = payload.get("sub")
            if not user_id:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            async with get_async_session_context() as session:
                async with get_user_db_context(session) as user_db:
                    admin = await user_db.get(uuid.UUID(user_id))

                    if not admin or not admin.is_superuser:
                        return JSONResponse({"detail": "Superuser access required"}, status_code=403)

                    # Get all users
                    from sqlalchemy import select
                    result = await session.execute(select(User))
                    users = result.scalars().all()

                    return JSONResponse({
                        "users": [
                            {
                                "id": str(u.id),
                                "email": u.email,
                                "display_name": u.display_name,
                                "is_active": u.is_active,
                                "is_superuser": u.is_superuser,
                                "max_tokens_per_hour": u.max_tokens_per_hour,
                                "max_requests_per_hour": u.max_requests_per_hour,
                            }
                            for u in users
                        ]
                    })

        except Exception as e:
            log.error("list_users_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=500)


    @mcp.custom_route("/auth/stats", methods=["GET"])
    async def auth_user_stats(request: Request) -> JSONResponse:
        """
        Get usage stats for authenticated user.

        GET /auth/stats
        Headers: Authorization: Bearer <token>
        Returns: User's usage statistics
        """
        try:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)

            token = auth_header.replace("Bearer ", "")

            # Decode token using helper
            payload = decode_jwt_token(token)
            if not payload:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            user_id = payload.get("sub")
            if not user_id:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            # Get user from database
            async with get_async_session_context() as session:
                async with get_user_db_context(session) as user_db:
                    user = await user_db.get(uuid.UUID(user_id))

                    if not user:
                        return JSONResponse({"detail": "User not found"}, status_code=404)

                    # Get tracking stats for this user
                    stats = tracker.get_user_stats(user.email)

                    return JSONResponse({
                        "user_id": str(user.id),
                        "email": user.email,
                        "quotas": {
                            "max_tokens_per_hour": user.max_tokens_per_hour,
                            "max_requests_per_hour": user.max_requests_per_hour,
                            "max_model_tier": user.max_model_tier,
                        },
                        "usage": {
                            "total_requests": stats.total_requests if stats else 0,
                            "total_tokens": stats.total_tokens if stats else 0,
                            "requests_this_hour": stats.requests_this_hour if stats else 0,
                            "tokens_this_hour": stats.tokens_this_hour if stats else 0,
                        },
                        "quota_remaining": {
                            "requests_remaining": user.max_requests_per_hour - (stats.requests_this_hour if stats else 0),
                            "tokens_remaining": user.max_tokens_per_hour - (stats.tokens_this_hour if stats else 0),
                        }
                    })

        except Exception as e:
            log.error("user_stats_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=500)


    @mcp.custom_route("/auth/stats/all", methods=["GET"])
    async def auth_all_stats(request: Request) -> JSONResponse:
        """
        Get usage stats for all users (superuser only).

        GET /auth/stats/all
        Headers: Authorization: Bearer <superuser_token>
        Returns: All users' usage statistics
        """
        try:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)

            token = auth_header.replace("Bearer ", "")

            # Decode token using helper
            payload = decode_jwt_token(token)
            if not payload:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            user_id = payload.get("sub")
            if not user_id:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            async with get_async_session_context() as session:
                async with get_user_db_context(session) as user_db:
                    admin = await user_db.get(uuid.UUID(user_id))

                    if not admin or not admin.is_superuser:
                        return JSONResponse({"detail": "Superuser access required"}, status_code=403)

                    # Get all user stats from tracker
                    all_stats = tracker.get_all_users()

                    return JSONResponse({
                        "users": [
                            {
                                "client_id": stats.client_id,
                                "total_requests": stats.total_requests,
                                "total_tokens": stats.total_tokens,
                                "requests_this_hour": stats.requests_this_hour,
                                "tokens_this_hour": stats.tokens_this_hour,
                                "first_request": stats.first_request.isoformat() if stats.first_request else None,
                                "last_request": stats.last_request.isoformat() if stats.last_request else None,
                            }
                            for stats in all_stats
                        ]
                    })

        except Exception as e:
            log.error("all_stats_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=500)


# Call to register auth routes if enabled
if AUTH_ENABLED:
    _register_auth_routes()
    log.info("auth_routes_registered", endpoints=["/auth/register", "/auth/jwt/login", "/auth/me", "/auth/stats"])
else:
    log.info("auth_disabled", message="Authentication routes not registered. Set DELIA_AUTH_ENABLED=true to enable.")


# ============================================================
# MICROSOFT OAUTH ROUTES (HTTP/SSE only)
# ============================================================

if AUTH_ENABLED:
    from auth import (
        microsoft_oauth_client, oauth_backend, get_user_manager,
        JWT_SECRET, MICROSOFT_REDIRECT_URL
    )
    from fastapi_users.router.oauth import get_oauth_router
    from fastapi import Request, Response
    from fastapi.responses import RedirectResponse
    import secrets

    # Create OAuth router instance to access its logic
    oauth_router_instance = get_oauth_router(
        microsoft_oauth_client,
        oauth_backend,
        get_user_manager,
        JWT_SECRET,
        redirect_url=MICROSOFT_REDIRECT_URL,
        associate_by_email=True,
    )

    @mcp.custom_route("/auth/microsoft/authorize", methods=["GET"])
    async def oauth_authorize(request: Request) -> RedirectResponse:
        """Initiate Microsoft OAuth login flow."""
        try:
            # Generate state for CSRF protection
            state = secrets.token_urlsafe(32)

            # Get authorization URL from OAuth client
            authorization_url = await microsoft_oauth_client.get_authorization_url(
                redirect_uri=MICROSOFT_REDIRECT_URL,
                state=state,
            )

            # Store state in session (simplified - in production use proper session storage)
            # For now, we'll skip state validation for simplicity

            return RedirectResponse(authorization_url)
        except Exception as e:
            log.error("oauth_authorize_failed", error=str(e))
            return JSONResponse({"detail": "OAuth authorization failed"}, status_code=500)

    @mcp.custom_route("/auth/microsoft/callback", methods=["GET"])
    async def oauth_callback(request: Request) -> RedirectResponse:
        """Handle Microsoft OAuth callback."""
        try:
            # Get authorization code from query params
            code = request.query_params.get("code")
            if not code:
                return JSONResponse({"detail": "Authorization code missing"}, status_code=400)

            # Exchange code for tokens
            tokens = await microsoft_oauth_client.get_access_token(code, MICROSOFT_REDIRECT_URL)

            # Get user info from Microsoft
            user_info = await microsoft_oauth_client.get_user_info(tokens["access_token"])

            # Create or get user in database
            async with get_async_session_context() as session:
                async with get_user_manager_context(get_user_db_context(session)) as user_manager:
                    # Try to find existing user by email
                    try:
                        user = await user_manager.get_by_email(user_info.email)
                    except Exception:
                        user = None  # User not found or database error

                    if user:
                        # Associate OAuth account with existing user
                        await user_manager.oauth_associate_account(
                            user,
                            "microsoft",
                            user_info.id,
                            tokens
                        )
                    else:
                        # Create new user
                        user = await user_manager.oauth_create_account(
                            "microsoft",
                            user_info.id,
                            user_info.email,
                            user_info.name,
                            tokens
                        )

                    # Generate JWT token
                    from auth import get_jwt_strategy
                    strategy = get_jwt_strategy()
                    token = await strategy.write_token(user)

                    # Redirect to frontend with token (or return JSON for API clients)
                    # For now, return JSON response with token
                    return JSONResponse({
                        "access_token": token,
                        "token_type": "bearer",
                        "user": {
                            "id": str(user.id),
                            "email": user.email,
                            "name": user_info.name,
                        }
                    })

        except Exception as e:
            log.error("oauth_callback_failed", error=str(e))
            return JSONResponse({"detail": "OAuth callback failed"}, status_code=500)

    log.info("oauth_routes_registered", provider="microsoft", endpoints=["/auth/microsoft/authorize", "/auth/microsoft/callback"])
else:
    log.info("oauth_disabled", message="OAuth routes not registered. Set DELIA_AUTH_ENABLED=true to enable.")


async def validate_delegate_request(
    task: str,
    content: str,
    file: Optional[str] = None,
    model: Optional[str] = None,
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
    context: Optional[str] = None,
    symbols: Optional[str] = None,
    include_references: bool = False,
) -> str:
    """Prepare content with context and symbol focus."""
    prepared_content = content

    # Load Serena memory context if specified
    if context:
        context_parts = []
        memory_names = [m.strip() for m in context.split(",")]
        for mem_name in memory_names:
            mem_content = _read_serena_memory(mem_name)
            if mem_content:
                context_parts.append(f"### Context from '{mem_name}':\n{mem_content}")
                log.info("context_memory_loaded", memory=mem_name)
        if context_parts:
            prepared_content = "\n\n".join(context_parts) + "\n\n---\n\n### Task:\n" + prepared_content

    # Add symbol focus hint if symbols specified
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        symbol_hint = f"\n\n### Focus Symbols: {', '.join(symbol_list)}"
        if include_references:
            symbol_hint += "\n_References to these symbols are included below._"
        prepared_content = symbol_hint + "\n\n" + prepared_content
        log.info("context_symbol_focus", symbols=symbol_list, include_references=include_references)

    return prepared_content


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
    model_override: Optional[str] = None,
    backend: Optional[str] = None,
    backend_obj: Optional[Any] = None,
) -> tuple[str, str, str]:
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
    target_backend: str,
    backend_obj: Optional[Any] = None,
) -> tuple[str, int, int]:
    """Execute the LLM call and return response with metadata."""
    # Call LLM
    enable_thinking = task_type in config.thinking_tasks
    # Create a preview for the recent calls log
    content_preview = content[:200].replace("\n", " ").strip()

    result = await call_llm(
        selected_model, content, system, enable_thinking,
        task_type=task_type, original_task=original_task, language=detected_language, content_preview=content_preview,
        backend=target_backend, backend_obj=backend_obj
    )

    if not result.get("success"):
        error_msg = result.get('error', 'Unknown error')
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
    target_backend: str,
    tier: str,
) -> str:
    """Add metadata footer and update tracking."""
    # Update tracker with actual token count and model tier
    client_id = current_client_id.get()
    if client_id:
        tracker.update_last_request(client_id, tokens=tokens, model_tier=tier)

    # Add metadata footer
    return f"""{response_text}

---
_Model: {selected_model} | Tokens: {tokens} | Time: {elapsed_ms}ms | Language: {detected_language} | Backend: {target_backend}_"""

async def _delegate_impl(
    task: str,
    content: str,
    file: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    context: Optional[str] = None,
    symbols: Optional[str] = None,
    include_references: bool = False,
    backend: Optional[str] = None,
    backend_obj: Optional[Any] = None,  # Backend object from backend_manager
) -> str:
    """
    Core implementation for delegate - can be called directly by batch().

    Enhanced context parameters:
        symbols: Comma-separated symbol names to focus on (e.g., "Foo,Foo/calculate")
        include_references: If True, indicates that references to symbols are included in content
        backend: Override backend ("ollama" or "llamacpp"), defaults to active_backend
    """
    start_time = time.time()

    # Validate request
    valid, error = await validate_delegate_request(task, content, file, model)
    if not valid:
        return error

    # Prepare content with context and symbols
    prepared_content = await prepare_delegate_content(content, context, symbols, include_references)

    # Map task to internal type
    task_type = determine_task_type(task)

    # Create enhanced, structured prompt with templates
    prepared_content = create_structured_prompt(
        task_type=task_type,
        content=prepared_content,
        file_path=file,
        language=language,
        symbols=symbols.split(',') if symbols else None,
        context_files=None  # TODO: Parse context parameter for file names if needed
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
            selected_model, prepared_content, system, task_type, task,
            detected_language, target_backend, backend_obj
        )
    except Exception as e:
        return f"Error: {str(e)}"

    # Calculate timing
    elapsed_ms = int((time.time() - start_time) * 1000)

    # Finalize response with metadata
    return finalize_delegate_response(
        response_text, selected_model, tokens, elapsed_ms,
        detected_language, target_backend, tier
    )


@mcp.tool()
async def delegate(
    task: str,
    content: str,
    file: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    context: Optional[str] = None,
    symbols: Optional[str] = None,
    include_references: bool = False,
    backend_type: Optional[str] = None,
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

    ROUTING LOGIC:
    1. Content > 32K tokens → Uses backend with largest context window
    2. Prefer local GPUs (lower latency) unless unavailable
    3. Falls back to remote if local circuit breaker is open
    4. Load balances across available backends based on priority weights

    Returns:
        LLM response with metadata footer showing model, tokens, time, backend

    Examples:
        delegate(task="review", content="<code>", language="python")
        delegate(task="generate", content="Write a REST API", backend_type="local")
        delegate(task="plan", content="Design caching strategy", model="moe")
        delegate(task="analyze", content="Debug this error", model="14b")
        delegate(task="quick", content="Summarize this article", model="fast")
    """
    # Smart backend selection using backend_manager
    backend_provider, backend_obj = await _select_optimal_backend_v2(content, file, task, backend_type)
    return await _delegate_impl(task, content, file, model, language, context, symbols, include_references, backend=backend_provider, backend_obj=backend_obj)


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
        enable_thinking = False
    elif depth == "deep":
        task_type = "plan"
        model_hint = "thinking"  # Use dedicated thinking model for deep reasoning
        enable_thinking = True
    else:  # normal
        task_type = "analyze"
        model_hint = "thinking"  # Use thinking model for normal depth too
        enable_thinking = True

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
    with _stats_thread_lock:
        TASK_STATS["think"] += 1
    await save_all_stats_async()

    return result


# ============================================================
# MULTI-BACKEND INTELLIGENT ROUTING
# ============================================================

async def _select_optimal_backend_v2(
    content: str,
    file_path: Optional[str] = None,
    task_type: str = "quick",
    backend_type: Optional[str] = None,
) -> tuple[Optional[str], Optional[Any]]:
    """
    Select optimal backend.
    
    If backend_type is specified ("local" or "remote"), tries to find a matching backend.
    Otherwise uses the active backend.
    """
    if backend_type:
        # Try to find a backend of the requested type
        for backend in backend_manager.get_enabled_backends():
            if backend.type == backend_type:
                return (None, backend)
    
    # Default to active backend
    backend = backend_manager.get_active_backend()
    return (None, backend)


async def _select_optimal_backend(content: str, file_path: Optional[str] = None) -> Optional[str]:
    """
    Legacy backend selection.
    """
    backend = backend_manager.get_active_backend()
    return backend.id if backend else None


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
    candidate_backends = [
        b.id for b in enabled_backends 
        if available.get(b.id, False)
    ]

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
            - model: Force tier - "quick"|"coder"|"moe"
            - language: Language hint for code tasks

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
    routing_counts = {}
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
        client_id: Optional[str],
        username: Optional[str],
    ) -> str:
        # Re-set context variables in this task
        # Child tasks don't inherit parent context, so we must restore them
        current_client_id.set(client_id)
        current_username.set(username)

        task_type = t.get("task", "analyze")
        content = t.get("content", "")
        file_path = t.get("file")
        model_hint = t.get("model")
        language = t.get("language")
        context = t.get("context")
        symbols = t.get("symbols")
        include_refs = t.get("include_references", False)

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
        )
        return f"### Task {i+1}: {task_type}\n\n{result}"

    results = await asyncio.gather(*[
        run_task(i, t, backend_assignments[i], captured_client_id, captured_username)
        for i, t in enumerate(task_list)
    ])

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Build routing summary
    routing_info = f"Backends used: {', '.join([f'{k}({v})' for k,v in routing_counts.items()])}"

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

    # Calculate usage stats (using new 4-tier keys: quick/coder/moe/thinking)
    total_quick_tokens = MODEL_USAGE["quick"]["tokens"]
    total_coder_tokens = MODEL_USAGE["coder"]["tokens"]
    total_moe_tokens = MODEL_USAGE["moe"]["tokens"]
    total_thinking_tokens = MODEL_USAGE["thinking"]["tokens"]
    local_tokens = total_quick_tokens + total_coder_tokens + total_moe_tokens + total_thinking_tokens
    local_calls = MODEL_USAGE["quick"]["calls"] + MODEL_USAGE["coder"]["calls"] + MODEL_USAGE["moe"]["calls"] + MODEL_USAGE["thinking"]["calls"]

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
                "calls": humanize.intcomma(MODEL_USAGE["quick"]["calls"]),
                "tokens": humanize.intword(total_quick_tokens),
            },
            "coder": {
                "calls": humanize.intcomma(MODEL_USAGE["coder"]["calls"]),
                "tokens": humanize.intword(total_coder_tokens),
            },
            "moe": {
                "calls": humanize.intcomma(MODEL_USAGE["moe"]["calls"]),
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
    if not hasattr(model_queue, 'loaded_models'):
        return json.dumps({
            "status": "queue_not_initialized",
            "message": "ModelQueue not yet initialized"
        }, indent=2)

    # Get current queue state
    loaded = model_queue.loaded_models.copy()
    queued_requests = []

    # Get queued requests from all model queues
    for model_name, queue in model_queue.request_queues.items():
        for queued_request in queue:
            queued_requests.append({
                "id": queued_request.request_id,
                "task": queued_request.task_type,
                "model": queued_request.model_name,
                "priority": queued_request.priority,
                "queued_at": queued_request.timestamp.isoformat() if queued_request.timestamp else None,
                "estimated_tokens": queued_request.content_length // 4,  # Rough estimate
            })

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
        }
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
        backends_info.append({
            "id": backend.id,
            "name": backend.name,
            "provider": backend.provider,
            "url": backend.url,
            "models": backend.models,
        })

    info = {
        "active_backend": get_active_backend_id(),
        "backends": backends_info,
        "currently_loaded": loaded,
        "selection_logic": {
            "quick_tasks": ["quick", "summarize"],
            "coder_tasks": ["generate", "review", "analyze"],
            "moe_tasks": ["plan", "critique"],
            "large_content_threshold": f"{config.large_content_threshold // 1024}KB",
        }
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

Available models: {', '.join(sorted(available_models))}

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
    updated_backend = backend_manager.update_backend(backend.id, {"models": new_models})

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
- **VRAM**: {model_info.get('vram_gb', 'Unknown')}GB
- **Context**: {model_info.get('context_tokens', 'Unknown')} tokens

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

    vram = info.get('vram_gb', 'Unknown')
    context = info.get('context_tokens', 'Unknown')
    tier = info.get('tier', 'unknown')

    # Format numbers nicely
    if isinstance(vram, (int, float)):
        vram_str = f"{vram}GB"
    else:
        vram_str = str(vram)

    if isinstance(context, int):
        if context >= 1000:
            context_str = f"{context:,} tokens"
        else:
            context_str = f"{context} tokens"
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
# SERENA MEMORY INTEGRATION (Internal helpers, not exposed as tools)
# Use Serena's memory tools directly - these are for internal Delia use
# ============================================================

# Memory directory - uses Serena's format for compatibility
MEMORY_DIR = Path(__file__).parent / ".serena" / "memories"


def _read_serena_memory(name: str) -> Optional[str]:
    """
    Internal: Read a Serena memory file.
    Returns content or None if not found.
    """
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    memory_path = MEMORY_DIR / f"{safe_name}.md"

    if memory_path.exists():
        return memory_path.read_text()
    return None


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
    from backend_manager import shutdown_backends
    await shutdown_backends()

    # Save tracker state on shutdown
    if TRACKING_ENABLED:
        await tracker.shutdown()


def main():
    """Entry point for the MCP server."""
    global log
    import argparse
    import atexit

    parser = argparse.ArgumentParser(
        description="Delia — Local LLM delegation server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run mcp_server.py                         # stdio (VS Code, default)
  uv run mcp_server.py --transport sse         # SSE on port 8200
  uv run mcp_server.py --transport http -p 9000  # HTTP on port 9000

Models (4-tier routing):
  quick (qwen3:14b)           - Fast general tasks, ~9GB VRAM
  coder (qwen2.5-coder:14b)   - Code generation/review, ~9GB VRAM
  moe (qwen3:30b-a3b)         - Complex planning/critique, ~17GB VRAM
  thinking (olmo3:7b-think)   - Chain-of-thought reasoning, ~6GB VRAM
        """
    )
    parser.add_argument(
        "-t", "--transport",
        choices=["stdio", "sse", "http", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8200,
        help="Port for HTTP/SSE transport (default: 8200)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind for HTTP/SSE (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    # Note: Stats and logs are already loaded at module import time (lines 181-182)

    # ✅ CRITICAL FIX: Reconfigure logging for STDIO transport
    # For STDIO, stdout MUST be reserved for JSON-RPC protocol messages only
    # Redirect logs to stderr to avoid polluting the protocol stream
    if args.transport == "stdio":
        _configure_structlog(use_stderr=True)
        # Get fresh logger reference after reconfiguration
        log = structlog.get_logger()
        log.info("stdio_logging_configured", destination="stderr")

    # Register graceful shutdown handler for all transports
    if args.transport == "stdio":
        # For STDIO transport, use atexit since there's no built-in shutdown event
        atexit.register(lambda: asyncio.run(_shutdown_handler()))
    # For HTTP/SSE, FastMCP (Starlette) handles shutdown via SIGTERM/SIGINT
    # and will call any registered shutdown handlers

    if args.transport in ("http", "streamable-http"):
        # ✅ Reconfigure logging to stderr for HTTP transports
        _configure_structlog(use_stderr=True)
        # Get fresh logger reference after reconfiguration
        log = structlog.get_logger()
        log.info("http_logging_configured", destination="stderr")

        # Initialize auth database for HTTP transport (if auth enabled)
        if AUTH_ENABLED:
            asyncio.run(_init_database())

        # Register startup/shutdown handlers with FastMCP/Starlette
        mcp.app.add_event_handler("startup", _startup_handler)
        mcp.app.add_event_handler("shutdown", _shutdown_handler)

        auth_endpoints = ["/auth/register", "/auth/jwt/login", "/auth/me"] if AUTH_ENABLED else []
        log.info("server_starting", transport="http", host=args.host, port=args.port,
                 auth_enabled=AUTH_ENABLED, endpoints=auth_endpoints)
        mcp.run(transport="http", host=args.host, port=args.port)
    elif args.transport == "sse":
        # ✅ Reconfigure logging to stderr for SSE transport
        _configure_structlog(use_stderr=True)
        # Get fresh logger reference after reconfiguration
        log = structlog.get_logger()
        log.info("sse_logging_configured", destination="stderr")

        # Initialize auth database for SSE transport (if auth enabled)
        if AUTH_ENABLED:
            asyncio.run(_init_database())

        # Register startup/shutdown handlers with FastMCP/Starlette
        mcp.app.add_event_handler("startup", _startup_handler)
        mcp.app.add_event_handler("shutdown", _shutdown_handler)

        auth_endpoints = ["/auth/register", "/auth/jwt/login", "/auth/me"] if AUTH_ENABLED else []
        log.info("server_starting", transport="sse", host=args.host, port=args.port,
                 auth_enabled=AUTH_ENABLED, endpoints=auth_endpoints)
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        # Default STDIO transport (already configured above)
        log.info("server_starting", transport="stdio", auth_enabled=False)
        # CRITICAL: show_banner=False keeps stdout clean for JSON-RPC protocol
        mcp.run(show_banner=False)  # Default STDIO transport (no auth needed - local only)


if __name__ == "__main__":
    main()
