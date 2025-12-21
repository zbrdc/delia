# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Logging Service for Delia - Handles dashboard log streaming and structlog configuration.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import structlog
from structlog.types import Processor

try:
    import aiofiles
except ImportError:
    aiofiles = None  # type: ignore


class LoggingService:
    """
    Handles logging configuration and dashboard log persistence.
    
    Eliminates globals: LIVE_LOGS, _live_logs_lock
    """

    def __init__(self, data_dir: Path, live_logs_file: Path, max_live_logs: int = 100):
        self.data_dir = data_dir
        self.live_logs_file = live_logs_file
        self.max_live_logs = max_live_logs
        self.live_logs: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._background_tasks: set[Any] = set()

    def load_live_logs(self) -> None:
        """Load live logs from disk into the buffer."""
        if self.live_logs_file.exists():
            try:
                content = self.live_logs_file.read_text()
                if content:
                    loaded = json.loads(content)[-self.max_live_logs :]
                    with self._lock:
                        self.live_logs = loaded
            except (json.JSONDecodeError, Exception) as e:
                # Use standard logging until structlog is configured
                logging.getLogger(__name__).debug(f"logs_load_failed: {e}")

    def dashboard_processor(self, logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Custom structlog processor that captures logs for dashboard streaming.
        """
        # Only process logs explicitly marked for dashboard
        log_type = event_dict.pop("log_type", None)
        if log_type:
            model = event_dict.pop("model", "")
            tokens = event_dict.pop("tokens", 0)
            message = event_dict.get("event", "")
            status_msg = event_dict.pop("status_msg", "")
            backend = event_dict.pop("backend", "")

            with self._lock:
                self.live_logs.append(
                    {
                        "ts": datetime.now().isoformat(),
                        "type": log_type,
                        "message": message,
                        "model": model,
                        "tokens": tokens,
                        "status_msg": status_msg,
                        "backend": backend,
                    }
                )
                if len(self.live_logs) > self.max_live_logs:
                    self.live_logs.pop(0)

            # Schedule async save if possible, otherwise sync
            self._trigger_save()

        return event_dict

    def _trigger_save(self) -> None:
        """Triggers log persistence."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self.save_live_logs_async())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            # No event loop, fallback to sync
            self.save_live_logs_sync()

    def save_live_logs_sync(self) -> None:
        """Save live logs to disk synchronously."""
        try:
            temp_file = self.live_logs_file.with_suffix(".tmp")
            with self._lock:
                content = json.dumps(self.live_logs[-self.max_live_logs :], indent=2)
            temp_file.write_text(content)
            temp_file.replace(self.live_logs_file)
        except Exception:
            pass

    async def save_live_logs_async(self) -> None:
        """Save live logs to disk asynchronously."""
        try:
            temp_file = self.live_logs_file.with_suffix(".tmp")
            with self._lock:
                content = json.dumps(self.live_logs[-self.max_live_logs :], indent=2)

            if aiofiles:
                async with aiofiles.open(temp_file, "w") as f:
                    await f.write(content)
            else:
                temp_file.write_text(content)

            temp_file.replace(self.live_logs_file)
        except Exception:
            pass

    def configure_structlog(self, use_stderr: bool = False) -> None:
        """
        Configure structlog with console + dashboard output.
        """
        is_debug = os.environ.get("DELIA_DEBUG", "").lower() in ("true", "1", "yes")
        log_level = logging.DEBUG if is_debug else logging.INFO

        structlog.reset_defaults()

        base_processors: list[Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            self.dashboard_processor,
        ]

        if use_stderr and not is_debug:
            # Silent factory for STDIO mode
            class SilentLoggerFactory:
                def __call__(self):
                    class SilentLogger:
                        def msg(self, *args, **kwargs): pass
                        def __getattr__(self, name): return self.msg
                    return SilentLogger()

            structlog.configure(
                processors=base_processors,
                wrapper_class=structlog.make_filtering_bound_logger(log_level),
                context_class=dict,
                logger_factory=SilentLoggerFactory(),
                cache_logger_on_first_use=False,
            )
        else:
            processors = base_processors + [structlog.dev.ConsoleRenderer(colors=True)]
            target_file = sys.stderr if use_stderr else sys.stdout

            structlog.configure(
                processors=processors,
                wrapper_class=structlog.make_filtering_bound_logger(log_level),
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(file=target_file),
                cache_logger_on_first_use=True,
            )


_logging_service: LoggingService | None = None

def get_logging_service() -> LoggingService:
    global _logging_service
    if _logging_service is None:
        from .paths import DATA_DIR, LIVE_LOGS_FILE
        _logging_service = LoggingService(DATA_DIR, LIVE_LOGS_FILE)
    return _logging_service
