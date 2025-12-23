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
Delia Statistics Service

Thread-safe service for tracking model usage, task statistics, and response times.
All data is persisted to disk asynchronously.
"""

import asyncio
import json
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import structlog

from . import paths

log = structlog.get_logger()

# Constants for deque/list size limits
MAX_RECENT_CALLS = 50
MAX_RESPONSE_TIMES = 100
MAX_TOOL_HISTORY = 200


class StatsService:
    """
    Thread-safe statistics tracking service.

    Tracks:
    - Model usage (calls and tokens per tier)
    - Task statistics (counts by task type)
    - Recent calls (last N calls with metadata)
    - Response times (last N response times per tier)

    All operations are thread-safe using a threading.Lock.
    Persistence is async using aiofiles with atomic writes.
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the stats service.

        Args:
            data_dir: Optional data directory path. Defaults to paths.DATA_DIR.
        """
        self.data_dir = data_dir or paths.DATA_DIR

        # Stats files
        self.stats_file = paths.STATS_FILE
        self.enhanced_stats_file = paths.ENHANCED_STATS_FILE

        # Model usage tracking
        self.model_usage: dict[str, dict[str, int]] = {
            "quick": {"calls": 0, "tokens": 0},
            "coder": {"calls": 0, "tokens": 0},
            "moe": {"calls": 0, "tokens": 0},
            "thinking": {"calls": 0, "tokens": 0},
        }

        # Task type tracking
        self.task_stats: dict[str, int] = {
            "review": 0,
            "analyze": 0,
            "generate": 0,
            "plan": 0,
            "think": 0,
            "other": 0,
        }

        # Recent calls log (deque for O(1) append/pop)
        self.recent_calls: deque[dict] = deque(maxlen=MAX_RECENT_CALLS)

        # Response time tracking
        self.response_times: dict[str, list[dict[str, Any]]] = {
            "quick": [],
            "coder": [],
            "moe": [],
            "thinking": [],
        }

        # MCP Tool call tracking
        self.tool_calls: dict[str, dict[str, int]] = {}  # {tool_name: {calls, total_ms}}
        self.recent_tool_calls: deque[dict] = deque(maxlen=MAX_TOOL_HISTORY)

        # Thread safety
        self._lock = threading.Lock()
        self._save_lock = asyncio.Lock()

        # Ensure data directories exist
        paths.ensure_directories()

    def record_call(
        self,
        model_tier: str,
        task_type: str,
        original_task: str,
        tokens: int,
        elapsed_ms: int,
        content_preview: str,
        enable_thinking: bool,
        backend: str = "ollama",
        backend_type: str = "local",
    ) -> None:
        """
        Thread-safe recording of a model call.

        Updates all in-memory stats atomically under a single lock.

        Args:
            model_tier: Model tier (quick, coder, moe, thinking)
            task_type: Task type (review, analyze, generate, etc.)
            original_task: Original task description
            tokens: Number of tokens used
            elapsed_ms: Response time in milliseconds
            content_preview: Preview of the response content
            enable_thinking: Whether thinking mode was enabled
            backend: Backend name (ollama, llama.cpp, gemini)
            backend_type: Backend type (local, remote)
        """
        with self._lock:
            # Track model usage
            self.model_usage[model_tier]["calls"] += 1
            self.model_usage[model_tier]["tokens"] += tokens

            # Track task type
            if task_type in self.task_stats:
                self.task_stats[task_type] += 1
            else:
                self.task_stats["other"] += 1

            # Track response time
            timestamp = datetime.now().isoformat()
            self.response_times[model_tier].append({"ts": timestamp, "ms": elapsed_ms})
            if len(self.response_times[model_tier]) > MAX_RESPONSE_TIMES:
                self.response_times[model_tier] = self.response_times[model_tier][
                    -MAX_RESPONSE_TIMES:
                ]

            # Add to recent calls log
            self.recent_calls.append(
                {
                    "ts": timestamp,
                    "tier": model_tier,
                    "task": original_task,
                    "type": task_type,
                    "tokens": tokens,
                    "ms": elapsed_ms,
                    "preview": content_preview[:100],
                    "thinking": enable_thinking,
                    "backend": backend,
                    "backend_type": backend_type,
                }
            )

    def increment_task(self, task_type: str) -> None:
        """
        Thread-safe increment of a task counter.

        Args:
            task_type: Task type to increment (e.g., "think")
        """
        with self._lock:
            if task_type in self.task_stats:
                self.task_stats[task_type] += 1
            else:
                self.task_stats["other"] += 1

    def record_tool_call(
        self,
        tool_name: str,
        elapsed_ms: int,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """
        Thread-safe recording of an MCP tool call.

        Args:
            tool_name: Name of the MCP tool called
            elapsed_ms: Execution time in milliseconds
            success: Whether the tool call succeeded
            error: Error message if failed
        """
        with self._lock:
            # Initialize tool entry if needed
            if tool_name not in self.tool_calls:
                self.tool_calls[tool_name] = {"calls": 0, "total_ms": 0, "errors": 0}

            # Update aggregate stats
            self.tool_calls[tool_name]["calls"] += 1
            self.tool_calls[tool_name]["total_ms"] += elapsed_ms
            if not success:
                self.tool_calls[tool_name]["errors"] += 1

            # Add to recent history
            self.recent_tool_calls.append({
                "ts": datetime.now().isoformat(),
                "tool": tool_name,
                "ms": elapsed_ms,
                "success": success,
                "error": error,
            })

    def get_tool_stats(self) -> dict[str, dict]:
        """
        Get aggregated tool call statistics.

        Returns:
            Dict of tool stats with avg_ms calculated.
        """
        with self._lock:
            result = {}
            for tool_name, stats in self.tool_calls.items():
                result[tool_name] = {
                    "calls": stats["calls"],
                    "errors": stats["errors"],
                    "avg_ms": round(stats["total_ms"] / stats["calls"], 1) if stats["calls"] > 0 else 0,
                    "total_ms": stats["total_ms"],
                }
            return result

    def get_snapshot(self) -> tuple[dict, dict, dict, list]:
        """
        Take atomic snapshot of all in-memory stats under lock.

        Returns:
            Tuple of (model_usage, task_stats, response_times, recent_calls)
            All as deep copies to prevent external modifications.
        """
        with self._lock:
            # Create deep copies to prevent external modifications
            model_usage_snapshot = {
                tier: data.copy() for tier, data in self.model_usage.items()
            }
            task_stats_snapshot = self.task_stats.copy()
            response_times_snapshot = {
                tier: times.copy() for tier, times in self.response_times.items()
            }
            recent_calls_snapshot = list(self.recent_calls)

        return (
            model_usage_snapshot,
            task_stats_snapshot,
            response_times_snapshot,
            recent_calls_snapshot,
        )

    def _save_usage_stats_sync(self) -> None:
        """
        Save usage stats to disk synchronously (atomic write).

        Uses snapshot to ensure consistent data even with concurrent updates.
        """
        try:
            model_usage_snapshot, _, _, _ = self.get_snapshot()
            temp_file = self.stats_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(model_usage_snapshot, indent=2))
            temp_file.replace(self.stats_file)  # Atomic on POSIX
        except Exception as e:
            log.warning("stats_save_failed", error=str(e))

    def _save_enhanced_stats_sync(self) -> None:
        """
        Save enhanced stats to disk synchronously (atomic write).

        Uses snapshots to ensure consistent data even with concurrent updates.
        """
        try:
            _, task_stats_snapshot, response_times_snapshot, recent_calls_snapshot = (
                self.get_snapshot()
            )
            tool_stats_snapshot = self.get_tool_stats()
            recent_tools = list(self.recent_tool_calls)[-MAX_TOOL_HISTORY:]

            data = {
                "task_stats": task_stats_snapshot,
                "recent_calls": recent_calls_snapshot[-MAX_RECENT_CALLS:],
                "response_times": {
                    "quick": response_times_snapshot["quick"][-MAX_RESPONSE_TIMES:],
                    "coder": response_times_snapshot["coder"][-MAX_RESPONSE_TIMES:],
                    "moe": response_times_snapshot["moe"][-MAX_RESPONSE_TIMES:],
                },
                "tool_stats": tool_stats_snapshot,
                "recent_tool_calls": recent_tools,
            }
            temp_file = self.enhanced_stats_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2))
            temp_file.replace(self.enhanced_stats_file)  # Atomic on POSIX
        except Exception as e:
            log.warning("enhanced_stats_save_failed", error=str(e))

    async def save_all(self) -> None:
        """
        Save all stats asynchronously with proper locking.

        Uses two-level locking strategy:
        1. Threading lock in get_snapshot() to atomically read in-memory stats
        2. Async lock here to prevent concurrent writes to disk

        This ensures:
        - Each save gets a consistent snapshot of stats
        - Only one save operation writes to disk at a time
        - Updates during save don't cause data loss
        """
        async with self._save_lock:
            await asyncio.to_thread(self._save_usage_stats_sync)
            await asyncio.to_thread(self._save_enhanced_stats_sync)

    def load(self) -> None:
        """
        Load stats from disk on startup.

        Handles migration from legacy tier names and corrupt files gracefully.
        """
        # Load basic stats
        if self.stats_file.exists():
            try:
                data = json.loads(self.stats_file.read_text())
                # Migration: Support legacy keys from pre-refactor versions
                # These map old tier names to current tier names (one-time migration)
                legacy_key_mapping = {
                    "quick": ["quick", "14b"],  # Legacy: "14b" → "quick"
                    "coder": ["coder", "30b"],  # Legacy: "30b" → "coder" (now moe)
                    "moe": ["moe"],
                }
                with self._lock:
                    for new_key, legacy_keys in legacy_key_mapping.items():
                        for legacy_key in legacy_keys:
                            if legacy_key in data:
                                self.model_usage[new_key]["calls"] += data[legacy_key].get(
                                    "calls", 0
                                )
                                self.model_usage[new_key]["tokens"] += data[legacy_key].get(
                                    "tokens", 0
                                )
                log.info(
                    "stats_loaded",
                    quick_calls=self.model_usage["quick"]["calls"],
                    coder_calls=self.model_usage["coder"]["calls"],
                    moe_calls=self.model_usage["moe"]["calls"],
                )
            except json.JSONDecodeError as e:
                log.warning("stats_load_failed", error=str(e), reason="invalid_json")
            except Exception as e:
                log.warning("stats_load_failed", error=str(e))

        # Load enhanced stats
        if self.enhanced_stats_file.exists():
            try:
                data = json.loads(self.enhanced_stats_file.read_text())
                with self._lock:
                    self.task_stats.update(data.get("task_stats", {}))
                    self.recent_calls.clear()
                    self.recent_calls.extend(
                        data.get("recent_calls", [])[-MAX_RECENT_CALLS:]
                    )
                    rt = data.get("response_times", {})
                    # Migration: Map legacy tier names to current names
                    self.response_times["quick"] = rt.get("quick", rt.get("14b", []))[
                        -MAX_RESPONSE_TIMES:
                    ]
                    self.response_times["coder"] = rt.get("coder", rt.get("30b", []))[
                        -MAX_RESPONSE_TIMES:
                    ]
                    self.response_times["moe"] = rt.get("moe", [])[-MAX_RESPONSE_TIMES:]
                log.info("enhanced_stats_loaded", recent_calls=len(self.recent_calls))
            except json.JSONDecodeError as e:
                log.warning(
                    "enhanced_stats_load_failed", error=str(e), reason="invalid_json"
                )
            except Exception as e:
                log.warning("enhanced_stats_load_failed", error=str(e))
