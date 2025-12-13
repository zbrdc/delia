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
Delia Response Cache

Hybrid in-memory + file-based cache for LLM responses with TTL expiration and LRU eviction.
Thread-safe implementation for concurrent access.
"""

import hashlib
import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import structlog

from . import paths

log = structlog.get_logger()

# Cache configuration constants
CACHE_TTL_SECONDS = 3600  # 1 hour
CACHE_MAX_ENTRIES = 2000


@dataclass
class CachedResponse:
    """
    Represents a cached LLM response.

    Attributes:
        response: The LLM response text
        tokens: Number of tokens used
        timestamp: ISO timestamp when response was cached
        model: Model name/tier used
        hits: Number of cache hits for this entry
        ttl_seconds: Time-to-live in seconds
    """

    response: str
    tokens: int
    timestamp: str
    model: str
    hits: int
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if this cached response has expired."""
        cached_time = datetime.fromisoformat(self.timestamp)
        expiry_time = cached_time + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CachedResponse":
        """Create CachedResponse from dictionary."""
        return cls(**data)


class ResponseCache:
    """
    Thread-safe hybrid cache for LLM responses.

    Features:
    - In-memory dict for fast lookups
    - TTL expiration (default 1 hour)
    - LRU eviction when exceeding max entries
    - Thread-safe operations with threading.Lock
    - Hit/miss statistics tracking
    """

    def __init__(
        self,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        max_entries: int = CACHE_MAX_ENTRIES,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the response cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            max_entries: Maximum number of entries before LRU eviction
            cache_dir: Optional cache directory path. Defaults to paths.CACHE_DIR
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cache_dir = cache_dir or paths.CACHE_DIR

        # In-memory cache storage
        self._memory_cache: dict[str, CachedResponse] = {}

        # Statistics tracking
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Thread safety
        self._lock = threading.Lock()

        # Ensure cache directory exists
        paths.ensure_directories()

        log.info(
            "response_cache_initialized",
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            cache_dir=str(self.cache_dir),
        )

    def _compute_key(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Compute a unique cache key using SHA-256 hash of parameters.

        Args:
            model: Model name/tier
            prompt: User prompt
            system: Optional system prompt
            enable_thinking: Whether thinking mode is enabled
            max_tokens: Optional max tokens limit

        Returns:
            SHA-256 hex digest as cache key
        """
        # Normalize parameters for consistent hashing
        key_parts = [
            f"model:{model}",
            f"prompt:{prompt}",
            f"system:{system or ''}",
            f"thinking:{enable_thinking}",
            f"max_tokens:{max_tokens or 'none'}",
        ]
        key_string = "|".join(key_parts)

        # Hash for compact key
        return hashlib.sha256(key_string.encode("utf-8")).hexdigest()

    def _evict_lru(self) -> None:
        """
        Evict least recently used entries (oldest 10% by hits+timestamp).

        This method should be called when the cache exceeds max_entries.
        Must be called while holding self._lock.
        """
        if len(self._memory_cache) <= self.max_entries:
            return

        # Calculate number of entries to evict (10%)
        evict_count = max(1, len(self._memory_cache) // 10)

        # Score entries by hits (lower is worse) and age (older is worse)
        # Combine hits and recency into a single score
        scored_entries = []
        for key, cached in self._memory_cache.items():
            cached_time = datetime.fromisoformat(cached.timestamp)
            age_seconds = (datetime.now() - cached_time).total_seconds()

            # Score: higher is better (more hits, more recent)
            # Normalize age to 0-1 range (assuming max 1 hour TTL)
            recency_score = 1.0 - min(age_seconds / self.ttl_seconds, 1.0)
            score = cached.hits + recency_score

            scored_entries.append((score, key))

        # Sort by score (lowest first) and take bottom evict_count entries
        scored_entries.sort()
        keys_to_evict = [key for _, key in scored_entries[:evict_count]]

        # Remove evicted entries
        for key in keys_to_evict:
            del self._memory_cache[key]
            self._evictions += 1

        log.debug(
            "cache_eviction",
            evicted=len(keys_to_evict),
            remaining=len(self._memory_cache),
            total_evictions=self._evictions,
        )

    def get(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Retrieve a cached response if available and not expired.

        Thread-safe operation. Increments hit count on cache hit.

        Args:
            model: Model name/tier
            prompt: User prompt
            system: Optional system prompt
            enable_thinking: Whether thinking mode is enabled
            max_tokens: Optional max tokens limit

        Returns:
            Cached response text if found and valid, None otherwise
        """
        key = self._compute_key(model, prompt, system, enable_thinking, max_tokens)

        with self._lock:
            cached = self._memory_cache.get(key)

            if cached is None:
                self._misses += 1
                log.debug("cache_miss", key_prefix=key[:16])
                return None

            # Check expiration
            if cached.is_expired():
                del self._memory_cache[key]
                self._misses += 1
                log.debug("cache_expired", key_prefix=key[:16])
                return None

            # Cache hit - increment counter
            cached.hits += 1
            self._hits += 1
            log.debug(
                "cache_hit",
                key_prefix=key[:16],
                hits=cached.hits,
                tokens_saved=cached.tokens,
            )
            return cached.response

    def put(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens: int,
        system: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Store a response in the cache.

        Thread-safe operation. Triggers LRU eviction if cache is full.

        Args:
            model: Model name/tier
            prompt: User prompt
            response: LLM response to cache
            tokens: Number of tokens used
            system: Optional system prompt
            enable_thinking: Whether thinking mode was enabled
            max_tokens: Optional max tokens limit
        """
        key = self._compute_key(model, prompt, system, enable_thinking, max_tokens)

        with self._lock:
            # Create new cache entry
            cached_response = CachedResponse(
                response=response,
                tokens=tokens,
                timestamp=datetime.now().isoformat(),
                model=model,
                hits=0,
                ttl_seconds=self.ttl_seconds,
            )

            # Store in memory
            self._memory_cache[key] = cached_response

            # Check if eviction is needed
            if len(self._memory_cache) > self.max_entries:
                self._evict_lru()

            log.debug(
                "cache_put",
                key_prefix=key[:16],
                model=model,
                tokens=tokens,
                cache_size=len(self._memory_cache),
            )

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Thread-safe operation.

        Returns:
            Dictionary with:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Hit rate as percentage (0-100)
            - entries: Current number of cached entries
            - evictions: Total number of evictions
            - max_entries: Maximum allowed entries
            - ttl_seconds: TTL in seconds
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "entries": len(self._memory_cache),
                "evictions": self._evictions,
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }

    def clear(self) -> None:
        """
        Clear all cached entries.

        Thread-safe operation. Resets statistics.
        """
        with self._lock:
            cleared_count = len(self._memory_cache)
            self._memory_cache.clear()
            log.info("cache_cleared", entries_cleared=cleared_count)

    def save_to_disk(self, filepath: Optional[Path] = None) -> None:
        """
        Save cache to disk as JSON file (atomic write).

        Thread-safe operation.

        Args:
            filepath: Optional custom file path. Defaults to cache_dir/response_cache.json
        """
        if filepath is None:
            filepath = self.cache_dir / "response_cache.json"

        with self._lock:
            # Convert cache to serializable format
            cache_data = {
                "entries": {
                    key: cached.to_dict() for key, cached in self._memory_cache.items()
                },
                "stats": {
                    "hits": self._hits,
                    "misses": self._misses,
                    "evictions": self._evictions,
                },
            }

            # Atomic write using temp file
            try:
                temp_file = filepath.with_suffix(".tmp")
                temp_file.write_text(json.dumps(cache_data, indent=2))
                temp_file.replace(filepath)  # Atomic on POSIX
                log.info("cache_saved", filepath=str(filepath), entries=len(self._memory_cache))
            except Exception as e:
                log.warning("cache_save_failed", error=str(e), filepath=str(filepath))

    def load_from_disk(self, filepath: Optional[Path] = None) -> None:
        """
        Load cache from disk JSON file.

        Thread-safe operation. Skips expired entries during load.

        Args:
            filepath: Optional custom file path. Defaults to cache_dir/response_cache.json
        """
        if filepath is None:
            filepath = self.cache_dir / "response_cache.json"

        if not filepath.exists():
            log.debug("cache_load_skipped", reason="file_not_found", filepath=str(filepath))
            return

        with self._lock:
            try:
                cache_data = json.loads(filepath.read_text())

                # Load entries, filtering out expired ones
                entries = cache_data.get("entries", {})
                loaded_count = 0
                expired_count = 0

                for key, entry_dict in entries.items():
                    cached = CachedResponse.from_dict(entry_dict)
                    if not cached.is_expired():
                        self._memory_cache[key] = cached
                        loaded_count += 1
                    else:
                        expired_count += 1

                # Load statistics
                stats = cache_data.get("stats", {})
                self._hits = stats.get("hits", 0)
                self._misses = stats.get("misses", 0)
                self._evictions = stats.get("evictions", 0)

                log.info(
                    "cache_loaded",
                    filepath=str(filepath),
                    loaded=loaded_count,
                    expired_filtered=expired_count,
                )
            except json.JSONDecodeError as e:
                log.warning(
                    "cache_load_failed",
                    error=str(e),
                    reason="invalid_json",
                    filepath=str(filepath),
                )
            except Exception as e:
                log.warning("cache_load_failed", error=str(e), filepath=str(filepath))


# Global cache instance
_global_cache: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    """
    Get or create the global response cache instance.

    Returns:
        Global ResponseCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ResponseCache()
        # Attempt to load from disk on first access
        _global_cache.load_from_disk()
    return _global_cache
