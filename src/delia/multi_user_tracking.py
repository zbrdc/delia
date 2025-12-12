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
Simplified Multi-User Tracking for Delia MCP Server

Provides:
- In-memory client tracking with background persistence
- Rate limiting using the 'limits' library
- Transport-aware client identification (HTTP: IP+key, STDIO: session)
- Aggregated per-user statistics
- Periodic background saves (every 60s) instead of per-request I/O
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict
from contextlib import suppress
import hashlib

import structlog
from limits import storage, strategies, parse

from . import paths

# Lazy logger to handle STDIO reconfiguration
def _get_log():
    return structlog.get_logger()


# ============================================================
# CONFIGURATION
# ============================================================

# Rate limit defaults (can be overridden per-user from auth database)
DEFAULT_REQUESTS_PER_HOUR = 1000
DEFAULT_TOKENS_PER_HOUR = 1_000_000
DEFAULT_CONCURRENT_REQUESTS = 10

# Persistence settings
SAVE_INTERVAL_SECONDS = 60  # Background save interval
DATA_DIR = paths.USER_DATA_DIR


# ============================================================
# DATA MODELS (Simplified)
# ============================================================

@dataclass
class ClientInfo:
    """Lightweight client identification."""
    client_id: str
    username: str
    ip_address: str = ""  # For HTTP mode
    api_key_hash: str = ""  # For authenticated HTTP mode
    transport: str = "stdio"  # "stdio" or "http"
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def touch(self):
        """Update last activity."""
        self.last_seen = time.time()


@dataclass
class UserStats:
    """Aggregated statistics per user."""
    username: str
    total_requests: int = 0
    total_tokens: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time_ms: int = 0
    task_counts: Dict[str, int] = field(default_factory=dict)
    model_counts: Dict[str, int] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class QuotaConfig:
    """Per-user quota configuration."""
    max_requests_per_hour: int = DEFAULT_REQUESTS_PER_HOUR
    max_tokens_per_hour: int = DEFAULT_TOKENS_PER_HOUR
    max_concurrent: int = DEFAULT_CONCURRENT_REQUESTS
    max_model_tier: str = "moe"


# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    """
    Rate limiter using the 'limits' library with in-memory storage.
    Supports both request count and token-based limits.
    """

    def __init__(self):
        # In-memory storage (fast, no I/O)
        self._storage = storage.MemoryStorage()
        self._strategy = strategies.MovingWindowRateLimiter(self._storage)

        # Per-client quota configs
        self._quotas: Dict[str, QuotaConfig] = {}

        # Concurrent request tracking
        self._concurrent: Dict[str, int] = {}

    def set_quota(self, client_id: str, quota: QuotaConfig):
        """Set quota for a client."""
        self._quotas[client_id] = quota

    def get_quota(self, client_id: str) -> QuotaConfig:
        """Get quota for client, creating default if needed."""
        if client_id not in self._quotas:
            self._quotas[client_id] = QuotaConfig()
        return self._quotas[client_id]

    def check_rate_limit(self, client_id: str) -> tuple[bool, str]:
        """
        Check if client can proceed with a request.
        Returns (allowed, reason).
        """
        quota = self.get_quota(client_id)

        # Check concurrent requests
        current = self._concurrent.get(client_id, 0)
        if current >= quota.max_concurrent:
            return False, f"Max concurrent requests ({quota.max_concurrent}) reached"

        # Check requests per hour using limits library
        rate_limit = parse(f"{quota.max_requests_per_hour}/hour")
        request_key = f"req:{client_id}"

        if not self._strategy.hit(rate_limit, request_key):
            stats = self._strategy.get_window_stats(rate_limit, request_key)
            reset_in = int(stats.reset_time - time.time()) if stats.reset_time else 3600
            return False, f"Rate limit exceeded ({quota.max_requests_per_hour}/hour). Resets in {reset_in}s"

        return True, "OK"

    def check_token_budget(self, client_id: str, estimated_tokens: int = 0) -> tuple[bool, str]:
        """Check if client has token budget remaining."""
        quota = self.get_quota(client_id)

        # Use limits library for token tracking too
        token_limit = parse(f"{quota.max_tokens_per_hour}/hour")
        token_key = f"tok:{client_id}"

        stats = self._strategy.get_window_stats(token_limit, token_key)
        remaining = quota.max_tokens_per_hour - stats.remaining if stats else quota.max_tokens_per_hour

        if remaining + estimated_tokens > quota.max_tokens_per_hour:
            return False, f"Token limit would be exceeded ({remaining}/{quota.max_tokens_per_hour} used this hour)"

        return True, "OK"

    def record_tokens(self, client_id: str, tokens: int):
        """Record token usage after request completion."""
        quota = self.get_quota(client_id)
        token_limit = parse(f"{quota.max_tokens_per_hour}/hour")
        token_key = f"tok:{client_id}"

        # Record token usage (hit the limiter 'tokens' times, but efficiently)
        # The limits library doesn't support weighted hits, so we track separately
        # and just use it for the time window management
        for _ in range(min(tokens, 100)):  # Batch to avoid excessive calls
            self._strategy.hit(token_limit, token_key)

    def start_request(self, client_id: str):
        """Mark request as started (for concurrent tracking)."""
        self._concurrent[client_id] = self._concurrent.get(client_id, 0) + 1

    def end_request(self, client_id: str):
        """Mark request as completed."""
        if client_id in self._concurrent:
            self._concurrent[client_id] = max(0, self._concurrent[client_id] - 1)

    def get_stats(self, client_id: str) -> dict:
        """Get current rate limit stats for a client."""
        quota = self.get_quota(client_id)

        rate_limit = parse(f"{quota.max_requests_per_hour}/hour")
        request_key = f"req:{client_id}"
        stats = self._strategy.get_window_stats(rate_limit, request_key)

        return {
            "requests_remaining": stats.remaining if stats else quota.max_requests_per_hour,
            "requests_limit": quota.max_requests_per_hour,
            "concurrent_current": self._concurrent.get(client_id, 0),
            "concurrent_limit": quota.max_concurrent,
        }


# ============================================================
# MAIN TRACKER CLASS
# ============================================================

class SimpleTracker:
    """
    Simplified tracking with in-memory state and background persistence.
    No per-request I/O - saves periodically in background.
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._clients: Dict[str, ClientInfo] = {}
        self._stats: Dict[str, UserStats] = {}

        # Rate limiter
        self.rate_limiter = RateLimiter()

        # Persistence
        self._clients_file = self.data_dir / "clients.json"
        self._stats_file = self.data_dir / "user_stats.json"
        self._dirty = False  # Track if we need to save
        self._save_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Load existing data
        self._load_from_disk()

    # ----------------------------------------------------------
    # Client Management
    # ----------------------------------------------------------

    def get_or_create_client(
        self,
        username: str,
        ip_address: str = "",
        api_key: Optional[str] = None,
        transport: str = "stdio",
    ) -> ClientInfo:
        """Get existing client or create new one."""

        # Generate client ID based on transport
        if transport == "http" and (ip_address or api_key):
            # HTTP mode: identify by IP + API key hash
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:12] if api_key else ""
            client_id = f"http:{ip_address}:{key_hash}" if key_hash else f"http:{ip_address}"
        else:
            # STDIO mode: identify by username (session-based)
            client_id = f"stdio:{username}"

        # Return existing or create new
        if client_id in self._clients:
            client = self._clients[client_id]
            client.touch()
            return client

        # Create new client
        client = ClientInfo(
            client_id=client_id,
            username=username,
            ip_address=ip_address,
            api_key_hash=hashlib.sha256(api_key.encode()).hexdigest()[:12] if api_key else "",
            transport=transport,
        )

        self._clients[client_id] = client
        self._dirty = True

        _get_log().info("client_registered",
                       client_id=client_id,
                       username=username,
                       transport=transport)

        return client

    def get_client(self, client_id: str) -> Optional[ClientInfo]:
        """Get client by ID."""
        return self._clients.get(client_id)

    # ----------------------------------------------------------
    # Quota & Rate Limiting
    # ----------------------------------------------------------

    def check_quota(self, client_id: str, estimated_tokens: int = 0) -> tuple[bool, str]:
        """Check if client can proceed with request."""
        # Check rate limit
        allowed, msg = self.rate_limiter.check_rate_limit(client_id)
        if not allowed:
            return False, msg

        # Check token budget if estimate provided
        if estimated_tokens > 0:
            allowed, msg = self.rate_limiter.check_token_budget(client_id, estimated_tokens)
            if not allowed:
                return False, msg

        return True, "OK"

    def set_user_quota(self, client_id: str, quota: QuotaConfig):
        """Set quota for a user (called when auth provides user limits)."""
        self.rate_limiter.set_quota(client_id, quota)

    # ----------------------------------------------------------
    # Request Recording
    # ----------------------------------------------------------

    def start_request(self, client_id: str):
        """Called when a request starts."""
        self.rate_limiter.start_request(client_id)

    def record_request(
        self,
        client_id: str,
        task_type: str,
        model_tier: str = "",
        tokens: int = 0,
        elapsed_ms: int = 0,
        backend: str = "ollama",
        success: bool = True,
        error: str = "",
    ):
        """Record completed request (updates stats, no I/O)."""

        # End concurrent tracking
        self.rate_limiter.end_request(client_id)

        # Get client info
        client = self._clients.get(client_id)
        if not client:
            _get_log().warning("request_from_unknown_client", client_id=client_id)
            return

        username = client.username
        client.touch()

        # Record tokens in rate limiter
        if tokens > 0:
            self.rate_limiter.record_tokens(client_id, tokens)

        # Update user stats
        if username not in self._stats:
            self._stats[username] = UserStats(username=username)

        stats = self._stats[username]
        stats.total_requests += 1
        stats.total_tokens += tokens
        stats.total_time_ms += elapsed_ms
        stats.last_seen = time.time()

        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1

        # Track task types
        stats.task_counts[task_type] = stats.task_counts.get(task_type, 0) + 1

        # Track model usage
        if model_tier:
            stats.model_counts[model_tier] = stats.model_counts.get(model_tier, 0) + 1

        self._dirty = True

        _get_log().debug("request_recorded",
                        client_id=client_id,
                        task=task_type,
                        tokens=tokens,
                        success=success)

    def update_request_tokens(self, client_id: str, tokens: int, model_tier: str = ""):
        """Update token count for most recent request (called after LLM response)."""
        client = self._clients.get(client_id)
        if not client:
            return

        username = client.username
        if username in self._stats:
            # Just add the tokens to total (we recorded 0 initially)
            self._stats[username].total_tokens += tokens

            if model_tier:
                stats = self._stats[username]
                stats.model_counts[model_tier] = stats.model_counts.get(model_tier, 0) + 1

        # Record in rate limiter
        if tokens > 0:
            self.rate_limiter.record_tokens(client_id, tokens)

        self._dirty = True

    # ----------------------------------------------------------
    # Stats Access
    # ----------------------------------------------------------

    def get_user_stats(self, username: str) -> Optional[UserStats]:
        """Get stats for a user."""
        return self._stats.get(username)

    def get_all_stats(self) -> Dict[str, UserStats]:
        """Get all user stats."""
        return self._stats.copy()

    def get_active_clients(self, idle_timeout: int = 3600) -> list[ClientInfo]:
        """Get clients active within timeout window."""
        now = time.time()
        return [c for c in self._clients.values() if (now - c.last_seen) < idle_timeout]

    def get_rate_limit_stats(self, client_id: str) -> dict:
        """Get rate limit stats for a client."""
        return self.rate_limiter.get_stats(client_id)

    # ----------------------------------------------------------
    # Persistence (Background)
    # ----------------------------------------------------------

    async def start_background_save(self):
        """Start background save task."""
        if self._save_task is None:
            self._save_task = asyncio.create_task(self._background_save_loop())
            _get_log().info("background_save_started", interval=SAVE_INTERVAL_SECONDS)

    async def _background_save_loop(self):
        """Periodically save to disk."""
        while not self._shutdown:
            await asyncio.sleep(SAVE_INTERVAL_SECONDS)
            if self._dirty:
                await self._save_to_disk_async()

    async def shutdown(self):
        """Shutdown tracker, saving final state."""
        self._shutdown = True

        if self._save_task:
            self._save_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._save_task

        # Final save
        if self._dirty:
            await self._save_to_disk_async()

        _get_log().info("tracker_shutdown")

    def save_sync(self):
        """Synchronous save for atexit handler."""
        self._save_to_disk_sync()

    async def _save_to_disk_async(self):
        """Save state to disk (async)."""
        try:
            # Run sync I/O in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_to_disk_sync)
            self._dirty = False
            _get_log().debug("state_saved")
        except Exception as e:
            _get_log().error("save_error", error=str(e))

    def _save_to_disk_sync(self):
        """Save state to disk (sync)."""
        try:
            # Save clients
            clients_data = {
                cid: asdict(c) for cid, c in self._clients.items()
            }
            with open(self._clients_file, "w") as f:
                json.dump(clients_data, f, indent=2)

            # Save stats
            stats_data = {
                uname: asdict(s) for uname, s in self._stats.items()
            }
            with open(self._stats_file, "w") as f:
                json.dump(stats_data, f, indent=2)

            self._dirty = False
        except Exception as e:
            _get_log().error("save_error", error=str(e))

    def _load_from_disk(self):
        """Load state from disk on startup."""
        try:
            if self._clients_file.exists():
                with open(self._clients_file) as f:
                    data = json.load(f)
                    for cid, cdata in data.items():
                        self._clients[cid] = ClientInfo(**cdata)
                _get_log().info("clients_loaded", count=len(self._clients))

            if self._stats_file.exists():
                with open(self._stats_file) as f:
                    data = json.load(f)
                    for uname, sdata in data.items():
                        self._stats[uname] = UserStats(**sdata)
                _get_log().info("stats_loaded", count=len(self._stats))

        except Exception as e:
            _get_log().error("load_error", error=str(e))


# ============================================================
# BACKWARDS COMPATIBILITY LAYER
# ============================================================

class MultiUserTracker(SimpleTracker):
    """
    Backwards-compatible wrapper providing the old API.
    New code should use SimpleTracker directly.
    """

    def register_client(
        self,
        username: str,
        hostname: str = "",
        ip_address: str = "",
        user_agent: str = "",
        api_key: Optional[str] = None,
    ) -> ClientInfo:
        """Legacy method - creates new client."""
        transport = "http" if ip_address else "stdio"
        return self.get_or_create_client(
            username=username,
            ip_address=ip_address,
            api_key=api_key,
            transport=transport,
        )

    def get_client_info(self, client_id: str) -> Optional[ClientInfo]:
        """Legacy alias."""
        return self.get_client(client_id)

    def get_all_users(self) -> list[UserStats]:
        """Legacy method - returns list of stats."""
        return list(self._stats.values())

    def get_recent_requests(self, limit: int = 100, username: str = None) -> list:
        """Legacy method - no longer tracks individual requests."""
        # Return empty list - we no longer track individual requests
        return []

    def update_last_request(self, client_id: str, tokens: int = 0, model_tier: str = None):
        """Legacy method - now just updates tokens."""
        self.update_request_tokens(client_id, tokens, model_tier or "")


# Global instance (backwards compatible)
tracker = MultiUserTracker()
