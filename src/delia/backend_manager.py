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
Backend Manager - Unified configuration from settings.json

Handles loading, managing, and health-checking backends defined in settings.json.
The WebGUI and CLI tools update this file directly.
"""
import json
import httpx
import asyncio
import time
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
import structlog

from . import paths

# Health check cache TTL in seconds
HEALTH_CHECK_TTL = 30

# Get logger lazily each time to ensure we use the current configuration
# (important for STDIO transport where logging is reconfigured after import)
def _get_log():
    return structlog.get_logger()

log = None  # Will be set to actual logger on first use

# Path to the unified configuration file
SETTINGS_FILE = paths.SETTINGS_FILE


@dataclass
class BackendConfig:
    """Configuration for a single backend."""
    id: str
    name: str
    provider: str  # "ollama", "llamacpp", "openai", etc.
    type: str  # "local" or "remote"
    url: str
    enabled: bool = True
    priority: int = 0
    models: dict[str, str] = field(default_factory=dict)
    health_endpoint: str = "/health"
    models_endpoint: str = "/v1/models"
    chat_endpoint: str = "/v1/chat/completions"
    context_limit: int = 4096
    timeout_seconds: float = 300.0
    connect_timeout: float = 10.0
    api_key: Optional[str] = None

    # Runtime state (not persisted)
    _available: bool = False
    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    def from_dict(cls, data: dict) -> "BackendConfig":
        """Create a BackendConfig from a dictionary."""
        return cls(
            id=data.get("id", "unknown"),
            name=data.get("name", "Unknown Backend"),
            provider=data.get("provider", "unknown"),
            type=data.get("type", "local"),
            url=data.get("url", "http://localhost:8080"),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 0),
            models=data.get("models", {}),
            health_endpoint=data.get("health_endpoint", "/health"),
            models_endpoint=data.get("models_endpoint", "/v1/models"),
            chat_endpoint=data.get("chat_endpoint", "/v1/chat/completions"),
            context_limit=data.get("context_limit", 4096),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            connect_timeout=data.get("connect_timeout", 10.0),
            api_key=data.get("api_key"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "type": self.type,
            "url": self.url,
            "enabled": self.enabled,
            "priority": self.priority,
            "models": self.models,
            "health_endpoint": self.health_endpoint,
            "models_endpoint": self.models_endpoint,
            "chat_endpoint": self.chat_endpoint,
            "context_limit": self.context_limit,
            "timeout_seconds": self.timeout_seconds,
            "connect_timeout": self.connect_timeout,
            "api_key": self.api_key,
        }

    def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for this backend."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.url,
                timeout=httpx.Timeout(self.timeout_seconds, connect=self.connect_timeout),
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
        return self._client

    async def close_client(self) -> None:
        """
        Properly close and cleanup the HTTP client.

        This prevents connection leaks and file descriptor exhaustion.
        Must be awaited to ensure complete cleanup before creating new client.
        """
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                _get_log().warning("client_close_failed", backend_id=self.id, error=str(e))
            finally:
                self._client = None

    async def check_health(self) -> bool:
        """Check if this backend is available."""
        if not self.enabled:
            self._available = False
            return False

        client = self.get_client()
        try:
            # Try the configured health endpoint
            response = await client.get(self.health_endpoint)
            if response.status_code == 200:
                self._available = True
                return True

            # Fallback: try models endpoint
            response = await client.get(self.models_endpoint)
            if response.status_code == 200:
                self._available = True
                return True

            # Ollama-specific: try /api/tags
            if self.provider == "ollama":
                response = await client.get("/api/tags")
                if response.status_code == 200:
                    self._available = True
                    return True

            # Gemini-specific health check
            if self.provider == "gemini":
                try:
                    import google.generativeai as genai
                    import os
                    
                    api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
                    if not api_key:
                        _get_log().warning("gemini_health_check_failed", reason="no_api_key")
                        self._available = False
                        return False
                        
                    genai.configure(api_key=api_key)
                    # Lightweight check - list models
                    # Run in thread to avoid blocking async loop
                    await asyncio.to_thread(genai.list_models)
                    self._available = True
                    return True
                except ImportError:
                    _get_log().warning("gemini_dependency_missing", instruction="Run 'uv add google-generativeai' to enable")
                    self._available = False
                    return False
                except Exception as e:
                    _get_log().warning("gemini_health_check_failed", error=str(e))
                    self._available = False
                    return False

            self._available = False
            return False
        except Exception as e:
            print(f"Health check for {self.id} failed with error: {e}")
            _get_log().debug("backend_health_check_failed", backend_id=self.id, error=str(e))
            self._available = False
            return False

    @property
    def available(self) -> bool:
        """Whether this backend is currently available."""
        return self._available


class BackendManager:
    """
    Manages all configured backends from settings.json.

    This is the single source of truth for backend configuration.
    The WebGUI updates settings.json, and this class reloads it.
    """

    def __init__(self, settings_file: Path = SETTINGS_FILE):
        self.settings_file = settings_file
        self.backends: dict[str, BackendConfig] = {}
        self.routing_config: dict = {}
        self.system_config: dict = {}
        self.models_config: dict = {}
        self._active_backend_id: Optional[str] = None

        # Health check cache with TTL
        self._health_cache: dict[str, bool] = {}
        self._health_cache_time: float = 0.0

        self._load_settings()

    def _load_settings(self) -> None:
        """Load settings from the JSON file."""
        if not self.settings_file.exists():
            _get_log().warning("settings_file_not_found", path=str(self.settings_file))
            self._create_default_settings()
            return

        try:
            with open(self.settings_file) as f:
                data = json.load(f)

            # Load backends
            self.backends = {}
            for backend_data in data.get("backends", []):
                backend = BackendConfig.from_dict(backend_data)
                self.backends[backend.id] = backend

            # Load other configs
            self.routing_config = data.get("routing", {})
            self.system_config = data.get("system", {})
            self.models_config = data.get("models", {})

            # Set active backend (first enabled one with highest priority)
            enabled = [b for b in self.backends.values() if b.enabled]
            if enabled:
                enabled.sort(key=lambda b: b.priority)
                self._active_backend_id = enabled[0].id

            _get_log().info("settings_loaded",
                     backends=len(self.backends),
                     enabled=[b.id for b in self.backends.values() if b.enabled])

        except Exception as e:
            _get_log().error("settings_load_failed", error=str(e))
            self._create_default_settings()

    def _create_default_settings(self) -> None:
        """Create default settings file."""
        default = {
            "version": "1.0",
            "system": {
                "gpu_memory_limit_gb": 8,
                "memory_buffer_gb": 1,
                "max_concurrent_requests_per_backend": 1,
            },
            "backends": [],
            "routing": {
                "prefer_local": True,
                "fallback_enabled": False,
                "load_balance": False,
            },
            "models": {},
            "auth": {
                "enabled": False,
                "tracking_enabled": True,
            }
        }
        self.save_settings(default)

    def save_settings(self, data: Optional[dict] = None) -> bool:
        """Save current settings to the JSON file."""
        if data is None:
            data = {
                "version": "1.0",
                "system": self.system_config,
                "backends": [b.to_dict() for b in self.backends.values()],
                "routing": self.routing_config,
                "models": self.models_config,
            }

        try:
            with open(self.settings_file, "w") as f:
                json.dump(data, f, indent=2)
            _get_log().info("settings_saved", path=str(self.settings_file))
            return True
        except Exception as e:
            _get_log().error("settings_save_failed", error=str(e))
            return False

    async def reload(self) -> None:
        """
        Reload settings from disk.

        Properly closes all existing clients before loading new settings.
        This prevents connection leaks when backends change.

        Must be awaited to ensure all clients are cleaned up before reload completes.
        """
        # Close all existing clients and wait for cleanup
        close_tasks = [
            backend.close_client()
            for backend in self.backends.values()
        ]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Now load new settings
        self._load_settings()

    def get_enabled_backends(self) -> list[BackendConfig]:
        """Get all enabled backends, sorted by priority."""
        enabled = [b for b in self.backends.values() if b.enabled]
        enabled.sort(key=lambda b: b.priority)
        return enabled

    def get_backend(self, backend_id: str) -> Optional[BackendConfig]:
        """Get a specific backend by ID."""
        return self.backends.get(backend_id)

    def get_active_backend(self) -> Optional[BackendConfig]:
        """Get the currently active backend."""
        if self._active_backend_id:
            return self.backends.get(self._active_backend_id)

        # Fall back to first enabled backend
        enabled = self.get_enabled_backends()
        return enabled[0] if enabled else None

    def set_active_backend(self, backend_id: str) -> bool:
        """Set the active backend by ID."""
        if backend_id in self.backends:
            self._active_backend_id = backend_id
            _get_log().info("active_backend_changed", backend_id=backend_id)
            return True
        return False

    async def check_all_health(self, use_cache: bool = True) -> dict[str, bool]:
        """
        Check health of all enabled backends.

        Args:
            use_cache: If True, return cached results if within TTL (default: True).
                      Set to False to force fresh health checks.

        Returns:
            Dict mapping backend_id to health status (True/False).
        """
        # Return cached results if still valid
        now = time.time()
        if use_cache and self._health_cache and (now - self._health_cache_time) < HEALTH_CHECK_TTL:
            _get_log().debug("health_check_cached", age_seconds=int(now - self._health_cache_time))
            return self._health_cache.copy()

        enabled = self.get_enabled_backends()
        if not enabled:
            return {}

        # Check all in parallel
        tasks = [backend.check_health() for backend in enabled]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health = {}
        for backend, result in zip(enabled, results):
            if isinstance(result, Exception):
                health[backend.id] = False
                _get_log().warning("health_check_exception", backend_id=backend.id, error=str(result))
            else:
                health[backend.id] = result

        # Update cache
        self._health_cache = health.copy()
        self._health_cache_time = now

        return health

    def invalidate_health_cache(self) -> None:
        """Invalidate the health check cache, forcing fresh checks on next call."""
        self._health_cache = {}
        self._health_cache_time = 0.0

    async def get_health_status(self) -> dict:
        """Get comprehensive health status for all backends."""
        health = await self.check_all_health()

        backends_status = []
        for backend in self.backends.values():
            backends_status.append({
                "id": backend.id,
                "name": backend.name,
                "provider": backend.provider,
                "type": backend.type,
                "url": backend.url,
                "enabled": backend.enabled,
                "available": health.get(backend.id, False) if backend.enabled else False,
                "models": backend.models,
            })

        # Determine overall status
        active = self.get_active_backend()
        active_available = health.get(active.id, False) if active else False
        any_available = any(health.values())

        if active_available:
            status = "healthy"
        elif any_available:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "active_backend": active.id if active else None,
            "backends": backends_status,
            "routing": self.routing_config,
        }

    def add_backend(self, backend_data: dict) -> BackendConfig:
        """Add a new backend from the WebGUI."""
        backend = BackendConfig.from_dict(backend_data)
        self.backends[backend.id] = backend
        self.save_settings()
        _get_log().info("backend_added", backend_id=backend.id, name=backend.name)
        return backend

    async def update_backend(self, backend_id: str, updates: dict) -> Optional[BackendConfig]:
        """
        Update an existing backend.

        Properly closes and recreates client if URL changes.
        Must be awaited to ensure client cleanup before updates apply.
        """
        if backend_id not in self.backends:
            return None

        backend = self.backends[backend_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(backend, key) and not key.startswith("_"):
                setattr(backend, key, value)

        # Reset client if URL changed (forces recreation on next access)
        if "url" in updates and backend._client:
            await backend.close_client()

        self.save_settings()
        _get_log().info("backend_updated", backend_id=backend_id)
        return backend

    async def remove_backend(self, backend_id: str) -> bool:
        """
        Remove a backend.

        Properly closes the client before removing the backend from the configuration.
        Must be awaited to ensure proper cleanup.
        """
        if backend_id not in self.backends:
            return False

        backend = self.backends[backend_id]

        # Close client properly before removal
        await backend.close_client()

        del self.backends[backend_id]

        # Update active backend if needed
        if self._active_backend_id == backend_id:
            enabled = self.get_enabled_backends()
            self._active_backend_id = enabled[0].id if enabled else None

        self.save_settings()
        _get_log().info("backend_removed", backend_id=backend_id)
        return True


# Global instance
backend_manager = BackendManager()


def get_backend_manager() -> BackendManager:
    """Get the global backend manager instance."""
    return backend_manager


async def shutdown_backends() -> None:
    """
    Gracefully shutdown all backends and close their clients.

    This should be called during server shutdown to prevent connection leaks.
    Waits for all clients to properly close before returning.

    Example usage in FastMCP/Starlette:
        @app.on_event("shutdown")
        async def shutdown():
            await shutdown_backends()
    """
    backends = list(backend_manager.backends.values())
    if not backends:
        return

    _get_log().info("backends_shutdown_starting", count=len(backends))

    close_tasks = [backend.close_client() for backend in backends]
    results = await asyncio.gather(*close_tasks, return_exceptions=True)

    # Log any errors but don't fail
    for backend, result in zip(backends, results):
        if isinstance(result, Exception):
            _get_log().warning("backend_close_failed", backend_id=backend.id, error=str(result))

    _get_log().info("backends_shutdown_complete", closed=len(backends))
