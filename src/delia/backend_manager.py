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
Backend Manager - Unified configuration from settings.json

Handles loading, managing, and health-checking backends defined in settings.json.
The WebGUI and CLI tools update this file directly.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import structlog

# Optional dependency - imported lazily in check_health()
genai: Any = None
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    pass

from . import paths

# Health check cache TTL in seconds
HEALTH_CHECK_TTL = 30


# Get logger lazily each time to ensure we use the current configuration
# (important for STDIO transport where logging is reconfigured after import)
def _get_log() -> structlog.stdlib.BoundLogger:
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
    api_key: str | None = None
    supports_native_tool_calling: bool = False

    # Runtime state (not persisted)
    _available: bool = False
    _client: httpx.AsyncClient | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackendConfig":
        """Create a BackendConfig from a dictionary."""
        provider = data.get("provider", "unknown")
        
        # Auto-detect native tool calling if not explicitly set
        if "supports_native_tool_calling" in data:
            supports_native = data["supports_native_tool_calling"]
        else:
            capabilities = cls.detect_capabilities(provider)
            supports_native = capabilities["supports_native_tool_calling"]
        
        return cls(
            id=data.get("id", "unknown"),
            name=data.get("name", "Unknown Backend"),
            provider=provider,
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
            supports_native_tool_calling=supports_native,
        )

    def to_dict(self) -> dict[str, Any]:
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
            "supports_native_tool_calling": self.supports_native_tool_calling,
        }

    @classmethod
    def detect_capabilities(cls, provider: str) -> dict[str, Any]:
        """
        Detect capabilities for a given provider.

        Returns a dictionary of capability flags based on the provider type.

        Args:
            provider: Backend provider name ("llamacpp", "lmstudio", "openai", "ollama", "gemini")

        Returns:
            Dictionary with capability flags:
            - supports_native_tool_calling: Whether the provider supports native tool/function calling
        """
        # Providers that support native tool calling via OpenAI-compatible API
        tool_calling_providers = {"llamacpp", "lmstudio", "openai"}

        return {
            "supports_native_tool_calling": provider.lower() in tool_calling_providers,
        }

    def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for this backend."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.url,
                timeout=httpx.Timeout(self.timeout_seconds, connect=self.connect_timeout),
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
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
                if not GEMINI_AVAILABLE:
                    _get_log().warning(
                        "gemini_dependency_missing", instruction="Run 'uv add google-generativeai' to enable"
                    )
                    self._available = False
                    return False

                # Use configured api_key, fall back to env var if not set
                api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    _get_log().warning("gemini_health_check_failed", reason="no_api_key")
                    self._available = False
                    return False

                try:
                    genai.configure(api_key=api_key)
                    # Lightweight check - list models
                    # Run in thread to avoid blocking async loop
                    await asyncio.to_thread(genai.list_models)
                    self._available = True
                    return True
                except Exception as e:
                    _get_log().warning("gemini_health_check_failed", error=str(e))
                    self._available = False
                    return False

            self._available = False
            return False
        except Exception as e:
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
        self.routing_config: dict[str, Any] = {}
        self.system_config: dict[str, Any] = {}
        self.models_config: dict[str, Any] = {}
        self.mcp_servers_config: list[dict[str, Any]] = []  # MCP server configurations
        self._active_backend_id: str | None = None

        # Health check cache with TTL
        self._health_cache: dict[str, bool] = {}
        self._health_cache_time: float = 0.0
        self._health_check_lock = asyncio.Lock()

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
            self.mcp_servers_config = data.get("mcp_servers", [])

            # Set active backend (first enabled one with highest priority)
            enabled = [b for b in self.backends.values() if b.enabled]
            if enabled:
                enabled.sort(key=lambda b: b.priority)
                self._active_backend_id = enabled[0].id

            _get_log().info(
                "settings_loaded",
                backends=len(self.backends),
                enabled=[b.id for b in self.backends.values() if b.enabled],
                mcp_servers=len(self.mcp_servers_config),
            )

        except Exception as e:
            _get_log().error("settings_load_failed", error=str(e))
            self._create_default_settings()

    def _create_default_settings(self) -> None:
        """Create default settings file with auto-detected backends."""
        default: dict[str, Any] = {
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
                "scoring": {
                    "latency": 0.35,
                    "throughput": 0.15,
                    "reliability": 0.35,
                    "availability": 0.15,
                    "cost": 0.0,
                },
                "hedging": {
                    "enabled": False,
                    "delay_ms": 50,
                    "max_backends": 2,
                },
                "prewarm": {
                    "enabled": False,
                    "threshold": 0.3,
                    "check_interval_minutes": 5,
                },
            },
            "models": {},
            "auth": {
                "enabled": False,
                "tracking_enabled": True,
            },
            "mcp_servers": [],  # External MCP servers for tool passthrough
        }

        # Try to auto-detect available backends
        detected = self._detect_available_backends()
        if detected:
            default["backends"] = detected
            _get_log().info("backends_auto_detected", count=len(detected))

        self.save_settings(default)

    def _detect_available_backends(self) -> list[dict[str, Any]]:
        """
        Auto-detect available LLM backends on common ports.

        Probes:
        - localhost:8080 (llama.cpp default)
        - localhost:11434 (Ollama default)

        Returns list of backend config dicts ready for settings.json.
        """
        detected = []

        # Common endpoints to probe
        endpoints = [
            {
                "url": "http://localhost:8080",
                "provider": "llamacpp",
                "name": "Local llama.cpp",
                "health": "/health",
                "models": "/v1/models",
            },
            {
                "url": "http://localhost:1234",
                "provider": "lmstudio",
                "name": "Local LM Studio",
                "health": "/v1/models",
                "models": "/v1/models",
            },
            {
                "url": "http://localhost:11434",
                "provider": "ollama",
                "name": "Local Ollama",
                "health": "/api/tags",
                "models": "/api/tags",
            },
        ]

        for endpoint in endpoints:
            try:
                backend = self._probe_endpoint(endpoint)
                if backend:
                    detected.append(backend)
            except Exception as e:
                _get_log().debug("endpoint_probe_failed", url=endpoint["url"], error=str(e))

        return detected

    def _probe_endpoint(self, endpoint: dict[str, Any]) -> dict[str, Any] | None:
        """
        Probe a single endpoint and return backend config if available.

        Queries the models endpoint to discover available models and
        intelligently assigns them to tiers.
        """
        import httpx

        url = endpoint["url"]
        try:
            with httpx.Client(timeout=5.0) as client:
                # Check if server is responding
                health_resp = client.get(f"{url}{endpoint['health']}")
                if health_resp.status_code != 200:
                    return None

                # Get available models
                models_resp = client.get(f"{url}{endpoint['models']}")
                if models_resp.status_code != 200:
                    return None

                data = models_resp.json()
                available_models = self._parse_models_response(data, endpoint["provider"])

                if not available_models:
                    return None

                # Assign models to tiers
                tier_assignments = self._assign_models_to_tiers(available_models)

                # Detect capabilities for this provider
                capabilities = BackendConfig.detect_capabilities(endpoint["provider"])

                backend_id = f"{endpoint['provider']}-local"
                return {
                    "id": backend_id,
                    "name": endpoint["name"],
                    "provider": endpoint["provider"],
                    "type": "local",
                    "url": url,
                    "enabled": True,
                    "priority": 0 if endpoint["provider"] == "llamacpp" else 1,
                    "models": tier_assignments,
                    "health_endpoint": endpoint["health"],
                    "models_endpoint": endpoint["models"],
                    "supports_native_tool_calling": capabilities["supports_native_tool_calling"],
                }

        except (httpx.ConnectError, httpx.TimeoutException):
            return None
        except Exception as e:
            _get_log().debug("probe_error", url=url, error=str(e))
            return None

    def _parse_models_response(self, data: dict[str, Any], provider: str) -> list[str]:
        """Parse models from API response based on provider format."""
        models = []

        if provider == "ollama":
            # Ollama format: {"models": [{"name": "model:tag", ...}]}
            for model in data.get("models", []):
                name = model.get("name", "")
                if name:
                    models.append(name.replace(":latest", ""))
        else:
            # OpenAI-compatible format: {"data": [{"id": "model-name", ...}]}
            for model in data.get("data", []):
                model_id = model.get("id", "")
                # Only include loaded models if status is available
                status = model.get("status", {})
                if isinstance(status, dict) and status.get("value") == "loaded":
                    models.append(model_id)
                elif not status or not isinstance(status, dict):
                    # No status info - include all
                    models.append(model_id)

        return models

    def _assign_models_to_tiers(self, available_models: list[str]) -> dict[str, str]:
        """Assign available models to tiers using shared logic."""
        from .model_detection import assign_models_to_tiers
        return assign_models_to_tiers(available_models)

    def save_settings(self, data: dict[str, Any] | None = None) -> bool:
        """Save current settings to the JSON file."""
        if data is None:
            data = {
                "version": "1.0",
                "system": self.system_config,
                "backends": [b.to_dict() for b in self.backends.values()],
                "routing": self.routing_config,
                "models": self.models_config,
                "mcp_servers": self.mcp_servers_config,
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
        close_tasks = [backend.close_client() for backend in self.backends.values()]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Now load new settings
        self._load_settings()

    def get_enabled_backends(self) -> list[BackendConfig]:
        """Get all enabled backends, sorted by priority."""
        enabled = [b for b in self.backends.values() if b.enabled]
        enabled.sort(key=lambda b: b.priority)
        return enabled

    def get_scoring_weights(self) -> "ScoringWeights":
        """Get scoring weights from routing config.

        Returns ScoringWeights with values from settings.json,
        falling back to defaults for missing keys.
        """
        from .routing import ScoringWeights

        scoring_config = self.routing_config.get("scoring", {})
        if scoring_config:
            return ScoringWeights.from_dict(scoring_config)
        return ScoringWeights()

    def get_backend(self, backend_id: str) -> BackendConfig | None:
        """Get a specific backend by ID."""
        return self.backends.get(backend_id)

    def get_active_backend(self) -> BackendConfig | None:
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
        async with self._health_check_lock:
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

            health: dict[str, bool] = {}
            for backend, result in zip(enabled, results, strict=True):
                if isinstance(result, Exception):
                    health[backend.id] = False
                    _get_log().warning("health_check_exception", backend_id=backend.id, error=str(result))
                else:
                    health[backend.id] = bool(result)

            # Update cache
            self._health_cache = health.copy()
            self._health_cache_time = now

            return health

    def invalidate_health_cache(self) -> None:
        """Invalidate the health check cache, forcing fresh checks on next call."""
        self._health_cache = {}
        self._health_cache_time = 0.0

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status for all backends."""
        health = await self.check_all_health()

        backends_status = []
        for backend in self.backends.values():
            backends_status.append(
                {
                    "id": backend.id,
                    "name": backend.name,
                    "provider": backend.provider,
                    "type": backend.type,
                    "url": backend.url,
                    "enabled": backend.enabled,
                    "available": health.get(backend.id, False) if backend.enabled else False,
                    "models": backend.models,
                }
            )

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

    def add_backend(self, backend_data: dict[str, Any]) -> BackendConfig:
        """Add a new backend from the WebGUI."""
        backend = BackendConfig.from_dict(backend_data)
        self.backends[backend.id] = backend
        self.save_settings()
        _get_log().info("backend_added", backend_id=backend.id, name=backend.name)
        return backend

    async def update_backend(self, backend_id: str, updates: dict[str, Any]) -> BackendConfig | None:
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

    # ===== MCP Server Management =====

    def get_mcp_servers(self) -> list[dict[str, Any]]:
        """Get configured MCP servers.

        Returns:
            List of MCP server configurations
        """
        return self.mcp_servers_config

    def add_mcp_server(self, server_config: dict[str, Any]) -> bool:
        """Add a new MCP server configuration.

        Args:
            server_config: Server configuration dict with at least 'id' and 'command'

        Returns:
            True if added successfully
        """
        if not server_config.get("id") or not server_config.get("command"):
            _get_log().error("mcp_server_invalid_config", config=server_config)
            return False

        # Check for duplicate ID
        for existing in self.mcp_servers_config:
            if existing.get("id") == server_config["id"]:
                _get_log().error("mcp_server_duplicate_id", server_id=server_config["id"])
                return False

        self.mcp_servers_config.append(server_config)
        self.save_settings()
        _get_log().info("mcp_server_added", server_id=server_config["id"])
        return True

    def update_mcp_server(self, server_id: str, updates: dict[str, Any]) -> bool:
        """Update an existing MCP server configuration.

        Args:
            server_id: ID of server to update
            updates: Fields to update

        Returns:
            True if updated successfully
        """
        for i, server in enumerate(self.mcp_servers_config):
            if server.get("id") == server_id:
                # Don't allow changing ID
                updates.pop("id", None)
                self.mcp_servers_config[i].update(updates)
                self.save_settings()
                _get_log().info("mcp_server_updated", server_id=server_id)
                return True

        _get_log().error("mcp_server_not_found", server_id=server_id)
        return False

    def remove_mcp_server(self, server_id: str) -> bool:
        """Remove an MCP server configuration.

        Args:
            server_id: ID of server to remove

        Returns:
            True if removed successfully
        """
        for i, server in enumerate(self.mcp_servers_config):
            if server.get("id") == server_id:
                del self.mcp_servers_config[i]
                self.save_settings()
                _get_log().info("mcp_server_removed", server_id=server_id)
                return True

        _get_log().error("mcp_server_not_found", server_id=server_id)
        return False


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
    for backend, result in zip(backends, results, strict=True):
        if isinstance(result, Exception):
            _get_log().warning("backend_close_failed", backend_id=backend.id, error=str(result))

    _get_log().info("backends_shutdown_complete", closed=len(backends))
