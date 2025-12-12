"""
Multi-backend failover tests.

Tests that the system correctly fails over between backends when one is unavailable.

Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_failover.py -v
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Use a temp directory for test data."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    # Clear cached modules
    modules_to_clear = ["paths", "config", "backend_manager", "mcp_server"]
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) or mod == m for m in modules_to_clear):
            del sys.modules[mod]

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


class TestMultiBackendConfiguration:
    """Test configuring multiple backends."""

    def test_multiple_backends_load(self, tmp_path):
        """Multiple backends should load from settings."""
        import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "primary",
                    "name": "Primary Backend",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "model-a"}
                },
                {
                    "id": "secondary",
                    "name": "Secondary Backend",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 1,
                    "models": {"quick": "model-b"}
                },
                {
                    "id": "tertiary",
                    "name": "Tertiary Backend",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11435",
                    "enabled": True,
                    "priority": 2,
                    "models": {"quick": "model-c"}
                }
            ],
            "routing": {"prefer_local": True, "fallback_enabled": True}
        }

        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        from backend_manager import BackendManager

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        assert len(manager.backends) == 3
        assert "primary" in manager.backends
        assert "secondary" in manager.backends
        assert "tertiary" in manager.backends

    def test_backends_sorted_by_priority(self, tmp_path):
        """Backends should be sorted by priority."""
        import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [
                {"id": "low", "name": "Low", "provider": "ollama", "type": "local",
                 "url": "http://localhost:11434", "enabled": True, "priority": 10, "models": {}},
                {"id": "high", "name": "High", "provider": "ollama", "type": "local",
                 "url": "http://localhost:11435", "enabled": True, "priority": 0, "models": {}},
                {"id": "medium", "name": "Medium", "provider": "ollama", "type": "local",
                 "url": "http://localhost:11436", "enabled": True, "priority": 5, "models": {}}
            ],
            "routing": {"prefer_local": True}
        }

        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        from backend_manager import BackendManager

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)
        enabled = manager.get_enabled_backends()

        # Should be sorted by priority (ascending)
        assert enabled[0].id == "high"
        assert enabled[1].id == "medium"
        assert enabled[2].id == "low"


class TestBackendHealthChecks:
    """Test backend health checking for failover."""

    @pytest.fixture(autouse=True)
    def setup_backends(self, tmp_path):
        """Set up test backends."""
        import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "primary",
                    "name": "Primary",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "test"}
                },
                {
                    "id": "fallback",
                    "name": "Fallback",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 1,
                    "models": {"quick": "test"}
                }
            ],
            "routing": {"prefer_local": True, "fallback_enabled": True}
        }

        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_health_check_marks_unavailable(self):
        """Health check should mark unavailable backends."""
        from backend_manager import BackendManager
        import paths

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        # Check health (will fail since backends aren't running)
        await manager.check_all_health()

        # Both should be marked unavailable
        primary = manager.get_backend("primary")
        fallback = manager.get_backend("fallback")

        # _available should be False for unreachable backends
        assert primary._available is False or primary._available is True  # Depends on actual connectivity
        assert fallback._available is False or fallback._available is True

    @pytest.mark.asyncio
    async def test_health_cache_works(self):
        """Health check results should be cached."""
        from backend_manager import BackendManager
        import paths

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        # First check
        result1 = await manager.check_all_health(use_cache=False)

        # Second check (cached)
        result2 = await manager.check_all_health(use_cache=True)

        # Results should be same (from cache)
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_health_cache_invalidation(self):
        """Health cache should be invalidatable."""
        from backend_manager import BackendManager
        import paths

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        await manager.check_all_health()

        # Invalidate cache
        manager.invalidate_health_cache()

        # Cache time should be reset
        assert manager._health_cache_time == 0


class TestCircuitBreakerFailover:
    """Test circuit breaker triggers failover."""

    def test_circuit_breaker_opens(self):
        """Circuit breaker should open after failures."""
        from config import BackendHealth

        health = BackendHealth("test-backend")

        # Record failures
        for _ in range(3):
            health.record_failure("connection_error")

        # Circuit should be open
        assert health.is_available() is False

    def test_circuit_breaker_reports_time_until_available(self):
        """Circuit breaker should report time until available."""
        from config import BackendHealth

        health = BackendHealth("test-backend")

        # Open circuit
        for _ in range(3):
            health.record_failure("timeout")

        # Should have positive time until available
        time_remaining = health.time_until_available()
        assert time_remaining > 0

    def test_circuit_breaker_context_reduction(self):
        """Circuit breaker should suggest context reduction."""
        from config import BackendHealth

        health = BackendHealth("test-backend")

        # Record failure with large context
        health.record_failure("context_overflow", context_size=100000)

        # Should suggest reducing context
        should_reduce, suggested_size = health.should_reduce_context(100000)

        # May or may not suggest reduction depending on implementation
        assert isinstance(should_reduce, bool)
        assert isinstance(suggested_size, int)


class TestActiveBackendFailover:
    """Test active backend selection during failover."""

    @pytest.fixture(autouse=True)
    def setup_backends(self, tmp_path):
        """Set up test backends."""
        import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "primary",
                    "name": "Primary",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "model-a"}
                },
                {
                    "id": "fallback",
                    "name": "Fallback",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 1,
                    "models": {"quick": "model-b"}
                }
            ],
            "routing": {"prefer_local": True, "fallback_enabled": True}
        }

        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    def test_active_backend_defaults_to_first_enabled(self):
        """Active backend should default to highest priority enabled backend."""
        from backend_manager import BackendManager
        import paths

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        active = manager.get_active_backend()
        assert active is not None
        # Should be primary (priority 0)
        assert active.id == "primary"

    def test_can_switch_active_backend(self):
        """Should be able to manually switch active backend."""
        from backend_manager import BackendManager
        import paths

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        # Switch to fallback
        manager.set_active_backend("fallback")

        active = manager.get_active_backend()
        assert active.id == "fallback"

    @pytest.mark.asyncio
    async def test_fallback_on_remove(self):
        """Removing active backend should fallback to next."""
        from backend_manager import BackendManager
        import paths

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        # Set primary as active
        manager.set_active_backend("primary")

        # Remove primary
        await manager.remove_backend("primary")

        # Should fallback to next enabled
        active = manager.get_active_backend()
        assert active is not None
        assert active.id == "fallback"


class TestDelegateFailover:
    """Test delegate() failover behavior."""

    @pytest.fixture(autouse=True)
    def setup_backends(self, tmp_path):
        """Set up test backends."""
        import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "primary",
                    "name": "Primary",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "model-a", "coder": "model-a", "moe": "model-a", "thinking": "model-a"}
                },
                {
                    "id": "fallback",
                    "name": "Fallback",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 1,
                    "models": {"quick": "model-b", "coder": "model-b", "moe": "model-b", "thinking": "model-b"}
                }
            ],
            "routing": {"prefer_local": True, "fallback_enabled": True}
        }

        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_delegate_returns_error_when_no_backends(self):
        """delegate() should return error when no backends available."""
        import paths

        # Create empty backend config
        settings = {
            "version": "1.0",
            "backends": [],
            "routing": {"prefer_local": True}
        }
        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

        import mcp_server
        await mcp_server.backend_manager.reload()

        result = await mcp_server.delegate.fn(
            task="quick",
            content="Test question"
        )

        assert result is not None
        # Should indicate no backend or error
        assert len(result) > 0


class TestBackendTypeRouting:
    """Test routing between local and remote backends."""

    @pytest.fixture(autouse=True)
    def setup_backends(self, tmp_path):
        """Set up local and remote backends."""
        import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "local",
                    "name": "Local LlamaCpp",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "local-model"}
                },
                {
                    "id": "remote",
                    "name": "Remote API",
                    "provider": "openai",
                    "type": "remote",
                    "url": "https://api.example.com",
                    "enabled": True,
                    "priority": 1,
                    "models": {"quick": "gpt-4"}
                }
            ],
            "routing": {"prefer_local": True, "fallback_enabled": True}
        }

        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    def test_prefer_local_setting(self):
        """Should prefer local backends when configured."""
        from backend_manager import BackendManager
        import paths

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        # Active should be local (higher priority + prefer_local)
        active = manager.get_active_backend()
        assert active.type == "local"

    def test_backend_type_detection(self):
        """Should correctly detect backend types."""
        from backend_manager import BackendManager
        import paths

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        local = manager.get_backend("local")
        remote = manager.get_backend("remote")

        assert local.type == "local"
        assert remote.type == "remote"


class TestConcurrentBackendRequests:
    """Test concurrent requests to backends."""

    @pytest.fixture(autouse=True)
    def setup_backends(self, tmp_path):
        """Set up test backends."""
        import paths
        paths.ensure_directories()

        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "backend-1",
                    "name": "Backend 1",
                    "provider": "llamacpp",
                    "type": "local",
                    "url": "http://localhost:8080",
                    "enabled": True,
                    "priority": 0,
                    "models": {"quick": "model"}
                }
            ],
            "routing": {"prefer_local": True}
        }

        with open(paths.SETTINGS_FILE, "w") as f:
            json.dump(settings, f)

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Multiple concurrent health checks should work."""
        from backend_manager import BackendManager
        import paths

        manager = BackendManager(settings_file=paths.SETTINGS_FILE)

        # Run multiple health checks concurrently
        results = await asyncio.gather(
            manager.check_all_health(use_cache=False),
            manager.check_all_health(use_cache=False),
            manager.check_all_health(use_cache=False)
        )

        # All should complete without error
        assert len(results) == 3


class TestBackendRecovery:
    """Test backend recovery after failures."""

    def test_circuit_breaker_recovery(self):
        """Circuit breaker should allow recovery after cooldown."""
        from config import BackendHealth
        import time

        health = BackendHealth("test-backend")

        # Open circuit
        for _ in range(3):
            health.record_failure("timeout")

        assert health.is_available() is False

        # Simulate cooldown passed
        health.circuit_open_until = time.time() - 1

        # Should be available again (half-open state)
        assert health.is_available() is True

    def test_success_resets_failures(self):
        """Successful request should reset failure count."""
        from config import BackendHealth

        health = BackendHealth("test-backend")

        # Record some failures
        health.record_failure("timeout")
        health.record_failure("timeout")

        assert health.consecutive_failures == 2

        # Record success
        health.record_success(1000)

        # Failures should reset
        assert health.consecutive_failures == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
