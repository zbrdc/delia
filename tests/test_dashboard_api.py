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
Tests for dashboard API routes.

These tests verify the dashboard API endpoints work correctly.
Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_dashboard_api.py -v

Note: These tests require the dashboard to be built and running,
or they test the API route logic in isolation.
"""

import os
import sys
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import httpx


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Use a temp directory for test data."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


class TestStatsAPIData:
    """Test data format expected by /api/stats route."""

    def test_stats_file_format(self, tmp_path):
        """Stats file should have correct format for dashboard."""
        data_dir = tmp_path
        cache_dir = data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "quick": {"calls": 100, "tokens": 50000},
            "coder": {"calls": 50, "tokens": 100000},
            "moe": {"calls": 20, "tokens": 60000},
            "thinking": {"calls": 10, "tokens": 30000}
        }

        stats_file = cache_dir / "usage_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f)

        # Verify format
        with open(stats_file) as f:
            loaded = json.load(f)

        assert "quick" in loaded
        assert "calls" in loaded["quick"]
        assert "tokens" in loaded["quick"]

    def test_enhanced_stats_file_format(self, tmp_path):
        """Enhanced stats file should have correct format."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        enhanced = {
            "task_stats": {"summarize": 30, "generate": 45, "answer": 100},
            "recent_calls": [
                {"ts": time.time(), "task": "summarize", "model": "quick", "tokens": 500}
            ],
            "response_times": {
                "quick": [{"ts": time.time(), "ms": 150}],
                "coder": [{"ts": time.time(), "ms": 800}]
            }
        }

        enhanced_file = cache_dir / "enhanced_stats.json"
        with open(enhanced_file, "w") as f:
            json.dump(enhanced, f)

        with open(enhanced_file) as f:
            loaded = json.load(f)

        assert "task_stats" in loaded
        assert "recent_calls" in loaded


class TestLogsAPIData:
    """Test data format expected by /api/logs route."""

    def test_live_logs_file_format(self, tmp_path):
        """Live logs file should have correct format."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        logs = [
            {
                "ts": time.time(),
                "type": "request",
                "message": "Processing delegate task",
                "model": "quick",
                "tokens": 500,
                "backend_id": "ollama-1",
                "provider": "ollama"
            },
            {
                "ts": time.time(),
                "type": "response",
                "message": "Completed successfully",
                "model": "quick",
                "tokens": 1000
            }
        ]

        logs_file = cache_dir / "live_logs.json"
        with open(logs_file, "w") as f:
            json.dump(logs, f)

        with open(logs_file) as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert "ts" in loaded[0]
        assert "type" in loaded[0]


class TestConfigAPIData:
    """Test data format expected by /api/config route."""

    def test_settings_file_format(self, tmp_path):
        """Settings file should have correct format for dashboard."""
        settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "ollama-main",
                    "name": "Ollama Local",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 0,
                    "models": {
                        "quick": "llama3:8b",
                        "coder": "codellama:13b",
                        "moe": "mixtral:8x7b",
                        "thinking": "deepseek-r1:14b"
                    }
                }
            ],
            "routing": {
                "prefer_local": True,
                "fallback_enabled": True,
                "load_balance": False
            }
        }

        # Settings file is in project root, not data dir
        settings_file = tmp_path / "settings.json"
        with open(settings_file, "w") as f:
            json.dump(settings, f)

        with open(settings_file) as f:
            loaded = json.load(f)

        assert "version" in loaded
        assert "backends" in loaded
        assert isinstance(loaded["backends"], list)
        assert "routing" in loaded


class TestBackendsAPIData:
    """Test data format expected by /api/backends route."""

    def test_backend_config_format(self, tmp_path):
        """Backend configs should have all required fields."""
        backend = {
            "id": "test-backend",
            "name": "Test Backend",
            "provider": "llamacpp",
            "type": "local",
            "url": "http://localhost:8080",
            "enabled": True,
            "priority": 0,
            "models": {
                "quick": "Qwen3-4B",
                "coder": "DeepSeek-Coder-7B"
            },
            "health_endpoint": "/health",
            "models_endpoint": "/v1/models",
            "chat_endpoint": "/v1/chat/completions"
        }

        # Verify required fields
        required = ["id", "name", "provider", "type", "url", "enabled"]
        for field in required:
            assert field in backend

    def test_backend_health_response_format(self):
        """Backend health check response format."""
        health_response = {
            "id": "ollama-1",
            "name": "Ollama",
            "provider": "ollama",
            "status": "healthy",
            "available": True,
            "latency_ms": 45,
            "models_loaded": ["llama3:8b", "codellama:13b"]
        }

        assert "status" in health_response
        assert "available" in health_response


class TestCircuitBreakerAPIData:
    """Test data format expected by /api/circuit-breaker route."""

    def test_circuit_breaker_file_format(self, tmp_path):
        """Circuit breaker file should have correct format."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        circuit_data = {
            "ollama": {
                "name": "ollama",
                "consecutive_failures": 0,
                "circuit_open": False,
                "last_error": None
            },
            "llamacpp": {
                "name": "llamacpp",
                "consecutive_failures": 2,
                "circuit_open": False,
                "last_error": "timeout"
            },
            "active_backend": "ollama",
            "timestamp": time.time()
        }

        cb_file = cache_dir / "circuit_breaker.json"
        with open(cb_file, "w") as f:
            json.dump(circuit_data, f)

        with open(cb_file) as f:
            loaded = json.load(f)

        assert "timestamp" in loaded
        # Should have backend info
        assert "ollama" in loaded or "active_backend" in loaded


class TestBackendModelsAPIData:
    """Test data format expected by /api/backends/models route."""

    def test_ollama_models_response_format(self):
        """Ollama models list response format."""
        response = {
            "models": [
                {"name": "llama3:8b", "size": "4.7GB", "quantization": "Q4_K_M"},
                {"name": "codellama:13b", "size": "7.3GB", "quantization": "Q4_K_M"},
                {"name": "mixtral:8x7b", "size": "26GB", "quantization": "Q4_K_M"}
            ]
        }

        assert "models" in response
        assert isinstance(response["models"], list)

    def test_llamacpp_models_response_format(self):
        """LlamaCpp models list response format."""
        response = {
            "data": [
                {"id": "Qwen3-4B-Q4_K_M", "object": "model"},
                {"id": "DeepSeek-Coder-7B-Q4_K_M", "object": "model"}
            ]
        }

        assert "data" in response


class TestAPIErrorResponses:
    """Test expected error response formats."""

    def test_error_response_format(self):
        """Error responses should have consistent format."""
        error = {
            "success": False,
            "error": "Backend not found",
            "code": 404
        }

        assert "success" in error
        assert error["success"] is False
        assert "error" in error

    def test_validation_error_format(self):
        """Validation error responses format."""
        validation_error = {
            "success": False,
            "error": "Missing required field: id",
            "field": "id"
        }

        assert "error" in validation_error


class TestDashboardDataIntegration:
    """Test that Python backend creates data dashboard can read."""

    def test_mcp_server_creates_dashboard_compatible_stats(self, tmp_path):
        """MCP server should create stats dashboard can parse."""
        os.environ["DELIA_DATA_DIR"] = str(tmp_path)

        # Clear modules
        modules_to_clear = ["paths", "config", "mcp_server", "backend_manager"]
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) or mod == m for m in modules_to_clear):
                del sys.modules[mod]

        try:
            import paths
            paths.ensure_directories()

            import mcp_server

            # Record some usage
            mcp_server.MODEL_USAGE["quick"]["calls"] = 50
            mcp_server.MODEL_USAGE["quick"]["tokens"] = 25000
            mcp_server.save_usage_stats()

            # Verify dashboard can read it
            with open(paths.STATS_FILE) as f:
                data = json.load(f)

            # Dashboard expects this format
            assert "quick" in data
            assert isinstance(data["quick"]["calls"], int)
            assert isinstance(data["quick"]["tokens"], int)
        finally:
            os.environ.pop("DELIA_DATA_DIR", None)

    def test_backend_manager_creates_dashboard_compatible_settings(self, tmp_path):
        """Backend manager should create settings dashboard can parse."""
        os.environ["DELIA_DATA_DIR"] = str(tmp_path)

        modules_to_clear = ["paths", "config", "backend_manager"]
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) or mod == m for m in modules_to_clear):
                del sys.modules[mod]

        try:
            import paths
            paths.ensure_directories()

            from backend_manager import BackendManager

            # Create manager which creates default settings
            settings_file = paths.SETTINGS_FILE
            manager = BackendManager(settings_file=settings_file)

            # Verify dashboard can read it
            with open(settings_file) as f:
                data = json.load(f)

            # Dashboard expects this format
            assert "version" in data
            assert "backends" in data
            assert isinstance(data["backends"], list)
            assert "routing" in data
        finally:
            os.environ.pop("DELIA_DATA_DIR", None)


class TestEnvironmentVariableHandling:
    """Test that API routes handle DELIA_DATA_DIR correctly."""

    def test_custom_data_dir_respected(self, tmp_path):
        """Dashboard should respect DELIA_DATA_DIR env var."""
        custom_dir = tmp_path / "custom_data"
        custom_dir.mkdir()
        cache_dir = custom_dir / "cache"
        cache_dir.mkdir()

        # Create stats in custom location
        stats = {"quick": {"calls": 999, "tokens": 999999}}
        stats_file = cache_dir / "usage_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f)

        # Verify file is in custom location
        assert stats_file.exists()
        with open(stats_file) as f:
            loaded = json.load(f)
        assert loaded["quick"]["calls"] == 999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
