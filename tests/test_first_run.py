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
Tests for first-run scenario - simulates git clone onto fresh machine.

Verifies:
- App works with no pre-existing data directory
- Settings.json is created if missing
- All required directories are created
- MCP server initializes correctly
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import pytest


class TestFreshClone:
    """Simulate a fresh git clone with no existing data."""

    @pytest.fixture(autouse=True)
    def setup_clean_environment(self, tmp_path):
        """Create a clean environment simulating fresh clone."""
        self.tmp_dir = tmp_path
        self.data_dir = tmp_path / "data"

        # Set env var to use temp directory
        os.environ["DELIA_DATA_DIR"] = str(self.data_dir)

        # Clear cached module imports
        modules_to_clear = [
            "delia.paths", "delia.config", "delia.backend_manager",
            "delia.multi_user_tracking", "delia.auth", "delia"
        ]
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) or mod == m for m in modules_to_clear):
                del sys.modules[mod]

        yield

        # Cleanup
        os.environ.pop("DELIA_DATA_DIR", None)
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) or mod == m for m in modules_to_clear):
                del sys.modules[mod]

    def test_no_data_dir_exists_initially(self):
        """Data directory should not exist before first run."""
        assert not self.data_dir.exists()

    def test_paths_module_loads_without_data_dir(self):
        """paths.py should load even when data dir doesn't exist."""
        from delia import paths
        assert paths.DATA_DIR == self.data_dir
        assert not self.data_dir.exists()  # Still doesn't exist

    def test_ensure_directories_creates_structure(self):
        """ensure_directories() should create full directory structure."""
        from delia import paths

        paths.ensure_directories()

        assert self.data_dir.exists()
        assert (self.data_dir / "cache").exists()
        assert (self.data_dir / "users").exists()
        assert (self.data_dir / "memories").exists()

    def test_config_loads_without_data_dir(self):
        """config.py should load and use correct paths."""
        from delia import paths
        from delia import config

        assert config.config.stats_file == paths.STATS_FILE
        assert str(config.config.stats_file).startswith(str(self.data_dir))

    def test_backend_manager_creates_default_settings(self, tmp_path):
        """BackendManager should create default settings.json if missing."""
        from delia import paths
        from delia.backend_manager import BackendManager

        # Use a temp settings file that doesn't exist
        settings_file = tmp_path / "new_settings.json"
        assert not settings_file.exists()

        manager = BackendManager(settings_file=settings_file)

        # Should have created default settings
        assert settings_file.exists()

        with open(settings_file) as f:
            data = json.load(f)

        assert "version" in data
        assert "backends" in data
        assert "routing" in data

    def test_multi_user_tracking_creates_data_dir(self):
        """MultiUserTracker should create its data directory."""
        from delia import paths
        paths.ensure_directories()

        from delia import multi_user_tracking

        # Tracker should have created its directory
        assert multi_user_tracking.DATA_DIR.exists()

    def test_full_initialization_sequence(self):
        """Test the full initialization sequence as mcp_server.py does."""
        from delia import paths

        # 1. Ensure directories (done early in mcp_server.py)
        paths.ensure_directories()

        # 2. Import config
        from delia import config
        assert config.config is not None

        # 3. Import backend manager
        from delia import backend_manager
        assert backend_manager.backend_manager is not None

        # 4. Import tracking
        from delia import multi_user_tracking
        assert multi_user_tracking.tracker is not None

        # All directories should exist
        assert paths.CACHE_DIR.exists()
        assert paths.USER_DATA_DIR.exists()


class TestSettingsJson:
    """Test settings.json handling."""

    def test_missing_settings_creates_default(self, tmp_path):
        """Missing settings.json should trigger default creation."""
        from delia.backend_manager import BackendManager

        settings_file = tmp_path / "settings.json"
        manager = BackendManager(settings_file=settings_file)

        assert settings_file.exists()

        with open(settings_file) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert isinstance(data["backends"], list)
        assert "routing" in data

    def test_invalid_settings_creates_default(self, tmp_path):
        """Invalid JSON in settings.json should trigger default creation."""
        from delia.backend_manager import BackendManager

        settings_file = tmp_path / "settings.json"
        settings_file.write_text("{ invalid json }")

        manager = BackendManager(settings_file=settings_file)

        # Should have overwritten with valid default
        with open(settings_file) as f:
            data = json.load(f)  # Should not raise

        assert "version" in data

    def test_valid_settings_preserved(self, tmp_path):
        """Valid settings.json should be loaded, not overwritten."""
        from delia.backend_manager import BackendManager

        settings_file = tmp_path / "settings.json"
        custom_settings = {
            "version": "1.0",
            "backends": [
                {
                    "id": "my-backend",
                    "name": "My Custom Backend",
                    "provider": "ollama",
                    "type": "local",
                    "url": "http://localhost:11434",
                    "enabled": True,
                    "priority": 0,
                    "models": {}
                }
            ],
            "routing": {"prefer_local": True}
        }

        with open(settings_file, "w") as f:
            json.dump(custom_settings, f)

        manager = BackendManager(settings_file=settings_file)

        assert "my-backend" in manager.backends
        assert manager.backends["my-backend"].name == "My Custom Backend"


class TestMCPServerInitialization:
    """Test MCP server can initialize without errors."""

    def test_mcp_server_imports(self):
        """mcp_server.py should import without errors."""
        # This tests that all module-level code runs without crashing
        from delia import mcp_server
        assert mcp_server is not None

    def test_mcp_instance_created(self):
        """FastMCP instance should be created."""
        from delia import mcp_server
        assert mcp_server.mcp is not None

    def test_tools_registered(self):
        """MCP tools should be registered."""
        from delia import mcp_server

        # Check that key tools exist
        # FastMCP stores tools internally
        mcp = mcp_server.mcp

        # The mcp object should have tools registered
        # We can't easily inspect them without running the server,
        # but we can verify the tool functions exist
        assert hasattr(mcp_server, 'delegate')
        assert hasattr(mcp_server, 'batch')
        assert hasattr(mcp_server, 'think')
        assert hasattr(mcp_server, 'health')
        assert hasattr(mcp_server, 'models')

    def test_stats_structures_initialized(self):
        """Stats tracking structures should be initialized."""
        from delia.mcp_server import stats_service

        # StatsService should be initialized with model tiers
        model_usage, task_stats, _, _ = stats_service.get_snapshot()
        assert "quick" in model_usage
        assert "coder" in model_usage
        assert "moe" in model_usage
        assert task_stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
