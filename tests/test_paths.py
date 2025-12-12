"""
Tests for paths.py - centralized path configuration.

Run with: uv run pytest tests/test_paths.py -v
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest import mock

import pytest


class TestPathsModule:
    """Test the paths module loads correctly and provides expected paths."""

    def test_import_paths(self):
        """paths.py should import without errors."""
        import paths
        assert paths is not None

    def test_project_root_exists(self):
        """PROJECT_ROOT should point to actual project directory."""
        import paths
        assert paths.PROJECT_ROOT.exists()
        assert (paths.PROJECT_ROOT / "paths.py").exists()

    def test_data_dir_default(self):
        """DATA_DIR should default to PROJECT_ROOT/data."""
        # Clear any cached imports
        import importlib
        import sys

        # Remove env var if set
        env_backup = os.environ.pop("DELIA_DATA_DIR", None)

        # Reload paths module to get fresh values
        if "paths" in sys.modules:
            del sys.modules["paths"]

        import paths

        try:
            expected = paths.PROJECT_ROOT / "data"
            assert paths.DATA_DIR == expected
        finally:
            # Restore env var
            if env_backup:
                os.environ["DELIA_DATA_DIR"] = env_backup

    def test_derived_directories(self):
        """Derived directories should be under DATA_DIR."""
        import paths

        assert paths.CACHE_DIR == paths.DATA_DIR / "cache"
        assert paths.USER_DATA_DIR == paths.DATA_DIR / "users"
        assert paths.MEMORIES_DIR == paths.DATA_DIR / "memories"

    def test_file_paths(self):
        """File paths should be correctly derived."""
        import paths

        assert paths.STATS_FILE == paths.CACHE_DIR / "usage_stats.json"
        assert paths.ENHANCED_STATS_FILE == paths.CACHE_DIR / "enhanced_stats.json"
        assert paths.LIVE_LOGS_FILE == paths.CACHE_DIR / "live_logs.json"
        assert paths.CIRCUIT_BREAKER_FILE == paths.CACHE_DIR / "circuit_breaker.json"
        assert paths.USER_DB_FILE == paths.USER_DATA_DIR / "users.db"

    def test_settings_file_in_project_root(self):
        """SETTINGS_FILE should be in PROJECT_ROOT, not DATA_DIR."""
        import paths

        assert paths.SETTINGS_FILE == paths.PROJECT_ROOT / "settings.json"
        assert paths.SETTINGS_FILE.parent == paths.PROJECT_ROOT


class TestCustomDataDir:
    """Test DELIA_DATA_DIR environment variable override."""

    def test_custom_data_dir_from_env(self):
        """DATA_DIR should respect DELIA_DATA_DIR env var."""
        import importlib
        import sys

        custom_path = "/tmp/custom-delia-test"

        # Set env var before importing
        env_backup = os.environ.get("DELIA_DATA_DIR")
        os.environ["DELIA_DATA_DIR"] = custom_path

        # Force reimport
        if "paths" in sys.modules:
            del sys.modules["paths"]

        try:
            import paths
            assert str(paths.DATA_DIR) == custom_path
            assert paths.CACHE_DIR == Path(custom_path) / "cache"
        finally:
            # Cleanup
            if env_backup:
                os.environ["DELIA_DATA_DIR"] = env_backup
            else:
                os.environ.pop("DELIA_DATA_DIR", None)

            # Reimport with original settings
            if "paths" in sys.modules:
                del sys.modules["paths"]


class TestEnsureDirectories:
    """Test ensure_directories() creates required directories."""

    def test_ensure_directories_creates_all(self):
        """ensure_directories() should create cache, users, and memories dirs."""
        import importlib
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set custom data dir
            os.environ["DELIA_DATA_DIR"] = tmpdir

            # Force reimport
            if "paths" in sys.modules:
                del sys.modules["paths"]

            try:
                import paths

                # Directories shouldn't exist yet
                assert not (Path(tmpdir) / "cache").exists()
                assert not (Path(tmpdir) / "users").exists()
                assert not (Path(tmpdir) / "memories").exists()

                # Create them
                paths.ensure_directories()

                # Now they should exist
                assert (Path(tmpdir) / "cache").exists()
                assert (Path(tmpdir) / "users").exists()
                assert (Path(tmpdir) / "memories").exists()
            finally:
                os.environ.pop("DELIA_DATA_DIR", None)
                if "paths" in sys.modules:
                    del sys.modules["paths"]

    def test_ensure_directories_idempotent(self):
        """ensure_directories() should be safe to call multiple times."""
        import importlib
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DELIA_DATA_DIR"] = tmpdir

            if "paths" in sys.modules:
                del sys.modules["paths"]

            try:
                import paths

                # Call multiple times - should not raise
                paths.ensure_directories()
                paths.ensure_directories()
                paths.ensure_directories()

                assert (Path(tmpdir) / "cache").exists()
            finally:
                os.environ.pop("DELIA_DATA_DIR", None)
                if "paths" in sys.modules:
                    del sys.modules["paths"]


class TestModuleImports:
    """Test that other modules correctly use paths."""

    @pytest.fixture(autouse=True)
    def clean_imports(self):
        """Clear cached imports before and after each test."""
        import sys

        modules_to_clear = [
            "paths", "config", "backend_manager",
            "multi_user_tracking", "auth", "mcp_server"
        ]

        # Clean before test
        os.environ.pop("DELIA_DATA_DIR", None)
        for mod in list(sys.modules.keys()):
            if any(mod == m or mod.startswith(m + ".") for m in modules_to_clear):
                del sys.modules[mod]

        yield

        # Clean after test
        os.environ.pop("DELIA_DATA_DIR", None)
        for mod in list(sys.modules.keys()):
            if any(mod == m or mod.startswith(m + ".") for m in modules_to_clear):
                del sys.modules[mod]

    def test_config_uses_paths(self):
        """config.py should use paths.STATS_FILE."""
        import paths
        import config

        assert config.config.stats_file == paths.STATS_FILE

    def test_backend_manager_uses_paths(self):
        """backend_manager.py should use paths.SETTINGS_FILE."""
        import paths
        import backend_manager

        assert backend_manager.SETTINGS_FILE == paths.SETTINGS_FILE

    def test_multi_user_tracking_uses_paths(self):
        """multi_user_tracking.py should use paths.USER_DATA_DIR."""
        import paths
        import multi_user_tracking

        assert multi_user_tracking.DATA_DIR == paths.USER_DATA_DIR


class TestEndToEnd:
    """End-to-end tests with custom DELIA_DATA_DIR."""

    def test_full_stack_with_custom_dir(self):
        """All modules should work with custom DELIA_DATA_DIR."""
        import importlib
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DELIA_DATA_DIR"] = tmpdir

            # Clear all related modules
            modules_to_clear = [
                "paths", "config", "backend_manager",
                "multi_user_tracking", "auth"
            ]
            for mod in modules_to_clear:
                if mod in sys.modules:
                    del sys.modules[mod]

            try:
                import paths
                paths.ensure_directories()

                import config
                import backend_manager
                import multi_user_tracking

                # Verify all paths point to custom dir
                assert str(paths.DATA_DIR) == tmpdir
                assert str(config.config.stats_file).startswith(tmpdir)
                assert str(multi_user_tracking.DATA_DIR).startswith(tmpdir)

                # Verify directories were created
                assert paths.CACHE_DIR.exists()
                assert paths.USER_DATA_DIR.exists()

            finally:
                os.environ.pop("DELIA_DATA_DIR", None)
                for mod in modules_to_clear:
                    if mod in sys.modules:
                        del sys.modules[mod]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
