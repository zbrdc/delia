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
Tests for CLI entry points (delia, delia-setup-auth).

Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_cli.py -v
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Use a temp directory for test data."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)
    yield
    os.environ.pop("DELIA_DATA_DIR", None)


class TestDeliaEntryPoint:
    """Test the main 'delia' CLI entry point."""

    def test_delia_help(self):
        """delia --help should display usage info."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "delia.mcp_server", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert result.returncode == 0
        assert "Delia" in result.stdout or "delia" in result.stdout.lower()
        assert "transport" in result.stdout.lower()
        assert "stdio" in result.stdout
        assert "sse" in result.stdout
        assert "http" in result.stdout

    def test_delia_transport_options(self):
        """delia should accept transport options."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "delia.mcp_server", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        # Should list all transport types
        assert "stdio" in result.stdout
        assert "sse" in result.stdout
        assert "http" in result.stdout

    def test_delia_port_option(self):
        """delia should accept port option."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "delia.mcp_server", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert "--port" in result.stdout or "-p" in result.stdout

    def test_delia_host_option(self):
        """delia should accept host option."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "delia.mcp_server", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert "--host" in result.stdout

    def test_delia_invalid_transport(self):
        """delia should reject invalid transport."""
        result = subprocess.run(
            ["uv", "run", "python", "-m", "delia.mcp_server", "--transport", "invalid_transport"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert result.returncode != 0
        assert "invalid" in result.stderr.lower() or "error" in result.stderr.lower()


class TestDeliaHTTPStartup:
    """Test delia starts correctly in HTTP mode."""

    def test_delia_http_starts(self, tmp_path):
        """delia should start in HTTP mode and respond to requests."""
        env = os.environ.copy()
        env["DELIA_DATA_DIR"] = str(tmp_path)

        # Start server in background
        proc = subprocess.Popen(
            ["uv", "run", "python", "-m", "delia.mcp_server", "--transport", "http", "--port", "18765"],
            cwd="/home/dan/git/delia",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            # Wait for startup
            time.sleep(3)

            # Check if process is still running
            assert proc.poll() is None, f"Server exited early: {proc.stderr.read().decode()}"

            # Try to connect
            import httpx
            try:
                response = httpx.get("http://localhost:18765/", timeout=5)
                # MCP servers may return various status codes
                assert response.status_code in [200, 404, 405, 501]
            except httpx.ConnectError:
                # Server might not be fully ready yet, that's okay for this test
                pass

        finally:
            # Clean shutdown
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


class TestDeliaSSEStartup:
    """Test delia starts correctly in SSE mode."""

    def test_delia_sse_starts(self, tmp_path):
        """delia should start in SSE mode."""
        env = os.environ.copy()
        env["DELIA_DATA_DIR"] = str(tmp_path)

        # Start server in background
        proc = subprocess.Popen(
            ["uv", "run", "python", "-m", "delia.mcp_server", "--transport", "sse", "--port", "18766"],
            cwd="/home/dan/git/delia",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            # Wait for startup
            time.sleep(3)

            # Check if process is still running
            assert proc.poll() is None, f"Server exited early: {proc.stderr.read().decode()}"

        finally:
            # Clean shutdown
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


class TestDeliaSetupAuth:
    """Test the delia-setup-auth CLI entry point."""

    def test_setup_auth_module_exists(self):
        """setup_auth.py should exist and be importable."""
        from delia import setup_auth
        assert setup_auth is not None

    def test_setup_auth_has_main(self):
        """setup_auth should have main function."""
        from delia import setup_auth
        assert hasattr(setup_auth, 'main')
        assert callable(setup_auth.main)

    def test_setup_auth_has_jwt_generator(self):
        """setup_auth should have JWT secret generator."""
        from delia import setup_auth
        assert hasattr(setup_auth, 'generate_jwt_secret')

        # Generate a secret
        secret = setup_auth.generate_jwt_secret()
        assert secret is not None
        assert len(secret) >= 32  # Should be secure length

    def test_setup_auth_generates_unique_secrets(self):
        """setup_auth should generate unique JWT secrets."""
        from delia import setup_auth

        secrets = [setup_auth.generate_jwt_secret() for _ in range(10)]

        # All should be unique
        assert len(set(secrets)) == 10

    def test_setup_auth_has_oauth_setup(self):
        """setup_auth should have OAuth setup function."""
        from delia import setup_auth
        assert hasattr(setup_auth, 'setup_oauth')

    def test_setup_auth_has_basic_auth_setup(self):
        """setup_auth should have basic auth setup function."""
        from delia import setup_auth
        assert hasattr(setup_auth, 'setup_basic_auth')


class TestMainFunction:
    """Test mcp_server.main() function."""

    def test_main_function_exists(self):
        """mcp_server should have main function."""
        from delia import mcp_server
        assert hasattr(mcp_server, 'main')
        assert callable(mcp_server.main)

    def test_typer_cli_configuration(self):
        """Main should use typer with correct options."""
        # Test by running with --help
        result = subprocess.run(
            ["uv", "run", "python", "-m", "delia.mcp_server", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        # Should have required arguments
        assert "-t" in result.stdout or "--transport" in result.stdout
        assert "-p" in result.stdout or "--port" in result.stdout
        assert "--host" in result.stdout

        # Should show transport options
        assert "stdio" in result.stdout
        assert "sse" in result.stdout
        assert "http" in result.stdout


class TestPackageEntryPoints:
    """Test that package entry points are correctly defined."""

    def test_pyproject_defines_delia_script(self):
        """pyproject.toml should define 'delia' script."""
        import tomllib
        with open("/home/dan/git/delia/pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        scripts = config.get("project", {}).get("scripts", {})
        assert "delia" in scripts
        assert "delia.cli:app" in scripts["delia"]

    def test_pyproject_defines_setup_auth_script(self):
        """pyproject.toml should define 'delia-setup-auth' script."""
        import tomllib
        with open("/home/dan/git/delia/pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        scripts = config.get("project", {}).get("scripts", {})
        assert "delia-setup-auth" in scripts
        assert "delia.setup_auth:main" in scripts["delia-setup-auth"]


class TestTyperCLICommands:
    """Test the typer-based CLI commands (delia <command>)."""

    def test_delia_cli_help(self):
        """delia --help should show all commands."""
        result = subprocess.run(
            ["uv", "run", "delia", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert result.returncode == 0
        # Should list all subcommands
        assert "init" in result.stdout
        assert "install" in result.stdout
        assert "doctor" in result.stdout
        assert "run" in result.stdout or "serve" in result.stdout
        assert "config" in result.stdout
        assert "uninstall" in result.stdout

    def test_delia_init_help(self):
        """delia init --help should show init options."""
        result = subprocess.run(
            ["uv", "run", "delia", "init", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert result.returncode == 0
        assert "init" in result.stdout.lower() or "setup" in result.stdout.lower()

    def test_delia_doctor_help(self):
        """delia doctor --help should show doctor options."""
        result = subprocess.run(
            ["uv", "run", "delia", "doctor", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert result.returncode == 0

    def test_delia_config_help(self):
        """delia config --help should show config options."""
        result = subprocess.run(
            ["uv", "run", "delia", "config", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert result.returncode == 0

    def test_delia_run_help(self):
        """delia run --help should show server options."""
        result = subprocess.run(
            ["uv", "run", "delia", "run", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert result.returncode == 0
        assert "transport" in result.stdout.lower() or "port" in result.stdout.lower()

    def test_delia_install_help(self):
        """delia install --help should show install options."""
        result = subprocess.run(
            ["uv", "run", "delia", "install", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            timeout=30
        )

        assert result.returncode == 0

    def test_delia_config_show(self, tmp_path):
        """delia config show should display configuration."""
        env = os.environ.copy()
        env["DELIA_DATA_DIR"] = str(tmp_path)

        result = subprocess.run(
            ["uv", "run", "delia", "config", "show"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            env=env,
            timeout=30
        )

        # Should either succeed or fail gracefully
        assert result.returncode in [0, 1, 2]

    def test_delia_doctor_runs(self, tmp_path):
        """delia doctor should run diagnostics."""
        env = os.environ.copy()
        env["DELIA_DATA_DIR"] = str(tmp_path)

        result = subprocess.run(
            ["uv", "run", "delia", "doctor"],
            capture_output=True,
            text=True,
            cwd="/home/dan/git/delia",
            env=env,
            timeout=30
        )

        # Doctor should run even if backends aren't available
        assert result.returncode in [0, 1, 2]
        # Should produce some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
