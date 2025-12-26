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
Tests for the confirmation prompt system (Phase 2.2).

Tests the bidirectional communication flow:
1. API emits confirm event for dangerous tools
2. CLI responds via /api/agent/confirm endpoint
3. Tool execution proceeds or is cancelled
"""

import asyncio
import pytest
from delia.api import (
    create_confirmation,
    resolve_confirmation,
    wait_for_confirmation,
    cleanup_confirmation,
    _pending_confirmations,
    CONFIRMATION_TIMEOUT,
)


class TestConfirmationStateManagement:
    """Test the confirmation state management functions."""

    def setup_method(self):
        """Clear pending confirmations before each test."""
        _pending_confirmations.clear()

    def test_create_confirmation(self):
        """Test creating a pending confirmation."""
        confirmation = create_confirmation("write_file", {"path": "/tmp/test.txt", "content": "hello"})

        assert confirmation.confirm_id is not None
        assert len(confirmation.confirm_id) == 8  # UUID prefix
        assert confirmation.tool_name == "write_file"
        assert confirmation.args == {"path": "/tmp/test.txt", "content": "hello"}
        assert confirmation.confirmed is None  # Pending
        assert confirmation.allow_all is False
        assert not confirmation.event.is_set()

        # Should be stored in pending dict
        assert confirmation.confirm_id in _pending_confirmations

    def test_resolve_confirmation_confirmed(self):
        """Test resolving a confirmation as confirmed."""
        confirmation = create_confirmation("shell_exec", {"command": "ls -la"})
        confirm_id = confirmation.confirm_id

        success = resolve_confirmation(confirm_id, confirmed=True, allow_all=False)

        assert success is True
        assert confirmation.confirmed is True
        assert confirmation.allow_all is False
        assert confirmation.event.is_set()

    def test_resolve_confirmation_denied(self):
        """Test resolving a confirmation as denied."""
        confirmation = create_confirmation("write_file", {"path": "test.txt", "content": "x"})
        confirm_id = confirmation.confirm_id

        success = resolve_confirmation(confirm_id, confirmed=False)

        assert success is True
        assert confirmation.confirmed is False
        assert confirmation.event.is_set()

    def test_resolve_confirmation_allow_all(self):
        """Test resolving with allow_all flag."""
        confirmation = create_confirmation("write_file", {"path": "test.txt", "content": "x"})
        confirm_id = confirmation.confirm_id

        success = resolve_confirmation(confirm_id, confirmed=True, allow_all=True)

        assert success is True
        assert confirmation.confirmed is True
        assert confirmation.allow_all is True
        assert confirmation.event.is_set()

    def test_resolve_confirmation_not_found(self):
        """Test resolving a non-existent confirmation."""
        success = resolve_confirmation("nonexistent", confirmed=True)
        assert success is False

    def test_cleanup_confirmation(self):
        """Test cleaning up a confirmation."""
        confirmation = create_confirmation("write_file", {"path": "test.txt", "content": "x"})
        confirm_id = confirmation.confirm_id

        assert confirm_id in _pending_confirmations
        cleanup_confirmation(confirm_id)
        assert confirm_id not in _pending_confirmations

    def test_cleanup_nonexistent_confirmation(self):
        """Test cleaning up a non-existent confirmation (should not raise)."""
        cleanup_confirmation("nonexistent")  # Should not raise


class TestConfirmationAsync:
    """Test async confirmation waiting."""

    def setup_method(self):
        """Clear pending confirmations before each test."""
        _pending_confirmations.clear()

    @pytest.mark.asyncio
    async def test_wait_for_confirmation_resolved(self):
        """Test waiting for a confirmation that gets resolved."""
        confirmation = create_confirmation("write_file", {"path": "test.txt", "content": "x"})
        confirm_id = confirmation.confirm_id

        # Resolve in background after a short delay
        async def resolve_after_delay():
            await asyncio.sleep(0.1)
            resolve_confirmation(confirm_id, confirmed=True)

        asyncio.create_task(resolve_after_delay())

        # Wait should complete when resolved
        result = await wait_for_confirmation(confirm_id)

        assert result.confirmed is True

    @pytest.mark.asyncio
    async def test_wait_for_confirmation_not_found(self):
        """Test waiting for a non-existent confirmation."""
        with pytest.raises(ValueError, match="No pending confirmation"):
            await wait_for_confirmation("nonexistent")


class TestConfirmationIntegration:
    """Integration tests simulating the full flow."""

    def setup_method(self):
        """Clear pending confirmations before each test."""
        _pending_confirmations.clear()

    @pytest.mark.asyncio
    async def test_full_confirmation_flow_confirmed(self):
        """Test the full flow: create -> wait (background) -> resolve."""
        # Simulate the tool creating a confirmation
        confirmation = create_confirmation(
            "write_file",
            {"path": "/home/user/test.txt", "content": "Hello, world!"}
        )
        confirm_id = confirmation.confirm_id

        # Simulate CLI responding after seeing the confirm event
        async def simulate_cli_response():
            await asyncio.sleep(0.05)
            # CLI calls POST /api/agent/confirm
            resolve_confirmation(confirm_id, confirmed=True, allow_all=False)

        asyncio.create_task(simulate_cli_response())

        # Tool waits for confirmation
        result = await wait_for_confirmation(confirm_id)

        assert result.confirmed is True
        assert result.allow_all is False

        # Cleanup
        cleanup_confirmation(confirm_id)
        assert confirm_id not in _pending_confirmations

    @pytest.mark.asyncio
    async def test_full_confirmation_flow_denied(self):
        """Test the flow when user denies."""
        confirmation = create_confirmation(
            "shell_exec",
            {"command": "rm -rf important_folder"}
        )
        confirm_id = confirmation.confirm_id

        async def simulate_cli_denial():
            await asyncio.sleep(0.05)
            resolve_confirmation(confirm_id, confirmed=False)

        asyncio.create_task(simulate_cli_denial())

        result = await wait_for_confirmation(confirm_id)

        assert result.confirmed is False

        cleanup_confirmation(confirm_id)

    @pytest.mark.asyncio
    async def test_allow_all_skips_future_confirmations(self):
        """Test that allow_all flag can be used to skip future confirmations."""
        # First confirmation with allow_all
        conf1 = create_confirmation("write_file", {"path": "file1.txt", "content": "a"})
        resolve_confirmation(conf1.confirm_id, confirmed=True, allow_all=True)

        result1 = await wait_for_confirmation(conf1.confirm_id)
        assert result1.allow_all is True

        # In real usage, the StreamingToolRegistry would check session_allow_all
        # and skip creating confirmations for subsequent dangerous tools

    @pytest.mark.asyncio
    async def test_multiple_confirmations_independent(self):
        """Test that multiple confirmations are independent."""
        conf1 = create_confirmation("write_file", {"path": "file1.txt", "content": "a"})
        conf2 = create_confirmation("write_file", {"path": "file2.txt", "content": "b"})

        # Resolve them in different order than created
        resolve_confirmation(conf2.confirm_id, confirmed=False)
        resolve_confirmation(conf1.confirm_id, confirmed=True)

        result1 = await wait_for_confirmation(conf1.confirm_id)
        result2 = await wait_for_confirmation(conf2.confirm_id)

        assert result1.confirmed is True
        assert result2.confirmed is False


class TestDangerousToolDetection:
    """Test that dangerous tools are properly marked."""

    def test_write_file_always_registered(self):
        """Test that write_file tool is always registered."""
        from delia.tools.builtins import get_default_tools

        # Without allow_write flag, write_file is registered but dangerous
        registry = get_default_tools(allow_write=False)
        tool = registry.get("write_file")
        assert tool is not None
        assert tool.dangerous is True  # Requires confirmation
        assert tool.permission_level == "write"

        # With allow_write flag, write_file is registered but NOT dangerous (auto-approved)
        registry = get_default_tools(allow_write=True)
        tool = registry.get("write_file")
        assert tool is not None
        assert tool.dangerous is False  # Auto-approved, no confirmation needed
        assert tool.permission_level == "write"

    def test_shell_exec_always_registered(self):
        """Test that shell_exec tool is always registered."""
        from delia.tools.builtins import get_default_tools

        # Without allow_exec flag, shell_exec is registered but dangerous
        registry = get_default_tools(allow_exec=False)
        tool = registry.get("shell_exec")
        assert tool is not None
        assert tool.dangerous is True  # Requires confirmation
        assert tool.permission_level == "exec"

        # With allow_exec flag, shell_exec is registered but NOT dangerous (auto-approved)
        registry = get_default_tools(allow_exec=True)
        tool = registry.get("shell_exec")
        assert tool is not None
        assert tool.dangerous is False  # Auto-approved, no confirmation needed
        assert tool.permission_level == "exec"

    def test_read_only_tools_not_dangerous(self):
        """Test that read-only tools are not marked dangerous."""
        from delia.tools.builtins import get_default_tools

        registry = get_default_tools()

        for tool_name in ["read_file", "list_directory", "search_code", "web_fetch"]:
            tool = registry.get(tool_name)
            assert tool is not None, f"{tool_name} should be registered"
            assert tool.dangerous is False, f"{tool_name} should not be dangerous"
            assert tool.permission_level == "read", f"{tool_name} should be read-only"

    def test_all_tools_registered_by_default(self):
        """Test that all tools including dangerous ones are registered by default."""
        from delia.tools.builtins import get_default_tools

        registry = get_default_tools()

        # All tools should be present
        expected_tools = [
            "read_file", "list_directory", "search_code",
            "web_fetch", "web_search",
            "write_file", "shell_exec"
            # NOTE: delete_file removed per ADR-010 (rarely used)
        ]
        for tool_name in expected_tools:
            assert tool_name in registry, f"{tool_name} should be registered"

    # NOTE: test_delete_file_always_registered removed - delete_file tool removed per ADR-010
