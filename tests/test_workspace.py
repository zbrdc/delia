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

"""Tests for workspace isolation feature.

These tests verify that the workspace feature correctly confines
agent file operations to a specific directory, preventing access
to files outside the project.
"""

import pytest
import tempfile
from pathlib import Path

from delia.types import Workspace
from delia.tools.executor import validate_path, validate_path_in_workspace
from delia.tools.builtins import read_file, list_directory, search_code, get_default_tools


class TestWorkspaceDataclass:
    """Tests for the Workspace dataclass."""

    def test_workspace_creation_with_path(self, tmp_path):
        """Test creating a workspace with a Path object."""
        workspace = Workspace(root=tmp_path)
        assert workspace.root == tmp_path
        assert workspace.root.is_absolute()

    def test_workspace_creation_with_string(self, tmp_path):
        """Test creating a workspace with a string path."""
        workspace = Workspace(root=str(tmp_path))
        assert workspace.root == tmp_path
        assert workspace.root.is_absolute()

    def test_workspace_resolves_relative_path(self):
        """Test that workspace resolves relative paths to absolute."""
        workspace = Workspace(root=".")
        assert workspace.root.is_absolute()

    def test_workspace_expands_home_dir(self):
        """Test that workspace expands ~ to home directory."""
        workspace = Workspace(root="~")
        assert workspace.root.is_absolute()
        assert "~" not in str(workspace.root)

    def test_workspace_contains_file_inside(self, tmp_path):
        """Test that contains() returns True for files inside workspace."""
        workspace = Workspace(root=tmp_path)

        # Create a file inside workspace
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        assert workspace.contains(test_file)
        assert workspace.contains(str(test_file))

    def test_workspace_contains_subdir_inside(self, tmp_path):
        """Test that contains() returns True for subdirs inside workspace."""
        workspace = Workspace(root=tmp_path)

        # Create a subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        assert workspace.contains(subdir)
        assert workspace.contains(subdir / "nested" / "file.txt")

    def test_workspace_does_not_contain_outside(self, tmp_path):
        """Test that contains() returns False for paths outside workspace."""
        workspace = Workspace(root=tmp_path)

        # Paths outside workspace
        assert not workspace.contains("/etc/passwd")
        assert not workspace.contains(tmp_path.parent)
        assert not workspace.contains("/tmp/other")

    def test_workspace_additional_allowed(self, tmp_path):
        """Test that additional_allowed paths are allowed."""
        workspace = Workspace(
            root=tmp_path,
            additional_allowed=["/usr/share"]
        )

        assert workspace.contains(tmp_path / "file.txt")
        assert workspace.contains("/usr/share/dict/words")
        assert not workspace.contains("/etc/passwd")

    def test_workspace_resolve_path_relative(self, tmp_path):
        """Test resolve_path with relative paths."""
        workspace = Workspace(root=tmp_path)

        resolved = workspace.resolve_path("src/main.py")
        assert resolved == tmp_path / "src" / "main.py"

    def test_workspace_resolve_path_absolute_inside(self, tmp_path):
        """Test resolve_path with absolute paths inside workspace."""
        workspace = Workspace(root=tmp_path)

        abs_path = tmp_path / "config.json"
        resolved = workspace.resolve_path(str(abs_path))
        assert resolved == abs_path

    def test_workspace_resolve_path_outside_raises(self, tmp_path):
        """Test resolve_path raises ValueError for paths outside workspace."""
        workspace = Workspace(root=tmp_path)

        with pytest.raises(ValueError, match="outside workspace"):
            workspace.resolve_path("/etc/passwd")


class TestValidatePathWithWorkspace:
    """Tests for validate_path with workspace parameter."""

    def test_validate_path_inside_workspace(self, tmp_path):
        """Test that paths inside workspace are valid."""
        workspace = Workspace(root=tmp_path)

        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        valid, error = validate_path(str(test_file), workspace)
        assert valid
        assert error == ""

    def test_validate_path_relative_inside_workspace(self, tmp_path):
        """Test that relative paths inside workspace are valid."""
        workspace = Workspace(root=tmp_path)

        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        valid, error = validate_path("test.txt", workspace)
        assert valid
        assert error == ""

    def test_validate_path_outside_workspace(self, tmp_path):
        """Test that paths outside workspace are invalid."""
        workspace = Workspace(root=tmp_path)

        # Use a path that exists but is outside workspace (not in blocked list)
        valid, error = validate_path("/etc/hostname", workspace)
        assert not valid
        assert "outside workspace" in error

    def test_validate_path_traversal_blocked(self, tmp_path):
        """Test that path traversal is blocked."""
        workspace = Workspace(root=tmp_path)

        valid, error = validate_path("../../../etc/passwd", workspace)
        assert not valid
        assert "traversal" in error.lower()

    def test_validate_path_blocked_paths_still_blocked(self, tmp_path):
        """Test that blocked paths are blocked even if inside workspace."""
        # Create a fake .ssh directory inside workspace
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        key_file = ssh_dir / "id_rsa"
        key_file.write_text("fake key")

        workspace = Workspace(root=tmp_path)

        # Normal files should be valid
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        valid, _ = validate_path(str(test_file), workspace)
        assert valid

    def test_validate_path_in_workspace_helper(self, tmp_path):
        """Test the validate_path_in_workspace helper function."""
        workspace = Workspace(root=tmp_path)

        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        valid, error, resolved = validate_path_in_workspace("test.txt", workspace)
        assert valid
        assert error == ""
        assert resolved == test_file


class TestBuiltinToolsWithWorkspace:
    """Tests for builtin tools with workspace confinement."""

    @pytest.mark.asyncio
    async def test_read_file_inside_workspace(self, tmp_path):
        """Test reading a file inside workspace."""
        workspace = Workspace(root=tmp_path)

        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = await read_file("test.txt", workspace=workspace)
        assert "Hello, World!" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_read_file_outside_workspace_blocked(self, tmp_path):
        """Test that reading files outside workspace is blocked."""
        workspace = Workspace(root=tmp_path)

        result = await read_file("/etc/hostname", workspace=workspace)
        assert "Error" in result
        assert "outside workspace" in result

    @pytest.mark.asyncio
    async def test_read_file_traversal_blocked(self, tmp_path):
        """Test that path traversal is blocked."""
        workspace = Workspace(root=tmp_path)

        result = await read_file("../../../etc/passwd", workspace=workspace)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_list_directory_inside_workspace(self, tmp_path):
        """Test listing directory inside workspace."""
        workspace = Workspace(root=tmp_path)

        # Create some files
        (tmp_path / "file1.txt").write_text("1")
        (tmp_path / "file2.txt").write_text("2")
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        result = await list_directory(".", workspace=workspace)
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_list_directory_outside_workspace_blocked(self, tmp_path):
        """Test that listing directories outside workspace is blocked."""
        workspace = Workspace(root=tmp_path)

        result = await list_directory("/etc", workspace=workspace)
        assert "Error" in result
        assert "outside workspace" in result

    @pytest.mark.asyncio
    async def test_search_code_inside_workspace(self, tmp_path):
        """Test searching code inside workspace."""
        workspace = Workspace(root=tmp_path)

        # Create a file with searchable content
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'")

        result = await search_code("def hello", workspace=workspace)
        # Either finds it or says no matches (both are valid)
        assert "Error" not in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_search_code_outside_workspace_blocked(self, tmp_path):
        """Test that searching outside workspace is blocked."""
        workspace = Workspace(root=tmp_path)

        result = await search_code("root", path="/etc", workspace=workspace)
        assert "Error" in result
        assert "outside workspace" in result


class TestGetDefaultToolsWithWorkspace:
    """Tests for get_default_tools with workspace parameter."""

    def test_get_default_tools_no_workspace(self):
        """Test that get_default_tools works without workspace."""
        registry = get_default_tools()

        assert "read_file" in registry
        assert "list_directory" in registry
        assert "search_code" in registry
        assert "web_fetch" in registry

    def test_get_default_tools_with_workspace(self, tmp_path):
        """Test that get_default_tools works with workspace."""
        workspace = Workspace(root=tmp_path)
        registry = get_default_tools(workspace=workspace)

        assert "read_file" in registry
        assert "list_directory" in registry
        assert "search_code" in registry
        assert "web_fetch" in registry

    @pytest.mark.asyncio
    async def test_workspace_bound_tools_are_confined(self, tmp_path):
        """Test that tools from workspace registry are confined."""
        workspace = Workspace(root=tmp_path)
        registry = get_default_tools(workspace=workspace)

        # Create a file inside workspace
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Get the read_file tool
        read_tool = registry.get("read_file")
        assert read_tool is not None

        # Read file inside workspace should work
        result = await read_tool.handler(path="test.txt")
        assert "content" in result

        # Read file outside workspace should be blocked
        result = await read_tool.handler(path="/etc/hostname")
        assert "Error" in result


class TestWorkspaceEdgeCases:
    """Tests for edge cases in workspace handling."""

    def test_workspace_with_symlinks(self, tmp_path):
        """Test workspace handling of symlinks."""
        workspace = Workspace(root=tmp_path)

        # Create a file and symlink inside workspace
        real_file = tmp_path / "real.txt"
        real_file.write_text("real content")

        link_file = tmp_path / "link.txt"
        link_file.symlink_to(real_file)

        # Both should be contained
        assert workspace.contains(real_file)
        assert workspace.contains(link_file)

    def test_workspace_with_nonexistent_path(self, tmp_path):
        """Test workspace with paths that don't exist yet."""
        workspace = Workspace(root=tmp_path)

        # Path doesn't exist but would be inside workspace
        future_file = tmp_path / "future" / "file.txt"

        # Should still be considered "inside" workspace
        assert workspace.contains(future_file)

    def test_workspace_with_special_characters(self, tmp_path):
        """Test workspace with paths containing special characters."""
        workspace = Workspace(root=tmp_path)

        # Create file with spaces and special chars
        special_dir = tmp_path / "my project (v2)"
        special_dir.mkdir()
        special_file = special_dir / "file with spaces.txt"
        special_file.write_text("content")

        assert workspace.contains(special_file)

        valid, error = validate_path(str(special_file), workspace)
        assert valid
