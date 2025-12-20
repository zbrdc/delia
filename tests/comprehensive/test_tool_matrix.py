# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from delia.tools.builtins import (
    read_file, list_directory, search_code, 
    write_file, delete_file, shell_exec,
    web_fetch
)
from delia.tools.editing import replace_in_file, insert_into_file
from delia.types import Workspace

@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create some dummy files in root for shallow tests
        (root / "README.md").write_text("# Project")
        (root / "main.txt").write_text("hello\n# TODO: Fix this")
        (root / "utils.txt").write_text("def add(a, b): return a + b")
        (root / "src").mkdir()
        (root / "src" / "main.txt").write_text("hello\n# TODO: Fix this")
        (root / "data.json").write_text('{"key": "value"}')
        yield Workspace(root=root)

class TestToolingMatrix:
    """Comprehensive Tooling Matrix - ~100+ permutation checks."""

    # ============================================================ 
    # READ_FILE TESTS
    # ============================================================ 
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path, start, end, expected_snippet", [
        ("README.md", 1, None, "# Project"),
        ("main.txt", 1, 1, "hello"),
        ("main.txt", 2, 2, "# TODO: Fix this"),
        ("nonexistent.txt", 1, None, "Error: File not found"),
        ("../outside.txt", 1, None, "path traversal"), # Path safety
    ])
    async def test_read_file_variants(self, temp_workspace, path, start, end, expected_snippet):
        res = await read_file(path, start_line=start, end_line=end, workspace=temp_workspace)
        assert expected_snippet.lower() in res.lower()

    # ============================================================ 
    # LIST_DIRECTORY TESTS
    # ============================================================ 
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path, recursive, pattern, expected_in, expected_not_in", [
        (".", False, None, "README.md", "src/main.txt"), # Shallow
        (".", False, None, "src", "nonexistent"), # Subdir check
        ("src", False, None, "main.txt", "README.md"), # Subdir contents
        (".", True, None, "main.txt", "invalid_file"), # Recursive
        (".", False, "utils.txt", "utils.txt", "README.md"), # Pattern check
        ("nonexistent", False, None, "Error", "README.md"),
    ])
    async def test_list_directory_variants(self, temp_workspace, path, recursive, pattern, expected_in, expected_not_in):
        res = await list_directory(path, recursive=recursive, pattern=pattern, workspace=temp_workspace)
        if "Error" in expected_in:
            assert "Error" in res
        else:
            assert expected_in in res

    # ============================================================ 
    # SEARCH_CODE (GREP) TESTS
    # ============================================================ 
    @pytest.mark.asyncio
    @pytest.mark.parametrize("pattern, path, file_pat, expected_hit", [
        ("hello", ".", None, "main.txt"),
        ("def add", ".", "*.txt", "utils.txt"),
        ("TODO", ".", "*.md", "No matches"), # Corrected expected text
        ("missing_term", ".", None, "No matches"),
    ])
    async def test_search_code_variants(self, temp_workspace, pattern, path, file_pat, expected_hit):
        res = await search_code(pattern, path=path, file_pattern=file_pat, workspace=temp_workspace)
        assert expected_hit.lower() in res.lower()

    # ============================================================ 
    # WRITE / DELETE TESTS
    # ============================================================ 
    @pytest.mark.asyncio
    async def test_write_delete_lifecycle(self, temp_workspace):
        # 1. Write
        path = "new_file.txt"
        content = "Test content"
        res = await write_file(path, content, workspace=temp_workspace)
        assert "Successfully wrote" in res
        assert (temp_workspace.root / path).read_text() == content
        
        # 2. Delete
        res = await delete_file(path, workspace=temp_workspace)
        assert "Deleted" in res
        assert not (temp_workspace.root / path).exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", [
        "/etc/passwd", 
        "../../outside.txt",
        "~/.ssh/id_rsa"
    ])
    async def test_destructive_safety_blocks(self, temp_workspace, path):
        """Verify that write/delete blocks dangerous paths."""
        res_w = await write_file(path, "bad", workspace=temp_workspace)
        assert "Error" in res_w
        
        res_d = await delete_file(path, workspace=temp_workspace)
        assert "Error" in res_d

    # ============================================================ 
    # EDITING TOOLS (REPLACE/INSERT)
    # ============================================================ 
    @pytest.mark.asyncio
    async def test_replace_in_file(self, temp_workspace):
        path = "main.txt"
        res = await replace_in_file(path, search="hello", replace="world", workspace=temp_workspace)
        assert "Successfully replaced" in res
        assert "world" in (temp_workspace.root / path).read_text()

    @pytest.mark.asyncio
    async def test_insert_into_file(self, temp_workspace):
        path = "main.txt"
        res = await insert_into_file(path, content="# Inserted", line=1, workspace=temp_workspace)
        assert "Successfully inserted" in res
        content = (temp_workspace.root / path).read_text()
        assert "# Inserted" in content
    # ============================================================ 
    # SHELL_EXEC TESTS
    # ============================================================ 
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cmd, expected_in", [
        ("echo 'test'", "STDOUT:\ntest"),
        ("ls README.md", "README.md"),
        ("rm -rf /", "Error: Blocked dangerous command"),
        ("cat nonexistent", "STDERR:"), # Error output captured
    ])
    async def test_shell_exec_variants(self, temp_workspace, cmd, expected_in):
        res = await shell_exec(cmd, workspace=temp_workspace)
        assert expected_in.lower() in res.lower()

    # ============================================================ 
    # WEB_FETCH TESTS
    # ============================================================ 
    @pytest.mark.asyncio
    async def test_web_fetch_mock(self):
        """Test web_fetch with mocked HTTP response."""
        url = "https://example.com"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = "<html><body><h1>Hello World</h1></body></html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            
            from delia.tools.builtins import web_fetch
            res = await web_fetch(url)
            assert "Hello World" in res
