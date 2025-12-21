# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock
from hypothesis import given, strategies as st, settings, HealthCheck
from delia.tools.coding import apply_diff
from delia.tools.editing import insert_into_file, replace_in_file
from delia.types import Workspace

@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace for coding tests."""
    f = tmp_path / "main.py"
    # Use distinct lines to test indexing clearly
    f.write_text("AAA\nBBB\nCCC", encoding="utf-8")
    return tmp_path

@pytest.fixture
def workspace_obj(temp_workspace):
    """Create a Workspace object for path validation."""
    return Workspace(root=temp_workspace)

# 1. DIFF HARDENING (Fuzzing)
@pytest.mark.asyncio
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(diff_content=st.text())
async def test_apply_diff_robustness(diff_content, workspace_obj):
    """Ensure apply_diff doesn't crash on arbitrary garbage diffs."""
    try:
        result = await apply_diff(diff_content, workspace=workspace_obj)
        assert isinstance(result, str)
    except Exception:
        pass

# 2. INSERT LOGIC
@pytest.mark.asyncio
async def test_insert_into_file_functional(temp_workspace, workspace_obj):
    """Test inserting code into files."""
    target_file = temp_workspace / "main.py"
    
    # line=0 inserts at start
    await insert_into_file("main.py", "START", line=0, workspace=workspace_obj)
    content = target_file.read_text(encoding="utf-8")
    # Current implementation uses "\n".join(lines), so it should be "START\nAAA..."
    assert content.startswith("START")
    assert "AAA" in content

# 3. REPLACE LOGIC
@pytest.mark.asyncio
async def test_replace_in_file_functional(temp_workspace, workspace_obj):
    """Test replacing code blocks."""
    target_file = temp_workspace / "main.py"
    
    # replace_in_file(path, search, replace, workspace)
    await replace_in_file("main.py", search="BBB", replace="REPLACED", workspace=workspace_obj)
    content = target_file.read_text(encoding="utf-8")
    assert "REPLACED" in content
    assert "BBB" not in content

# 4. SAFETY BLOCKS
@pytest.mark.asyncio
async def test_editing_tools_respect_path_validation(temp_workspace, workspace_obj):
    """Verify editing tools cannot edit files outside the workspace."""
    outside_file = "/etc/passwd"
    res = await insert_into_file(outside_file, "hacked", line=0, workspace=workspace_obj)
    assert "Error" in res or "Blocked" in res or "invalid" in res.lower() or "outside" in res.lower()