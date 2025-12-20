# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import tempfile
from unittest.mock import AsyncMock, patch
from pathlib import Path
from delia.tools.lsp import lsp_goto_definition, lsp_find_references, lsp_hover
from delia.types import Workspace

@pytest.mark.asyncio
async def test_lsp_goto_definition_success():
    """Test successful goto definition."""
    mock_results = [
        {"path": "src/delia/api.py", "line": 10, "character": 5}
    ]
    
    with patch("delia.tools.lsp.get_lsp_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.goto_definition.return_value = mock_results
        
        result = await lsp_goto_definition("src/delia/api.py", 124, 5)
        
        assert "Found 1 definition(s)" in result
        assert "src/delia/api.py line 10, char 5" in result
        mock_client.goto_definition.assert_called_once_with("src/delia/api.py", 124, 5)

@pytest.mark.asyncio
async def test_lsp_goto_definition_none():
    """Test goto definition with no results."""
    with patch("delia.tools.lsp.get_lsp_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.goto_definition.return_value = []
        
        result = await lsp_goto_definition("src/delia/api.py", 124, 5)
        
        assert "No definition found." in result

@pytest.mark.asyncio
async def test_lsp_find_references():
    """Test finding references."""
    mock_results = [
        {"path": "src/delia/api.py", "line": 10, "character": 5},
        {"path": "src/delia/orchestration/service.py", "line": 20, "character": 0}
    ]
    
    with patch("delia.tools.lsp.get_lsp_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.find_references.return_value = mock_results
        
        result = await lsp_find_references("src/delia/api.py", 124, 5)
        
        assert "Found 2 reference(s)" in result
        assert "src/delia/api.py line 10, char 5" in result
        assert "src/delia/orchestration/service.py line 20, char 0" in result

@pytest.mark.asyncio
async def test_lsp_hover():
    """Test hover info."""
    mock_hover_text = "### function get_orchestration_service\nReturns the global service instance."
    
    with patch("delia.tools.lsp.get_lsp_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        mock_client.hover.return_value = mock_hover_text
        
        result = await lsp_hover("src/delia/api.py", 124, 5)
        
        assert result == mock_hover_text
        assert "get_orchestration_service" in result

@pytest.mark.asyncio
async def test_lsp_tools_with_workspace():
    """Test LSP tools respect workspace root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Workspace(root=Path(tmpdir))
        
        with patch("delia.tools.lsp.get_lsp_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            mock_client.goto_definition.return_value = []
            
            await lsp_goto_definition("test.py", 1, 1, workspace=workspace)
            mock_get_client.assert_called_once_with(Path(tmpdir))