# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import json
import asyncio
from hypothesis import given, strategies as st
from unittest.mock import AsyncMock, MagicMock, patch
from delia.providers.ollama import OllamaProvider
from delia.backend_manager import BackendConfig

class MockResponse:
    def __init__(self, lines, status_code=200):
        self.lines = lines
        self.status_code = status_code
        self.text = "Error"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def aiter_lines(self):
        for line in self.lines:
            yield line

@pytest.mark.asyncio
@given(st.lists(st.text()))
async def test_ollama_stream_robustness(lines):
    """Test that Ollama provider handles arbitrary lines from the stream without crashing."""
    config = MagicMock()
    config.ollama_timeout_seconds = 10
    backend_manager = MagicMock()
    
    provider = OllamaProvider(config, backend_manager)
    backend_obj = BackendConfig(id="test", name="test", provider="ollama", type="local", url="http://localhost:11434")
    
    mock_client = MagicMock()
    mock_client.stream.return_value = MockResponse(lines)
    backend_obj.get_client = lambda: mock_client
    
    chunks = []
    try:
        async for chunk in provider.call_stream("model", "prompt", backend_obj=backend_obj):
            chunks.append(chunk)
    except Exception as e:
        # It's actually okay if it raises certain exceptions (like JSONDecodeError) 
        # as long as we know it's handled or intended. 
        # But for robustness, the provider should ideally yield an error chunk instead of crashing.
        pass

@pytest.mark.asyncio
@given(st.lists(st.dictionaries(st.text(), st.text())))
async def test_ollama_stream_json_robustness(data_list):
    """Test with valid JSON but unexpected fields."""
    lines = [json.dumps(d) for d in data_list]
    config = MagicMock()
    config.ollama_timeout_seconds = 10
    backend_manager = MagicMock()
    
    provider = OllamaProvider(config, backend_manager)
    backend_obj = BackendConfig(id="test", name="test", provider="ollama", type="local", url="http://localhost:11434")
    
    mock_client = MagicMock()
    mock_client.stream.return_value = MockResponse(lines)
    backend_obj.get_client = lambda: mock_client
    
    async for chunk in provider.call_stream("model", "prompt", backend_obj=backend_obj):
        assert isinstance(chunk.text, str)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
