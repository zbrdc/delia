# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import asyncio
import json
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch, AsyncMock
from delia.api import app

@pytest.mark.asyncio
async def test_api_health_endpoint():
    """Test that the health endpoint returns success."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_api_sessions_list():
    """Test listing sessions via API."""
    with patch("delia.session_manager.SessionManager") as mock_sm_class:
        mock_sm_instance = mock_sm_class.return_value
        mock_sm_instance.list_sessions.return_value = []
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

@pytest.mark.asyncio
async def test_api_agent_run_sse_trigger():
    """Test that agent run endpoint initiates SSE stream."""
    payload = {
        "task": "hello", # handler expects 'task' not 'message'
        "session_id": "test-session"
    }
    
    async def mock_generator(*args, **kwargs):
        yield "event: thought\ndata: {\"text\": \"thinking\"}\n\n"
        yield "event: response\ndata: {\"text\": \"done\"}\n\n"

    # Patch the async generator function used by the handler
    with patch("delia.api.agent_run_stream") as mock_stream:
        mock_stream.side_effect = mock_generator
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/agent/run", json=payload)
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]