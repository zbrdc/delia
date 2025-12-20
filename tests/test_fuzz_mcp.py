# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import json
import asyncio
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import AsyncMock, MagicMock, patch
from delia.mcp_server import delegate, batch, think, session_create, session_get

# Mock results for the orchestration service
mock_success_result = MagicMock()
mock_success_result.result.success = True
mock_success_result.result.response = "Mocked Response"
mock_success_result.result.model_used = "mock-model"
mock_success_result.result.tokens = 10
mock_success_result.result.elapsed_ms = 100
mock_success_result.quality_score = 0.95

@pytest.mark.asyncio
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(
    task=st.text(),
    content=st.text(),
    file=st.one_of(st.none(), st.text()),
    model=st.one_of(st.none(), st.text()),
    language=st.one_of(st.none(), st.text()),
    context=st.one_of(st.none(), st.text()),
    symbols=st.one_of(st.none(), st.text()),
    include_references=st.booleans(),
    backend_type=st.one_of(st.none(), st.text()),
    files=st.one_of(st.none(), st.text()),
    include_metadata=st.booleans(),
    max_tokens=st.one_of(st.none(), st.integers()),
    dry_run=st.booleans(),
    session_id=st.one_of(st.none(), st.text()),
    stream=st.booleans()
)
async def test_delegate_interface_robustness(
    task, content, file, model, language, context, symbols, 
    include_references, backend_type, files, include_metadata, 
    max_tokens, dry_run, session_id, stream
):
    """Test the delegate tool with arbitrary inputs to ensure no unhandled crashes."""
    # We mock the orchestration service to avoid real LLM calls but keep the tool logic
    with patch("delia.mcp_server.get_orchestration_service") as mock_service_getter:
        mock_service = AsyncMock()
        mock_service.process.return_value = mock_success_result
        mock_service_getter.return_value = mock_service
        
        # We also need to mock some internal globals/dependencies that delegate might hit
        with patch("delia.mcp_server.get_router") as mock_router_getter:
            mock_router = AsyncMock()
            mock_router.select_optimal_backend.return_value = (None, MagicMock(id="mock-backend", type="local"))
            mock_router_getter.return_value = mock_router
            
            try:
                # Call the tool via .fn (the actual function)
                await delegate.fn(
                    task=task,
                    content=content,
                    file=file,
                    model=model,
                    language=language,
                    context=context,
                    symbols=symbols,
                    include_references=include_references,
                    backend_type=backend_type,
                    files=files,
                    include_metadata=include_metadata,
                    max_tokens=max_tokens,
                    dry_run=dry_run,
                    session_id=session_id,
                    stream=stream
                )
            except Exception as e:
                # Top level tools should return error strings, not raise exceptions
                # EXCEPT if it's a known validation error that we want to catch
                pytest.fail(f"delegate crashed with unhandled exception: {e}")

@pytest.mark.asyncio
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(
    tasks_json=st.text(),
    include_metadata=st.booleans(),
    max_tokens=st.one_of(st.none(), st.integers()),
    session_id=st.one_of(st.none(), st.text())
)
async def test_batch_interface_robustness(tasks_json, include_metadata, max_tokens, session_id):
    """Test the batch tool with arbitrary JSON (or garbage)."""
    with patch("delia.mcp_server.get_orchestration_service") as mock_service_getter:
        mock_service = AsyncMock()
        mock_service.process.return_value = mock_success_result
        mock_service_getter.return_value = mock_service
        
        with patch("delia.mcp_server.backend_manager") as mock_bm:
            mock_bm.check_all_health.return_value = {}
            
            try:
                await batch.fn(
                    tasks=tasks_json,
                    include_metadata=include_metadata,
                    max_tokens=max_tokens,
                    session_id=session_id
                )
            except Exception as e:
                pytest.fail(f"batch crashed with input {repr(tasks_json)}: {e}")

@pytest.mark.asyncio
@given(st.text(), st.text(), st.text(), st.one_of(st.none(), st.text()))
async def test_think_interface_robustness(problem, context, depth, session_id):
    """Test the think tool."""
    with patch("delia.mcp_server.get_orchestration_service") as mock_service_getter:
        mock_service = AsyncMock()
        mock_service.process.return_value = mock_success_result
        mock_service_getter.return_value = mock_service
        
        try:
            await think.fn(problem=problem, context=context, depth=depth, session_id=session_id)
        except Exception as e:
            pytest.fail(f"think crashed: {e}")

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
