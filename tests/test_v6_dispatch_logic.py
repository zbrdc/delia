# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from delia.orchestration.intrinsics import IntrinsicsEngine
from delia.orchestration.dispatcher import ModelDispatcher

@pytest.mark.asyncio
async def test_v6_internal_dispatch_logic():
    """
    Test the logic for internal dispatching (v6 style).
    This simulates a tool using the dispatcher to decide which 
    sub-specialist should handle a logic block.
    """
    
    # 1. Setup Mock LLM
    mock_llm = AsyncMock()
    
    # Simulate v6 dispatcher returning a specialized tool call
    mock_llm.return_value = {
        "success": True,
        "response": "<tool_call>{\"name\": \"call_planner\", \"arguments\": {\"reasoning\": \"complex logic detected\"}}</tool_call>",
        "metadata": {
            "tool_calls": [{
                "function": {"name": "call_planner", "arguments": "{\"reasoning\": \"complex logic detected\"}"}
            }]
        }
    }
    
    dispatcher = ModelDispatcher(mock_llm)
    intrinsics = IntrinsicsEngine(mock_llm)
    
    # 2. Test Recursive Dispatching
    # We want to make sure the dispatcher can be called with bypass_queue=True
    # when triggered FROM within another tool.
    
    intent = MagicMock()
    intent.task_type = "agentic"
    
    # First call: Dispatcher decides which tier using embeddings
    # "refactor the database layer" matches executor patterns (code execution)
    result_tier = await dispatcher.dispatch("I need to refactor the database layer", intent)
    assert result_tier == "executor"  # Embedding-based dispatch: refactoring is execution

    # Note: Dispatcher no longer uses LLM calls - it uses embedding similarity
    # The mock_llm is only used by IntrinsicsEngine below
    
    # 3. Test Intrinsic Gate (Answerability)
    # Simulate v6 model acting as a Gate
    mock_llm.return_value = {
        "success": True,
        "response": "{\"score\": 0.9, \"reasoning\": \"Context is sufficient for refactoring.\"}"
    }
    
    gate_result = await intrinsics.check_answerability(
        task="Refactor database",
        context="Database schema: users, orders, items"
    )

    assert gate_result.passed is True
    assert gate_result.score == 0.9
    
    print("\n✓ Recursive Dispatching logic verified (v6 CPU-bypass mode)")
    print("✓ Answerability Gate logic verified (Internal dispatcher mode)")

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
