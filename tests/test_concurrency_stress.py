# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import asyncio
import time
import random
from unittest.mock import MagicMock, patch, AsyncMock
from delia.delegation import delegate_impl as delegate
from delia.container import get_container

container = get_container()
stats_service = container.stats_service
model_queue = container.model_queue

from delia.backend_manager import backend_manager, BackendConfig

@pytest.mark.asyncio
async def test_high_concurrency_load():
    """Simulate 20 concurrent users hitting the system."""
    
    # 1. Setup mock backends
    mock_backend = BackendConfig(
        id="stress-backend",
        name="Stress Backend",
        provider="ollama",
        type="local",
        url="http://localhost:11434",
        enabled=True,
        models={"quick": "qwen3:7b", "coder": "qwen2.5-coder:7b", "moe": "qwen3:14b"}
    )

    # Mocking essential components - patch backend_manager which is used throughout
    with patch.object(backend_manager, "get_active_backend") as mock_ab, \
         patch.object(backend_manager, "get_enabled_backends") as mock_eb, \
         patch("delia.llm.call_llm") as mock_call, \
         patch("delia.delegation.get_affinity_tracker") as mock_at:
        
        mock_ab.return_value = mock_backend
        mock_eb.return_value = [mock_backend]
        mock_at.return_value = MagicMock()
        
        # Simulate varying latency for LLM calls
        async def slow_call(*args, **kwargs):
            await asyncio.sleep(random.uniform(0.01, 0.1))
            # Manually record stats since we've patched call_llm
            stats_service.record_call(
                model_tier=kwargs.get("task_type", "quick"),
                task_type=kwargs.get("task_type", "quick"),
                original_task=kwargs.get("original_task", "stress-test"),
                tokens=10,
                elapsed_ms=50,
                content_preview="...",
                enable_thinking=False,
                backend="stress-backend"
            )
            return {"success": True, "response": "OK", "tokens": 10}
        
        mock_call.side_effect = slow_call
        
        # 2. Define concurrent worker
        async def simulated_user(user_id):
            task_type = random.choice(["quick", "generate", "analyze", "plan"])
            try:
                # Call implementation directly
                result = await delegate(
                    task=task_type,
                    content=f"Stress test task {user_id}",
                    session_id=f"stress-session-{user_id}"
                )
                return True
            except Exception as e:
                print(f"User {user_id} failed: {e}")
                return False

        # 3. Launch 20 concurrent requests
        start_time = time.time()
        tasks = [simulated_user(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # 4. Verify results
        assert all(results), "Not all concurrent requests succeeded"
        print(f"Concurrency test finished: 20 requests in {duration:.2f}s")
        
        # Check if stats were tracked correctly (thread safety)
        # 20 requests + whatever was there before
        usage, _, _, _ = stats_service.get_snapshot()
        total_after = sum(t["calls"] for t in usage.values())
        assert total_after >= 20, f"Expected at least 20 calls, got {total_after}"

@pytest.mark.asyncio
async def test_model_queue_contention():
    """Specifically test the ModelQueue's ability to handle simultaneous acquisition requests."""
    # Use a fresh queue to avoid interference
    from delia.queue import ModelQueue
    queue = ModelQueue()
    
    # Simulate many tasks trying to acquire different and same models
    async def acquire_release(model_name, tid):
        is_avail, future = await queue.acquire_model(model_name, "test", 100, "ollama")
        if not is_avail and future:
            await future
        
        # Simulate work
        await asyncio.sleep(0.01)
        
        # In real code, release_model is ALWAYS called after use
        await queue.release_model(model_name, True, "ollama")
        return True

    models = ["model-a", "model-b", "model-a", "model-c", "model-b"]
    tasks = [acquire_release(m, i) for i, m in enumerate(models * 4)] # 20 contention points
    
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        assert all(results)
    except asyncio.TimeoutError:
        pytest.fail("ModelQueue contention test timed out - likely a deadlock!")
    
    assert len(queue.loading_models) == 0
    # print(f"DEBUG: queue stats: {queue.get_queue_status()}")

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-s"])
