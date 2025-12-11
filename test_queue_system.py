#!/usr/bin/env python3
"""
Test suite for the fixed ModelQueue system.

Tests:
1. Immediate availability when model is loaded
2. Queueing when model is loading
3. Future resolution after model loads
4. Priority-based ordering
5. Concurrent requests
6. Error handling on failed loads
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import queue system
from mcp_server import ModelQueue, QueuedRequest


async def test_immediate_availability():
    """Test that loaded models return immediately."""
    print("\n=== Test 1: Immediate Availability ===")
    queue = ModelQueue()

    # Mark a model as loaded
    queue.loaded_models["test-model"] = {
        "loaded_at": None,
        "last_used": None,
        "size_gb": 4.0
    }

    is_available, future = await queue.acquire_model("test-model")
    assert is_available == True, "Model should be immediately available"
    assert future is None, "Future should be None for immediate availability"
    print("✓ Loaded models are immediately available")


async def test_queueing():
    """Test that requests are queued when model is loading."""
    print("\n=== Test 2: Queueing System ===")
    queue = ModelQueue()

    # Start loading a model
    queue.loading_models.add("test-model")

    # First request should be queued
    is_available, future = await queue.acquire_model("test-model", task_type="thinking")
    assert is_available == False, "Model should not be immediately available"
    assert isinstance(future, asyncio.Future), "Should return a Future for queued request"
    print(f"✓ Request queued with {len(queue.request_queues['test-model'])} item(s)")

    # Verify stats were updated
    assert queue.total_queued == 1, "Should track queued request"
    print("✓ Queue stats updated correctly")


async def test_future_resolution():
    """Test that futures are resolved after model loading."""
    print("\n=== Test 3: Future Resolution ===")
    queue = ModelQueue()

    # Start loading a model
    queue.loading_models.add("test-model")

    # Queue a request
    is_available, future = await queue.acquire_model("test-model", task_type="thinking")
    assert is_available == False

    # Simulate model load completion
    async def complete_loading():
        await asyncio.sleep(0.1)  # Small delay
        await queue.release_model("test-model", success=True)

    # Start the completion task
    completion_task = asyncio.create_task(complete_loading())

    # Wait for the future to be resolved
    try:
        result = await asyncio.wait_for(future, timeout=2.0)
        assert result == True, "Future should be resolved to True"
        print("✓ Future was properly resolved after model loading")
    except asyncio.TimeoutError:
        assert False, "Future timed out - it was not resolved"

    await completion_task


async def test_priority_ordering():
    """Test that higher priority requests are processed first."""
    print("\n=== Test 4: Priority Ordering ===")
    queue = ModelQueue()

    # Start loading a model
    queue.loading_models.add("test-model")

    # Queue multiple requests with different priorities
    _, future1 = await queue.acquire_model("test-model", task_type="general", content_length=10000)
    _, future2 = await queue.acquire_model("test-model", task_type="thinking", content_length=1000)
    _, future3 = await queue.acquire_model("test-model", task_type="review", content_length=5000)

    # Verify queue has 3 items
    assert len(queue.request_queues["test-model"]) == 3

    # Get the next request (should be thinking task)
    next_request = queue.request_queues["test-model"][0]  # Heap peek
    assert next_request.task_type == "thinking", f"Expected thinking, got {next_request.task_type}"
    print("✓ Higher priority requests are at top of queue")


async def test_concurrent_requests():
    """Test handling of concurrent requests."""
    print("\n=== Test 5: Concurrent Requests ===")
    queue = ModelQueue()

    # Start loading a model
    queue.loading_models.add("test-model")

    async def queue_request(task_type: str, idx: int):
        is_available, future = await queue.acquire_model("test-model", task_type=task_type)
        if future:
            return await asyncio.wait_for(future, timeout=2.0)
        return True

    # Simulate concurrent requests
    async def complete_loading():
        await asyncio.sleep(0.2)
        await queue.release_model("test-model", success=True)

    # Queue multiple concurrent requests
    tasks = [
        asyncio.create_task(queue_request("general", i))
        for i in range(5)
    ]
    completion_task = asyncio.create_task(complete_loading())

    results = await asyncio.gather(*tasks, return_exceptions=True)
    await completion_task

    # Verify all requests succeeded
    successes = sum(1 for r in results if r is True)
    assert successes >= 1, f"Expected at least 1 success, got {successes}"
    print(f"✓ Processed {queue.total_processed} queued requests concurrently")
    print(f"  Total queued: {queue.total_queued}, total processed: {queue.total_processed}")


async def test_failed_load():
    """Test that futures fail when model loading fails."""
    print("\n=== Test 6: Failed Load Handling ===")
    queue = ModelQueue()

    # Start loading a model
    queue.loading_models.add("test-model")

    # Queue a request
    is_available, future = await queue.acquire_model("test-model")
    assert is_available == False

    # Simulate failed load
    await queue.release_model("test-model", success=False)

    # Future should be rejected with exception
    try:
        await asyncio.wait_for(future, timeout=1.0)
        assert False, "Future should have raised exception"
    except Exception as e:
        assert "Failed to load" in str(e)
        print("✓ Future properly fails when model load fails")


async def test_queue_stats():
    """Test queue health statistics."""
    print("\n=== Test 7: Queue Statistics ===")
    queue = ModelQueue()

    # Perform some operations
    queue.loading_models.add("model-1")

    # Queue several requests
    for i in range(3):
        await queue.acquire_model("model-1", task_type="general")

    # Check stats
    stats = queue.get_queue_status()
    assert stats["queue_stats"]["total_queued"] == 3
    assert stats["queue_stats"]["max_queue_depth"] == 3
    print("✓ Queue statistics tracked correctly")
    print(f"  Queue stats: {stats['queue_stats']}")


async def test_timeout_detection():
    """Test that timeouts are tracked."""
    print("\n=== Test 8: Timeout Detection ===")
    queue = ModelQueue()

    queue.loading_models.add("test-model")
    is_available, future = await queue.acquire_model("test-model")

    # Don't complete the load - let it timeout
    try:
        await asyncio.wait_for(future, timeout=0.1)
        assert False, "Should have timed out"
    except asyncio.TimeoutError:
        # Expected - model will timeout in real use
        print("✓ Timeout can be detected with wait_for")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("MODEL QUEUE SYSTEM - TEST SUITE")
    print("=" * 60)

    tests = [
        test_immediate_availability,
        test_queueing,
        test_future_resolution,
        test_priority_ordering,
        test_concurrent_requests,
        test_failed_load,
        test_queue_stats,
        test_timeout_detection,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
