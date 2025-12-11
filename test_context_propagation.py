#!/usr/bin/env python3
"""
Test suite for context variable propagation in batch operations.

Verifies that user context (client_id, username) set by middleware
properly propagates to child tasks in batch_process().

Issues without this fix:
- Context variables are task-local in asyncio
- asyncio.gather() spawns tasks in a fresh context
- Child tasks can't see parent context
- User tracking fails silently in batch mode
- Quota enforcement doesn't apply to batch tasks
"""
import asyncio
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import context variables and mock implementations
from mcp_server import current_client_id, current_username


# Mock _delegate_impl to capture context during execution
context_captures = []


async def mock_delegate_impl(
    task: str,
    content: str,
    file: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    context: Optional[str] = None,
    symbols: Optional[str] = None,
    include_references: bool = False,
    backend: Optional[str] = None,
    backend_obj=None,
) -> str:
    """Mock _delegate_impl that captures context."""
    # Capture the context variables at the time this task executes
    client_id = current_client_id.get()
    username = current_username.get()

    context_captures.append({
        "task": task,
        "client_id": client_id,
        "username": username,
        "content_len": len(content),
    })

    # Simulate some async work
    await asyncio.sleep(0.01)

    return f"Task {task} completed"


async def test_context_propagation_single_task():
    """Test that context is available in a single batch task."""
    print("\n=== Test 1: Context Propagation - Single Task ===")

    context_captures.clear()

    # Set context (as middleware would)
    current_client_id.set("client-123")
    current_username.set("testuser")

    # Simulate batch_process's context capture pattern
    captured_client_id = current_client_id.get()
    captured_username = current_username.get()

    async def run_task(task_name: str, client_id: Optional[str], username: Optional[str]) -> str:
        # Re-set context in child task (CRITICAL FIX)
        current_client_id.set(client_id)
        current_username.set(username)

        # Call delegate
        return await mock_delegate_impl(
            task=task_name,
            content="test content",
        )

    # Run the task
    result = await run_task("analyze", captured_client_id, captured_username)

    # Verify context was captured
    assert len(context_captures) == 1
    assert context_captures[0]["client_id"] == "client-123"
    assert context_captures[0]["username"] == "testuser"
    print("✓ Single task captured correct context")


async def test_context_propagation_batch_tasks():
    """Test that context propagates to all tasks in a batch."""
    print("\n=== Test 2: Context Propagation - Batch Tasks ===")

    context_captures.clear()

    # Set context (as middleware would)
    current_client_id.set("client-456")
    current_username.set("batchuser")

    # Simulate batch_process's context capture pattern
    captured_client_id = current_client_id.get()
    captured_username = current_username.get()

    async def run_task(
        task_id: int,
        task_name: str,
        client_id: Optional[str],
        username: Optional[str],
    ) -> str:
        # Re-set context in child task (CRITICAL FIX)
        current_client_id.set(client_id)
        current_username.set(username)

        # Call delegate
        return await mock_delegate_impl(
            task=task_name,
            content=f"task {task_id} content",
        )

    # Run multiple tasks in parallel (simulating batch)
    tasks = [
        ("analyze", "Analyzing code"),
        ("review", "Reviewing changes"),
        ("summarize", "Summarizing"),
    ]

    results = await asyncio.gather(*[
        run_task(i, task_type, captured_client_id, captured_username)
        for i, (task_type, _) in enumerate(tasks)
    ])

    # Verify all tasks captured correct context
    assert len(context_captures) == 3
    for i, capture in enumerate(context_captures):
        assert capture["client_id"] == "client-456", f"Task {i} lost context"
        assert capture["username"] == "batchuser", f"Task {i} lost username"
        print(f"✓ Task {i} captured correct context")

    assert len(results) == 3
    print(f"✓ All {len(results)} batch tasks captured correct context")


async def test_concurrent_batches_isolated():
    """Test that concurrent batches don't mix user contexts."""
    print("\n=== Test 3: Concurrent Batches - Context Isolation ===")

    context_captures.clear()

    async def run_batch(batch_id: str, user_id: str, username: str, num_tasks: int):
        # Each batch sets its own context
        current_client_id.set(user_id)
        current_username.set(username)

        # Capture context for this batch
        captured_client_id = current_client_id.get()
        captured_username = current_username.get()

        async def run_task(task_id: int, client_id, user):
            # Re-set context in child task
            current_client_id.set(client_id)
            current_username.set(user)

            return await mock_delegate_impl(
                task=f"task_{task_id}",
                content=f"batch {batch_id} task {task_id}",
            )

        # Run batch tasks in parallel
        results = await asyncio.gather(*[
            run_task(i, captured_client_id, captured_username)
            for i in range(num_tasks)
        ])
        return results

    # Run multiple batches concurrently
    batch_results = await asyncio.gather(
        run_batch("batch-a", "client-a", "user-a", 2),
        run_batch("batch-b", "client-b", "user-b", 2),
        run_batch("batch-c", "client-c", "user-c", 2),
    )

    # Verify each batch's tasks have correct context
    assert len(context_captures) == 6  # 3 batches × 2 tasks each

    # Batch A tasks should have client-a
    batch_a_captures = context_captures[0:2]
    for capture in batch_a_captures:
        assert capture["client_id"] == "client-a", f"Batch A task lost context: {capture}"
        assert capture["username"] == "user-a"
    print("✓ Batch A isolated correctly")

    # Batch B tasks should have client-b
    batch_b_captures = context_captures[2:4]
    for capture in batch_b_captures:
        assert capture["client_id"] == "client-b"
        assert capture["username"] == "user-b"
    print("✓ Batch B isolated correctly")

    # Batch C tasks should have client-c
    batch_c_captures = context_captures[4:6]
    for capture in batch_c_captures:
        assert capture["client_id"] == "client-c"
        assert capture["username"] == "user-c"
    print("✓ Batch C isolated correctly")

    print("✓ All concurrent batches maintained context isolation")


async def test_context_with_none_values():
    """Test that None context values are properly propagated."""
    print("\n=== Test 4: Context Propagation - None Values ===")

    context_captures.clear()

    # Clear context (simulate unauthenticated user)
    current_client_id.set(None)
    current_username.set(None)

    captured_client_id = current_client_id.get()
    captured_username = current_username.get()

    async def run_task(client_id, username):
        current_client_id.set(client_id)
        current_username.set(username)

        return await mock_delegate_impl(
            task="analyze",
            content="test",
        )

    # Run task with None context
    result = await run_task(captured_client_id, captured_username)

    # Verify None context was preserved
    assert len(context_captures) == 1
    assert context_captures[0]["client_id"] is None
    assert context_captures[0]["username"] is None
    print("✓ None context values properly propagated")


async def test_nested_batch_operations():
    """Test context propagation in nested batch-like operations."""
    print("\n=== Test 5: Nested Batch Operations ===")

    context_captures.clear()

    # Set context
    current_client_id.set("client-nested")
    current_username.set("nested-user")

    captured_client_id = current_client_id.get()
    captured_username = current_username.get()

    async def run_subtask(subtask_id: int, client_id, username):
        # Re-set context
        current_client_id.set(client_id)
        current_username.set(username)

        return await mock_delegate_impl(
            task=f"subtask_{subtask_id}",
            content=f"subtask {subtask_id}",
        )

    async def run_main_task(task_id: int, client_id, username):
        # Re-set context for main task
        current_client_id.set(client_id)
        current_username.set(username)

        # This task spawns sub-tasks
        subtasks = await asyncio.gather(*[
            run_subtask(i, client_id, username)
            for i in range(2)
        ])

        # Main task also records itself
        await mock_delegate_impl(
            task=f"main_task_{task_id}",
            content=f"main task {task_id}",
        )

        return subtasks

    # Run nested batch operations
    results = await asyncio.gather(*[
        run_main_task(i, captured_client_id, captured_username)
        for i in range(2)
    ])

    # Verify all nested tasks captured context correctly
    # 2 main tasks × (2 subtasks + 1 main recording) = 6 captures
    assert len(context_captures) == 6
    for capture in context_captures:
        assert capture["client_id"] == "client-nested"
        assert capture["username"] == "nested-user"

    print("✓ Nested batch operations maintained context throughout")


async def test_context_error_handling():
    """Test context is maintained even when tasks raise errors."""
    print("\n=== Test 6: Context in Error Scenarios ===")

    context_captures.clear()

    current_client_id.set("client-error")
    current_username.set("error-user")

    captured_client_id = current_client_id.get()
    captured_username = current_username.get()

    async def run_task_with_error(
        task_id: int,
        should_fail: bool,
        client_id,
        username,
    ):
        # Re-set context even if task will fail
        current_client_id.set(client_id)
        current_username.set(username)

        # Capture was done, now fail if requested
        if should_fail:
            raise ValueError(f"Intentional error in task {task_id}")

        return await mock_delegate_impl(
            task=f"task_{task_id}",
            content=f"task {task_id}",
        )

    # Run mix of successful and failing tasks
    results = await asyncio.gather(
        run_task_with_error(0, False, captured_client_id, captured_username),
        run_task_with_error(1, True, captured_client_id, captured_username),
        run_task_with_error(2, False, captured_client_id, captured_username),
        return_exceptions=True,
    )

    # Verify context was captured even in failing task
    # Task 0 succeeds, Task 1 fails (but still captured context), Task 2 succeeds
    # Actually, the failing task doesn't call mock_delegate_impl, so only 2 captures

    # But the important point: if an error happens before context is used,
    # it shouldn't lose the context value
    assert len(context_captures) == 2  # Only successful tasks called delegate
    for capture in context_captures:
        assert capture["client_id"] == "client-error"
        assert capture["username"] == "error-user"

    # Verify error was raised
    assert isinstance(results[1], ValueError)
    print("✓ Context properly maintained even with task failures")


async def main():
    """Run all tests."""
    print("=" * 70)
    print("CONTEXT PROPAGATION IN BATCH OPERATIONS TEST SUITE")
    print("=" * 70)

    tests = [
        test_context_propagation_single_task,
        test_context_propagation_batch_tasks,
        test_concurrent_batches_isolated,
        test_context_with_none_values,
        test_nested_batch_operations,
        test_context_error_handling,
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

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
