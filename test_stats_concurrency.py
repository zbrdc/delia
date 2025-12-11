#!/usr/bin/env python3
"""
Test suite for concurrent stats updates.

Verifies that the stats system properly handles:
1. Concurrent stat updates
2. Concurrent saves
3. Atomic snapshots
4. No data loss under high concurrency
5. Stats consistency
"""
import asyncio
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import stats functions
from mcp_server import (
    _update_stats_sync,
    _snapshot_stats,
    save_all_stats_async,
    MODEL_USAGE,
    TASK_STATS,
    RECENT_CALLS,
    _stats_thread_lock,
    _stats_lock,
)


async def test_concurrent_updates():
    """Test concurrent stat updates don't cause corruption."""
    print("\n=== Test 1: Concurrent Updates ===")

    # Reset stats
    with _stats_thread_lock:
        MODEL_USAGE["quick"]["calls"] = 0
        TASK_STATS["general"] = 0
        RECENT_CALLS.clear()

    # Simulate concurrent updates
    async def update_stats(idx: int):
        for i in range(10):
            _update_stats_sync(
                model_tier="quick",
                task_type="general",
                original_task="test",
                tokens=100,
                elapsed_ms=50,
                content_preview=f"Request {idx}-{i}",
                enable_thinking=False,
                backend="ollama"
            )
            # Small delay to increase chance of concurrent access
            await asyncio.sleep(0.001)

    # Run 5 concurrent update tasks
    await asyncio.gather(*[update_stats(i) for i in range(5)])

    # Verify all updates were recorded
    with _stats_thread_lock:
        total_calls = MODEL_USAGE["quick"]["calls"]
        total_tasks = TASK_STATS["general"]
        recent_count = len(RECENT_CALLS)

    expected = 50  # 5 tasks × 10 updates each
    assert total_calls == expected, f"Expected {expected} calls, got {total_calls}"
    assert total_tasks == expected, f"Expected {expected} tasks, got {total_tasks}"
    assert recent_count == expected, f"Expected {expected} recent calls, got {recent_count}"

    print(f"✓ All {expected} concurrent updates recorded correctly")
    print(f"  - MODEL_USAGE calls: {total_calls}")
    print(f"  - TASK_STATS: {total_tasks}")
    print(f"  - RECENT_CALLS: {recent_count}")


async def test_snapshot_consistency():
    """Test that snapshots are consistent even during updates."""
    print("\n=== Test 2: Snapshot Consistency ===")

    # Reset stats
    with _stats_thread_lock:
        MODEL_USAGE["coder"]["calls"] = 5
        TASK_STATS["thinking"] = 3
        RECENT_CALLS.clear()
        for i in range(5):
            RECENT_CALLS.append({"idx": i})

    # Take a snapshot
    usage_snap, task_snap, _, recent_snap = _snapshot_stats()

    # Verify snapshot has expected values
    assert usage_snap["coder"]["calls"] == 5, "Usage snapshot incorrect"
    assert task_snap["thinking"] == 3, "Task snapshot incorrect"
    assert len(recent_snap) == 5, "Recent calls snapshot incorrect"

    print("✓ Snapshot captures all stats correctly")
    print(f"  - Usage snapshot: {usage_snap['coder']}")
    print(f"  - Task snapshot: {task_snap}")
    print(f"  - Recent calls: {len(recent_snap)} items")


async def test_concurrent_snapshots():
    """Test that concurrent snapshots don't corrupt stats."""
    print("\n=== Test 3: Concurrent Snapshots ===")

    # Reset stats
    with _stats_thread_lock:
        MODEL_USAGE["moe"]["calls"] = 0
        TASK_STATS["analysis"] = 0
        RECENT_CALLS.clear()

    # Concurrent updates and snapshots
    async def update_and_snapshot(idx: int, results: list):
        for i in range(20):
            _update_stats_sync(
                model_tier="moe",
                task_type="analysis",
                original_task="test",
                tokens=100,
                elapsed_ms=50,
                content_preview=f"Request {idx}-{i}",
                enable_thinking=False,
                backend="ollama"
            )

            # Take snapshots while updates happen
            if i % 5 == 0:
                snap = _snapshot_stats()
                results.append(snap)

            await asyncio.sleep(0.001)

    results = []
    await asyncio.gather(
        update_and_snapshot(0, results),
        update_and_snapshot(1, results),
        update_and_snapshot(2, results),
    )

    # Verify all snapshots are internally consistent
    for snap_idx, (usage_snap, task_snap, _, recent_snap) in enumerate(results):
        # In each snapshot, values should match
        calls = usage_snap["moe"]["calls"]
        tasks = task_snap["analysis"]
        recent = len(recent_snap)

        # These should be in sync
        assert calls == tasks, f"Snapshot {snap_idx}: calls ({calls}) != tasks ({tasks})"
        assert recent <= calls, f"Snapshot {snap_idx}: recent ({recent}) > calls ({calls})"

    final_snap = _snapshot_stats()
    final_calls = final_snap[0]["moe"]["calls"]
    final_tasks = final_snap[1]["analysis"]

    print(f"✓ All snapshots internally consistent")
    print(f"  - Snapshots taken: {len(results)}")
    print(f"  - Final calls: {final_calls}")
    print(f"  - Final tasks: {final_tasks}")


async def test_save_during_updates():
    """Test that saves work correctly during concurrent updates."""
    print("\n=== Test 4: Save During Updates ===")

    # Reset stats
    with _stats_thread_lock:
        MODEL_USAGE["quick"]["calls"] = 0
        TASK_STATS["general"] = 0
        RECENT_CALLS.clear()

    # Concurrent updates and saves
    async def updates_task():
        for i in range(30):
            _update_stats_sync(
                model_tier="quick",
                task_type="general",
                original_task="test",
                tokens=100,
                elapsed_ms=50,
                content_preview=f"Request {i}",
                enable_thinking=False,
                backend="ollama"
            )
            await asyncio.sleep(0.001)

    async def saves_task():
        for i in range(3):
            await asyncio.sleep(0.01)
            await save_all_stats_async()

    # Run updates and saves concurrently
    await asyncio.gather(
        updates_task(),
        saves_task(),
    )

    # Verify final state
    final_snap = _snapshot_stats()
    final_calls = final_snap[0]["quick"]["calls"]

    assert final_calls == 30, f"Expected 30 calls, got {final_calls}"

    print("✓ Saves work correctly during concurrent updates")
    print(f"  - Final stats saved: {final_calls} calls")


async def test_no_data_loss():
    """Test that no updates are lost even under high concurrency."""
    print("\n=== Test 5: No Data Loss Under High Concurrency ===")

    # Import to check max limit
    from mcp_server import MAX_RECENT_CALLS

    # Reset stats
    with _stats_thread_lock:
        MODEL_USAGE["coder"]["calls"] = 0
        TASK_STATS["coding"] = 0
        RECENT_CALLS.clear()

    NUM_TASKS = 20
    UPDATES_PER_TASK = 50

    async def update_task(task_id: int):
        for i in range(UPDATES_PER_TASK):
            _update_stats_sync(
                model_tier="coder",
                task_type="coding",
                original_task="test",
                tokens=100,
                elapsed_ms=50,
                content_preview=f"Task {task_id} Request {i}",
                enable_thinking=False,
                backend="ollama"
            )
            # Minimal sleep for max concurrency
            await asyncio.sleep(0)

    # Run many concurrent updates
    await asyncio.gather(*[update_task(i) for i in range(NUM_TASKS)])

    # Verify no data loss
    final_snap = _snapshot_stats()
    final_calls = final_snap[0]["coder"]["calls"]
    final_tasks = final_snap[1]["coding"]
    final_recent = len(final_snap[3])

    expected_total = NUM_TASKS * UPDATES_PER_TASK
    expected_recent = min(expected_total, MAX_RECENT_CALLS)  # Recent calls are bounded

    assert final_calls == expected_total, f"Expected {expected_total} calls, got {final_calls}"
    assert final_tasks == expected_total, f"Expected {expected_total} tasks, got {final_tasks}"
    assert final_recent == expected_recent, f"Expected {expected_recent} recent calls, got {final_recent}"

    print(f"✓ No data loss with {NUM_TASKS} concurrent tasks")
    print(f"  - Total updates: {expected_total}")
    print(f"  - Calls recorded: {final_calls}")
    print(f"  - Tasks recorded: {final_tasks}")
    print(f"  - Recent calls (bounded): {final_recent}/{MAX_RECENT_CALLS}")


async def test_token_accumulation():
    """Test that tokens accumulate correctly under concurrency."""
    print("\n=== Test 6: Token Accumulation ===")

    # Reset stats
    with _stats_thread_lock:
        MODEL_USAGE["moe"]["tokens"] = 0

    TOKEN_COUNTS = [100, 200, 150, 300, 250]

    async def update_with_tokens(tokens: int):
        _update_stats_sync(
            model_tier="moe",
            task_type="analysis",
            original_task="test",
            tokens=tokens,
            elapsed_ms=50,
            content_preview="test",
            enable_thinking=False,
            backend="ollama"
        )
        await asyncio.sleep(0)

    # Concurrent updates with different token counts
    await asyncio.gather(*[update_with_tokens(t) for t in TOKEN_COUNTS])

    # Verify tokens accumulated
    final_snap = _snapshot_stats()
    final_tokens = final_snap[0]["moe"]["tokens"]
    expected_tokens = sum(TOKEN_COUNTS)

    assert final_tokens == expected_tokens, f"Expected {expected_tokens} tokens, got {final_tokens}"

    print(f"✓ Tokens accumulated correctly")
    print(f"  - Updates: {len(TOKEN_COUNTS)}")
    print(f"  - Token counts: {TOKEN_COUNTS}")
    print(f"  - Total tokens: {final_tokens}")


async def main():
    """Run all tests."""
    print("=" * 70)
    print("STATS CONCURRENCY TEST SUITE")
    print("=" * 70)

    tests = [
        test_concurrent_updates,
        test_snapshot_consistency,
        test_concurrent_snapshots,
        test_save_during_updates,
        test_no_data_loss,
        test_token_accumulation,
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
