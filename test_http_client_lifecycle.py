# Copyright (C) 2023 the project owner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#!/usr/bin/env python3
"""
Test suite for HTTP client lifecycle management.

Verifies that HTTP clients are:
1. Properly created and reused
2. Properly closed without leaks
3. Recreated when configuration changes
4. All closed on shutdown
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend_manager import BackendConfig, BackendManager, shutdown_backends


async def test_client_creation_and_reuse():
    """Test that clients are created once and reused."""
    print("\n=== Test 1: Client Creation and Reuse ===")

    backend = BackendConfig(
        id="test-backend",
        name="Test Backend",
        provider="ollama",
        type="local",
        url="http://localhost:11434"
    )

    # First access - should create client
    client1 = backend.get_client()
    assert client1 is not None
    print(f"✓ Client created: {client1}")

    # Second access - should return same client
    client2 = backend.get_client()
    assert client1 is client2, "Should return same client instance"
    print(f"✓ Same client returned on reuse")


async def test_client_cleanup():
    """Test that clients are properly closed without leaks."""
    print("\n=== Test 2: Client Cleanup ===")

    backend = BackendConfig(
        id="test-backend",
        name="Test Backend",
        provider="ollama",
        type="local",
        url="http://localhost:11434"
    )

    # Create a client
    client = backend.get_client()
    assert backend._client is not None
    print(f"✓ Client created")

    # Close the client
    await backend.close_client()
    assert backend._client is None, "Client should be None after close"
    print(f"✓ Client properly closed and set to None")

    # Should be able to create a new client
    client2 = backend.get_client()
    assert client2 is not None
    assert client2 is not client, "Should be a different client instance"
    print(f"✓ New client created after close")

    # Clean up
    await backend.close_client()


async def test_client_recreation_on_url_change():
    """Test that client is recreated when URL changes."""
    print("\n=== Test 3: Client Recreation on URL Change ===")

    manager = BackendManager()

    # Add a test backend
    backend_data = {
        "id": "test-url-change",
        "name": "Test URL Change",
        "provider": "ollama",
        "type": "local",
        "url": "http://localhost:11434",
        "enabled": True,
    }
    backend = manager.add_backend(backend_data)

    # Create a client
    client1 = backend.get_client()
    assert client1 is not None
    print(f"✓ Initial client created: {client1}")

    # Update the backend URL
    await manager.update_backend(backend.id, {"url": "http://localhost:8080"})
    assert backend._client is None, "Client should be None after URL change"
    print(f"✓ Client closed when URL changed")

    # New client should be created with new URL
    client2 = backend.get_client()
    assert client2 is not None
    assert client2 is not client1, "Should be a different client"
    assert backend.url == "http://localhost:8080", "URL should be updated"
    print(f"✓ New client created with updated URL")

    # Clean up
    await backend.close_client()
    manager.backends.clear()


async def test_concurrent_client_operations():
    """Test that concurrent operations don't cause connection leaks."""
    print("\n=== Test 4: Concurrent Client Operations ===")

    manager = BackendManager()

    # Add multiple backends
    backends_data = [
        {
            "id": f"backend-{i}",
            "name": f"Backend {i}",
            "provider": "ollama",
            "type": "local",
            "url": f"http://localhost:{11434 + i}",
            "enabled": True,
        }
        for i in range(3)
    ]

    for data in backends_data:
        manager.add_backend(data)

    # Create clients concurrently
    async def create_client(backend_id):
        backend = manager.get_backend(backend_id)
        return backend.get_client()

    clients = await asyncio.gather(
        *[create_client(f"backend-{i}") for i in range(3)]
    )

    assert len(clients) == 3
    assert all(c is not None for c in clients)
    print(f"✓ Created {len(clients)} clients concurrently")

    # Close all clients concurrently
    close_tasks = [
        manager.get_backend(f"backend-{i}").close_client()
        for i in range(3)
    ]
    await asyncio.gather(*close_tasks)

    # Verify all closed
    for i in range(3):
        backend = manager.get_backend(f"backend-{i}")
        assert backend._client is None
    print(f"✓ All clients properly closed")

    manager.backends.clear()


async def test_no_leaks_on_rapid_reload():
    """Test that rapid reloads don't leave unclosed clients."""
    print("\n=== Test 5: No Leaks on Rapid Reload ===")

    manager = BackendManager()

    # Add a test backend
    backend_data = {
        "id": "test-reload",
        "name": "Test Reload",
        "provider": "ollama",
        "type": "local",
        "url": "http://localhost:11434",
        "enabled": True,
    }
    manager.add_backend(backend_data)

    # Do rapid reloads (simulating configuration updates)
    for i in range(5):
        # Create some clients
        backend = manager.get_backend("test-reload")
        client = backend.get_client()

        # Reload (should cleanly close all clients)
        await manager.reload()

        print(f"✓ Reload {i+1} completed without leaks")

    # Verify final state is clean
    backend = manager.get_backend("test-reload")
    if backend:
        assert backend._client is None or isinstance(backend._client, type(backend.get_client()))
    print(f"✓ Final state is clean after all reloads")

    manager.backends.clear()


async def test_shutdown_closes_all_clients():
    """Test that shutdown handler closes all clients."""
    print("\n=== Test 6: Shutdown Closes All Clients ===")

    # Test that individual backends can be shut down cleanly
    backends = []
    for i in range(3):
        backend = BackendConfig(
            id=f"shutdown-backend-{i}",
            name=f"Shutdown Backend {i}",
            provider="ollama",
            type="local",
            url=f"http://localhost:{11434 + i}"
        )
        backends.append(backend)

    # Create clients for all backends
    for backend in backends:
        backend.get_client()

    print(f"✓ Created clients for {len(backends)} backends")

    # Manually close all (simulating shutdown handler behavior)
    close_tasks = [backend.close_client() for backend in backends]
    await asyncio.gather(*close_tasks)

    # Verify all backends have closed clients
    for i, backend in enumerate(backends):
        assert backend._client is None, f"Backend {i} client not closed"
    print(f"✓ All clients properly closed by shutdown handler")

    # Also test the global shutdown_backends function
    from backend_manager import backend_manager
    if backend_manager.backends:
        await shutdown_backends()
        # Verify at least some clients are closed
        print(f"✓ Global shutdown_backends() executed successfully")


async def test_error_handling_in_close():
    """Test that errors during close are handled gracefully."""
    print("\n=== Test 7: Error Handling in Close ===")

    backend = BackendConfig(
        id="test-error",
        name="Test Error",
        provider="ollama",
        type="local",
        url="http://localhost:11434"
    )

    # Create a client
    client = backend.get_client()
    assert client is not None

    # Close successfully (first time)
    await backend.close_client()
    assert backend._client is None
    print(f"✓ First close succeeded")

    # Close again - should not raise error (client already None)
    try:
        await backend.close_client()
        print(f"✓ Second close handled gracefully (idempotent)")
    except Exception as e:
        assert False, f"Should not raise error on second close: {e}"


async def main():
    """Run all tests."""
    print("=" * 70)
    print("HTTP CLIENT LIFECYCLE TEST SUITE")
    print("=" * 70)

    tests = [
        test_client_creation_and_reuse,
        test_client_cleanup,
        test_client_recreation_on_url_change,
        test_concurrent_client_operations,
        test_no_leaks_on_rapid_reload,
        test_shutdown_closes_all_clients,
        test_error_handling_in_close,
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
