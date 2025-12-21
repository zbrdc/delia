# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from hypothesis import given, strategies as st
from delia.discovery import DeliaServiceListener, DiscoveryEngine

@pytest.fixture
def mock_loop():
    """Create or get an event loop for testing."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# 1. DISCOVERY LISTENER FUZZING
@given(
    host=st.ip_addresses(v=4),
    port=st.integers(min_value=1, max_value=65535)
)
def test_discovery_listener_add_service_fuzz(host, port):
    """Fuzz the service listener with various IP/Port combinations."""
    # Create event loop for this test
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    listener = DeliaServiceListener(loop, local_ip="127.0.0.1", local_port=8000)
    zc = MagicMock()

    info = MagicMock()
    info.addresses = [host.packed]
    info.port = port
    zc.get_service_info.return_value = info

    with patch.object(listener, "_register_backend", new_callable=AsyncMock):
        listener.add_service(zc, "_delia._tcp.local.", "TestDelia")
        # Ensure no crashes occur

# 2. DISCOVERY ENGINE LIFECYCLE
@pytest.mark.asyncio
async def test_discovery_engine_lifecycle():
    """Test start/stop of the discovery engine."""
    engine = DiscoveryEngine()

    with patch("delia.discovery.Zeroconf") as mock_zc, \
         patch("delia.discovery.ServiceBrowser"):

        # start() doesn't take local_port in current implementation
        await engine.start()
        assert engine._running == True

        await engine.stop()
        assert engine._running == False
        mock_zc.return_value.close.assert_called()

# 3. SELF-REGISTRATION PROTECTION
def test_discovery_ignores_self():
    """Ensure discovery doesn't add the local instance as a remote backend."""
    # Create event loop for this test
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    listener = DeliaServiceListener(loop, local_ip="192.168.1.50", local_port=34589)
    zc = MagicMock()

    import socket
    info = MagicMock()
    info.addresses = [socket.inet_aton("192.168.1.50")]
    info.port = 34589
    zc.get_service_info.return_value = info

    with patch("delia.discovery.log") as mock_log:
        listener.add_service(zc, "_delia._tcp.local.", "self")
        mock_log.debug.assert_any_call("discovery_ignoring_self", host="192.168.1.50", port=34589)