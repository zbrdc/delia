# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Distributed Backend Discovery via mDNS (Zeroconf).

Allows Delia instances on the same local subnet to find each other
automatically, enabling multi-device VRAM pooling and distributed
inference without manual configuration.
"""

from __future__ import annotations

import socket
import asyncio
from typing import Any, cast

import structlog
from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf, ServiceListener

from .backend_manager import backend_manager, BackendConfig

log = structlog.get_logger()

SERVICE_TYPE = "_delia._tcp.local."

class DeliaServiceListener(ServiceListener):
    """Listens for Delia services on the network."""
    
    def __init__(self, loop: asyncio.AbstractEventLoop, local_ip: str, local_port: int):
        self.loop = loop
        self.local_ip = local_ip
        self.local_port = local_port
    
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if not info:
            return
            
        # Extract backend details
        addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
        if not addresses:
            return
            
        host = addresses[0]
        port = info.port
        
        # AVOID SELF-REGISTRATION: Don't add if it's our own local IP and port
        if host == self.local_ip and port == self.local_port:
            log.debug("discovery_ignoring_self", host=host, port=port)
            return
            
        url = f"http://{host}:{port}"
        
        # Check if we already have this backend
        backend_id = f"remote-{host.replace('.', '-')}"
        
        if backend_id not in backend_manager.backends:
            log.info("discovery_new_backend_found", name=name, url=url)
            
            # Register as a remote backend
            new_backend = BackendConfig(
                id=backend_id,
                name=f"Delia @ {host}",
                provider="ollama", # Assume ollama-compatible API for now
                type="remote",
                url=url,
                enabled=True,
                priority=20, # Higher number = lower priority than local
            )
            
            # Schedule registration in the main event loop from this thread
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._register_backend(new_backend))
            )

    async def _register_backend(self, backend: BackendConfig) -> None:
        """Asynchronously add and probe the new backend."""
        backend_manager.add_backend(backend)
        # Probe to see what models it has
        await backend_manager.probe_backend(backend.id)
        log.info("discovery_backend_registered", id=backend.id, models=len(backend.models))

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Handle service updates (not currently used)."""
        pass

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Handle service removal."""
        log.info("discovery_backend_removed", name=name)


class DiscoveryEngine:
    """Manages mDNS advertisement and browsing."""
    
    def __init__(self):
        self.zc: Zeroconf | None = None
        self.browser: ServiceBrowser | None = None
        self.info: ServiceInfo | None = None
        self._running = False
        self.loop = asyncio.get_event_loop()

    async def start(self, port: int = 34589) -> None:
        """Start advertising and browsing for Delia services."""
        if self._running:
            return
            
        try:
            # Zeroconf initialization can be very slow/blocking (e.g. 10s timeout)
            # Run it in a separate thread to keep the main loop responsive
            self.zc = await asyncio.to_thread(Zeroconf)
            
            # 1. Advertise local service
            hostname = socket.gethostname()
            local_ip = self._get_local_ip()
            
            self.info = ServiceInfo(
                SERVICE_TYPE,
                f"{hostname}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton(local_ip)],
                port=port,
                properties={"version": "1.0.0"},
                server=f"{hostname}.local.",
            )
            
            self.zc.register_service(self.info)
            log.info("discovery_advertising_started", ip=local_ip, port=port)
            
            # 2. Browse for other services
            self.browser = ServiceBrowser(
                self.zc, 
                SERVICE_TYPE, 
                DeliaServiceListener(self.loop, local_ip, port)
            )
            
            self._running = True
        except Exception as e:
            log.error("discovery_start_failed", error=str(e))

    async def stop(self) -> None:
        """Stop advertising and browsing."""
        if not self._running or not self.zc:
            return
            
        try:
            if self.info:
                self.zc.unregister_service(self.info)
            self.zc.close()
            self._running = False
            log.info("discovery_stopped")
        except Exception as e:
            log.error("discovery_stop_failed", error=str(e))

    def _get_local_ip(self) -> str:
        """Get the primary local IP address."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

# Global singleton
_discovery: DiscoveryEngine | None = None

def get_discovery_engine() -> DiscoveryEngine:
    """Get the global discovery engine."""
    global _discovery
    if _discovery is None:
        _discovery = DiscoveryEngine()
    return _discovery
