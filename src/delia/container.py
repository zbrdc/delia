# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Service Container for Delia - Dependency Injection and Lifecycle Management.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend_manager import BackendManager
    from .config import Config
    from .language import LanguageDetector
    from .logging_service import LoggingService
    from .multi_user_tracking import UserTracker
    from .queue import ModelQueue
    from .routing import ModelRouter
    from .stats import StatsService
    from .hardware import HardwareMonitor
    from .tools.registry import MCPClientManager


class ServiceContainer:
    """
    Central container for all Delia services.
    
    Manages singletons and dependency wiring to eliminate global state coupling.
    """

    def __init__(self) -> None:
        from .paths import DATA_DIR, LIVE_LOGS_FILE
        from .config import config
        from .backend_manager import backend_manager
        from .stats import StatsService
        from .logging_service import LoggingService
        from .language import LanguageDetector
        from .routing import ModelRouter
        from .queue import ModelQueue
        from .hardware import get_hardware_monitor
        from .tools.mcp_client import MCPClientManager
        from .multi_user_tracking import tracker

        self.data_dir = DATA_DIR
        self.config: Config = config
        self.backend_manager: BackendManager = backend_manager
        self.stats_service: StatsService = StatsService()
        self.logging_service: LoggingService = LoggingService(DATA_DIR, LIVE_LOGS_FILE)
        self.language_detector: LanguageDetector = LanguageDetector()
        self.hardware_monitor: HardwareMonitor = get_hardware_monitor()
        self.model_queue: ModelQueue = ModelQueue()
        self.model_router: ModelRouter = ModelRouter(config, backend_manager, model_queue=self.model_queue)
        self.mcp_client_manager: MCPClientManager = MCPClientManager()
        self.user_tracker: UserTracker = tracker

    def initialize(self, use_stderr: bool = True) -> None:
        """Initialize all services in the correct order."""
        # 1. Logging first
        self.logging_service.load_live_logs()
        self.logging_service.configure_structlog(use_stderr=use_stderr)
        
        # 2. Stats and Config are already loaded by their respective modules
        # but we ensure directories exist
        from .paths import ensure_directories
        ensure_directories()

    async def shutdown(self) -> None:
        """Gracefully shutdown all services."""
        await self.stats_service.save_all_stats_async()
        await self.logging_service.save_live_logs_async()
        await self.mcp_client_manager.stop_all_servers()


_container: ServiceContainer | None = None

def get_container() -> ServiceContainer:
    """Get the global service container singleton."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container
