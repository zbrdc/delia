# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Service Container for Delia - Dependency Injection and Lifecycle Management.

Uses lazy initialization to reduce memory footprint for proxy/secondary instances.
Heavy services (model_queue, model_router, etc.) are only created when first accessed.
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
    Uses lazy initialization to reduce memory for secondary instances.
    """

    def __init__(self) -> None:
        from .paths import DATA_DIR, LIVE_LOGS_FILE

        # Store paths for lazy initialization
        self._data_dir = DATA_DIR
        self._live_logs_file = LIVE_LOGS_FILE

        # Lazy-loaded service instances (None until first access)
        self._config: Config | None = None
        self._backend_manager: BackendManager | None = None
        self._stats_service: StatsService | None = None
        self._logging_service: LoggingService | None = None
        self._language_detector: LanguageDetector | None = None
        self._hardware_monitor: HardwareMonitor | None = None
        self._model_queue: ModelQueue | None = None
        self._model_router: ModelRouter | None = None
        self._mcp_client_manager: MCPClientManager | None = None
        self._user_tracker: UserTracker | None = None

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def config(self) -> Config:
        if self._config is None:
            from .config import config
            self._config = config
        return self._config

    @property
    def backend_manager(self) -> BackendManager:
        if self._backend_manager is None:
            from .backend_manager import backend_manager
            self._backend_manager = backend_manager
        return self._backend_manager

    @property
    def stats_service(self) -> StatsService:
        if self._stats_service is None:
            from .stats import StatsService
            self._stats_service = StatsService()
        return self._stats_service

    @property
    def logging_service(self) -> LoggingService:
        if self._logging_service is None:
            from .logging_service import LoggingService
            self._logging_service = LoggingService(self._data_dir, self._live_logs_file)
        return self._logging_service

    @property
    def language_detector(self) -> LanguageDetector:
        if self._language_detector is None:
            from .language import LanguageDetector
            self._language_detector = LanguageDetector()
        return self._language_detector

    @property
    def hardware_monitor(self) -> HardwareMonitor:
        if self._hardware_monitor is None:
            from .hardware import get_hardware_monitor
            self._hardware_monitor = get_hardware_monitor()
        return self._hardware_monitor

    @property
    def model_queue(self) -> ModelQueue:
        if self._model_queue is None:
            from .queue import ModelQueue
            self._model_queue = ModelQueue()
        return self._model_queue

    @property
    def model_router(self) -> ModelRouter:
        if self._model_router is None:
            from .routing import ModelRouter
            self._model_router = ModelRouter(
                self.config,
                self.backend_manager,
                model_queue=self.model_queue
            )
        return self._model_router

    @property
    def mcp_client_manager(self) -> MCPClientManager:
        if self._mcp_client_manager is None:
            from .tools.mcp_client import MCPClientManager
            self._mcp_client_manager = MCPClientManager()
        return self._mcp_client_manager

    @property
    def user_tracker(self) -> UserTracker:
        if self._user_tracker is None:
            from .multi_user_tracking import tracker
            self._user_tracker = tracker
        return self._user_tracker

    def initialize(self, use_stderr: bool = True) -> None:
        """Initialize essential services in the correct order."""
        # 1. Logging first (this triggers lazy load)
        self.logging_service.load_live_logs()
        self.logging_service.configure_structlog(use_stderr=use_stderr)

        # 2. Ensure directories exist
        from .paths import ensure_directories
        ensure_directories()

        # Note: Other services remain lazy until first access

    async def shutdown(self) -> None:
        """Gracefully shutdown all initialized services."""
        # Only shutdown services that were actually initialized
        if self._stats_service is not None:
            await self._stats_service.save_all_stats_async()
        if self._logging_service is not None:
            await self._logging_service.save_live_logs_async()
        if self._mcp_client_manager is not None:
            await self._mcp_client_manager.stop_all_servers()


_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    """Get the global service container singleton."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container
