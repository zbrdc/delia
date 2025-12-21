# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Hardware Telemetry for Delia - GPU and System Resource Monitoring.

Provides real-time access to VRAM, RAM, and CPU usage to enable
intelligent model routing and OOM prevention.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import psutil
import structlog

log = structlog.get_logger()


@dataclass
class GPUStats:
    """Statistics for a single GPU."""
    index: int
    name: str
    vram_total_mb: float
    vram_used_mb: float
    vram_free_mb: float
    temperature: int
    utilization: int


@dataclass
class HardwareStats:
    """Consolidated hardware statistics."""
    cpu_usage_percent: float
    ram_total_mb: float
    ram_used_mb: float
    ram_free_mb: float
    gpus: list[GPUStats]


class HardwareMonitor:
    """
    Monitors system and GPU resources for intelligent load balancing.
    """

    def __init__(self) -> None:
        self._has_nvidia = shutil.which("nvidia-smi") is not None
        self._last_stats: HardwareStats | None = None
        self._last_update_time: float = 0
        self._cache_ttl = 1.0  # 1 second cache for hardware stats

    def get_stats(self, force: bool = False) -> HardwareStats:
        """
        Get current hardware statistics (cached for 1s).
        """
        import time
        now = time.time()
        
        if not force and self._last_stats and (now - self._last_update_time) < self._cache_ttl:
            return self._last_stats

        # 1. System RAM and CPU
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent()

        # 2. GPU Stats (if available)
        gpus = []
        if self._has_nvidia:
            gpus = self._get_nvidia_stats()

        stats = HardwareStats(
            cpu_usage_percent=cpu,
            ram_total_mb=vm.total / (1024 * 1024),
            ram_used_mb=vm.used / (1024 * 1024),
            ram_free_mb=vm.available / (1024 * 1024),
            gpus=gpus,
        )

        self._last_stats = stats
        self._last_update_time = now
        return stats

    def _get_nvidia_stats(self) -> list[GPUStats]:
        """Query nvidia-smi for VRAM and utilization."""
        try:
            # Optimized nvidia-smi query for speed
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    gpus.append(GPUStats(
                        index=int(parts[0]),
                        name=parts[1],
                        vram_total_mb=float(parts[2]),
                        vram_used_mb=float(parts[3]),
                        vram_free_mb=float(parts[4]),
                        temperature=int(parts[5]),
                        utilization=int(parts[6]),
                    ))
            return gpus
        except Exception as e:
            log.debug("nvidia_smi_query_failed", error=str(e))
            return []

    def get_total_free_vram_mb(self) -> float:
        """Get sum of free VRAM across all GPUs."""
        stats = self.get_stats()
        if not stats.gpus:
            return 0.0
        return sum(gpu.vram_free_mb for gpu in stats.gpus)

    def is_oom_risk(self, required_mb: float, buffer_percent: float = 0.1) -> bool:
        """
        Check if loading a model of required_mb poses an OOM risk.
        
        Args:
            required_mb: VRAM required for the model
            buffer_percent: Safety margin (default 10%)
        """
        stats = self.get_stats()
        
        # If no GPU, we check system RAM (for CPU/GGUF usage)
        if not stats.gpus:
            available = stats.ram_free_mb
        else:
            # For multi-GPU, we assume the model must fit on ONE GPU
            # (Simplification for now, as Delia doesn't do tensor parallelism yet)
            available = max(gpu.vram_free_mb for gpu in stats.gpus)

        safe_available = available * (1.0 - buffer_percent)
        return required_mb > safe_available


# Singleton holder
_monitor: HardwareMonitor | None = None

def get_hardware_monitor() -> HardwareMonitor:
    """Get the global HardwareMonitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = HardwareMonitor()
    return _monitor
