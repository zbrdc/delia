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
"""
Delia Path Configuration

All paths derived from DELIA_DATA_DIR environment variable.
Default: ./data/ relative to project root.

Usage:
    from paths import DATA_DIR, CACHE_DIR, SETTINGS_FILE
"""

import os
from pathlib import Path

# Project root (go up from src/delia/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Base data directory - override with DELIA_DATA_DIR
DATA_DIR = Path(os.environ.get("DELIA_DATA_DIR", PROJECT_ROOT / "data"))

# Derived directories
CACHE_DIR = DATA_DIR / "cache"
USER_DATA_DIR = DATA_DIR / "users"
MEMORIES_DIR = DATA_DIR / "memories"

# Specific files
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
STATS_FILE = CACHE_DIR / "usage_stats.json"
ENHANCED_STATS_FILE = CACHE_DIR / "enhanced_stats.json"
LIVE_LOGS_FILE = CACHE_DIR / "live_logs.json"
CIRCUIT_BREAKER_FILE = CACHE_DIR / "circuit_breaker.json"
RESPONSE_CACHE_FILE = CACHE_DIR / "response_cache.json"
USER_DB_FILE = USER_DATA_DIR / "users.db"


def ensure_directories() -> None:
    """Create all required directories."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
