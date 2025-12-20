# Copyright (C) 2024 Delia Contributors
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

IMPORTANT: All paths use lazy evaluation via get_*() functions to support
test isolation. Tests can set DELIA_DATA_DIR and DELIA_SETTINGS_FILE
env vars AFTER import and paths will resolve correctly.

Usage:
    from delia.paths import get_settings_file, get_data_dir

    # Lazy evaluation - respects env vars set after import
    settings = get_settings_file()
    data = get_data_dir()

    # Legacy module-level constants (evaluated once at first access)
    from delia.paths import DATA_DIR, CACHE_DIR, SETTINGS_FILE
"""

import os
from functools import lru_cache
from pathlib import Path

# Project root (go up from src/delia/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# User config root
USER_DELIA_DIR = Path.home() / ".delia"


def get_settings_file() -> Path:
    """
    Get the settings file path with lazy evaluation.

    Priority:
    1. DELIA_SETTINGS_FILE environment variable (allows explicit override)
    2. settings.json in ~/.delia/ (global user config - preferred)
    3. settings.json in Current Working Directory (local override for dev)
    4. settings.json in Project Root (legacy/dev mode fallback)

    Returns:
        Path to settings.json
    """
    # Environment variable takes absolute priority (used by tests and subprocess)
    env_path = os.environ.get("DELIA_SETTINGS_FILE")
    if env_path:
        return Path(env_path)

    # User config is the preferred location (consistent across all contexts)
    user_path = USER_DELIA_DIR / "settings.json"
    if user_path.exists():
        return user_path

    # CWD for local development overrides
    cwd_path = Path.cwd() / "settings.json"
    if cwd_path.exists():
        return cwd_path

    # Project root fallback for dev mode
    project_path = PROJECT_ROOT / "settings.json"
    if project_path.exists():
        return project_path

    # Default to user path (will be created on first run)
    return user_path


def get_data_dir() -> Path:
    """
    Get the data directory path with lazy evaluation.

    Priority:
    1. DELIA_DATA_DIR environment variable
    2. Project Root data (legacy/dev compatibility - if it exists)
    3. ~/.delia/data (global user data - default)

    Returns:
        Path to data directory
    """
    env_path = os.environ.get("DELIA_DATA_DIR")
    if env_path:
        return Path(env_path)

    # If running from source/dev and data exists, use it
    project_data = PROJECT_ROOT / "data"
    if project_data.exists():
        return project_data

    return USER_DELIA_DIR / "data"


# Module-level lazy attributes using __getattr__ (Python 3.7+)
# This allows tests to set DELIA_DATA_DIR/DELIA_SETTINGS_FILE after import
def __getattr__(name: str) -> Path:
    """Lazy attribute access for backward compatibility."""
    if name == "SETTINGS_FILE":
        return get_settings_file()
    elif name == "DATA_DIR":
        return get_data_dir()
    elif name == "CACHE_DIR":
        return get_data_dir() / "cache"
    elif name == "USER_DATA_DIR":
        return get_data_dir() / "users"
    elif name == "MEMORIES_DIR":
        return get_data_dir() / "memories"
    elif name == "SESSIONS_DIR":
        return get_data_dir() / "sessions"
    elif name == "STATS_FILE":
        return get_data_dir() / "cache" / "usage_stats.json"
    elif name == "ENHANCED_STATS_FILE":
        return get_data_dir() / "cache" / "enhanced_stats.json"
    elif name == "LIVE_LOGS_FILE":
        return get_data_dir() / "cache" / "live_logs.json"
    elif name == "CIRCUIT_BREAKER_FILE":
        return get_data_dir() / "cache" / "circuit_breaker.json"
    elif name == "BACKEND_METRICS_FILE":
        return get_data_dir() / "cache" / "backend_metrics.json"
    elif name == "AFFINITY_FILE":
        return get_data_dir() / "cache" / "affinity.json"
    elif name == "PREWARM_FILE":
        return get_data_dir() / "cache" / "prewarm.json"
    elif name == "USER_DB_FILE":
        return get_data_dir() / "users" / "users.db"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ensure_directories() -> None:
    """Create all required directories."""
    data_dir = get_data_dir()
    cache_dir = data_dir / "cache"
    user_data_dir = data_dir / "users"
    memories_dir = data_dir / "memories"
    sessions_dir = data_dir / "sessions"

    # Ensure base global dir exists if we are using it
    if USER_DELIA_DIR in data_dir.parents or USER_DELIA_DIR == data_dir.parent:
        USER_DELIA_DIR.mkdir(parents=True, exist_ok=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    user_data_dir.mkdir(parents=True, exist_ok=True)
    memories_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
