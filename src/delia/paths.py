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

Usage:
    from paths import DATA_DIR, CACHE_DIR, SETTINGS_FILE
"""

import os
from pathlib import Path

# Project root (go up from src/delia/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# User config root
USER_DELIA_DIR = Path.home() / ".delia"

# Determine Settings File
# Priority:
# 1. DELIA_SETTINGS_FILE environment variable
# 2. settings.json in Current Working Directory (local override)
# 3. settings.json in ~/.delia/ (global user config)
# 4. settings.json in Project Root (legacy/dev mode)
def _find_settings_file() -> Path:
    env_path = os.environ.get("DELIA_SETTINGS_FILE")
    if env_path:
        return Path(env_path)
    
    cwd_path = Path.cwd() / "settings.json"
    if cwd_path.exists():
        return cwd_path
        
    user_path = USER_DELIA_DIR / "settings.json"
    if user_path.exists():
        return user_path
        
    return PROJECT_ROOT / "settings.json"

SETTINGS_FILE = _find_settings_file()


# Determine Data Directory
# Priority:
# 1. DELIA_DATA_DIR environment variable
# 2. Project Root data (legacy/dev compatibility - if it exists)
# 3. ~/.delia/data (global user data - default)
def _find_data_dir() -> Path:
    env_path = os.environ.get("DELIA_DATA_DIR")
    if env_path:
        return Path(env_path)

    # If running from source/dev and data exists, use it
    project_data = PROJECT_ROOT / "data"
    if project_data.exists():
        return project_data
        
    return USER_DELIA_DIR / "data"

DATA_DIR = _find_data_dir()

# Derived directories
CACHE_DIR = DATA_DIR / "cache"
USER_DATA_DIR = DATA_DIR / "users"
MEMORIES_DIR = DATA_DIR / "memories"
SESSIONS_DIR = DATA_DIR / "sessions"

# Specific files
STATS_FILE = CACHE_DIR / "usage_stats.json"
ENHANCED_STATS_FILE = CACHE_DIR / "enhanced_stats.json"
LIVE_LOGS_FILE = CACHE_DIR / "live_logs.json"
CIRCUIT_BREAKER_FILE = CACHE_DIR / "circuit_breaker.json"
USER_DB_FILE = USER_DATA_DIR / "users.db"


def ensure_directories() -> None:
    """Create all required directories."""
    # Ensure base global dir exists if we are using it
    if USER_DELIA_DIR in DATA_DIR.parents or USER_DELIA_DIR == DATA_DIR.parent:
        USER_DELIA_DIR.mkdir(parents=True, exist_ok=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
