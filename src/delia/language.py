#!/usr/bin/env python3
# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Language detection and system prompt generation for Delia.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound

from .prompts import DELIA_IDENTITY
import structlog

log = structlog.get_logger()

LANGUAGE_CONFIGS = {
    "python": {"extensions": [".py"], "keywords": ["def ", "import ", "class "], "system_prompt": "Expert Python developer"},
    "rust": {"extensions": [".rs"], "keywords": ["fn ", "impl ", "use std::"], "system_prompt": "Expert Rust developer"},
    "react": {"extensions": [".jsx", ".tsx"], "keywords": ["useState", "useEffect", "import React"], "system_prompt": "Expert React developer"},
}

PYGMENTS_LANGUAGE_MAP = {
    "python": "python",
    "javascript": "nodejs",
    "typescript": "nodejs",
    "rust": "rust",
}

class LanguageDetector:
    def __init__(self):
        self.configs = LANGUAGE_CONFIGS

    def detect(self, content: str, file_path: str = "", hint: str | None = None) -> str:
        return detect_language(content, file_path, hint)

    def get_system_prompt(self, language: str, task_type: str) -> str:
        return get_system_prompt(language, task_type)

def get_current_time_context() -> str:
    """Generate current time context for LLM."""
    utc_now = datetime.now(timezone.utc)
    return f"[System time: {utc_now.strftime('%A, %B %d, %Y at %H:%M UTC')}]"

def detect_language(content: str, file_path: str = "", hint: str | None = None) -> str:
    if hint: return hint
    if ".py" in file_path or "def " in content: return "python"
    if ".rs" in file_path or "fn " in content: return "rust"
    return "python"

def get_system_prompt(language: str, task_type: str) -> str:
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["python"])
    time_ctx = get_current_time_context()
    return f"{DELIA_IDENTITY}\n{time_ctx}\nRole: {config['system_prompt']}\nTask: {task_type}"

def optimize_prompt(content: str, task_type: str) -> str:
    return content.strip()