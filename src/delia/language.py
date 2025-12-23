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
    # Python
    "python": {"extensions": [".py", ".pyi", ".pyx"], "keywords": ["def ", "import ", "class ", "async def"], "system_prompt": "Expert Python developer"},
    # Rust
    "rust": {"extensions": [".rs"], "keywords": ["fn ", "impl ", "use std::", "pub fn", "struct "], "system_prompt": "Expert Rust developer"},
    # JavaScript/TypeScript
    "javascript": {"extensions": [".js", ".mjs", ".cjs"], "keywords": ["function ", "const ", "let ", "=>", "require("], "system_prompt": "Expert JavaScript developer"},
    "typescript": {"extensions": [".ts"], "keywords": ["interface ", "type ", ": string", ": number", "export "], "system_prompt": "Expert TypeScript developer"},
    "react": {"extensions": [".jsx", ".tsx"], "keywords": ["useState", "useEffect", "import React", "<div", "className="], "system_prompt": "Expert React developer"},
    # Go
    "go": {"extensions": [".go"], "keywords": ["func ", "package ", "import (", "type ", "struct {"], "system_prompt": "Expert Go developer"},
    # Java/Kotlin
    "java": {"extensions": [".java"], "keywords": ["public class", "private ", "void ", "@Override", "import java"], "system_prompt": "Expert Java developer"},
    "kotlin": {"extensions": [".kt", ".kts"], "keywords": ["fun ", "val ", "var ", "class ", "suspend fun"], "system_prompt": "Expert Kotlin developer"},
    # C/C++
    "c": {"extensions": [".c", ".h"], "keywords": ["#include", "int main", "void ", "printf(", "malloc("], "system_prompt": "Expert C developer"},
    "cpp": {"extensions": [".cpp", ".hpp", ".cc"], "keywords": ["#include", "std::", "class ", "template<", "virtual "], "system_prompt": "Expert C++ developer"},
    # Ruby
    "ruby": {"extensions": [".rb", ".rake"], "keywords": ["def ", "end", "class ", "require ", "attr_"], "system_prompt": "Expert Ruby developer"},
    # PHP
    "php": {"extensions": [".php"], "keywords": ["<?php", "function ", "class ", "$", "->"], "system_prompt": "Expert PHP developer"},
    # Shell
    "shell": {"extensions": [".sh", ".bash", ".zsh"], "keywords": ["#!/bin/", "if [", "then", "fi", "echo "], "system_prompt": "Expert Shell scripting developer"},
    # SQL
    "sql": {"extensions": [".sql"], "keywords": ["SELECT ", "FROM ", "WHERE ", "INSERT ", "CREATE TABLE"], "system_prompt": "Expert SQL developer"},
    # Docker
    "dockerfile": {"extensions": [], "keywords": ["FROM ", "RUN ", "COPY ", "ENTRYPOINT", "WORKDIR"], "system_prompt": "Expert Docker/container developer"},
    # YAML/Config
    "yaml": {"extensions": [".yaml", ".yml"], "keywords": ["- ", ": ", "name:", "version:"], "system_prompt": "Expert configuration developer"},
    # Vue/Svelte
    "vue": {"extensions": [".vue"], "keywords": ["<template>", "<script>", "v-if", "v-for", "@click"], "system_prompt": "Expert Vue.js developer"},
    "svelte": {"extensions": [".svelte"], "keywords": ["<script>", "{#if", "{#each", "on:click", "$:"], "system_prompt": "Expert Svelte developer"},
    # Solidity/Web3
    "solidity": {"extensions": [".sol"], "keywords": ["pragma solidity", "contract ", "function ", "mapping(", "require(", "emit "], "system_prompt": "Expert Solidity/Web3 developer"},
}

# Extension to language mapping (comprehensive)
EXTENSION_TO_LANGUAGE = {
    ".py": "python", ".pyi": "python", ".pyx": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "react", ".jsx": "react",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".rb": "ruby", ".rake": "ruby",
    ".php": "php",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".sql": "sql",
    ".yaml": "yaml", ".yml": "yaml",
    ".vue": "vue",
    ".svelte": "svelte",
    ".css": "css", ".scss": "css", ".less": "css",
    ".html": "html", ".htm": "html",
    ".md": "markdown",
    ".json": "json",
    ".toml": "toml",
    ".xml": "xml",
    ".sol": "solidity",
}

PYGMENTS_LANGUAGE_MAP = {
    "python": "python",
    "javascript": "nodejs",
    "typescript": "nodejs",
    "rust": "rust",
    "go": "go",
    "java": "java",
    "cpp": "cpp",
    "c": "c",
    "ruby": "ruby",
    "php": "php",
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
    """
    Detect programming language from file path and/or content.

    Priority:
    1. Explicit hint
    2. File extension
    3. Special filenames (Dockerfile, Makefile, etc.)
    4. Content keyword analysis
    5. Pygments fallback
    6. Default to 'unknown'
    """
    if hint:
        return hint

    # Check file extension first
    if file_path:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[ext]

        # Special filenames
        name = path.name.lower()
        if name == "dockerfile" or name.startswith("dockerfile."):
            return "dockerfile"
        if name == "makefile" or name == "gnumakefile":
            return "makefile"
        if name == "jenkinsfile":
            return "groovy"
        if name == "vagrantfile":
            return "ruby"
        if name.endswith(".config.js") or name.endswith(".config.ts"):
            return "javascript" if name.endswith(".js") else "typescript"

    # Content-based detection
    if content:
        # Score each language by keyword matches
        scores: dict[str, int] = {}
        for lang, config in LANGUAGE_CONFIGS.items():
            for keyword in config["keywords"]:
                if keyword in content:
                    scores[lang] = scores.get(lang, 0) + 1

        if scores:
            best_lang = max(scores.items(), key=lambda x: x[1])
            if best_lang[1] >= 2:  # At least 2 keyword matches
                return best_lang[0]

    # Try pygments as fallback
    if file_path:
        try:
            lexer = get_lexer_for_filename(file_path)
            pygments_name = lexer.name.lower()
            for key, mapped in PYGMENTS_LANGUAGE_MAP.items():
                if key in pygments_name:
                    return key
        except ClassNotFound:
            pass

    return "unknown"

def get_system_prompt(language: str, task_type: str) -> str:
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["python"])
    time_ctx = get_current_time_context()
    return f"{DELIA_IDENTITY}\n{time_ctx}\nRole: {config['system_prompt']}\nTask: {task_type}"

def optimize_prompt(content: str, task_type: str) -> str:
    return content.strip()