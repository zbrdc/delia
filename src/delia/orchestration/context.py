# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unified Context Engine for Delia Orchestration.

This module provides high-performance context assembly for LLMs,
unifying file reads, Serena memory retrieval, and symbol focus
into a single pipeline used by the OrchestrationService.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from ..file_helpers import read_files, read_serena_memory
from ..language import detect_language

log = structlog.get_logger()

class ContextEngine:
    """
    Assembles the "World View" for Delia's models.
    
    Handles:
    - Lazy-loading files from disk
    - Retrieving Serena memories (Project Knowledge)
    - Formatting symbol focus for IDE-like behavior
    - Injecting session history for conversation continuity
    """

    @staticmethod
    async def prepare_content(
        content: str,
        context: str | None = None,
        symbols: str | None = None,
        include_references: bool = False,
        files: str | None = None,
        session_context: str | None = None,
    ) -> str:
        """
        Assembles all contextual parts into a single prompt string.
        """
        parts = []

        # 1. Session History (Continuity)
        if session_context:
            parts.append(f"### Previous Conversation\n{session_context}\n")

        # 2. File Context (Direct Disk Access)
        if files:
            file_contents = read_files(files)
            if file_contents:
                for path, file_content in file_contents:
                    ext = Path(path).suffix.lstrip(".")
                    lang_hint = ext if ext else ""
                    parts.append(f"### File: `{path}`\n```{lang_hint}\n{file_content}\n```")
                log.info("context_files_loaded", count=len(file_contents))

        # 3. Serena Memories (Project Knowledge)
        if context:
            memory_names = [m.strip() for m in context.split(",")]
            for mem_name in memory_names:
                mem_content = read_serena_memory(mem_name)
                if mem_content:
                    parts.append(f"### Project Context ({mem_name}):\n{mem_content}")
                    log.info("context_memory_loaded", memory=mem_name)

        # 4. Symbol Focus (Targeted Coding)
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(",")]
            symbol_hint = f"### Focus Symbols: {', '.join(symbol_list)}"
            if include_references:
                symbol_hint += "\n_References to these symbols are included in the files above._"
            parts.append(symbol_hint)
            log.info("context_symbols_focused", symbols=symbol_list)

        # 5. The Task
        if parts:
            parts.append(f"---\n\n### Task:\n{content}")
            return "\n\n".join(parts)
        
        return content

    @staticmethod
    def get_context_signals(content: str, file_path: str | None = None) -> dict[str, Any]:
        """Detect language and size signals for routing."""
        return {
            "language": detect_language(content, file_path or ""),
            "size_bytes": len(content.encode("utf-8")),
            "is_large": len(content.encode("utf-8")) > 50000 # Config threshold
        }
