# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unified Context Engine for Delia Orchestration.

This module provides high-performance context assembly for LLMs,
unifying file reads, Delia's memory retrieval, and symbol focus
into a single pipeline used by the OrchestrationService.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from ..file_helpers import read_files, read_memory
from ..language import detect_language
from ..project_memory import get_project_context
from .summarizer import get_summarizer
from .graph import get_symbol_graph

log = structlog.get_logger()

class ContextEngine:
    """
    Assembles the "World View" for Delia's models.
    
    Handles:
    - Lazy-loading files from disk
    - Retrieving Delia's memories (Project Knowledge)
    - Formatting symbol focus for IDE-like behavior
    - Injecting session history for conversation continuity
    - Hierarchical project summarization (Infinite Context)
    - Dependency Graph navigation (GraphRAG)
    """

    @staticmethod
    async def prepare_content(
        content: str,
        context: str | None = None,
        symbols: str | None = None,
        include_references: bool = False,
        files: str | None = None,
        session_context: str | None = None,
        include_project_overview: bool = False,
        include_project_instructions: bool = True,
    ) -> str:
        """
        Assembles all contextual parts into a single prompt string.

        Args:
            content: The main task/prompt
            context: Comma-separated memory names to load
            symbols: Comma-separated symbols to focus on
            include_references: Include symbol references
            files: Comma-separated file paths to load
            session_context: Previous conversation history
            include_project_overview: Include hierarchical code summary
            include_project_instructions: Auto-load DELIA.md project instructions
        """
        parts = []

        # 0. Project Instructions (Claude Code-like DELIA.md auto-loading)
        if include_project_instructions:
            project_context = get_project_context()
            if project_context:
                parts.append(project_context)
                log.debug("project_instructions_injected")

        # 0.5. Project Overview (Gemini-class Hierarchical Context)
        if include_project_overview:
            summarizer = get_summarizer()
            await summarizer.initialize()
            overview = summarizer.get_project_overview()
            if overview:
                parts.append(overview)

        # 1. Session History (Continuity)
        if session_context:
            parts.append(f"### Previous Conversation\n{session_context}\n")

        # 2. File Context (Direct Disk Access)
        explicit_files = set()
        if files:
            file_contents = read_files(files)
            if file_contents:
                for path, file_content in file_contents:
                    explicit_files.add(path)
                    ext = Path(path).suffix.lstrip(".")
                    lang_hint = ext if ext else ""
                    parts.append(f"### File: `{path}`\n```{lang_hint}\n{file_content}\n```")
                log.info("context_files_loaded", count=len(file_contents))

        # 2.5 Dynamic Windowing & Graph Neighbors (GraphRAG Dependency Mapping) 🌐
        # Objective: Prioritize "Graph Neighbors" over "Recent History" for better structural awareness.
        if explicit_files:
            try:
                graph = get_symbol_graph()
                await graph.initialize()
                summarizer = get_summarizer()
                await summarizer.initialize()
                
                related_files = set()
                for f in explicit_files:
                    related = graph.get_related_files(f)
                    related_files.update(related)
                
                # Identify neighbors that aren't already included
                neighbors = related_files - explicit_files
                if neighbors:
                    # Sort neighbors by importance (e.g. number of connections)
                    neighbor_lines = []
                    for nf in sorted(neighbors):
                        summary_text = summarizer.summaries[nf].summary if nf in summarizer.summaries else "(Direct dependency)"
                        neighbor_lines.append(f"- `{nf}`: {summary_text}")
                            
                    if neighbor_lines:
                        # Dynamic windowing: Injected as high-priority architectural context
                        parts.append("### Architectural Context (Dependency Graph)\n" + "\n".join(neighbor_lines))
                        log.info("dynamic_window_neighbors_injected", count=len(neighbor_lines))
            except Exception as e:
                log.debug("graph_context_failed", error=str(e))

        # 3. Delia's Memories (Project Knowledge)
        if context:
            memory_names = [m.strip() for m in context.split(",")]
            for mem_name in memory_names:
                mem_content = read_memory(mem_name)
                if mem_content:
                    parts.append(f"### Project Context ({mem_name}):\n{mem_content}")
                    log.info("context_memory_loaded", memory=mem_name)

        # 4. Symbol Focus (Targeted Coding) - Now with actual symbol definitions!
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(",")]
            symbol_parts = [f"### Focus Symbols: {', '.join(symbol_list)}"]

            try:
                graph = get_symbol_graph()
                await graph.initialize()

                for sym_name in symbol_list:
                    found_symbols = graph.find_symbol(sym_name)
                    if found_symbols:
                        # Take the first match (most likely the definition)
                        sym = found_symbols[0]
                        # Read a snippet around the symbol definition (±20 lines)
                        try:
                            with open(sym.file_path, 'r') as f:
                                lines = f.readlines()
                                start = max(0, sym.line - 3)  # 2 lines before
                                end = min(len(lines), sym.line + 50)  # 50 lines after
                                snippet = ''.join(lines[start:end])
                                ext = Path(sym.file_path).suffix.lstrip(".")
                                symbol_parts.append(
                                    f"\n#### `{sym_name}` ({sym.kind} in `{sym.file_path}:{sym.line}`)\n"
                                    f"```{ext}\n{snippet.rstrip()}\n```"
                                )
                                log.debug("symbol_definition_loaded", symbol=sym_name, file=sym.file_path)
                        except Exception as e:
                            log.debug("symbol_read_failed", symbol=sym_name, error=str(e))
                            symbol_parts.append(f"\n- `{sym_name}`: {sym.kind} at {sym.file_path}:{sym.line}")
                    else:
                        symbol_parts.append(f"\n- `{sym_name}`: (not found in index)")

                # Find references if requested
                if include_references and found_symbols:
                    try:
                        from ..lsp_client import get_lsp_client
                        lsp = get_lsp_client()
                        for sym_name in symbol_list:
                            found_symbols = graph.find_symbol(sym_name)
                            if found_symbols:
                                sym = found_symbols[0]
                                refs = await lsp.find_references(sym.file_path, sym.line, 0)
                                if refs:
                                    ref_lines = [f"  - `{r['uri']}:{r['line']}`" for r in refs[:10]]
                                    symbol_parts.append(f"\n##### References to `{sym_name}`:\n" + "\n".join(ref_lines))
                    except Exception as e:
                        log.debug("lsp_references_failed", error=str(e))
                        symbol_parts.append("\n_LSP not available for reference lookup._")
            except Exception as e:
                log.debug("symbol_lookup_failed", error=str(e))
                # Fallback to simple hint
                if include_references:
                    symbol_parts.append("\n_References to these symbols could not be fetched._")

            parts.append("\n".join(symbol_parts))
            log.info("context_symbols_focused", symbols=symbol_list, with_definitions=True)

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
