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

import re
from pydantic import BaseModel, Field

from ..file_helpers import read_files, read_memory
from ..language import detect_language
from ..project_memory import get_project_context
from .summarizer import get_summarizer
from .graph import get_symbol_graph

log = structlog.get_logger()


# =============================================================================
# SMART SYMBOL INJECTION HELPERS
# =============================================================================

class SymbolOutline(BaseModel):
    """Lightweight symbol representation for context injection."""
    name: str
    kind: str
    line: int
    children: list["SymbolOutline"] = Field(default_factory=list)
    
    def format(self, indent: int = 0) -> str:
        """Format symbol as indented outline."""
        prefix = "  " * indent
        result = f"{prefix}- {self.kind} {self.name} (line {self.line})"
        for child in self.children:
            result += "\n" + child.format(indent + 1)
        return result


# File path patterns to detect in prompts
_FILE_PATTERNS = [
    r'`([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,5})`',  # `path/to/file.py`
    r'"([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,5})"',  # "path/to/file.py"
    r"'([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,5})'",  # 'path/to/file.py'
    r'\b(src/[a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,5})\b',  # src/path/file.py
    r'\b([a-zA-Z0-9_]+\.(?:py|ts|js|tsx|jsx|rs|go|java))\b',  # file.py (common extensions)
]

# Symbol/class/function patterns
_SYMBOL_PATTERNS = [
    r'\b(?:class|function|def|method|func)\s+([A-Za-z_][A-Za-z0-9_]*)',  # class Foo, def bar
    r'\b([A-Z][a-zA-Z0-9_]*(?:Service|Manager|Handler|Controller|Client|Provider))\b',  # FooService
    r'`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)`',  # `ClassName` or `module.func`
]


def extract_file_mentions(content: str) -> list[str]:
    """Extract file paths mentioned in the content.
    
    Returns unique file paths that appear to be referenced in the prompt.
    """
    mentions = set()
    for pattern in _FILE_PATTERNS:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            path = match.group(1)
            # Filter out common false positives
            if not path.startswith('.') and '/' in path or '.' in path:
                mentions.add(path)
    return list(mentions)


def extract_symbol_mentions(content: str) -> list[str]:
    """Extract symbol names (classes, functions) mentioned in the content."""
    mentions = set()
    for pattern in _SYMBOL_PATTERNS:
        for match in re.finditer(pattern, content):
            symbol = match.group(1)
            if len(symbol) > 2:  # Filter out very short matches
                mentions.add(symbol)
    return list(mentions)


def format_symbol_outline(symbols: list[dict], max_depth: int = 2) -> str:
    """Format LSP symbols as a readable outline.
    
    Args:
        symbols: List of symbol dicts from LSP client
        max_depth: Maximum nesting depth to show
    
    Returns:
        Formatted string showing symbol structure
    """
    if not symbols:
        return "(no symbols found)"
    
    lines = []
    for sym in symbols:
        depth = sym.get("depth", 0)
        if depth > max_depth:
            continue
        
        indent = "  " * depth
        kind = sym.get("kind", "symbol")
        name = sym.get("name", "?")
        line = sym.get("range", {}).get("start_line", sym.get("location", {}).get("line", "?"))
        
        lines.append(f"{indent}- {kind} `{name}` (line {line})")
    
    return "\n".join(lines) if lines else "(no symbols found)"


async def get_symbols_for_files(
    file_paths: list[str],
    project_root: Path | None = None,
) -> dict[str, list[dict]]:
    """Get LSP symbols for multiple files.
    
    Returns dict mapping file path to list of symbols.
    """
    from ..lsp_client import get_lsp_client
    
    root = project_root or Path.cwd()
    results = {}
    
    try:
        client = get_lsp_client(root)
        
        for file_path in file_paths[:5]:  # Limit to 5 files to avoid overload
            # Resolve relative paths
            full_path = root / file_path if not Path(file_path).is_absolute() else Path(file_path)
            if full_path.exists():
                rel_path = str(full_path.relative_to(root))
                symbols = await client.document_symbols(rel_path)
                if symbols:
                    results[rel_path] = symbols
                    log.debug("symbols_loaded_for_file", file=rel_path, count=len(symbols))
    except Exception as e:
        log.debug("symbol_loading_failed", error=str(e))
    
    return results

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
        include_playbook_bullets: bool = True,
        include_smart_symbols: bool = True,
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
            include_playbook_bullets: Auto-inject ACE playbook bullets (ENFORCED by default)
            include_smart_symbols: Auto-inject symbol outlines for mentioned files (default True)
        """
        parts = []

        # 0. Project Instructions (Claude Code-like DELIA.md auto-loading)
        if include_project_instructions:
            project_context = get_project_context()
            if project_context:
                parts.append(project_context)
                log.debug("project_instructions_injected")

        # 0.1 ACE PLAYBOOK BULLETS (Strategic Guidance - ENFORCED)
        # This ensures playbook guidance is ALWAYS injected, not relying on agents to remember
        if include_playbook_bullets:
            try:
                from ..playbook import get_playbook_manager
                pm = get_playbook_manager()
                playbook_content = pm.get_auto_injection_bullets(content)
                if playbook_content:
                    parts.append(playbook_content)
                    log.info("ace_playbook_bullets_injected", task_type=pm.detect_task_type(content))
            except Exception as e:
                log.debug("playbook_injection_failed", error=str(e))

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

        # 2.7 SMART SYMBOL INJECTION (LSP-powered code awareness)
        # Detect file mentions in the prompt and inject their symbol outlines
        # This gives agents immediate structural awareness without loading full files
        if include_smart_symbols:
            mentioned_files = extract_file_mentions(content)
            if mentioned_files:
                try:
                    file_symbols = await get_symbols_for_files(mentioned_files)
                    if file_symbols:
                        symbol_parts = ["### Symbol Outline (auto-detected from prompt)"]
                        for file_path, syms in file_symbols.items():
                            if file_path not in explicit_files:  # Don't duplicate
                                outline = format_symbol_outline(syms)
                                symbol_parts.append(f"\n#### `{file_path}`\n{outline}")
                        
                        if len(symbol_parts) > 1:  # More than just the header
                            parts.append("\n".join(symbol_parts))
                            log.info("smart_symbols_injected", files=list(file_symbols.keys()))
                except Exception as e:
                    log.debug("smart_symbol_injection_failed", error=str(e))

        # 3. Delia's Memories (Project Knowledge)
        manual_memories = set()
        if context:
            manual_memories = {m.strip() for m in context.split(",")}
            for mem_name in manual_memories:
                mem_content = read_memory(mem_name)
                if mem_content:
                    parts.append(f"### Project Context ({mem_name}):\n{mem_content}")
                    log.info("context_memory_loaded", memory=mem_name)
                    
        # 3.5. Additive Semantic Memory (New for P3.1)
        # Search for relevant memories that weren't manually requested
        try:
            summarizer = get_summarizer()
            await summarizer.initialize()
            
            # Search only in memory:// entries
            semantic_results = await summarizer.search(content, top_k=5)
            added_semantic = 0
            for res in semantic_results:
                path = res["path"]
                if path.startswith("memory://"):
                    mem_name = path.replace("memory://", "")
                    if mem_name not in manual_memories and res["score"] > 0.6:
                        mem_content = read_memory(mem_name)
                        if mem_content:
                            parts.append(f"### Relevant Context ({mem_name}):\n{mem_content}")
                            added_semantic += 1
                            if added_semantic >= 2: # Limit to top 2 to avoid noise
                                break
            if added_semantic > 0:
                log.info("additive_semantic_memories_injected", count=added_semantic)
        except Exception as e:
            log.debug("additive_memory_failed", error=str(e))

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
