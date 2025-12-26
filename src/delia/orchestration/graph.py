# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Project Symbol Graph and Dependency Mapping.

Extracts a global graph of symbols (classes, functions) and their
relationships across the codebase. This powers GraphRAG by allowing
Delia to follow dependencies when gathering context.
"""

from __future__ import annotations

import ast
import re
import json
import asyncio
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import structlog

from ..context import get_project_path
from ..file_helpers import read_file_safe
from .constants import CODE_EXTENSIONS, IGNORE_DIRS

log = structlog.get_logger()


def _get_graph_cache_file(project_path: Path | None = None) -> Path:
    """Get the graph cache file path for a project."""
    root = get_project_path(project_path)
    return root / ".delia" / "symbol_graph.json"

@dataclass
class Symbol:
    """A named entity within a code file."""
    name: str
    kind: str  # class, function, method, async_function
    line: int
    file_path: str
    summary: str | None = None

@dataclass
class FileNode:
    """A node in the dependency graph representing a file."""
    path: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    edge_summaries: dict[str, str] = field(default_factory=dict) # target_path -> explanation
    mtime: float = 0.0

class SymbolGraph:
    """
    Builds and maintains a graph of code symbols and file dependencies.
    """

    def __init__(self, root: Path | None = None):
        self.root = get_project_path(root)
        # Automatic Root Discovery
        if not root:
            for parent in [self.root] + list(self.root.parents):
                if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                    self.root = parent
                    break
        self.nodes: dict[str, FileNode] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    def _cache_file(self) -> Path:
        """Get the cache file path for this graph."""
        return _get_graph_cache_file(self.root)

    async def initialize(self) -> None:
        """Load graph from cache or perform initial scan."""
        async with self._lock:
            if self._initialized:
                return

            cache_file = self._cache_file()
            if cache_file.exists():
                try:
                    self._load_cache()
                    log.info("symbol_graph_loaded", files=len(self.nodes))
                except Exception as e:
                    log.warning("symbol_graph_load_failed", error=str(e))
            
            # Trigger background sync (non-blocking)
            asyncio.create_task(self.sync())
            self._initialized = True

    async def sync(self, force: bool = False) -> int:
        """Scan project and update graph for changed files."""
        async with self._lock:
            updated_count = 0
            
            for path in self.root.rglob("*"):
                if any(part in IGNORE_DIRS for part in path.parts):
                    continue
                if path.suffix not in CODE_EXTENSIONS or not path.is_file():
                    continue
                    
                rel_path = str(path.relative_to(self.root))
                mtime = path.stat().st_mtime
                
                if force or rel_path not in self.nodes or self.nodes[rel_path].mtime < mtime:
                    success = self._parse_file(rel_path, path, mtime)
                    if success:
                        updated_count += 1
            
            if updated_count > 0:
                self._save_cache()
                log.info("symbol_graph_sync_complete", updated=updated_count)
            
            return updated_count

    async def explain_dependency(self, source_path: str, target_path: str) -> str:
        """
        Lazily explain why source_path depends on target_path.
        
        Uses a quick LLM model to analyze the relationship and caches the result.
        """
        if source_path not in self.nodes:
            return "Source file not indexed."
            
        node = self.nodes[source_path]
        if target_path in node.edge_summaries:
            return node.edge_summaries[target_path]
            
        # Not in cache, generate it
        log.info("generating_edge_explanation", source=source_path, target=target_path)
        
        from ..llm import call_llm
        from ..file_helpers import read_file_safe
        
        source_content, _ = read_file_safe(str(self.root / source_path), max_size=10000)
        target_content, _ = read_file_safe(str(self.root / target_path), max_size=10000)
        
        if not source_content or not target_content:
            return "Could not read files to explain dependency."
            
        prompt = f"""Explain the relationship between these two files.
How does `{source_path}` use `{target_path}`?
Focus on the primary interaction (e.g. inherits from, calls factory, uses utility).

FILE A ({source_path}):
{source_content[:2000]}

FILE B ({target_path}):
{target_content[:2000]}

Output ONE short sentence of architectural context."""

        res = await call_llm(model="quick", prompt=prompt, task_type="summarize")
        explanation = res.get("response", "Relationship could not be summarized.").strip()
        
        async with self._lock:
            node.edge_summaries[target_path] = explanation
            self._save_cache()
            
        return explanation

    def get_related_files(self, file_path: str, max_depth: int = 2) -> list[str]:
        """
        Get files related by dependency graph using BFS traversal.
        
        Args:
            file_path: Source file to start traversal from
            max_depth: How many hops to follow (default 2 for 'Butterfly Effect' awareness)
            
        Returns:
            List of related file paths, sorted by proximity.
        """
        if file_path not in self.nodes:
            return []
            
        visited = {file_path: 0} # path -> depth
        queue = [(file_path, 0)]
        
        while queue:
            current_path, current_depth = queue.pop(0)
            
            if current_depth >= max_depth:
                continue
                
            # Find all neighbors (out-edges and in-edges)
            neighbors = set()
            
            # 1. Out-edges (imports)
            if current_path in self.nodes:
                for imp in self.nodes[current_path].imports:
                    resolved = self._resolve_import(imp, current_path)
                    if resolved:
                        neighbors.add(resolved)
                        
            # 2. In-edges (dependents) - slower but necessary for depth
            module_name = self._path_to_module(current_path)
            for path, node in self.nodes.items():
                if module_name in node.imports:
                    neighbors.add(path)
                    
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = current_depth + 1
                    queue.append((neighbor, current_depth + 1))
                    
        # Remove source file from results
        if file_path in visited:
            del visited[file_path]
            
        # Return sorted by depth (closest first)
        return [p for p, d in sorted(visited.items(), key=lambda x: x[1])]

    def find_symbol(self, name: str, semantic_fallback: bool = False) -> list[Symbol]:
        """Find a symbol by name across the entire project."""
        results = []
        for node in self.nodes.values():
            for sym in node.symbols:
                if sym.name == name:
                    results.append(sym)
                    
        # If no exact match and semantic fallback enabled, try searching by meaning
        if not results and semantic_fallback:
            # Note: This is a synchronous wrapper for what will likely be 
            # an async operation in the caller. We'll provide a separate async method.
            pass
            
        return results

    async def search_symbols_semantic(self, query: str, top_k: int = 5) -> list[Symbol]:
        """
        Search for symbols semantically using the CodeSummarizer index.
        
        Args:
            query: Natural language query (e.g. "auth logic")
            top_k: Number of relevant files to inspect for symbols
            
        Returns:
            List of symbols from the most semantically relevant files.
        """
        from .summarizer import get_summarizer
        summarizer = get_summarizer()
        await summarizer.initialize()
        
        # 1. Find semantically relevant files
        file_results = await summarizer.search(query, top_k=top_k)
        
        results = []
        for res in file_results:
            path = res["path"]
            if path in self.nodes:
                # Add all top-level symbols from this relevant file
                # In the future, we could further filter these by LLM/embeddings
                results.extend(self.nodes[path].symbols)
                
        return results[:20] # Cap total symbols returned

    def get_hot_files(self, limit: int = 10, since_hours: float = 24.0) -> list[tuple[str, float]]:
        """Get recently modified files sorted by modification time.
        
        Args:
            limit: Maximum number of files to return
            since_hours: Only include files modified within this many hours
            
        Returns:
            List of (file_path, mtime) tuples, most recent first
        """
        import time
        cutoff = time.time() - (since_hours * 3600)
        
        recent = [
            (path, node.mtime) 
            for path, node in self.nodes.items()
            if node.mtime > cutoff
        ]
        recent.sort(key=lambda x: x[1], reverse=True)
        return recent[:limit]

    def get_file_recency_score(self, file_path: str, decay_hours: float = 24.0) -> float:
        """Get a recency score for a file (1.0 = just modified, 0.0 = old).
        
        Uses exponential decay based on how recently the file was modified.
        
        Args:
            file_path: Path to check
            decay_hours: Hours until score drops to ~37% (1/e)
            
        Returns:
            Float from 0.0 to 1.0 indicating recency
        """
        import time
        import math
        
        if file_path not in self.nodes:
            return 0.0
            
        mtime = self.nodes[file_path].mtime
        age_hours = (time.time() - mtime) / 3600
        
        # Exponential decay: score = e^(-age/decay)
        return math.exp(-age_hours / decay_hours)

    def _parse_file(self, rel_path: str, full_path: Path, mtime: float) -> bool:
        """Parse symbols and imports from a file based on its extension."""
        content, error = read_file_safe(str(full_path))
        if not content:
            if error:
                log.debug("file_read_failed", file=rel_path, error=error)
            return False

        node = FileNode(path=rel_path, mtime=mtime)
        
        try:
            if rel_path.endswith(".py"):
                self._parse_python(content, rel_path, node)
            elif rel_path.endswith((".js", ".ts", ".tsx", ".jsx")):
                self._parse_js_ts(content, rel_path, node)
            else:
                # Generic regex-based fallback
                self._parse_generic(content, rel_path, node)
                
            self.nodes[rel_path] = node
            return True
        except Exception as e:
            log.debug("file_parse_failed", file=rel_path, error=str(e))
            return False

    def _parse_python(self, content: str, rel_path: str, node: FileNode) -> None:
        """Use AST to extract symbols and imports from Python code."""
        try:
            tree = ast.parse(content)
            for item in tree.body:
                if isinstance(item, ast.ClassDef):
                    node.symbols.append(Symbol(item.name, "class", item.lineno, rel_path))
                elif isinstance(item, ast.FunctionDef):
                    node.symbols.append(Symbol(item.name, "function", item.lineno, rel_path))
                elif isinstance(item, ast.AsyncFunctionDef):
                    node.symbols.append(Symbol(item.name, "async_function", item.lineno, rel_path))
                
                # Imports
                if isinstance(item, ast.Import):
                    for alias in item.names:
                        node.imports.add(alias.name)
                elif isinstance(item, ast.ImportFrom):
                    # Store level explicitly to avoid dot-counting ambiguity
                    module_name = item.module if item.module else ""
                    if item.level > 0:
                        node.imports.add(f"REL:{item.level}:{module_name}")
                    else:
                        node.imports.add(module_name)
        except SyntaxError:
            # Fallback to regex if AST fails (e.g. invalid syntax in some files)
            self._parse_generic(content, rel_path, node)

    def _parse_js_ts(self, content: str, rel_path: str, node: FileNode) -> None:
        """Regex-based extraction for JS/TS (basic)."""
        # Classes
        for match in re.finditer(r"class\s+(\w+)", content):
            node.symbols.append(Symbol(match.group(1), "class", content[:match.start()].count("\n") + 1, rel_path))
        
        # Functions (function name() or const name = ()) 
        for match in re.finditer(r"function\s+(\w+)\s*\(", content):
            node.symbols.append(Symbol(match.group(1), "function", content[:match.start()].count("\n") + 1, rel_path))
            
        # Imports (from "module" or require("module"))
        for match in re.finditer(r"from\s+['\"](.+?)['\"]", content):
            node.imports.add(match.group(1))
        for match in re.finditer(r"require\s*\(\s*['\"](.+?)['\"]", content):
            node.imports.add(match.group(1))

    def _parse_generic(self, content: str, rel_path: str, node: FileNode) -> None:
        """Simple regex fallback for unknown languages."""
        # Common patterns for functions/classes in many languages
        patterns = {
            "class": r"\bclass\s+(\w+)",
            "function": r"\b(?:def|func|fn|function)\s+(\w+)\s*\(",
        }
        
        for kind, pattern in patterns.items():
            for match in re.finditer(pattern, content):
                node.symbols.append(Symbol(match.group(1), kind, content[:match.start()].count("\n") + 1, rel_path))

    def _resolve_import(self, imp: str, from_path: str) -> str | None:
        """Convert a module import to a relative file path if it exists locally."""
        log.debug("resolving_import", imp=imp, from_path=from_path)
        
        # 1. Relative import resolution (e.g. REL:2:result -> ../result.py)
        if imp.startswith("REL:"):
            try:
                _, level_str, module_name = imp.split(":", 2)
                level = int(level_str)
                
                parent = Path(from_path).parent
                # level=1 means same dir, level=2 means parent dir
                target_parent = parent
                for _ in range(level - 1):
                    target_parent = target_parent.parent
                
                if not module_name:
                    # from .. import x
                    for ext in ["/__init__.py", ".py"]:
                        test_path = str(target_parent) + ext
                        if test_path in self.nodes:
                            log.debug("import_resolved", path=test_path)
                            return test_path
                else:
                    # from ..module import x
                    sub_path = module_name.replace(".", "/")
                    for ext in [".py", "/__init__.py"]:
                        test_path = str(target_parent / sub_path) + ext
                        if test_path in self.nodes:
                            log.debug("import_resolved", path=test_path)
                            return test_path
            except Exception as e:
                log.debug("relative_resolve_failed", error=str(e))
                return None
        
        # 2. Absolute package resolution (e.g. delia.orchestration -> src/delia/orchestration)
        if imp.startswith("delia"):
            parts = imp.split(".")
            sub_path = "/".join(parts[1:])
            
            for ext in [".py", "/__init__.py"]:
                test_path = "src/delia/" + sub_path + ext
                if test_path in self.nodes:
                    log.debug("import_resolved", path=test_path)
                    return test_path
                
        return None

    def _path_to_module(self, path: str) -> str:
        """Convert a file path to a Python-style module name."""
        p = Path(path)
        if p.name == "__init__.py":
            p = p.parent
        else:
            p = p.with_suffix("")
            
        # Strip common roots
        parts = list(p.parts)
        if parts[0] == "src":
            parts = parts[1:]
            
        return ".".join(parts)

    def _load_cache(self) -> None:
        """Load graph from disk."""
        cache_file = self._cache_file()
        data = json.loads(cache_file.read_text())
        for path, n_data in data.items():
            node = FileNode(path=path, mtime=n_data["mtime"])
            node.imports = set(n_data["imports"])
            node.symbols = [Symbol(**s) for s in n_data["symbols"]]
            node.edge_summaries = n_data.get("edge_summaries", {})
            self.nodes[path] = node

    def _save_cache(self) -> None:
        """Save graph to disk."""
        try:
            cache_file = self._cache_file()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for path, node in self.nodes.items():
                data[path] = {
                    "mtime": node.mtime,
                    "imports": list(node.imports),
                    "symbols": [asdict(s) for s in node.symbols],
                    "edge_summaries": node.edge_summaries
                }
            cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.warning("graph_save_failed", error=str(e))

# Singleton holder
_graph: SymbolGraph | None = None

def get_symbol_graph() -> SymbolGraph:
    """Get the global SymbolGraph instance."""
    global _graph
    if _graph is None:
        _graph = SymbolGraph()
    return _graph
