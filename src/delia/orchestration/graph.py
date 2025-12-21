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

from .. import paths
from ..file_helpers import read_file_safe
from .constants import CODE_EXTENSIONS, IGNORE_DIRS

log = structlog.get_logger()

# File to store the symbol graph
GRAPH_CACHE_FILE = paths.DATA_DIR / "symbol_graph.json"

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
    mtime: float = 0.0

class SymbolGraph:
    """
    Builds and maintains a graph of code symbols and file dependencies.
    """

    def __init__(self, root: Path | None = None):
        self.root = root or Path.cwd()
        # Automatic Root Discovery
        if not root:
            for parent in [self.root] + list(self.root.parents):
                if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                    self.root = parent
                    break
        self.nodes: dict[str, FileNode] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Load graph from cache or perform initial scan."""
        async with self._lock:
            if self._initialized:
                return

            if GRAPH_CACHE_FILE.exists():
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

    def get_related_files(self, file_path: str) -> list[str]:
        """Get files that are dependencies of or depend on the given file."""
        if file_path not in self.nodes:
            return []
            
        related = set()
        
        # 1. Files this file imports (out-edges)
        for imp in self.nodes[file_path].imports:
            # Try to resolve import to a local file
            resolved = self._resolve_import(imp, file_path)
            if resolved:
                related.add(resolved)
                
        # 2. Files that import this file (in-edges)
        # (This is slower, but good for context)
        module_name = self._path_to_module(file_path)
        for path, node in self.nodes.items():
            if module_name in node.imports:
                related.add(path)
                
        return list(related)

    def find_symbol(self, name: str) -> list[Symbol]:
        """Find a symbol by name across the entire project."""
        results = []
        for node in self.nodes.values():
            for sym in node.symbols:
                if sym.name == name:
                    results.append(sym)
        return results

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
        data = json.loads(GRAPH_CACHE_FILE.read_text())
        for path, n_data in data.items():
            node = FileNode(path=path, mtime=n_data["mtime"])
            node.imports = set(n_data["imports"])
            node.symbols = [Symbol(**s) for s in n_data["symbols"]]
            self.nodes[path] = node

    def _save_cache(self) -> None:
        """Save graph to disk."""
        try:
            GRAPH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for path, node in self.nodes.items():
                data[path] = {
                    "mtime": node.mtime,
                    "imports": list(node.imports),
                    "symbols": [asdict(s) for s in node.symbols]
                }
            GRAPH_CACHE_FILE.write_text(json.dumps(data, indent=2))
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
