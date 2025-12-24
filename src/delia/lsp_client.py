# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
LSP Client for Delia.

Provides a robust client to interact with various Language Servers (pyright, tsserver, etc.)
to provide deep code intelligence to Delia agents.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import structlog
from pathlib import Path
from typing import Any, List, Optional

from pygls.lsp.client import LanguageClient
from lsprotocol import types as lsp

log = structlog.get_logger()

class DeliaLSPClient:
    """
    A unified LSP client that can manage multiple language servers.
    """

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.clients: dict[str, LanguageClient] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get_client(self, language_id: str) -> Optional[LanguageClient]:
        """Get or start a client for the given language."""
        if language_id in self.clients:
            return self.clients[language_id]

        # Determine server command
        cmd = self._get_server_command(language_id)
        if not cmd:
            log.warning("lsp_server_not_found", language=language_id)
            return None

        async with self._get_lock(language_id):
            if language_id in self.clients:
                return self.clients[language_id]

            client = LanguageClient("delia-lsp", "1.0.0")
            try:
                log.info("lsp_server_starting", language=language_id, cmd=cmd)
                await client.start_io(*cmd)
                
                # Initialize with minimal capabilities to avoid slow workspace analysis
                # (Full capabilities like workspace_folders=True cause Pyright to hang)
                params = lsp.InitializeParams(
                    root_uri=self.root_path.as_uri(),
                    capabilities=lsp.ClientCapabilities(),
                    workspace_folders=[lsp.WorkspaceFolder(uri=self.root_path.as_uri(), name="workspace")]
                )
                await client.initialize_async(params)
                client.initialized(lsp.InitializedParams())

                # Give the language server time to start indexing
                # This is especially important for workspace/symbol queries
                await asyncio.sleep(1.0)

                self.clients[language_id] = client
                log.info("lsp_server_ready", language=language_id)
                return client
            except Exception as e:
                log.error("lsp_server_failed", language=language_id, error=str(e))
                return None

    def _get_lock(self, language_id: str) -> asyncio.Lock:
        if language_id not in self._locks:
            self._locks[language_id] = asyncio.Lock()
        return self._locks[language_id]

    def _get_server_command(self, language_id: str) -> Optional[List[str]]:
        """Map language IDs to server commands."""
        if language_id == "python":
            # Prefer based on what's installed
            for srv in ["pyright-langserver", "pylsp"]:
                cmd = self._find_executable(srv)
                if cmd:
                    return [cmd, "--stdio"]
        elif language_id in ("typescript", "javascript"):
            cmd = self._find_executable("typescript-language-server")
            if cmd:
                return [cmd, "--stdio"]
        elif language_id == "rust":
            cmd = self._find_executable("rust-analyzer")
            if cmd:
                return [cmd]
        elif language_id == "go":
            cmd = self._find_executable("gopls")
            if cmd:
                return [cmd]
        return None

    def _find_executable(self, name: str) -> Optional[str]:
        """Find executable in PATH or project's venv."""
        # Check PATH first
        found = shutil.which(name)
        if found:
            return found

        # Check project's .venv/bin
        venv_bin = self.root_path / ".venv" / "bin" / name
        if venv_bin.exists() and os.access(venv_bin, os.X_OK):
            return str(venv_bin)

        # Check common venv locations
        for venv_dir in ["venv", ".venv", "env"]:
            venv_bin = self.root_path / venv_dir / "bin" / name
            if venv_bin.exists() and os.access(venv_bin, os.X_OK):
                return str(venv_bin)

        return None

    async def goto_definition(self, file_path: str, line: int, character: int) -> List[dict]:
        """Go to definition tool."""
        lang_id = self._guess_language(file_path)
        client = await self.get_client(lang_id)
        if not client:
            return []

        abs_path = (self.root_path / file_path).resolve()
        if not abs_path.exists():
            return []

        # Open the document first (required by LSP)
        await self._open_document(client, abs_path, lang_id)

        params = lsp.DefinitionParams(
            text_document=lsp.TextDocumentIdentifier(uri=abs_path.as_uri()),
            position=lsp.Position(line=line - 1, character=character)
        )

        try:
            result = await client.text_document_definition_async(params)
            return self._format_locations(result)
        except Exception as e:
            log.error("lsp_definition_failed", error=str(e))
            return []

    async def find_references(self, file_path: str, line: int, character: int) -> List[dict]:
        """Find references tool."""
        lang_id = self._guess_language(file_path)
        client = await self.get_client(lang_id)
        if not client:
            return []

        abs_path = (self.root_path / file_path).resolve()
        if not abs_path.exists():
            return []

        # Ensure workspace is indexed for cross-file references (Pyright issue #10086)
        await self._ensure_workspace_indexed(client, lang_id)
        
        # Open the target document
        await self._open_document(client, abs_path, lang_id)

        params = lsp.ReferenceParams(
            text_document=lsp.TextDocumentIdentifier(uri=abs_path.as_uri()),
            position=lsp.Position(line=line - 1, character=character),
            context=lsp.ReferenceContext(include_declaration=True)
        )

        try:
            result = await client.text_document_references_async(params)
            return self._format_locations(result)
        except Exception as e:
            log.error("lsp_references_failed", error=str(e))
            return []

    async def hover(self, file_path: str, line: int, character: int) -> Optional[str]:
        """Get hover information (type info, docs)."""
        lang_id = self._guess_language(file_path)
        client = await self.get_client(lang_id)
        if not client:
            return None

        abs_path = (self.root_path / file_path).resolve()
        if not abs_path.exists():
            return None

        # Open the document first (required by LSP)
        await self._open_document(client, abs_path, lang_id)

        params = lsp.HoverParams(
            text_document=lsp.TextDocumentIdentifier(uri=abs_path.as_uri()),
            position=lsp.Position(line=line - 1, character=character)
        )

        try:
            result = await client.text_document_hover_async(params)
            if result and result.contents:
                if isinstance(result.contents, str):
                    return result.contents
                if isinstance(result.contents, lsp.MarkupContent):
                    return result.contents.value
                # Handle list of contents
                return str(result.contents)
            return None
        except Exception as e:
            log.error("lsp_hover_failed", error=str(e))
            return None

    async def document_symbols(self, file_path: str) -> List[dict] | dict:
        """Get all symbols in a document.

        Returns a hierarchical list of symbols (classes, functions, etc.)
        with their ranges and kinds. Returns a dict with 'error' key on failure.
        """
        lang_id = self._guess_language(file_path)
        client = await self.get_client(lang_id)
        if not client:
            return {"error": f"No language server for '{lang_id}'. Install pyright: uv add pyright"}

        abs_path = (self.root_path / file_path).resolve()
        if not abs_path.exists():
            return {"error": f"File not found: {file_path}"}

        # Open the document first (required by LSP)
        await self._open_document(client, abs_path, lang_id)

        params = lsp.DocumentSymbolParams(
            text_document=lsp.TextDocumentIdentifier(uri=abs_path.as_uri())
        )

        try:
            result = await client.text_document_document_symbol_async(params)
            return self._format_symbols(result)
        except Exception as e:
            log.error("lsp_symbols_failed", error=str(e))
            return {"error": f"LSP error: {str(e)}"}

    async def workspace_symbol(self, query: str, language_id: str = "python") -> List[dict] | dict:
        """Search for symbols across the entire workspace.

        Uses LSP workspace/symbol request which is much faster than
        file-by-file document symbol scanning.

        Args:
            query: Symbol name pattern to search for
            language_id: Language server to use (default: python)

        Returns:
            List of matching symbols with name, kind, location, and container info.
            Returns a dict with 'error' key on failure.
        """
        client = await self.get_client(language_id)
        if not client:
            return {"error": f"No language server for '{language_id}'. Install pyright: uv add pyright"}

        params = lsp.WorkspaceSymbolParams(query=query)

        try:
            result = await client.workspace_symbol_async(params)
            return self._format_workspace_symbols(result)
        except Exception as e:
            log.error("lsp_workspace_symbol_failed", error=str(e), query=query)
            return {"error": f"LSP error: {str(e)}"}

    def _format_workspace_symbols(self, result: Any) -> List[dict]:
        """Format workspace symbol results into a consistent structure."""
        if not result:
            return []

        symbols = []
        for item in result:
            # Handle both SymbolInformation and WorkspaceSymbol
            if hasattr(item, "location"):
                # SymbolInformation format
                loc = item.location
                uri = loc.uri if hasattr(loc, "uri") else str(loc)
                path = uri.replace("file://", "") if uri.startswith("file://") else uri
                if os.name == "nt" and path.startswith("/"):
                    path = path[1:]
                # Make path relative to root
                try:
                    path = str(Path(path).relative_to(self.root_path))
                except ValueError:
                    pass  # Keep absolute if not under root

                sym = {
                    "name": item.name,
                    "kind": self._symbol_kind_name(item.kind),
                    "file": path,
                    "range": {
                        "start_line": loc.range.start.line + 1,
                        "start_char": loc.range.start.character,
                        "end_line": loc.range.end.line + 1,
                        "end_char": loc.range.end.character,
                    },
                }
                if hasattr(item, "container_name") and item.container_name:
                    sym["container"] = item.container_name
                symbols.append(sym)
            elif hasattr(item, "name"):
                # WorkspaceSymbol format (lazy resolution)
                sym = {
                    "name": item.name,
                    "kind": self._symbol_kind_name(item.kind) if hasattr(item, "kind") else "unknown",
                }
                if hasattr(item, "container_name") and item.container_name:
                    sym["container"] = item.container_name
                symbols.append(sym)

        return symbols

    async def _open_document(self, client: LanguageClient, abs_path: Path, lang_id: str) -> None:
        """Open a document in the language server if not already open."""
        uri = abs_path.as_uri()
        if not hasattr(self, '_opened_docs'):
            self._opened_docs: set[str] = set()

        if uri in self._opened_docs:
            return

        try:
            content = abs_path.read_text()
            params = lsp.DidOpenTextDocumentParams(
                text_document=lsp.TextDocumentItem(
                    uri=uri,
                    language_id=lang_id,
                    version=1,
                    text=content
                )
            )
            client.text_document_did_open(params)
            self._opened_docs.add(uri)
            # Minimal delay - Pyright queues these internally
            await asyncio.sleep(0.01)
        except Exception as e:
            log.warning("lsp_open_failed", path=str(abs_path), error=str(e))

    async def _ensure_workspace_indexed(self, client: LanguageClient, lang_id: str) -> None:
        """Open all workspace files to ensure Pyright indexes them for references.
        
        This is a workaround for Pyright issue #10086 where find_references only
        returns results from files that have been explicitly opened with didOpen.
        """
        if not hasattr(self, '_workspace_indexed'):
            self._workspace_indexed: set[str] = set()
        
        if lang_id in self._workspace_indexed:
            return
        
        # Find all files of this language type in the workspace
        extensions = {
            'python': ['.py'],
            'typescript': ['.ts', '.tsx'],
            'javascript': ['.js', '.jsx'],
            'rust': ['.rs'],
            'go': ['.go'],
        }
        
        exts = extensions.get(lang_id, [])
        if not exts:
            return
        
        # Collect files, limiting to src/ and tests/ to avoid opening too many
        files_to_open: list[Path] = []
        search_dirs = ['src', 'tests', 'lib', 'app']
        
        for search_dir in search_dirs:
            search_path = self.root_path / search_dir
            if search_path.exists():
                for ext in exts:
                    files_to_open.extend(search_path.rglob(f'*{ext}'))
        
        # Limit to reasonable number of files
        MAX_FILES = 200
        if len(files_to_open) > MAX_FILES:
            log.warning("lsp_workspace_too_large", 
                       count=len(files_to_open), 
                       limit=MAX_FILES,
                       msg="Limiting workspace indexing to first 200 files")
            files_to_open = files_to_open[:MAX_FILES]
        
        log.info("lsp_workspace_indexing", language=lang_id, file_count=len(files_to_open))
        
        # Open files in batches - Pyright processes these quickly
        BATCH_SIZE = 50
        for i in range(0, len(files_to_open), BATCH_SIZE):
            batch = files_to_open[i:i + BATCH_SIZE]
            for file_path in batch:
                await self._open_document(client, file_path, lang_id)
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        self._workspace_indexed.add(lang_id)
        log.info("lsp_workspace_indexed", language=lang_id, file_count=len(files_to_open))
        
        # Give Pyright time to process all opened files before querying
        await asyncio.sleep(1.0)

    async def rename_symbol(
        self, file_path: str, line: int, character: int, new_name: str
    ) -> dict[str, Any]:
        """Rename a symbol across the codebase.

        Returns a dict with the edits to apply, grouped by file.
        """
        lang_id = self._guess_language(file_path)
        client = await self.get_client(lang_id)
        if not client:
            return {"error": "No LSP client available"}

        abs_path = (self.root_path / file_path).resolve()
        params = lsp.RenameParams(
            text_document=lsp.TextDocumentIdentifier(uri=abs_path.as_uri()),
            position=lsp.Position(line=line - 1, character=character),
            new_name=new_name
        )

        try:
            result = await client.text_document_rename_async(params)
            if not result:
                return {"error": "Rename not supported or no changes"}
            return self._format_workspace_edit(result)
        except Exception as e:
            log.error("lsp_rename_failed", error=str(e))
            return {"error": str(e)}

    async def prepare_rename(
        self, file_path: str, line: int, character: int
    ) -> Optional[dict]:
        """Check if rename is valid at position and get the symbol range."""
        lang_id = self._guess_language(file_path)
        client = await self.get_client(lang_id)
        if not client:
            return None

        abs_path = (self.root_path / file_path).resolve()
        params = lsp.PrepareRenameParams(
            text_document=lsp.TextDocumentIdentifier(uri=abs_path.as_uri()),
            position=lsp.Position(line=line - 1, character=character)
        )

        try:
            result = await client.text_document_prepare_rename_async(params)
            if not result:
                return None
            if isinstance(result, lsp.Range):
                return {
                    "start_line": result.start.line + 1,
                    "start_char": result.start.character,
                    "end_line": result.end.line + 1,
                    "end_char": result.end.character,
                }
            if isinstance(result, lsp.PrepareRenameResult_Type1):
                return {
                    "start_line": result.range.start.line + 1,
                    "start_char": result.range.start.character,
                    "end_line": result.range.end.line + 1,
                    "end_char": result.range.end.character,
                    "placeholder": result.placeholder,
                }
            return None
        except Exception as e:
            log.debug("lsp_prepare_rename_failed", error=str(e))
            return None

    def _format_symbols(self, result: Any, depth: int = 0) -> List[dict]:
        """Convert LSP symbols to simple dicts with hierarchy."""
        if not result:
            return []

        symbols = []
        items = result if isinstance(result, list) else [result]

        for item in items:
            if isinstance(item, lsp.DocumentSymbol):
                sym = {
                    "name": item.name,
                    "kind": self._symbol_kind_name(item.kind),
                    "range": {
                        "start_line": item.range.start.line + 1,
                        "end_line": item.range.end.line + 1,
                    },
                    "selection_range": {
                        "start_line": item.selection_range.start.line + 1,
                        "start_char": item.selection_range.start.character,
                        "end_line": item.selection_range.end.line + 1,
                        "end_char": item.selection_range.end.character,
                    },
                    "depth": depth,
                }
                if item.detail:
                    sym["detail"] = item.detail
                symbols.append(sym)
                # Recurse for children
                if item.children:
                    symbols.extend(self._format_symbols(item.children, depth + 1))
            elif isinstance(item, lsp.SymbolInformation):
                # Flat symbol format (older LSP)
                symbols.append({
                    "name": item.name,
                    "kind": self._symbol_kind_name(item.kind),
                    "location": {
                        "line": item.location.range.start.line + 1,
                        "character": item.location.range.start.character,
                    },
                    "depth": depth,
                })

        return symbols

    def _symbol_kind_name(self, kind: lsp.SymbolKind) -> str:
        """Convert SymbolKind enum to readable name."""
        names = {
            lsp.SymbolKind.File: "file",
            lsp.SymbolKind.Module: "module",
            lsp.SymbolKind.Namespace: "namespace",
            lsp.SymbolKind.Package: "package",
            lsp.SymbolKind.Class: "class",
            lsp.SymbolKind.Method: "method",
            lsp.SymbolKind.Property: "property",
            lsp.SymbolKind.Field: "field",
            lsp.SymbolKind.Constructor: "constructor",
            lsp.SymbolKind.Enum: "enum",
            lsp.SymbolKind.Interface: "interface",
            lsp.SymbolKind.Function: "function",
            lsp.SymbolKind.Variable: "variable",
            lsp.SymbolKind.Constant: "constant",
            lsp.SymbolKind.String: "string",
            lsp.SymbolKind.Number: "number",
            lsp.SymbolKind.Boolean: "boolean",
            lsp.SymbolKind.Array: "array",
            lsp.SymbolKind.Object: "object",
            lsp.SymbolKind.Key: "key",
            lsp.SymbolKind.Null: "null",
            lsp.SymbolKind.EnumMember: "enum_member",
            lsp.SymbolKind.Struct: "struct",
            lsp.SymbolKind.Event: "event",
            lsp.SymbolKind.Operator: "operator",
            lsp.SymbolKind.TypeParameter: "type_parameter",
        }
        return names.get(kind, f"unknown({kind})")

    def _format_workspace_edit(self, edit: lsp.WorkspaceEdit) -> dict[str, Any]:
        """Convert WorkspaceEdit to dict of file -> edits."""
        result: dict[str, Any] = {"files": {}, "total_edits": 0}

        if edit.changes:
            for uri, text_edits in edit.changes.items():
                path = uri.replace("file://", "")
                if os.name == 'NT' and path.startswith('/'):
                    path = path[1:]
                rel_path = os.path.relpath(path, self.root_path)

                edits = []
                for te in text_edits:
                    edits.append({
                        "start_line": te.range.start.line + 1,
                        "start_char": te.range.start.character,
                        "end_line": te.range.end.line + 1,
                        "end_char": te.range.end.character,
                        "new_text": te.new_text,
                    })
                result["files"][rel_path] = edits
                result["total_edits"] += len(edits)

        if edit.document_changes:
            for change in edit.document_changes:
                if isinstance(change, lsp.TextDocumentEdit):
                    uri = change.text_document.uri
                    path = uri.replace("file://", "")
                    if os.name == 'NT' and path.startswith('/'):
                        path = path[1:]
                    rel_path = os.path.relpath(path, self.root_path)

                    edits = []
                    for te in change.edits:
                        if isinstance(te, lsp.TextEdit):
                            edits.append({
                                "start_line": te.range.start.line + 1,
                                "start_char": te.range.start.character,
                                "end_line": te.range.end.line + 1,
                                "end_char": te.range.end.character,
                                "new_text": te.new_text,
                            })
                    if rel_path not in result["files"]:
                        result["files"][rel_path] = []
                    result["files"][rel_path].extend(edits)
                    result["total_edits"] += len(edits)

        return result

    def _guess_language(self, file_path: str) -> str:
        """Guess language from file path using centralized language detection."""
        from .language import detect_language, EXTENSION_TO_LANGUAGE

        ext = Path(file_path).suffix.lower()

        # Use centralized extension map first
        if ext in EXTENSION_TO_LANGUAGE:
            lang = EXTENSION_TO_LANGUAGE[ext]
            # Normalize for LSP (react -> typescript, vue -> typescript, etc.)
            if lang in ("react", "vue", "svelte", "angular"):
                return "typescript"
            return lang

        # Fallback to content-based detection (without content)
        lang = detect_language("", file_path)
        if lang != "unknown":
            return lang

        return "plain"

    def _format_locations(self, result: Any) -> List[dict]:
        """Convert LSP locations to simple dicts."""
        if not result:
            return []
        
        locations = []
        if isinstance(result, lsp.Location):
            locations = [result]
        elif isinstance(result, (list, tuple)):
            locations = list(result)

        formatted = []
        for loc in locations:
            uri = None
            range_obj = None
            
            if isinstance(loc, lsp.Location):
                uri = loc.uri
                range_obj = loc.range
            elif isinstance(loc, lsp.LocationLink):
                uri = loc.target_uri
                range_obj = loc.target_range
            elif hasattr(loc, 'uri') and hasattr(loc, 'range'):
                # Generic fallback for Location-like objects
                uri = loc.uri
                range_obj = loc.range
            
            if uri and range_obj:
                path = uri.replace("file://", "")
                if os.name == 'nt' and path.startswith('/'):
                    path = path[1:]
                
                formatted.append({
                    "path": os.path.relpath(path, self.root_path),
                    "line": range_obj.start.line + 1,
                    "character": range_obj.start.character
                })
        return formatted

    async def shutdown(self):
        """Shutdown all clients."""
        for lang, client in self.clients.items():
            try:
                await client.shutdown_async()
                client.exit()
                log.info("lsp_server_stopped", language=lang)
            except:
                pass
        self.clients.clear()

# Singleton
_lsp_client: DeliaLSPClient | None = None

def get_lsp_client(root: Path) -> DeliaLSPClient:
    global _lsp_client
    if _lsp_client is None:
        _lsp_client = DeliaLSPClient(root)
    return _lsp_client
