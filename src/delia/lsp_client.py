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
                
                # Initialize
                params = lsp.InitializeParams(
                    root_uri=self.root_path.as_uri(),
                    capabilities=lsp.ClientCapabilities(),
                    workspace_folders=[lsp.WorkspaceFolder(uri=self.root_path.as_uri(), name="workspace")]
                )
                await client.initialize_async(params)
                client.initialized(lsp.InitializedParams())
                
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

        # Open the document first (required by LSP)
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
            # Give pyright a moment to process the file
            await asyncio.sleep(0.1)
        except Exception as e:
            log.warning("lsp_open_failed", path=str(abs_path), error=str(e))

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
        elif isinstance(result, list):
            locations = result

        formatted = []
        for loc in locations:
            if isinstance(loc, lsp.Location):
                uri = loc.uri
                path = uri.replace("file://", "")
                if os.name == 'NT' and path.startswith('/'):
                    path = path[1:]
                
                formatted.append({
                    "path": os.path.relpath(path, self.root_path),
                    "line": loc.range.start.line + 1,
                    "character": loc.range.start.character
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
