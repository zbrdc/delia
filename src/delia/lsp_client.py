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
                if shutil.which(srv):
                    return [srv, "--stdio"]
        elif language_id in ("typescript", "javascript"):
            if shutil.which("typescript-language-server"):
                return ["typescript-language-server", "--stdio"]
        elif language_id == "rust":
            if shutil.which("rust-analyzer"):
                return ["rust-analyzer"]
        elif language_id == "go":
            if shutil.which("gopls"):
                return ["gopls"]
        return None

    async def goto_definition(self, file_path: str, line: int, character: int) -> List[dict]:
        """Go to definition tool."""
        lang_id = self._guess_language(file_path)
        client = await self.get_client(lang_id)
        if not client:
            return []

        abs_path = (self.root_path / file_path).resolve()
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

    def _guess_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix
        if ext == ".py": return "python"
        if ext in (".ts", ".tsx"): return "typescript"
        if ext in (".js", ".jsx"): return "javascript"
        if ext == ".rs": return "rust"
        if ext == ".go": return "go"
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
