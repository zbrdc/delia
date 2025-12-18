# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Code Retrieval-Augmented Generation (CodeRAG).

Indexes the codebase to allow semantic search over source code.
Uses a sliding window chunking strategy to handle large files.
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from . import paths
from .embeddings import EmbeddingsClient, cosine_similarity, DEFAULT_EMBEDDINGS_URL

log = structlog.get_logger()

# File to store code index
CODE_INDEX_FILE = paths.DATA_DIR / "code_index.json"

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".c", ".cpp", ".h", 
    ".java", ".kt", ".swift", ".rb", ".php", ".sh", ".json", ".toml", ".yaml", ".yml"
}

# Directories to ignore
IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv", 
    ".next", "target", "vendor", ".idea", ".vscode"
}

@dataclass
class CodeChunk:
    """A chunk of code from a file."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    mtime: float
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChunk":
        return cls(**data)


class CodeRAG:
    """
    RAG system for the codebase.
    
    - Scans project root for code files
    - Chunks files into segments
    - Embeds chunks
    - Performs semantic search
    """

    def __init__(self, embeddings_url: str = DEFAULT_EMBEDDINGS_URL):
        self.client = EmbeddingsClient(embeddings_url)
        self.chunks: list[CodeChunk] = []
        self._file_mtimes: dict[str, float] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
        
        # Determine project root (fallback to CWD if paths.PROJECT_ROOT is library loc)
        self.root = Path.cwd()

    async def initialize(self) -> bool:
        """Initialize the CodeRAG index."""
        async with self._lock:
            if self._initialized:
                return True
                
            # Load cache
            self._load_index()
            
            # Sync with disk (background task in real app, but synchronous here for simplicity)
            # Only sync if embeddings are available
            if await self.client.health_check():
                updated = await self._sync_codebase()
                if updated:
                    self._save_index()
            else:
                log.warning("code_rag_embeddings_unavailable")
                
            self._initialized = True
            log.info("code_rag_initialized", chunks=len(self.chunks))
            return True

    async def _sync_codebase(self) -> bool:
        """Scan codebase and update index."""
        updated = False
        current_files = set()
        
        # Walk the directory tree
        for path in self.root.rglob("*"):
            # Check exclusions
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
                
            if path.suffix not in CODE_EXTENSIONS or not path.is_file():
                continue
                
            # Get relative path for ID
            try:
                rel_path = str(path.relative_to(self.root))
            except ValueError:
                continue
                
            current_files.add(rel_path)
            
            try:
                stat = path.stat()
                mtime = stat.st_mtime
                
                # Check if file needs re-indexing
                if rel_path not in self._file_mtimes or self._file_mtimes[rel_path] < mtime:
                    # Remove old chunks for this file
                    self.chunks = [c for c in self.chunks if c.file_path != rel_path]
                    
                    # Read and chunk file
                    content = path.read_text(encoding="utf-8", errors="replace")
                    new_chunks = await self._chunk_and_embed(rel_path, content, mtime)
                    
                    self.chunks.extend(new_chunks)
                    self._file_mtimes[rel_path] = mtime
                    updated = True
                    log.info("code_rag_indexed_file", file=rel_path, chunks=len(new_chunks))
                    
            except Exception as e:
                log.debug("code_rag_file_error", file=rel_path, error=str(e))
        
        # Remove deleted files
        original_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.file_path in current_files]
        if len(self.chunks) != original_count:
            updated = True
            # Cleanup mtimes
            self._file_mtimes = {k: v for k, v in self._file_mtimes.items() if k in current_files}
            
        return updated

    async def _chunk_and_embed(self, file_path: str, content: str, mtime: float) -> list[CodeChunk]:
        """Split code into chunks and embed them."""
        chunks = []
        lines = content.splitlines()
        total_lines = len(lines)
        
        # Sliding window parameters
        CHUNK_SIZE = 60  # lines
        OVERLAP = 10     # lines
        
        if total_lines <= CHUNK_SIZE:
            # Small file: single chunk
            chunk_content = content
            embedding_array = await self.client.embed(chunk_content)
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=1,
                end_line=total_lines,
                content=chunk_content,
                mtime=mtime,
                embedding=embedding_array.tolist()
            ))
        else:
            # Sliding window
            start = 0
            while start < total_lines:
                end = min(start + CHUNK_SIZE, total_lines)
                chunk_lines = lines[start:end]
                chunk_content = "\n".join(chunk_lines)
                
                # Context header to help embedding understand file location
                context_header = f"# File: {file_path} (Lines {start+1}-{end})\n"
                full_text = context_header + chunk_content
                
                embedding_array = await self.client.embed(full_text)
                
                chunks.append(CodeChunk(
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end,
                    content=chunk_content,
                    mtime=mtime,
                    embedding=embedding_array.tolist()
                ))
                
                start += (CHUNK_SIZE - OVERLAP)
                
        return chunks

    async def search(self, query: str, top_k: int = 3, threshold: float = 0.45) -> list[tuple[str, float]]:
        """Search for relevant code chunks."""
        if not self._initialized:
            if not await self.initialize():
                return []
        
        if not self.chunks:
            return []
            
        try:
            query_vec = await self.client.embed(query)
            
            scores = []
            for chunk in self.chunks:
                if chunk.embedding:
                    chunk_vec = np.array(chunk.embedding, dtype=np.float32)
                    score = cosine_similarity(query_vec, chunk_vec)
                    if score >= threshold:
                        scores.append((chunk, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            # Deduplicate by file to avoid showing same file multiple times?
            # Maybe show top 2 chunks per file max.
            seen_files = {}
            
            for chunk, score in scores:
                if len(results) >= top_k:
                    break
                    
                # Limit chunks per file
                if seen_files.get(chunk.file_path, 0) >= 2:
                    continue
                seen_files[chunk.file_path] = seen_files.get(chunk.file_path, 0) + 1
                
                context = f"### Code: {chunk.file_path} ({chunk.start_line}-{chunk.end_line})\n```{"{Path(chunk.file_path).suffix[1:]}"}\n{chunk.content}\n```"
                results.append((context, score))
                log.debug("code_rag_hit", file=chunk.file_path, score=f"{score:.2f}")
                
            return results
            
        except Exception as e:
            log.warning("code_rag_search_failed", error=str(e))
            return []

    def _load_index(self) -> None:
        """Load index from disk."""
        if not CODE_INDEX_FILE.exists():
            return
            
        try:
            with open(CODE_INDEX_FILE, "r") as f:
                data = json.load(f)
                self.chunks = [CodeChunk.from_dict(c) for c in data.get("chunks", [])]
                self._file_mtimes = data.get("mtimes", {})
        except Exception as e:
            log.warning("code_rag_load_error", error=str(e))

    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            data = {
                "chunks": [c.to_dict() for c in self.chunks],
                "mtimes": self._file_mtimes
            }
            temp_file = CODE_INDEX_FILE.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f)
            temp_file.replace(CODE_INDEX_FILE)
        except Exception as e:
            log.warning("code_rag_save_error", error=str(e))


# Global singleton
_code_rag: CodeRAG | None = None

def get_code_rag() -> CodeRAG:
    """Get the global CodeRAG instance."""
    global _code_rag
    if _code_rag is None:
        _code_rag = CodeRAG()
    return _code_rag
