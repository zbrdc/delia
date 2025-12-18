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
Retrieval-Augmented Generation (RAG) for Delia Memories.

Provides semantic search capabilities over .serena documentation files.
Reuses existing EmbeddingsClient and paths configuration.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from . import paths
from .embeddings import EmbeddingsClient, cosine_similarity, DEFAULT_EMBEDDINGS_URL

log = structlog.get_logger()

# File to store memory index
MEMORY_INDEX_FILE = paths.DATA_DIR / "memory_index.json"


@dataclass
class MemoryDocument:
    """A single memory document."""
    name: str
    content: str
    mtime: float  # Modification time for cache invalidation
    embedding: list[float] | None = None  # Stored as list for JSON serialization

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryDocument":
        return cls(**data)


class MemoryRAG:
    """
    RAG system for .serena memories.
    
    - Indexes .md files in MEMORIES_DIR
    - Uses local embeddings model
    - Caches embeddings to disk
    - Performs cosine similarity search
    """

    def __init__(self, embeddings_url: str = DEFAULT_EMBEDDINGS_URL):
        self.client = EmbeddingsClient(embeddings_url)
        self.documents: dict[str, MemoryDocument] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> bool:
        """Initialize the RAG index (lazy load)."""
        async with self._lock:
            if self._initialized:
                return True
                
            # Load cache
            self._load_index()
            
            # Sync with disk
            updated = await self._sync_memories()
            
            if updated:
                self._save_index()
                
            self._initialized = True
            log.info("memory_rag_initialized", documents=len(self.documents))
            return True

    async def _sync_memories(self) -> bool:
        """Scan MEMORIES_DIR and update index."""
        if not paths.MEMORIES_DIR.exists():
            return False
            
        updated = False
        current_files = set()
        
        # Check embeddings availability
        if not await self.client.health_check():
            log.warning("memory_rag_embeddings_unavailable")
            return False

        # Scan files
        for file_path in paths.MEMORIES_DIR.glob("*.md"):
            name = file_path.stem
            current_files.add(name)
            
            try:
                stat = file_path.stat()
                mtime = stat.st_mtime
                
                # Check if needs update (new or modified)
                if name not in self.documents or self.documents[name].mtime < mtime:
                    content = file_path.read_text(encoding="utf-8")
                    
                    # Generate embedding
                    embedding_array = await self.client.embed(content)
                    embedding_list = embedding_array.tolist()
                    
                    self.documents[name] = MemoryDocument(
                        name=name,
                        content=content,
                        mtime=mtime,
                        embedding=embedding_list
                    )
                    updated = True
                    log.info("memory_rag_indexed", memory=name)
                    
            except Exception as e:
                log.warning("memory_rag_index_failed", memory=name, error=str(e))
        
        # Remove deleted files
        to_remove = [name for name in self.documents if name not in current_files]
        for name in to_remove:
            del self.documents[name]
            updated = True
            log.info("memory_rag_removed", memory=name)
            
        return updated

    async def search(self, query: str, top_k: int = 3, threshold: float = 0.4) -> list[tuple[str, float]]:
        """
        Search for relevant memories.
        
        Args:
            query: The search query (user message)
            top_k: Number of results to return
            threshold: Minimum similarity score (0.0 - 1.0)
            
        Returns:
            List of (content, score) tuples
        """
        if not self._initialized:
            if not await self.initialize():
                return []
        
        if not self.documents:
            return []
            
        try:
            # Embed query
            query_vec = await self.client.embed(query)
            
            scores = []
            for doc in self.documents.values():
                if doc.embedding:
                    doc_vec = np.array(doc.embedding, dtype=np.float32)
                    score = cosine_similarity(query_vec, doc_vec)
                    if score >= threshold:
                        scores.append((doc, score))
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Format results
            results = []
            for doc, score in scores[:top_k]:
                context = f"### Memory: {doc.name}\n{doc.content}"
                results.append((context, score))
                log.debug("memory_rag_hit", memory=doc.name, score=f"{score:.2f}")
                
            return results
            
        except Exception as e:
            log.warning("memory_rag_search_failed", error=str(e))
            return []

    def _load_index(self) -> None:
        """Load index from disk."""
        if not MEMORY_INDEX_FILE.exists():
            return
            
        try:
            with open(MEMORY_INDEX_FILE, "r") as f:
                data = json.load(f)
                self.documents = {
                    name: MemoryDocument.from_dict(doc_data)
                    for name, doc_data in data.items()
                }
        except Exception as e:
            log.warning("memory_rag_load_error", error=str(e))

    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            data = {name: doc.to_dict() for name, doc in self.documents.items()}
            # Atomic write
            temp_file = MEMORY_INDEX_FILE.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f)
            temp_file.replace(MEMORY_INDEX_FILE)
        except Exception as e:
            log.warning("memory_rag_save_error", error=str(e))


# Global singleton
_rag: MemoryRAG | None = None

def get_rag() -> MemoryRAG:
    """Get the global RAG instance."""
    global _rag
    if _rag is None:
        _rag = MemoryRAG()
    return _rag
