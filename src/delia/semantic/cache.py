# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Semantic Response Cache.

Uses embeddings to find semantically similar past interactions to avoid
redundant LLM calls and provide instant responses for common queries.
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ..embeddings import HybridEmbeddingsClient, cosine_similarity

log = structlog.get_logger()

# Project-specific semantic cache (.delia/ in CWD)
CACHE_FILE = Path.cwd() / ".delia" / "semantic_cache.json"

class SemanticCache:
    """
    Semantic cache for LLM responses.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.client = HybridEmbeddingsClient()
        self.entries: list[dict[str, Any]] = []
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            
            if CACHE_FILE.exists():
                try:
                    self.entries = json.loads(CACHE_FILE.read_text())
                    # Convert embeddings back to numpy
                    for entry in self.entries:
                        entry["embedding"] = np.array(entry["embedding"], dtype=np.float32)
                except Exception as e:
                    log.warning("semantic_cache_load_failed", error=str(e))
            
            self._initialized = True

    async def get(self, query: str) -> str | None:
        """Find a semantically similar entry in the cache."""
        if not self._initialized:
            await self.initialize()

        query_vec = await self.client.embed(query)
        
        best_score = 0.0
        best_response = None
        
        for entry in self.entries:
            score = cosine_similarity(query_vec, entry["embedding"])
            if score > best_score:
                best_score = score
                best_response = entry["response"]
        
        if best_score >= self.threshold:
            log.info("semantic_cache_hit", score=round(best_score, 3))
            return best_response
        
        return None

    async def set(self, query: str, response: str) -> None:
        """Add a new entry to the cache."""
        if not self._initialized:
            await self.initialize()

        query_vec = await self.client.embed(query)
        
        async with self._lock:
            self.entries.append({
                "query": query,
                "response": response,
                "embedding": query_vec
            })
            self._save()

    def _save(self) -> None:
        try:
            # Convert numpy to list for JSON
            data = []
            for entry in self.entries:
                data.append({
                    "query": entry["query"],
                    "response": entry["response"],
                    "embedding": entry["embedding"].tolist()
                })
            CACHE_FILE.write_text(json.dumps(data))
        except Exception as e:
            log.warning("semantic_cache_save_failed", error=str(e))

# Singleton
_cache: SemanticCache | None = None

def get_semantic_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
