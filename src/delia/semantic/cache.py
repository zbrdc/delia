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

from ..context import get_project_path
from ..embeddings import get_embeddings_client, cosine_similarity

log = structlog.get_logger()


def _get_cache_file(project_path: Path | None = None) -> Path:
    """Get the semantic cache file path for a project."""
    root = get_project_path(project_path)
    return root / ".delia" / "semantic_cache.json"


class SemanticCache:
    """
    Semantic cache for LLM responses.
    """

    def __init__(self, threshold: float = 0.95, project_path: Path | None = None):
        self.threshold = threshold
        self.project_path = project_path
        self._client = None  # Lazy singleton
        self.entries: list[dict[str, Any]] = []
        self._initialized = False
        self._lock = asyncio.Lock()

    def _cache_file(self) -> Path:
        """Get the cache file path for this instance."""
        return _get_cache_file(self.project_path)

    async def _get_client(self):
        if self._client is None:
            self._client = await get_embeddings_client()
        return self._client

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return

            cache_file = self._cache_file()
            if cache_file.exists():
                try:
                    self.entries = json.loads(cache_file.read_text())
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

        client = await self._get_client()
        query_vec = await client.embed_query(query)  # Use cached query embedding

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

        client = await self._get_client()
        query_vec = await client.embed(query)  # Store with regular embed
        
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
            cache_file = self._cache_file()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data))
        except Exception as e:
            log.warning("semantic_cache_save_failed", error=str(e))

# Per-project cache instances (keyed by resolved project path)
_caches: dict[str, SemanticCache] = {}


def get_semantic_cache(project_path: Path | str | None = None) -> SemanticCache:
    """Get the SemanticCache instance for a specific project.

    Args:
        project_path: Project root directory. Defaults to project context.

    Returns:
        SemanticCache instance for that project (cached).
    """
    resolved = get_project_path(project_path)
    key = str(resolved.resolve())

    if key not in _caches:
        _caches[key] = SemanticCache(project_path=resolved)
        log.debug("created_semantic_cache", project=key)

    return _caches[key]


def reset_semantic_cache(project_path: Path | str | None = None) -> None:
    """Reset the SemanticCache instance for a project.

    Args:
        project_path: Project to reset. If None, resets all.
    """
    global _caches
    if project_path is None:
        _caches.clear()
    else:
        key = str(Path(project_path).resolve())
        _caches.pop(key, None)
