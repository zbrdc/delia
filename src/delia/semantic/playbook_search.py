# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Playbook Vector Search.

Enables semantic retrieval of strategic playbook bullets, allowing Delia
to find relevant lessons learned even when keywords don't match exactly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import structlog

from ..embeddings import HybridEmbeddingsClient, cosine_similarity
from ..playbook import playbook_manager, PlaybookBullet

log = structlog.get_logger()

class PlaybookVectorSearch:
    """
    Semantic search for playbook bullets.
    """

    def __init__(self):
        self.client = HybridEmbeddingsClient()

    async def find_relevant_strategies(
        self, 
        task_type: str, 
        query: str, 
        limit: int = 3,
        threshold: float = 0.5
    ) -> list[PlaybookBullet]:
        """Find the most semantically relevant bullets for a task."""
        bullets = playbook_manager.load_playbook(task_type)
        if not bullets:
            return []
            
        try:
            query_vec = await self.client.embed(query)
            
            scored_bullets = []
            for bullet in bullets:
                # We lazily embed bullets if they don't have one (hypothetically)
                # For now we embed on the fly for search
                bullet_vec = await self.client.embed(bullet.content)
                score = cosine_similarity(query_vec, bullet_vec)
                
                if score >= threshold:
                    scored_bullets.append((score, bullet))
            
            # Sort by score descending
            scored_bullets.sort(key=lambda x: x[0], reverse=True)
            
            return [b for score, b in scored_bullets[:limit]]
            
        except Exception as e:
            log.warning("playbook_vector_search_failed", error=str(e))
            return bullets[:limit] # Fallback to top bullets

# Singleton
_search: PlaybookVectorSearch | None = None

def get_playbook_search() -> PlaybookVectorSearch:
    global _search
    if _search is None:
        _search = PlaybookVectorSearch()
    return _search
