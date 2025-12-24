# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Hybrid Retrieval for Playbook Bullets.

Implements scoring formula: score = relevance^α × utility^β × recency^γ

Uses HybridEmbeddingsClient for embeddings (Ollama/API/local fallback).
Pre-computed bullet embeddings stored in .delia/embeddings.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from delia.playbook import PlaybookBullet, PlaybookManager

log = structlog.get_logger()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class ScoredBullet:
    """A bullet with hybrid retrieval score and breakdown."""
    bullet: "PlaybookBullet"
    final_score: float        # Combined score
    relevance_score: float    # Semantic similarity to query (0-1)
    utility_score: float      # helpful_count / total (0-1)
    recency_score: float      # Time decay factor (0-1)
    explanation: str | None = None  # Human-readable breakdown


class HybridRetriever:
    """
    Hybrid retrieval combining semantic relevance, utility, and recency.

    Uses HybridEmbeddingsClient (Ollama/API/local) for embeddings.
    Pre-computed bullet embeddings loaded from .delia/embeddings.json.

    Formula: score = relevance^α × utility^β × recency^γ
    """

    ALPHA = 1.0   # Relevance weight
    BETA = 0.5    # Utility weight
    GAMMA = 0.3   # Recency weight
    RECENCY_HALF_LIFE = 30.0

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        recency_half_life: float = 30.0,
    ):
        self.ALPHA = alpha
        self.BETA = beta
        self.GAMMA = gamma
        self.RECENCY_HALF_LIFE = recency_half_life

        # Embeddings client (lazy init)
        self._client = None
        self._client_available: bool | None = None

        # Pre-computed embeddings cache: bullet_id -> embedding
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._cache_loaded_for: Path | None = None

    async def _get_client(self):
        """Get or create HybridEmbeddingsClient."""
        if self._client is None:
            from delia.embeddings import HybridEmbeddingsClient
            self._client = HybridEmbeddingsClient()
            self._client_available = await self._client.initialize()
            if self._client_available:
                log.debug("retrieval_embeddings_available")
            else:
                log.debug("retrieval_embeddings_unavailable_fallback_to_utility")
        return self._client if self._client_available else None

    def _load_embeddings(self, project_path: Path) -> None:
        """Load pre-computed embeddings from .delia/embeddings.json."""
        if self._cache_loaded_for == project_path:
            return

        embeddings_file = project_path / ".delia" / "embeddings.json"
        if embeddings_file.exists():
            try:
                with open(embeddings_file) as f:
                    data = json.load(f)
                self._embedding_cache = {
                    k: np.array(v, dtype=np.float32)
                    for k, v in data.items()
                }
                log.debug("embeddings_loaded", count=len(self._embedding_cache), path=str(embeddings_file))
            except Exception as e:
                log.warning("embeddings_load_failed", error=str(e))
                self._embedding_cache = {}
        else:
            self._embedding_cache = {}

        self._cache_loaded_for = project_path

    def _save_embeddings(self, project_path: Path) -> None:
        """Save embeddings cache to .delia/embeddings.json."""
        embeddings_file = project_path / ".delia" / "embeddings.json"
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {k: v.tolist() for k, v in self._embedding_cache.items()}
            with open(embeddings_file, "w") as f:
                json.dump(data, f)
            log.debug("embeddings_saved", count=len(data))
        except Exception as e:
            log.warning("embeddings_save_failed", error=str(e))

    async def generate_embedding(self, text: str) -> np.ndarray | None:
        """Generate embedding for text using HybridEmbeddingsClient."""
        client = await self._get_client()
        if not client:
            return None
        try:
            return await client.embed(text)
        except Exception as e:
            log.debug("embedding_generation_failed", error=str(e))
            return None

    async def add_bullet_embedding(
        self,
        bullet_id: str,
        content: str,
        project_path: Path,
    ) -> bool:
        """Generate and store embedding for a new bullet."""
        embedding = await self.generate_embedding(content)
        if embedding is None:
            return False

        self._load_embeddings(project_path)
        self._embedding_cache[bullet_id] = embedding
        self._save_embeddings(project_path)
        return True

    def get_bullet_embedding(self, bullet_id: str) -> np.ndarray | None:
        """Get pre-computed embedding for a bullet."""
        return self._embedding_cache.get(bullet_id)

    def compute_utility_score(self, bullet: "PlaybookBullet") -> float:
        """Compute utility score with Laplace smoothing."""
        helpful = getattr(bullet, "helpful_count", 0) or 0
        harmful = getattr(bullet, "harmful_count", 0) or 0
        return (helpful + 1) / (helpful + harmful + 2)

    def compute_recency_score(self, bullet: "PlaybookBullet") -> float:
        """Compute recency score with exponential decay."""
        last_used = getattr(bullet, "last_used", None)
        if not last_used:
            last_used = getattr(bullet, "created_at", None)

        if not last_used:
            return 0.5

        try:
            if isinstance(last_used, str):
                last_date = datetime.fromisoformat(last_used[:19])
            else:
                last_date = last_used
            days_since = (datetime.now() - last_date).days
            return math.exp(-days_since / self.RECENCY_HALF_LIFE)
        except Exception:
            return 0.5

    def compute_final_score(
        self,
        relevance: float,
        utility: float,
        recency: float,
    ) -> float:
        """Combine scores: relevance^α × utility^β × recency^γ"""
        relevance = max(relevance, 0.001)
        utility = max(utility, 0.001)
        recency = max(recency, 0.001)
        return (
            pow(relevance, self.ALPHA) *
            pow(utility, self.BETA) *
            pow(recency, self.GAMMA)
        )

    async def retrieve(
        self,
        bullets: list["PlaybookBullet"],
        query: str,
        project_path: Path,
        limit: int = 5,
        min_score: float = 0.1,
    ) -> list[ScoredBullet]:
        """
        Retrieve bullets using hybrid scoring.

        Uses semantic relevance if embeddings available, otherwise utility × recency.
        """
        if not bullets:
            return []

        # Load pre-computed embeddings
        self._load_embeddings(project_path)

        # Try to get query embedding
        query_embedding = await self.generate_embedding(query)
        use_semantic = query_embedding is not None and len(self._embedding_cache) > 0

        scored: list[ScoredBullet] = []

        for bullet in bullets:
            utility = self.compute_utility_score(bullet)
            recency = self.compute_recency_score(bullet)

            # Compute relevance if we have embeddings
            if use_semantic:
                bullet_emb = self.get_bullet_embedding(bullet.id)
                if bullet_emb is not None:
                    relevance = cosine_similarity(query_embedding, bullet_emb)
                else:
                    relevance = 0.5  # No embedding for this bullet
            else:
                relevance = 0.5  # No semantic matching

            final = self.compute_final_score(relevance, utility, recency)

            if final < min_score:
                continue

            scored.append(ScoredBullet(
                bullet=bullet,
                final_score=final,
                relevance_score=relevance,
                utility_score=utility,
                recency_score=recency,
            ))

        scored.sort(key=lambda s: s.final_score, reverse=True)

        log.debug(
            "hybrid_retrieval",
            total=len(bullets),
            scored=len(scored),
            semantic=use_semantic,
            cached_embeddings=len(self._embedding_cache),
        )

        return scored[:limit]

    def retrieve_by_utility(
        self,
        bullets: list["PlaybookBullet"],
        limit: int = 5,
    ) -> list[ScoredBullet]:
        """Retrieve by utility × recency only (sync, no embeddings)."""
        scored = []
        for bullet in bullets:
            utility = self.compute_utility_score(bullet)
            recency = self.compute_recency_score(bullet)
            final = utility * recency

            scored.append(ScoredBullet(
                bullet=bullet,
                final_score=final,
                relevance_score=0.5,
                utility_score=utility,
                recency_score=recency,
            ))

        scored.sort(key=lambda s: s.final_score, reverse=True)
        return scored[:limit]

    async def retrieve_from_manager(
        self,
        manager: "PlaybookManager",
        task_type: str,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
    ) -> list[ScoredBullet]:
        """Retrieve from a PlaybookManager."""
        bullets = manager.load_playbook(task_type)
        project_path = Path(manager.playbook_dir).parent if manager.playbook_dir else Path.cwd()
        return await self.retrieve(
            bullets=bullets,
            query=query,
            project_path=project_path,
            limit=limit,
            min_score=min_score,
        )

    def invalidate_cache(self, bullet_id: str) -> None:
        """Remove a bullet from embedding cache."""
        self._embedding_cache.pop(bullet_id, None)

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache.clear()
        self._cache_loaded_for = None

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.close()
            self._client = None


# Singleton
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
