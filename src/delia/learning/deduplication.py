# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Semantic Deduplication for Playbook Bullets.

Uses HybridEmbeddingsClient for semantic similarity when available.
Falls back to string-based matching otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog

if TYPE_CHECKING:
    from delia.playbook import PlaybookBullet

log = structlog.get_logger()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class SimilarityMatch:
    """Result of a similarity check against a single bullet."""
    bullet_id: str
    content: str
    similarity: float
    is_duplicate: bool


@dataclass
class DeduplicationResult:
    """Full result of deduplication check."""
    is_duplicate: bool
    best_match: SimilarityMatch | None
    all_matches: list[SimilarityMatch]
    recommended_action: Literal["add", "skip", "merge"]
    merge_suggestion: str | None = None


@dataclass
class Cluster:
    """A cluster of similar bullets."""
    bullets: list["PlaybookBullet"]
    centroid_id: str
    avg_similarity: float


class SemanticDeduplicator:
    """
    Deduplication for playbook bullets.

    Uses HybridEmbeddingsClient for semantic similarity when available.
    Falls back to string-based matching otherwise.

    Thresholds:
    - DUPLICATE_THRESHOLD (0.90): Skip add entirely
    - MERGE_THRESHOLD (0.85): Consider merging
    - SIMILAR_THRESHOLD (0.75): Flag as related
    """

    DUPLICATE_THRESHOLD = 0.90
    MERGE_THRESHOLD = 0.85
    SIMILAR_THRESHOLD = 0.75

    def __init__(
        self,
        duplicate_threshold: float = 0.90,
        merge_threshold: float = 0.85,
        similar_threshold: float = 0.75,
    ):
        self.DUPLICATE_THRESHOLD = duplicate_threshold
        self.MERGE_THRESHOLD = merge_threshold
        self.SIMILAR_THRESHOLD = similar_threshold

        # Embeddings client (lazy init)
        self._client = None
        self._client_available: bool | None = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    async def _get_client(self):
        """Get or create HybridEmbeddingsClient."""
        if self._client is None:
            from delia.embeddings import HybridEmbeddingsClient
            self._client = HybridEmbeddingsClient()
            self._client_available = await self._client.initialize()
        return self._client if self._client_available else None

    async def _embed(self, text: str) -> np.ndarray | None:
        """Get embedding for text."""
        client = await self._get_client()
        if not client:
            return None
        try:
            return await client.embed(text)
        except Exception as e:
            log.debug("dedup_embed_failed", error=str(e))
            return None

    async def _embed_bullet(self, bullet: "PlaybookBullet") -> np.ndarray | None:
        """Get or compute embedding for a bullet (cached)."""
        if bullet.id in self._embedding_cache:
            return self._embedding_cache[bullet.id]

        embedding = await self._embed(bullet.content)
        if embedding is not None:
            self._embedding_cache[bullet.id] = embedding
        return embedding

    def _string_similarity(self, a: str, b: str) -> float:
        """Compute string similarity using SequenceMatcher."""
        a_norm = a.lower().strip()
        b_norm = b.lower().strip()
        if a_norm == b_norm:
            return 1.0
        return SequenceMatcher(None, a_norm, b_norm).ratio()

    async def check_similarity(
        self,
        new_content: str,
        existing_bullets: list["PlaybookBullet"],
        threshold: float | None = None,
    ) -> DeduplicationResult:
        """
        Check if new content is similar to existing bullets.

        Uses semantic similarity if embeddings available, else string matching.
        """
        if threshold is None:
            threshold = self.SIMILAR_THRESHOLD

        if not existing_bullets:
            return DeduplicationResult(
                is_duplicate=False,
                best_match=None,
                all_matches=[],
                recommended_action="add",
            )

        # Try semantic similarity first
        new_embedding = await self._embed(new_content)
        use_semantic = new_embedding is not None

        matches: list[SimilarityMatch] = []

        for bullet in existing_bullets:
            if use_semantic:
                bullet_emb = await self._embed_bullet(bullet)
                if bullet_emb is not None:
                    similarity = cosine_similarity(new_embedding, bullet_emb)
                else:
                    similarity = self._string_similarity(new_content, bullet.content)
            else:
                similarity = self._string_similarity(new_content, bullet.content)

            if similarity >= threshold:
                matches.append(SimilarityMatch(
                    bullet_id=bullet.id,
                    content=bullet.content,
                    similarity=similarity,
                    is_duplicate=similarity >= self.DUPLICATE_THRESHOLD,
                ))

        matches.sort(key=lambda m: m.similarity, reverse=True)

        if not matches:
            return DeduplicationResult(
                is_duplicate=False,
                best_match=None,
                all_matches=[],
                recommended_action="add",
            )

        best_match = matches[0]

        if best_match.similarity >= self.DUPLICATE_THRESHOLD:
            return DeduplicationResult(
                is_duplicate=True,
                best_match=best_match,
                all_matches=matches,
                recommended_action="skip",
            )
        elif best_match.similarity >= self.MERGE_THRESHOLD:
            return DeduplicationResult(
                is_duplicate=False,
                best_match=best_match,
                all_matches=matches,
                recommended_action="merge",
            )
        else:
            return DeduplicationResult(
                is_duplicate=False,
                best_match=best_match,
                all_matches=matches,
                recommended_action="add",
            )

    def check_similarity_sync(
        self,
        new_content: str,
        existing_bullets: list["PlaybookBullet"],
        threshold: float | None = None,
    ) -> DeduplicationResult:
        """Sync version using string matching only."""
        if threshold is None:
            threshold = self.SIMILAR_THRESHOLD

        if not existing_bullets:
            return DeduplicationResult(
                is_duplicate=False,
                best_match=None,
                all_matches=[],
                recommended_action="add",
            )

        matches: list[SimilarityMatch] = []

        for bullet in existing_bullets:
            similarity = self._string_similarity(new_content, bullet.content)
            if similarity >= threshold:
                matches.append(SimilarityMatch(
                    bullet_id=bullet.id,
                    content=bullet.content,
                    similarity=similarity,
                    is_duplicate=similarity >= self.DUPLICATE_THRESHOLD,
                ))

        matches.sort(key=lambda m: m.similarity, reverse=True)

        if not matches:
            return DeduplicationResult(
                is_duplicate=False,
                best_match=None,
                all_matches=[],
                recommended_action="add",
            )

        best_match = matches[0]

        if best_match.similarity >= self.DUPLICATE_THRESHOLD:
            return DeduplicationResult(
                is_duplicate=True,
                best_match=best_match,
                all_matches=matches,
                recommended_action="skip",
            )
        elif best_match.similarity >= self.MERGE_THRESHOLD:
            return DeduplicationResult(
                is_duplicate=False,
                best_match=best_match,
                all_matches=matches,
                recommended_action="merge",
            )
        else:
            return DeduplicationResult(
                is_duplicate=False,
                best_match=best_match,
                all_matches=matches,
                recommended_action="add",
            )

    def find_clusters(
        self,
        bullets: list["PlaybookBullet"],
        threshold: float | None = None,
    ) -> list[Cluster]:
        """Find clusters of similar bullets using string similarity."""
        if threshold is None:
            threshold = self.SIMILAR_THRESHOLD

        if len(bullets) < 2:
            return []

        used = set()
        clusters = []

        for i, bullet in enumerate(bullets):
            if bullet.id in used:
                continue

            cluster_bullets = [bullet]
            used.add(bullet.id)

            for other in bullets[i + 1:]:
                if other.id in used:
                    continue

                sim = self._string_similarity(bullet.content, other.content)
                if sim >= threshold:
                    cluster_bullets.append(other)
                    used.add(other.id)

            if len(cluster_bullets) > 1:
                centroid = max(cluster_bullets, key=lambda b: len(b.content))
                avg_sim = sum(
                    self._string_similarity(centroid.content, b.content)
                    for b in cluster_bullets
                ) / len(cluster_bullets)

                clusters.append(Cluster(
                    bullets=cluster_bullets,
                    centroid_id=centroid.id,
                    avg_similarity=avg_sim,
                ))

        return clusters

    def invalidate_cache(self, bullet_id: str) -> None:
        """Remove a bullet from embedding cache."""
        self._embedding_cache.pop(bullet_id, None)


# Singleton
_deduplicator: SemanticDeduplicator | None = None


def get_deduplicator() -> SemanticDeduplicator:
    """Get or create the global deduplicator instance."""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = SemanticDeduplicator()
    return _deduplicator
