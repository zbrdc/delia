# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Hybrid Retrieval for Playbook Bullets.

Implements scoring formula: score = relevance^α × utility^β × recency^γ

Uses HybridEmbeddingsClient for embeddings (Ollama/API/local fallback).
Primary storage: ChromaDB vector database for efficient similarity search.
Fallback: Pre-computed bullet embeddings in .delia/embeddings.json.
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
        """Get the global embeddings client singleton."""
        if self._client is None:
            from delia.embeddings import get_embeddings_client
            try:
                self._client = await get_embeddings_client()
                self._client_available = True
                log.debug("retrieval_embeddings_available")
            except Exception as e:
                log.debug("retrieval_embeddings_unavailable_fallback_to_utility", error=str(e))
                self._client_available = False
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

        # Try to get query embedding with caching
        client = await self._get_client()
        query_embedding = await client.embed_query(query) if client else None
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
        """Clean up local resources (singleton client is not closed)."""
        # Don't close the shared singleton client - just clear reference
        self._client = None
        self._client_available = None

    # =========================================================================
    # CHROMADB-BASED RETRIEVAL (Preferred)
    # =========================================================================

    def _get_vector_store(self, project_path: Path | str | None = None):
        """Get the VectorStore instance for a specific project."""
        from delia.orchestration.vector_store import get_vector_store
        return get_vector_store(project_path)

    async def index_bullets_to_chromadb(
        self,
        bullets: list["PlaybookBullet"],
        task_type: str,
        project: str | None = None,
        project_path: Path | str | None = None,
    ) -> int:
        """Index playbook bullets to ChromaDB using batch embeddings.

        Args:
            bullets: Bullets to index
            task_type: Task type (coding, testing, etc.)
            project: Optional project identifier for metadata filtering
            project_path: Project path for per-project ChromaDB storage

        Returns:
            Number of bullets indexed
        """
        if not bullets:
            return 0

        store = self._get_vector_store(project_path)

        # Use batch embedding for efficiency
        client = await self._get_client()
        if not client:
            log.warning("indexing_failed_no_client")
            return 0

        contents = [b.content for b in bullets]
        embeddings = await client.embed_batch(contents, input_type="document")

        indexed = 0
        for bullet, embedding in zip(bullets, embeddings):
            # Skip zero vectors (failed embeddings)
            if embedding is None or (hasattr(embedding, 'sum') and embedding.sum() == 0):
                continue

            store.add_playbook_bullet(
                bullet_id=bullet.id,
                content=bullet.content,
                embedding=embedding.tolist(),
                task_type=task_type,
                project=project,
                utility_score=self.compute_utility_score(bullet),
            )
            indexed += 1

        log.info("indexed_bullets_to_chromadb", count=indexed, task_type=task_type, project=project)
        return indexed

    async def retrieve_from_chromadb(
        self,
        query: str,
        task_type: str | None = None,
        project: str | None = None,
        project_path: Path | str | None = None,
        limit: int = 5,
        min_score: float = 0.1,
    ) -> list[ScoredBullet]:
        """Retrieve bullets using ChromaDB semantic search.

        Args:
            query: Natural language query
            task_type: Optional filter by task type
            project: Optional project context for metadata filtering
            project_path: Project path for per-project ChromaDB storage
            limit: Max results
            min_score: Minimum similarity score

        Returns:
            Scored bullets with semantic relevance
        """
        # Get query embedding with caching
        client = await self._get_client()
        if not client:
            log.debug("chromadb_retrieve_no_client")
            return []

        query_embedding = await client.embed_query(query)
        if query_embedding is None:
            log.debug("chromadb_retrieve_no_embedding")
            return []

        store = self._get_vector_store(project_path)
        results = store.search_playbook(
            query_embedding=query_embedding.tolist(),
            task_type=task_type,
            project=project,
            n_results=limit * 2,  # Get extra for filtering
        )

        if not results:
            return []

        # Convert to ScoredBullets with utility/recency applied
        from delia.playbook import PlaybookBullet

        scored = []
        for r in results:
            if r["score"] < min_score:
                continue

            # Reconstruct bullet from metadata
            meta = r.get("metadata", {})
            bullet = PlaybookBullet(
                id=r["id"],
                content=r["content"],
                section=meta.get("task_type", "coding"),
            )

            # Apply hybrid scoring
            relevance = r["score"]
            utility = meta.get("utility_score", 0.5)
            recency = 0.8  # ChromaDB doesn't store recency, use default

            final = self.compute_final_score(relevance, utility, recency)

            scored.append(ScoredBullet(
                bullet=bullet,
                final_score=final,
                relevance_score=relevance,
                utility_score=utility,
                recency_score=recency,
            ))

        scored.sort(key=lambda s: s.final_score, reverse=True)

        log.debug(
            "chromadb_retrieval",
            query=query[:50],
            results=len(scored),
            task_type=task_type,
        )

        return scored[:limit]

    async def migrate_json_to_chromadb(
        self,
        project_path: Path,
        task_types: list[str] | None = None,
    ) -> dict[str, int]:
        """Migrate existing JSON embeddings to ChromaDB.

        Args:
            project_path: Project path containing .delia/
            task_types: Task types to migrate (default: all)

        Returns:
            Dict of task_type -> count migrated
        """
        from delia.playbook import get_playbook_manager

        if task_types is None:
            task_types = [
                "coding", "testing", "debugging", "security",
                "architecture", "deployment", "performance",
                "api", "git", "project",
            ]

        # Load existing JSON embeddings
        self._load_embeddings(project_path)

        pm = get_playbook_manager()
        pm.set_project(project_path)

        migrated = {}
        for task_type in task_types:
            bullets = pm.load_playbook(task_type)
            if not bullets:
                continue

            count = await self.index_bullets_to_chromadb(
                bullets=bullets,
                task_type=task_type,
                project=str(project_path),
                project_path=project_path,
            )
            migrated[task_type] = count

        total = sum(migrated.values())
        log.info("migrated_to_chromadb", total=total, by_type=migrated)
        return migrated


# Singleton
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
