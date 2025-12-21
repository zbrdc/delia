# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Semantic Intent Matching using Sentence Transformers.

This is Layer 2 of Delia's tiered intent detection system.
When regex patterns (Layer 1) are uncertain, we use semantic
similarity to match user messages against canonical exemplars.

Design:
- Pre-compute embeddings for all exemplars at startup
- At runtime, embed user message and find nearest neighbors
- Return intent with confidence based on cosine similarity
- GPU-accelerated when available (~50-100ms latency)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import structlog

from ..prompts import ModelRole, OrchestrationMode
from .exemplars import IntentExemplar, get_all_exemplars
from .result import DetectedIntent

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger()

# Model configuration - use the shared model from embeddings module
SIMILARITY_THRESHOLD = 0.65  # Minimum similarity to trust


@dataclass
class SemanticMatch:
    """Result of semantic matching."""
    exemplar: IntentExemplar
    similarity: float
    
    @property
    def confidence(self) -> float:
        """Convert similarity to confidence (0-1)."""
        # Similarity of 0.65+ maps to confidence 0.5+
        # Similarity of 0.85+ maps to confidence 0.9+
        return min(1.0, (self.similarity - 0.5) * 2.5)


class SemanticIntentMatcher:
    """
    Semantic intent detection using sentence embeddings.

    Uses the SHARED embedding model from embeddings module to avoid
    loading multiple models. All embedding operations share one model.

    Thread-safe and caches exemplar embeddings for reuse.

    Usage:
        matcher = get_semantic_matcher()
        matches = matcher.find_matches("verify this code is correct")
        # matches[0].exemplar.orchestration_mode == VOTING
        # matches[0].confidence == 0.87
    """

    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._exemplars: list[IntentExemplar] = []
        self._exemplar_embeddings: np.ndarray | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization - get shared model and compute exemplar embeddings."""
        if self._initialized:
            return

        start = time.time()

        try:
            # Use the SHARED model from embeddings module (loaded once for entire system)
            from ..embeddings import get_shared_model, SHARED_EMBEDDING_MODEL

            log.info("semantic_using_shared_model", model=SHARED_EMBEDDING_MODEL)
            self._model = get_shared_model()

            # Load exemplars
            self._exemplars = get_all_exemplars()

            # Pre-compute embeddings for all exemplars
            exemplar_texts = [e.text for e in self._exemplars]
            self._exemplar_embeddings = self._model.encode(
                exemplar_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,  # For cosine similarity
            )

            self._initialized = True
            elapsed = (time.time() - start) * 1000

            log.info(
                "semantic_initialized",
                model=SHARED_EMBEDDING_MODEL,
                exemplars=len(self._exemplars),
                elapsed_ms=int(elapsed),
            )

        except ImportError as e:
            log.warning("semantic_import_error", error=str(e))
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from e
    
    def find_matches(
        self,
        message: str,
        top_k: int = 5,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[SemanticMatch]:
        """
        Find the closest matching exemplars for a message.
        
        Args:
            message: User message to match
            top_k: Maximum number of matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of SemanticMatch sorted by similarity (descending)
        """
        self._ensure_initialized()
        
        if not self._model or self._exemplar_embeddings is None:
            return []
        
        start = time.time()
        
        # Encode the user message
        message_embedding = self._model.encode(
            message,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        
        # Compute cosine similarities (dot product since normalized)
        similarities = np.dot(self._exemplar_embeddings, message_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build matches above threshold
        matches = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                matches.append(SemanticMatch(
                    exemplar=self._exemplars[idx],
                    similarity=sim,
                ))
        
        elapsed = (time.time() - start) * 1000
        
        if matches:
            log.debug(
                "semantic_match",
                message=message[:50],
                top_match=matches[0].exemplar.text[:30],
                similarity=f"{matches[0].similarity:.3f}",
                elapsed_ms=int(elapsed),
            )
        
        return matches
    
    def detect_intent(
        self,
        message: str,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> DetectedIntent | None:
        """
        Detect intent from message using semantic matching.
        
        Returns None if no confident match found.
        
        Args:
            message: User message
            threshold: Minimum similarity to return a result
            
        Returns:
            DetectedIntent if match found, None otherwise
        """
        matches = self.find_matches(message, top_k=3, threshold=threshold)
        
        if not matches:
            return None
        
        # Aggregate signals from top matches
        best = matches[0]
        
        # Start with values from best match
        orchestration_mode = best.exemplar.orchestration_mode
        model_role = best.exemplar.model_role
        task_type = best.exemplar.task_type
        
        # Check for consensus among top matches
        for match in matches[1:]:
            if match.similarity >= threshold:
                # If multiple matches agree, boost confidence
                if match.exemplar.orchestration_mode == orchestration_mode:
                    pass  # Consensus
                elif match.exemplar.orchestration_mode is not None and orchestration_mode is None:
                    orchestration_mode = match.exemplar.orchestration_mode
                    
                if match.exemplar.model_role == model_role:
                    pass  # Consensus
                elif match.exemplar.model_role is not None and model_role is None:
                    model_role = match.exemplar.model_role
                    
                if match.exemplar.task_type == task_type:
                    pass  # Consensus
                elif match.exemplar.task_type is not None and task_type is None:
                    task_type = match.exemplar.task_type
        
        return DetectedIntent(
            task_type=task_type or "quick",
            orchestration_mode=orchestration_mode or OrchestrationMode.NONE,
            model_role=model_role or ModelRole.ASSISTANT,
            confidence=best.confidence,
            reasoning=f"semantic match: '{best.exemplar.text}' ({best.similarity:.2f})",
            trigger_keywords=[best.exemplar.text],
        )
    
    @property
    def is_initialized(self) -> bool:
        """Check if the matcher has been initialized."""
        return self._initialized
    
    @property
    def exemplar_count(self) -> int:
        """Number of exemplars loaded."""
        return len(self._exemplars)


# =============================================================================
# SINGLETON + LAZY LOADING
# =============================================================================

_semantic_matcher: SemanticIntentMatcher | None = None


def get_semantic_matcher() -> SemanticIntentMatcher:
    """
    Get the global semantic matcher instance.
    
    Lazy-loads the model on first use.
    Thread-safe via Python's GIL.
    """
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = SemanticIntentMatcher()
    return _semantic_matcher


def reset_semantic_matcher() -> None:
    """Reset the global matcher (for testing)."""
    global _semantic_matcher
    _semantic_matcher = None


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def semantic_detect(message: str) -> DetectedIntent | None:
    """
    Convenience function for semantic intent detection.
    
    Returns None if no confident match found.
    """
    return get_semantic_matcher().detect_intent(message)


__all__ = [
    "SemanticIntentMatcher",
    "SemanticMatch",
    "get_semantic_matcher",
    "reset_semantic_matcher",
    "semantic_detect",
]

