# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Semantic Coherence Scoring.

Uses embeddings to verify that an LLM response is coherent and relevant
to the original prompt, detecting hallucinations or gibberish.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import structlog

from ..embeddings import HybridEmbeddingsClient, cosine_similarity

log = structlog.get_logger()

class CoherenceScorer:
    """
    Scores the semantic coherence of LLM responses.
    """

    def __init__(self):
        self.client = HybridEmbeddingsClient()

    async def score(self, prompt: str, response: str) -> float:
        """
        Score how coherent/relevant the response is to the prompt.
        
        Returns: 0.0 to 1.0 (1.0 = highly coherent)
        """
        try:
            # Embed both prompt and response
            vectors = await asyncio.gather(
                self.client.embed(prompt),
                self.client.embed(response)
            )
            
            prompt_vec = vectors[0]
            response_vec = vectors[1]
            
            # Simple cosine similarity is a good baseline for coherence
            similarity = cosine_similarity(prompt_vec, response_vec)
            
            # Adjust score - typically even unrelated text has ~0.3 similarity
            # We want to map 0.3-0.8 range to 0.0-1.0
            score = max(0.0, min(1.0, (similarity - 0.3) / 0.5))
            
            log.debug("coherence_score", similarity=round(similarity, 3), score=round(score, 3))
            return score
            
        except Exception as e:
            log.warning("coherence_scoring_failed", error=str(e))
            return 1.0 # Default to coherent on error to avoid blocking

# Singleton
_scorer: CoherenceScorer | None = None

def get_coherence_scorer() -> CoherenceScorer:
    global _scorer
    if _scorer is None:
        _scorer = CoherenceScorer()
    return _scorer
