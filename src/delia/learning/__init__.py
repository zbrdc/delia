# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Delia Learning Framework

This module implements adaptive context optimization for the Delia Framework:

Components:
- SemanticDeduplicator: Embedding-based duplicate detection
- HybridRetriever: Relevance × Utility × Recency scoring
- Reflector: Task outcome analysis and insight extraction
- Curator: Playbook maintenance via delta updates

Based on context engineering research and Anthropic's principles.
"""

from delia.learning.deduplication import SemanticDeduplicator, SimilarityMatch, DeduplicationResult
from delia.learning.retrieval import HybridRetriever, ScoredBullet
from delia.learning.reflector import Reflector, ReflectionResult, ExtractedInsight, InsightType
from delia.learning.curator import Curator, CurationResult, CurationDelta, CurationAction

__all__ = [
    # Deduplication
    "SemanticDeduplicator",
    "SimilarityMatch",
    "DeduplicationResult",
    # Retrieval
    "HybridRetriever",
    "ScoredBullet",
    # Reflector
    "Reflector",
    "ReflectionResult",
    "ExtractedInsight",
    "InsightType",
    # Curator
    "Curator",
    "CurationResult",
    "CurationDelta",
    "CurationAction",
]
