# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
ACE Framework: Agentic Context Engineering

This module implements the ACE framework for adaptive context optimization:

Components:
- SemanticDeduplicator: Embedding-based duplicate detection
- HybridRetriever: Relevance × Utility × Recency scoring
- Reflector: Task outcome analysis and insight extraction
- Curator: Playbook maintenance via delta updates

Based on Stanford ACE research and Anthropic's context engineering principles.
"""

from delia.ace.deduplication import SemanticDeduplicator, SimilarityMatch, DeduplicationResult
from delia.ace.retrieval import HybridRetriever, ScoredBullet
from delia.ace.reflector import Reflector, ReflectionResult, ExtractedInsight, InsightType
from delia.ace.curator import Curator, CurationResult, CurationDelta, CurationAction

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
