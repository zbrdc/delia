# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Delia Semantic Intelligence Layer.

Unified embedding-powered infrastructure for:
- Response caching (semantic similarity retrieval)
- Playbook vector search (strategy retrieval)
- Quality coherence scoring (semantic validation)
- Conversation compression (clustering + summarization)

All components share the HybridEmbeddingsClient from embeddings.py.
"""

from .cache import SemanticCache, get_semantic_cache
from .playbook_search import PlaybookVectorSearch, get_playbook_search
from .coherence import CoherenceScorer, get_coherence_scorer
from .compression import ConversationCompressor, get_conversation_compressor

__all__ = [
    "SemanticCache",
    "get_semantic_cache",
    "PlaybookVectorSearch",
    "get_playbook_search",
    "CoherenceScorer",
    "get_coherence_scorer",
    "ConversationCompressor",
    "get_conversation_compressor",
]
