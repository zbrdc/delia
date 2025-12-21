# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Embedding-based Dispatcher - Fast and Accurate.

Uses sentence embeddings to classify prompts by similarity
to example prompts. Much faster and more reliable than LLM-based.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache

import structlog

log = structlog.get_logger()

# Example prompts for each tool type
EXECUTOR_EXAMPLES = [
    "Write a python function",
    "Implement the feature",
    "Implement caching layer",
    "Implement authentication",
    "Fix the bug",
    "Debug the error",
    "Refactor the code",
    "Add unit tests",
    "Create a class",
    "Create a REST endpoint",
    "Create a new API endpoint",
    "Create a function for",
    "Generate code for",
    "Optimize the function",
    "Write a script",
    "Code a solution",
    "Build a component",
    "Build an API",
    "Patch the vulnerability",
    "Clean up the code",
    "Add logging to",
    "Handle errors in",
    "Set up the database connection",
    "Configure the middleware",
]

PLANNER_EXAMPLES = [
    "Design a system architecture",
    "Plan the migration",
    "Create a roadmap for the project",
    "Create a roadmap for API",
    "Create a project roadmap",
    "Evaluate tradeoffs between",
    "Compare options for",
    "Strategy for implementing",
    "Scale our system",
    "Design a solution",
    "Architecture for",
    "Long-term plan for",
    "How should we structure",
    "Disaster recovery plan",
    "CI/CD pipeline design",
    "What's the best approach for",
    "How to architect",
    "Technical design for",
]

STATUS_EXAMPLES = [
    "Show me the melon leaderboard",
    "Check system health",
    "What models are available",
    "How many tokens saved",
    "Display usage statistics",
    "Is the backend healthy",
    "List available models",
    "Show model rankings",
    "System status",
    "Backend connectivity",
    "Queue depth",
    "Cost savings",
]


class EmbeddingDispatcher:
    """
    Fast embedding-based dispatcher.

    Classifies prompts by computing cosine similarity to example
    embeddings for each tool type.
    """

    def __init__(self):
        self._model = None
        self._executor_embeddings = None
        self._planner_embeddings = None
        self._status_embeddings = None
        self._initialized = False

    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use a small, fast model
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                log.debug("embedding_dispatcher_model_loaded", model="all-MiniLM-L6-v2")
            except ImportError:
                log.warning("sentence_transformers_not_installed")
                return None
        return self._model

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        model = self._get_model()
        if model is None:
            return None
        return model.encode(texts, normalize_embeddings=True)

    def initialize(self):
        """Pre-compute example embeddings."""
        if self._initialized:
            return

        model = self._get_model()
        if model is None:
            return

        self._executor_embeddings = self._compute_embeddings(EXECUTOR_EXAMPLES)
        self._planner_embeddings = self._compute_embeddings(PLANNER_EXAMPLES)
        self._status_embeddings = self._compute_embeddings(STATUS_EXAMPLES)
        self._initialized = True
        log.info("embedding_dispatcher_initialized")

    def dispatch(self, message: str) -> str:
        """
        Classify a message to the appropriate tool.

        Returns: "executor", "planner", or "status"
        """
        if not self._initialized:
            self.initialize()

        if self._executor_embeddings is None:
            # Fallback to keyword matching
            return self._keyword_fallback(message)

        # Compute message embedding
        msg_embedding = self._compute_embeddings([message])[0]

        # Compute similarities
        executor_sim = np.max(msg_embedding @ self._executor_embeddings.T)
        planner_sim = np.max(msg_embedding @ self._planner_embeddings.T)
        status_sim = np.max(msg_embedding @ self._status_embeddings.T)

        # Return highest similarity
        scores = {
            "executor": executor_sim,
            "planner": planner_sim,
            "status": status_sim,
        }
        result = max(scores, key=scores.get)

        log.debug(
            "embedding_dispatch",
            result=result,
            executor=f"{executor_sim:.3f}",
            planner=f"{planner_sim:.3f}",
            status=f"{status_sim:.3f}",
        )

        return result

    def _keyword_fallback(self, message: str) -> str:
        """Simple keyword-based fallback."""
        msg_lower = message.lower()

        # Status keywords
        status_keywords = ["melon", "leaderboard", "health", "status", "models", "tokens", "stats", "queue"]
        if any(k in msg_lower for k in status_keywords):
            return "status"

        # Planner keywords
        planner_keywords = ["design", "architecture", "plan", "strategy", "scale", "migrate", "roadmap", "compare", "tradeoff"]
        if any(k in msg_lower for k in planner_keywords):
            return "planner"

        # Default to executor
        return "executor"


# Singleton instance
_dispatcher: EmbeddingDispatcher | None = None


def get_embedding_dispatcher() -> EmbeddingDispatcher:
    """Get the singleton dispatcher instance."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = EmbeddingDispatcher()
    return _dispatcher


def dispatch(message: str) -> str:
    """Convenience function for dispatching."""
    return get_embedding_dispatcher().dispatch(message)
