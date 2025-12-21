# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Embedding-Based Dispatcher.

Uses sentence embeddings for fast, accurate task classification.
Routes requests to the appropriate tier (Planner, Executor, or Status).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from .result import DetectedIntent, OrchestrationMode

if TYPE_CHECKING:
    from ..backend_manager import BackendConfig

log = structlog.get_logger()

# Lazy-loaded embedding dispatcher
_embedding_dispatcher = None


def get_embedding_dispatcher():
    """Get or create the embedding dispatcher singleton."""
    global _embedding_dispatcher
    if _embedding_dispatcher is None:
        try:
            from .dispatcher_embed import EmbeddingDispatcher
            _embedding_dispatcher = EmbeddingDispatcher()
            _embedding_dispatcher.initialize()
            log.info("embedding_dispatcher_initialized")
        except ImportError as e:
            log.warning("embedding_dispatcher_unavailable", error=str(e))
            _embedding_dispatcher = False  # Mark as unavailable
    return _embedding_dispatcher if _embedding_dispatcher else None


class ModelDispatcher:
    """
    Routes tasks to the appropriate model tier using embeddings.

    Classifies prompts as: executor, planner, or status.
    """

    def __init__(self, call_llm_fn: Any = None):
        # call_llm_fn kept for API compatibility but not used
        pass

    async def dispatch(
        self,
        message: str,
        intent: DetectedIntent,
        backend_obj: BackendConfig | None = None,
        use_embeddings: bool = True,  # Kept for API compatibility
    ) -> str:
        """
        Classify a message to the appropriate tier.

        Args:
            message: The user message to classify
            intent: Pre-detected intent information (used as fallback)
            backend_obj: Unused, kept for API compatibility
            use_embeddings: Unused, always uses embeddings

        Returns:
            "executor", "planner", or "status"
        """
        # Primary: Use embedding-based dispatcher
        embed_dispatcher = get_embedding_dispatcher()
        if embed_dispatcher:
            try:
                result = embed_dispatcher.dispatch(message)
                log.debug(
                    "dispatch_result",
                    result=result,
                    message_preview=message[:50],
                )
                return result
            except Exception as e:
                log.warning("embedding_dispatch_failed", error=str(e))

        # Fallback: Use keyword matching from dispatcher_embed
        log.debug("dispatch_using_keyword_fallback")
        return self._keyword_fallback(message, intent)

    def _keyword_fallback(self, message: str, intent: DetectedIntent) -> str:
        """Simple keyword-based fallback when embeddings unavailable."""
        from ..config import config

        msg_lower = message.lower()

        # Status keywords
        status_keywords = ["melon", "leaderboard", "health", "status", "models", "tokens", "stats", "queue"]
        if any(k in msg_lower for k in status_keywords):
            return "status"

        # Planner keywords
        planner_keywords = ["design", "architecture", "plan", "strategy", "scale", "migrate", "roadmap", "compare", "tradeoff"]
        if any(k in msg_lower for k in planner_keywords):
            return "planner"

        # Intent-based fallback
        if intent.task_type in config.moe_tasks or intent.orchestration_mode != OrchestrationMode.NONE:
            return "planner"

        # Default to executor
        return "executor"
