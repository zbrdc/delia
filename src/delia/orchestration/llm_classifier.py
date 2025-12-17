# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
LLM-Based Intent Classification (Layer 3).

When regex (Layer 1) and semantic matching (Layer 2) are uncertain,
we use Delia's own quick model to classify intent. This is the most
expensive layer (~500ms-1s) but handles complex/ambiguous cases.

Design:
- Use the quick model (already loaded) for classification
- Cache results by message hash (LRU, 1000 entries)
- Return structured JSON for parsing
- Fallback to defaults on errors
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from functools import lru_cache
from typing import Any

import structlog

from ..prompts import ModelRole, OrchestrationMode
from .result import DetectedIntent

log = structlog.get_logger()

# Classification prompt template
CLASSIFICATION_PROMPT = """Classify the user's intent for routing to the appropriate AI model.

User Message: "{message}"

Output ONLY valid JSON (no markdown, no explanation):
{{
  "task_type": "quick|coder|moe|thinking|status",
  "orchestration_mode": "none|voting|comparison|deep_thinking|agentic",
  "model_role": "assistant|code_reviewer|code_generator|architect|explainer|debugger|analyst|summarizer",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence explanation"
}}

Guidelines:
- task_type: "quick" for simple questions, "coder" for programming, "moe" for complex analysis, "thinking" for deep reasoning, "status" for system queries
- orchestration_mode: "voting" if user wants verification/certainty, "comparison" for multiple viewpoints, "deep_thinking" for thorough analysis, "agentic" for file/shell operations,code, coding quesitons, "none" otherwise
- model_role: Choose based on what the user needs (reviewing code? generating code? explaining? debugging?)
- confidence: Your certainty about this classification (0.5 = uncertain, 0.9 = very confident)

JSON only, no other text:"""


def _hash_message(message: str) -> str:
    """Create a hash for cache lookup."""
    return hashlib.sha256(message.encode()).hexdigest()[:16]


class LLMIntentClassifier:
    """
    LLM-based intent classification with caching.
    
    Uses Delia's quick model to classify ambiguous messages.
    Results are cached to avoid repeated LLM calls for the same message.
    
    Usage:
        classifier = get_llm_classifier()
        intent = await classifier.classify("complex ambiguous message")
    """
    
    def __init__(self, cache_size: int = 1000):
        self._cache_size = cache_size
        self._cache: dict[str, DetectedIntent] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
    async def classify(
        self,
        message: str,
        backend_type: str | None = None,
    ) -> DetectedIntent | None:
        """
        Classify intent using LLM.
        
        Args:
            message: User message to classify
            backend_type: Optional backend preference
            
        Returns:
            DetectedIntent if classification succeeds, None on error
        """
        # Check cache first
        cache_key = _hash_message(message)
        if cache_key in self._cache:
            self._cache_hits += 1
            log.debug("llm_classifier_cache_hit", key=cache_key[:8])
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        start = time.time()
        
        try:
            # Import here to avoid circular imports
            from ..llm import call_llm
            from ..routing import select_model
            from ..mcp_server import _select_optimal_backend_v2
            
            # Select quick model for classification
            _, backend_obj = await _select_optimal_backend_v2(
                message, None, "quick", backend_type
            )
            
            if not backend_obj:
                log.warning("llm_classifier_no_backend")
                return None
            
            selected_model = await select_model(
                task_type="quick",
                content_size=len(message),
            )
            
            # Format prompt
            prompt = CLASSIFICATION_PROMPT.format(message=message[:500])  # Truncate long messages
            
            # Call LLM
            response = await call_llm(
                prompt=prompt,
                model=selected_model,
                backend=backend_obj,
                system="You are a JSON-only classifier. Output valid JSON only.",
                max_tokens=200,
                temperature=0.1,  # Low temperature for consistent classification
            )
            
            elapsed = (time.time() - start) * 1000
            
            # Parse response
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            intent = self._parse_response(response_text, message)
            
            if intent:
                # Cache the result
                self._add_to_cache(cache_key, intent)
                
                log.debug(
                    "llm_classifier_success",
                    task_type=intent.task_type,
                    mode=intent.orchestration_mode.value,
                    role=intent.model_role.value,
                    confidence=f"{intent.confidence:.2f}",
                    elapsed_ms=int(elapsed),
                )
            
            return intent
            
        except Exception as e:
            log.warning("llm_classifier_error", error=str(e))
            return None
    
    def _parse_response(self, response_text: str, original_message: str) -> DetectedIntent | None:
        """Parse LLM response into DetectedIntent."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if not json_match:
                log.warning("llm_classifier_no_json", response=response_text[:100])
                return None
            
            data = json.loads(json_match.group())
            
            # Parse orchestration mode
            mode_str = data.get("orchestration_mode", "none").lower()
            try:
                orchestration_mode = OrchestrationMode(mode_str)
            except ValueError:
                orchestration_mode = OrchestrationMode.NONE
            
            # Parse model role
            role_str = data.get("model_role", "assistant").lower()
            try:
                model_role = ModelRole(role_str)
            except ValueError:
                model_role = ModelRole.ASSISTANT
            
            # Parse task type
            task_type = data.get("task_type", "quick").lower()
            if task_type not in ("quick", "coder", "moe", "thinking", "status"):
                task_type = "quick"
            
            # Parse confidence
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            reasoning = data.get("reasoning", "LLM classification")
            
            return DetectedIntent(
                task_type=task_type,
                orchestration_mode=orchestration_mode,
                model_role=model_role,
                confidence=confidence,
                reasoning=f"LLM: {reasoning}",
            )
            
        except json.JSONDecodeError as e:
            log.warning("llm_classifier_json_error", error=str(e))
            return None
        except Exception as e:
            log.warning("llm_classifier_parse_error", error=str(e))
            return None
    
    def _add_to_cache(self, key: str, intent: DetectedIntent) -> None:
        """Add to cache with LRU eviction."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (first key in dict)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = intent
    
    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
        }
    
    def clear_cache(self) -> None:
        """Clear the classification cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# =============================================================================
# SINGLETON
# =============================================================================

_llm_classifier: LLMIntentClassifier | None = None


def get_llm_classifier() -> LLMIntentClassifier:
    """Get the global LLM classifier instance."""
    global _llm_classifier
    if _llm_classifier is None:
        _llm_classifier = LLMIntentClassifier()
    return _llm_classifier


def reset_llm_classifier() -> None:
    """Reset the global classifier (for testing)."""
    global _llm_classifier
    _llm_classifier = None


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def llm_classify(message: str) -> DetectedIntent | None:
    """
    Convenience function for LLM-based classification.
    
    Returns None if classification fails.
    """
    return await get_llm_classifier().classify(message)


__all__ = [
    "LLMIntentClassifier",
    "get_llm_classifier",
    "reset_llm_classifier",
    "llm_classify",
]

