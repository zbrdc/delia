# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Content detection and routing utilities for Delia."""

import os
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from .config import (
    get_affinity_tracker,
    get_backend_health,
    get_backend_metrics,
    get_provider_cost,
)

if TYPE_CHECKING:
    from .backend_manager import BackendConfig, BackendManager
    from .config import Config

log = structlog.get_logger()

# Environment variable to enable semantic routing
SEMANTIC_ROUTING_ENABLED = os.environ.get("DELIA_SEMANTIC_ROUTING", "true").lower() in ("true", "1", "yes")


# Code indicators with weights for confidence scoring
# Pre-compiled regex patterns for performance (avoids recompilation on each call)
CODE_INDICATORS = {
    # Strong indicators (weight 3) - almost certainly code
    "strong": [
        re.compile(r"\bdef\s+\w+\s*\(", re.MULTILINE),  # Python function
        re.compile(r"\bclass\s+\w+[\s:(]", re.MULTILINE),  # Class definition
        re.compile(r"\bimport\s+\w+", re.MULTILINE),  # Import statement
        re.compile(r"\bfrom\s+\w+\s+import", re.MULTILINE),  # From import
        re.compile(r"\bfunction\s+\w+\s*\(", re.MULTILINE),  # JS function
        re.compile(r"\bconst\s+\w+\s*=", re.MULTILINE),  # JS const
        re.compile(r"\blet\s+\w+\s*=", re.MULTILINE),  # JS let
        re.compile(r"\bexport\s+(default\s+)?", re.MULTILINE),  # JS export
        re.compile(r"^\s*@\w+", re.MULTILINE),  # Decorator
        re.compile(r"\basync\s+(def|function)", re.MULTILINE),  # Async
        re.compile(r"\bawait\s+\w+", re.MULTILINE),  # Await
        re.compile(r"\breturn\s+[\w{(\[]", re.MULTILINE),  # Return statement
        re.compile(r"if\s*\(.+\)\s*{", re.MULTILINE),  # C-style if
        re.compile(r"for\s*\(.+\)\s*{", re.MULTILINE),  # C-style for
        re.compile(r"\bwhile\s*\(.+\)", re.MULTILINE),  # While loop
        re.compile(r"\btry\s*[:{]", re.MULTILINE),  # Try block
        re.compile(r"\bcatch\s*\(", re.MULTILINE),  # Catch block
        re.compile(r"\bexcept\s+\w*:", re.MULTILINE),  # Python except
        re.compile(r"=>\s*{", re.MULTILINE),  # Arrow function
        re.compile(r"\.map\(|\.filter\(|\.reduce\(", re.MULTILINE),  # Array methods
    ],
    # Medium indicators (weight 2) - likely code
    "medium": [
        re.compile(r"\bself\.", re.MULTILINE),  # Python self
        re.compile(r"\bthis\.", re.MULTILINE),  # JS this
        re.compile(r"===|!==", re.MULTILINE),  # Strict equality
        re.compile(r"&&|\|\|", re.MULTILINE),  # Logical operators
        re.compile(r"\bnull\b|\bundefined\b", re.MULTILINE),  # Null/undefined
        re.compile(r"\bTrue\b|\bFalse\b|\bNone\b", re.MULTILINE),  # Python booleans
        re.compile(r":\s*\w+\s*[,)\]]", re.MULTILINE),  # Type annotations
        re.compile(r"\[\w+\]", re.MULTILINE),  # Array indexing
        re.compile(r"\{\s*\w+:\s*", re.MULTILINE),  # Object literal
        re.compile(r"console\.|print\(|logger\.", re.MULTILINE),  # Logging
        re.compile(r"\braise\s+\w+", re.MULTILINE),  # Python raise
        re.compile(r"\bthrow\s+new", re.MULTILINE),  # JS throw
        re.compile(r"`[^`]+\$\{", re.MULTILINE),  # Template literal
        re.compile(r'f"[^"]*\{', re.MULTILINE),  # Python f-string
    ],
    # Weak indicators (weight 1) - could be code
    "weak": [
        re.compile(r";$", re.MULTILINE),  # Semicolon ending
        re.compile(r"\{|\}", re.MULTILINE),  # Braces
        re.compile(r"\[|\]", re.MULTILINE),  # Brackets
        re.compile(r"==|!=", re.MULTILINE),  # Equality
        re.compile(r"->", re.MULTILINE),  # Arrow (type hints, etc)
        re.compile(r"\bint\b|\bstr\b|\bbool\b", re.MULTILINE),  # Type names
        re.compile(r"\bvar\b", re.MULTILINE),  # Var keyword
    ],
}


def detect_code_content(content: str) -> tuple[bool, float, str]:
    """
    Detect if content is primarily code or text.

    Returns:
        (is_code, confidence, reasoning)
        - is_code: True if content appears to be code
        - confidence: 0.0-1.0 score
        - reasoning: Brief explanation
    """
    if not content or len(content.strip()) < 20:
        return False, 0.0, "Content too short"

    lines = content.strip().split("\n")

    # Count code indicators
    strong_matches = 0
    medium_matches = 0
    weak_matches = 0

    for pattern in CODE_INDICATORS["strong"]:
        matches = len(pattern.findall(content))  # Use pre-compiled pattern
        strong_matches += min(matches, 5)  # Cap per pattern

    for pattern in CODE_INDICATORS["medium"]:
        matches = len(pattern.findall(content))  # Use pre-compiled pattern
        medium_matches += min(matches, 5)

    for pattern in CODE_INDICATORS["weak"]:
        matches = len(pattern.findall(content))  # Use pre-compiled pattern
        weak_matches += min(matches, 5)

    # Weighted score
    score = strong_matches * 3 + medium_matches * 2 + weak_matches * 1

    # Normalize by content length (per 1000 chars)
    normalized = score / max(1, len(content) / 1000)

    # Additional heuristics
    avg_line_length = sum(len(line) for line in lines) / max(1, len(lines))
    indent_lines = sum(1 for line in lines if line.startswith("  ") or line.startswith("\t"))
    indent_ratio = indent_lines / max(1, len(lines))

    # Adjust score based on structure
    if indent_ratio > 0.3:  # Lots of indentation = code
        normalized *= 1.3
    if avg_line_length < 100:  # Code lines tend to be shorter
        normalized *= 1.1

    # Determine threshold
    if normalized > 3.0:
        return True, min(1.0, normalized / 5), f"Strong code signals (score={normalized:.1f})"
    elif normalized > 1.5:
        return True, normalized / 4, f"Likely code (score={normalized:.1f})"
    elif normalized > 0.8:
        return False, 0.4, f"Mixed content (score={normalized:.1f})"
    else:
        return False, max(0, 0.3 - normalized / 3), f"Primarily text (score={normalized:.1f})"



def detect_chat_task_type(message: str) -> tuple[str, float, str]:
    """
    Detect the appropriate task type for a chat message.

    Returns:
        (task_type, confidence, reasoning)
        - task_type: "quick", "coder", or "moe"
        - confidence: 0.0-1.0 score
        - reasoning: Brief explanation
    """
    if not message or len(message.strip()) < 5:
        return "quick", 0.5, "Very short message"

    message_lower = message.lower()

    # MoE indicators (complex reasoning tasks)
    moe_patterns = [
        (r"\b(plan|design|architect|strategy)\b", "planning task"),
        (r"\b(compare|contrast|analyze pros|trade.?offs?)\b", "comparison/analysis"),
        (r"\b(explain|why|how does|what is the reason)\b.*\b(work|happen|cause)\b", "deep explanation"),
        (r"\b(critique|evaluate|assess|review)\b.*\b(approach|design|architecture)\b", "evaluation task"),
        (r"\b(step.?by.?step|detailed|comprehensive|thorough)\b", "detailed analysis requested"),
    ]

    for pattern, reason in moe_patterns:
        if re.search(pattern, message_lower):
            return "moe", 0.7, f"Complex task: {reason}"

    # Coder indicators (code-related tasks)
    coder_patterns = [
        (r"\b(code|coding|coder|function|class|method|script|program|programming)\b", "code keyword"),
        (r"\b(write|create|implement|build|develop)\b.*\b(function|class|code|app)\b", "code generation"),
        (r"\b(fix|debug|error|bug|issue)\b.*\b(code|function|script)\b", "debugging"),
        (r"\b(refactor|optimize|improve)\b.*\b(code|function|performance)\b", "code improvement"),
        (r"```|\bdef\s|\bclass\s|\bfunction\s|\bconst\s|\blet\s", "code syntax"),
        (r"\b(python|javascript|typescript|java|rust|go|c\+\+|react|node)\b", "language mention"),
        (r"\b(api|endpoint|database|sql|query|json|xml|http|rest|graphql)\b", "technical term"),
        (r"\b(import|export|module|package|library|dependency|npm|pip)\b", "module keyword"),
        (r"\b(coding model|coder model|code model)\b", "explicit model request"),
    ]

    for pattern, reason in coder_patterns:
        if re.search(pattern, message_lower):
            # Double-check with code content detection
            is_code, code_conf, _ = detect_code_content(message)
            if is_code or code_conf > 0.3:
                return "coder", max(0.6, code_conf), f"Code-related: {reason}"
            return "coder", 0.5, f"Code-related: {reason}"

    # Check for actual code in the message
    is_code, confidence, code_reason = detect_code_content(message)
    if is_code and confidence > 0.5:
        return "coder", confidence, f"Contains code: {code_reason}"

    # Quick task indicators (simple questions, casual chat)
    quick_patterns = [
        (r"^(hi|hello|hey|thanks|thank you|ok|okay)\b", "greeting/acknowledgment"),
        (r"\b(what is|who is|when|where|how many|how much)\b", "simple question"),
        (r"\b(tell me|give me|show me)\b.*\b(about|example)\b", "information request"),
        (r"\b(list|summarize|brief|short)\b", "summary request"),
        (r"^.{0,50}\?$", "short question"),  # Short questions
    ]

    for pattern, reason in quick_patterns:
        if re.search(pattern, message_lower):
            return "quick", 0.7, f"Simple task: {reason}"

    # Default to quick for general chat
    return "quick", 0.5, "General chat message"


def parse_model_override(model_hint: str | None, content: str) -> str | None:
    """Parse explicit model request from content or hint.

    Recognizes tier keywords (moe, coder, quick, thinking) and natural language model references.
    Supports size-based hints (7b, 14b, 30b), capability hints (coder, reasoning), and descriptive terms.
    """
    # Check explicit hint first (tier names and natural language)
    if model_hint:
        hint = model_hint.lower().strip()

        # Direct tier names
        if "moe" in hint or "30b" in hint or "large" in hint or "complex" in hint or "reasoning" in hint:
            return "moe"
        if "coder" in hint or "code" in hint or "programming" in hint or "14b" in hint:
            return "coder"
        if "quick" in hint or "7b" in hint or "small" in hint or "fast" in hint:
            return "quick"
        if "thinking" in hint or "think" in hint or "chain" in hint:
            return "thinking"

        # Pass through other hints as-is (might be specific model name)
        return model_hint

    # Scan content for tier keywords and natural language using word boundaries
    if content:
        content_lower = content.lower()

        # Size and capability-based patterns
        if re.search(r"\b(30b|large|big|complex|reasoning|deep|planning|critique)\b", content_lower) or re.search(
            r"\bmoe\b", content_lower
        ):
            log.info("model_override_detected", tier="moe", source="content", pattern="size/capability")
            return "moe"

        if re.search(r"\b(14b|coder|code|programming|development|review|analyze)\b", content_lower):
            log.info("model_override_detected", tier="coder", source="content", pattern="size/capability")
            return "coder"

        if re.search(r"\b(7b|small|fast|quick|simple|basic|summarize)\b", content_lower):
            log.info("model_override_detected", tier="quick", source="content", pattern="size/capability")
            return "quick"

        if re.search(r"\b(thinking|think|chain|reason|step.*by.*step)\b", content_lower):
            log.info("model_override_detected", tier="thinking", source="content", pattern="capability")
            return "thinking"

    return None


# =============================================================================
# Backend Scoring for Latency-Aware Routing
# =============================================================================


@dataclass
class ScoringWeights:
    """Configurable weights for backend scoring.

    All weights should sum to approximately 1.0 for normalized scoring.
    Cost weight defaults to 0.0 (disabled) - enable it by reducing other weights.
    """

    latency: float = 0.35  # Lower latency = better
    throughput: float = 0.15  # Higher tokens/sec = better
    reliability: float = 0.35  # Higher success rate = better
    availability: float = 0.15  # Circuit breaker state
    cost: float = 0.0  # Lower cost = better (disabled by default)

    def __post_init__(self) -> None:
        """Validate weights are non-negative."""
        for attr in ("latency", "throughput", "reliability", "availability", "cost"):
            if getattr(self, attr) < 0:
                raise ValueError(f"Weight {attr} must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "ScoringWeights":
        """Create ScoringWeights from a dictionary.

        Only uses recognized keys; ignores unknown keys.
        """
        valid_keys = {"latency", "throughput", "reliability", "availability", "cost"}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class BackendScorer:
    """Score backends for optimal routing based on performance metrics.

    Higher scores indicate better backend choices. Scoring considers:
    - Latency: Lower P50 latency = better
    - Throughput: Higher tokens per second = better
    - Reliability: Higher success rate = better
    - Availability: Circuit breaker state (open = unavailable)
    - Cost: Lower cost per token = better (optional, disabled by default)
    """

    # Reference values for normalization
    _REFERENCE_LATENCY_MS: float = 1000.0  # 1s = 0.5 score
    _REFERENCE_THROUGHPUT_TPS: float = 100.0  # 100 tok/s = 1.0 score cap
    _REFERENCE_COST_PER_1K: float = 0.01  # $0.01/1K = 0.5 score

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        """Initialize the scorer with optional custom weights.

        Args:
            weights: Custom scoring weights. Uses defaults if not specified.
        """
        self.weights = weights or ScoringWeights()

    def score(
        self, backend: "BackendConfig", task_type: str | None = None
    ) -> float:
        """Calculate a 0.0-1.0 score for a backend.

        Higher score = better backend choice. New backends with no metrics
        start with a neutral/optimistic score.

        Args:
            backend: Backend configuration to score
            task_type: Optional task type for affinity-based scoring boost

        Returns:
            Score from 0.0 to 1.0 (may exceed 1.0 with affinity boost)
        """
        metrics = get_backend_metrics(backend.id)
        health = get_backend_health(backend.id)

        # Calculate component scores (all 0.0 to 1.0)
        latency_score = self._score_latency(metrics.latency_p50)
        throughput_score = self._score_throughput(metrics.tokens_per_second)
        reliability_score = metrics.success_rate
        availability_score = 1.0 if health.is_available() else 0.0
        cost_score = self._score_cost(backend.provider)

        # Weighted sum
        total = (
            self.weights.latency * latency_score
            + self.weights.throughput * throughput_score
            + self.weights.reliability * reliability_score
            + self.weights.availability * availability_score
            + self.weights.cost * cost_score
        )

        # Apply task-specific affinity boost if task_type provided
        affinity = None
        if task_type:
            tracker = get_affinity_tracker()
            affinity = tracker.get_affinity(backend.id, task_type)
            total = tracker.boost_score(total, backend.id, task_type)

        log.debug(
            "backend_scored",
            backend=backend.id,
            score=round(total, 3),
            latency=round(latency_score, 3),
            throughput=round(throughput_score, 3),
            reliability=round(reliability_score, 3),
            availability=availability_score,
            cost=round(cost_score, 3),
            affinity=round(affinity, 3) if affinity is not None else None,
            task_type=task_type,
        )

        return total

    def _score_latency(self, latency_ms: float) -> float:
        """Convert latency to 0-1 score (lower latency = higher score).

        Uses inverse relationship: score = 1 / (1 + latency/reference)
        - 0ms = 1.0 (optimal)
        - 500ms = 0.67
        - 1000ms = 0.5
        - 2000ms = 0.33

        Args:
            latency_ms: P50 latency in milliseconds (0 = no data)

        Returns:
            Score from 0.0 to 1.0
        """
        if latency_ms <= 0:
            return 1.0  # No data = optimistic
        return 1.0 / (1.0 + latency_ms / self._REFERENCE_LATENCY_MS)

    def _score_throughput(self, tps: float) -> float:
        """Convert tokens per second to 0-1 score (higher = better).

        Linear scaling capped at reference throughput.
        - 0 tok/s = 0.5 (neutral for no data)
        - 50 tok/s = 0.5
        - 100+ tok/s = 1.0

        Args:
            tps: Tokens per second (0 = no data)

        Returns:
            Score from 0.0 to 1.0
        """
        if tps <= 0:
            return 0.5  # No data = neutral
        return min(tps / self._REFERENCE_THROUGHPUT_TPS, 1.0)

    def _score_cost(self, provider: str) -> float:
        """Convert provider cost to 0-1 score (lower cost = higher score).

        Uses inverse relationship similar to latency scoring.
        - $0.00/1K (local) = 1.0 (optimal)
        - $0.005/1K = 0.67
        - $0.01/1K = 0.5
        - $0.02/1K = 0.33

        Args:
            provider: Provider name (e.g., 'ollama', 'gemini', 'openai')

        Returns:
            Score from 0.0 to 1.0
        """
        cost_per_1k = get_provider_cost(provider)
        if cost_per_1k <= 0:
            return 1.0  # Free = optimal
        return 1.0 / (1.0 + cost_per_1k / self._REFERENCE_COST_PER_1K)

    def select_best(
        self,
        backends: list["BackendConfig"],
        backend_type: str | None = None,
        task_type: str | None = None,
    ) -> "BackendConfig | None":
        """Select the best backend from a list based on scoring.

        Filters by enabled status and optionally by backend type,
        then scores remaining candidates and returns the best.

        Args:
            backends: List of backends to choose from
            backend_type: Optional filter ("local" or "remote")
            task_type: Optional task type for affinity-based scoring

        Returns:
            Best backend or None if none available
        """
        # Filter to enabled backends matching type
        candidates = [
            b
            for b in backends
            if b.enabled and (backend_type is None or b.type == backend_type)
        ]

        if not candidates:
            log.debug("no_candidates", backend_type=backend_type)
            return None

        # Further filter by circuit breaker availability
        available = [b for b in candidates if get_backend_health(b.id).is_available()]

        if not available:
            # All circuit breakers open - return highest priority as fallback
            log.warning(
                "all_backends_unavailable",
                backend_type=backend_type,
                falling_back_to_priority=True,
            )
            return max(candidates, key=lambda b: b.priority)

        # Score and select best
        best = max(available, key=lambda b: self.score(b, task_type))
        log.info(
            "backend_selected",
            backend=best.id,
            score=round(self.score(best, task_type), 3),
            candidates=len(available),
            task_type=task_type,
        )
        return best

    def select_weighted(
        self,
        backends: list["BackendConfig"],
        backend_type: str | None = None,
        task_type: str | None = None,
    ) -> "BackendConfig | None":
        """Select a backend using weighted random selection for load distribution.

        Uses scores as weights for random selection - higher scoring backends
        are more likely to be chosen, but lower scoring backends still get
        some traffic for load balancing.

        Args:
            backends: List of backends to choose from
            backend_type: Optional filter ("local" or "remote")
            task_type: Optional task type for affinity-based scoring

        Returns:
            Selected backend or None if none available
        """
        # Filter to enabled backends matching type
        candidates = [
            b
            for b in backends
            if b.enabled and (backend_type is None or b.type == backend_type)
        ]

        if not candidates:
            log.debug("no_candidates", backend_type=backend_type)
            return None

        # Further filter by circuit breaker availability
        available = [b for b in candidates if get_backend_health(b.id).is_available()]

        if not available:
            # All circuit breakers open - return highest priority as fallback
            log.warning(
                "all_backends_unavailable_weighted",
                backend_type=backend_type,
                falling_back_to_priority=True,
            )
            return max(candidates, key=lambda b: b.priority)

        # Single backend - no need for weighted selection
        if len(available) == 1:
            return available[0]

        # Calculate scores as weights (ensure non-zero minimum)
        scores = [max(self.score(b, task_type), 0.01) for b in available]

        # Weighted random selection
        selected = random.choices(available, weights=scores, k=1)[0]
        log.info(
            "backend_selected_weighted",
            backend=selected.id,
            score=round(self.score(selected, task_type), 3),
            candidates=len(available),
            weights=[round(s, 3) for s in scores],
            task_type=task_type,
        )
        return selected

    def select_top_n(
        self,
        backends: list["BackendConfig"],
        n: int = 2,
        backend_type: str | None = None,
        task_type: str | None = None,
    ) -> list["BackendConfig"]:
        """Select top N backends by score for hedged requests.

        Returns multiple backends sorted by score (best first).
        Used for speculative execution where requests are sent
        to multiple backends with staggered starts.

        Args:
            backends: List of backends to choose from
            n: Maximum number of backends to return (default 2)
            backend_type: Optional filter ("local" or "remote")
            task_type: Optional task type for affinity-based scoring

        Returns:
            List of up to N backends, sorted by score (best first)
        """
        # Filter to enabled backends matching type
        candidates = [
            b
            for b in backends
            if b.enabled and (backend_type is None or b.type == backend_type)
        ]

        if not candidates:
            log.debug("no_candidates_for_hedging", backend_type=backend_type)
            return []

        # Filter by circuit breaker availability
        available = [b for b in candidates if get_backend_health(b.id).is_available()]

        if not available:
            log.debug("no_available_backends_for_hedging", backend_type=backend_type)
            return []

        # Sort by score descending and take top N
        scored = [(b, self.score(b, task_type)) for b in available]
        scored.sort(key=lambda x: x[1], reverse=True)

        result = [b for b, _ in scored[:n]]
        log.debug(
            "hedging_candidates_selected",
            backends=[b.id for b in result],
            scores=[round(s, 3) for _, s in scored[:n]],
            task_type=task_type,
        )
        return result


class ModelRouter:
    """Intelligent model and backend selection for Delia."""

    def __init__(self, config: "Config", backend_manager: "BackendManager", use_semantic_routing: bool = True):
        """Initialize the model router with configuration and backend manager.

        Args:
            config: Configuration object with model tiers and task mappings
            backend_manager: Backend manager for retrieving active/enabled backends
            use_semantic_routing: Whether to use embeddings-based semantic routing
        """
        self.config = config
        self.backend_manager = backend_manager
        self.use_semantic_routing = use_semantic_routing and SEMANTIC_ROUTING_ENABLED
        self._semantic_router = None

    async def _get_semantic_router(self):
        """Lazy-load the semantic router."""
        if self._semantic_router is None and self.use_semantic_routing:
            try:
                from .embeddings import get_semantic_router
                self._semantic_router = get_semantic_router()
            except ImportError:
                log.warning("semantic_routing_import_failed")
                self.use_semantic_routing = False
        return self._semantic_router

    async def select_model(
        self, task_type: str, content_size: int = 0, model_override: str | None = None, content: str = ""
    ) -> str:
        """
        Select the best model for the task with intelligent routing.

        Tiers (configured in settings.json):
        - quick: Fast general tasks, text analysis, summarize
        - coder: Code generation, review, analysis
        - moe: Complex reasoning - plan, critique, large text

        Strategy:
        1. Honor explicit overrides
        2. MoE tasks (plan, critique) always use MoE model
        3. **SEMANTIC ROUTING** (if enabled): Use embeddings to classify content
        4. Fallback: Regex-based code detection for content classification
        5. Default to quick for everything else
        """
        # Get models from active backend (settings.json is the single source of truth)
        backend = self.backend_manager.get_active_backend()
        if not backend:
            log.error("no_backend_configured", hint="Run 'delia serve' to auto-detect backends")
            raise RuntimeError("No backend configured. Check ~/.cache/delia/settings.json or run 'delia serve' to auto-detect.")

        model_quick = backend.models.get("quick", "current")
        model_coder = backend.models.get("coder", "current")
        model_moe = backend.models.get("moe", "current")
        model_thinking = backend.models.get("thinking", "current")

        # Helper to resolve tier name to model name
        def resolve_tier(tier_name):
            if tier_name == "quick":
                return model_quick
            if tier_name == "coder":
                return model_coder
            if tier_name == "moe":
                return model_moe
            if tier_name == "thinking":
                return model_thinking
            return tier_name  # Assume it's a model name if not a tier

        # Priority 1: Explicit override
        if model_override:
            resolved = resolve_tier(model_override)
            log.info("model_selected", source="override", tier=model_override, model=resolved)
            return resolved

        # Priority 2: Tasks that REQUIRE MoE (complex multi-step reasoning)
        if task_type in self.config.moe_tasks:
            log.info("model_selected", source="moe_task", task=task_type, tier="moe")
            return model_moe

        # Priority 3: Semantic routing (embeddings-based classification)
        if content and self.use_semantic_routing:
            semantic_router = await self._get_semantic_router()
            if semantic_router:
                try:
                    tier, confidence, reasoning = await semantic_router.get_recommended_tier(content)
                    if confidence > 0.5:  # Only use semantic routing if confident
                        resolved = resolve_tier(tier)
                        log.info(
                            "model_selected",
                            source="semantic",
                            task=task_type,
                            tier=tier,
                            confidence=f"{confidence:.0%}",
                            reasoning=reasoning,
                            model=resolved,
                        )
                        return resolved
                    else:
                        log.debug(
                            "semantic_routing_low_confidence",
                            confidence=f"{confidence:.0%}",
                            reasoning=reasoning,
                        )
                except Exception as e:
                    log.warning("semantic_routing_failed", error=str(e))

        # Priority 4 (fallback): Regex-based code detection
        code_detection = None
        if content and (content_size > self.config.large_content_threshold or task_type in self.config.coder_tasks):
            code_detection = detect_code_content(content)

        # Large content handling
        if content_size > self.config.large_content_threshold and code_detection:
            is_code, confidence, reasoning = code_detection
            if is_code and confidence > 0.5:
                log.info(
                    "model_selected",
                    source="large_code_regex",
                    content_kb=content_size // 1000,
                    confidence=f"{confidence:.0%}",
                    tier="coder",
                    reasoning=reasoning,
                )
                return model_coder
            else:
                # Large text content benefits from MoE's reasoning
                log.info(
                    "model_selected",
                    source="large_text_regex",
                    content_kb=content_size // 1000,
                    confidence=f"{1 - confidence:.0%}",
                    tier="moe",
                    reasoning=reasoning,
                )
                return model_moe

        # Code-focused tasks with regex detection
        if task_type in self.config.coder_tasks and code_detection:
            is_code, confidence, reasoning = code_detection
            if is_code:
                log.info("model_selected", source="coder_task_regex", task=task_type, tier="coder", reasoning=reasoning)
                return model_coder
            else:
                log.info("model_selected", source="coder_task_text_regex", task=task_type, tier="quick", reasoning=reasoning)
                # Fall through to default

        # Priority 5: Default to quick (fastest model)
        log.info("model_selected", source="default", task=task_type, tier="quick")
        return model_quick

    async def select_optimal_backend(
        self,
        content: str,
        file_path: str | None = None,
        task_type: str = "quick",
        backend_type: str | None = None,
    ) -> tuple[str | None, Any | None]:
        """
        Select optimal backend using performance-based scoring.

        Uses BackendScorer to select the best backend based on:
        - Latency (lower is better)
        - Throughput (higher is better)
        - Reliability (success rate)
        - Availability (circuit breaker status)

        Args:
            content: The content to process (for future content-based routing)
            file_path: Optional file path for context
            task_type: Type of task being performed
            backend_type: Optional backend type constraint ("local" or "remote")

        Returns:
            Tuple of (None, backend_obj) where backend_obj is the selected backend
        """
        enabled_backends = self.backend_manager.get_enabled_backends()

        if not enabled_backends:
            return (None, None)

        # Use BackendScorer for intelligent selection
        scorer = BackendScorer()
        selected = scorer.select_best(enabled_backends, backend_type=backend_type)

        if selected:
            return (None, selected)

        # Fallback to active backend if no matching type found
        active_backend = self.backend_manager.get_active_backend()
        return (None, active_backend)


# Module-level singleton for convenience
_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    """Get or create the global ModelRouter instance."""
    global _router
    if _router is None:
        from .backend_manager import backend_manager
        from .config import config
        _router = ModelRouter(config, backend_manager)
    return _router


async def select_model(
    task_type: str, content_size: int = 0, model_override: str | None = None, content: str = ""
) -> str:
    """Module-level select_model that uses the global router.
    
    This is the canonical model selection function - use this instead of
    creating ModelRouter instances directly.
    """
    router = get_router()
    return await router.select_model(task_type, content_size, model_override, content)
