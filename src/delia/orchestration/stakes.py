# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
High-Stakes Detection for ToT Triggering.

A sophisticated multi-signal system for detecting when a task warrants
Tree of Thoughts meta-orchestration. Goes far beyond simple keyword matching.

Detection Layers:
1. Semantic Similarity - Match against high-stakes exemplars
2. Compound Patterns - Domain + Action combinations with weights
3. Complexity Signals - File count, scope, architectural indicators
4. Context Signals - Session history, previous failures, frustration
5. Negative Patterns - Educational/exploratory contexts that reduce stakes

All signals are aggregated using Bayesian weighting to produce a final
stakes score in [0, 1] where higher = more critical.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from .result import DetectedIntent

log = structlog.get_logger()


class RiskDomain(Enum):
    """Risk domains with base severity weights."""
    # Critical - Always high stakes
    SECURITY = ("security", 0.9)
    CRYPTOGRAPHY = ("cryptography", 0.95)
    AUTHENTICATION = ("authentication", 0.85)
    AUTHORIZATION = ("authorization", 0.85)
    PAYMENTS = ("payments", 0.9)
    FINANCIAL = ("financial", 0.85)
    MEDICAL = ("medical", 0.95)

    # High - Often high stakes
    PRODUCTION = ("production", 0.8)
    DATABASE = ("database", 0.7)
    INFRASTRUCTURE = ("infrastructure", 0.75)
    API_DESIGN = ("api_design", 0.6)

    # Medium - Context dependent
    PERFORMANCE = ("performance", 0.5)
    ARCHITECTURE = ("architecture", 0.6)
    MIGRATION = ("migration", 0.65)
    REFACTORING = ("refactoring", 0.4)

    # Low - Usually not high stakes
    TESTING = ("testing", 0.3)
    DOCUMENTATION = ("documentation", 0.1)
    STYLING = ("styling", 0.1)

    def __init__(self, domain_name: str, base_weight: float):
        self.domain_name = domain_name
        self.base_weight = base_weight


class ActionType(Enum):
    """Action types with risk multipliers."""
    # High risk actions
    IMPLEMENT = ("implement", 1.5)
    DEPLOY = ("deploy", 1.8)
    DELETE = ("delete", 1.6)
    MIGRATE = ("migrate", 1.4)
    MODIFY = ("modify", 1.3)

    # Medium risk actions
    REVIEW = ("review", 1.2)
    FIX = ("fix", 1.1)
    REFACTOR = ("refactor", 1.0)
    OPTIMIZE = ("optimize", 1.0)

    # Low risk actions (reduce stakes)
    EXPLAIN = ("explain", 0.3)
    UNDERSTAND = ("understand", 0.3)
    EXAMPLE = ("example", 0.2)
    TUTORIAL = ("tutorial", 0.2)
    QUESTION = ("question", 0.4)

    def __init__(self, action_name: str, multiplier: float):
        self.action_name = action_name
        self.multiplier = multiplier


# Semantic exemplars for high-stakes scenarios
# These get embedded and matched via cosine similarity
HIGH_STAKES_EXEMPLARS = [
    # Security Critical
    ("Implement JWT authentication with refresh token rotation for our production API", 0.95),
    ("Review this authentication code for security vulnerabilities before deployment", 0.9),
    ("Fix the SQL injection vulnerability in the user search endpoint", 0.95),
    ("Implement CSRF protection across all form submissions", 0.85),
    ("Add rate limiting to prevent brute force attacks on login", 0.85),
    ("Implement secure session management with proper cookie flags", 0.85),
    ("Review password hashing implementation for compliance with OWASP", 0.9),

    # Cryptography
    ("Implement AES-256 encryption for storing sensitive user data", 0.95),
    ("Review the cryptographic key derivation function implementation", 0.95),
    ("Generate and rotate API keys securely for external integrations", 0.85),
    ("Implement end-to-end encryption for user messages", 0.95),

    # Payments/Financial
    ("Implement Stripe payment processing for subscription billing", 0.9),
    ("Fix the race condition in order processing that causes duplicate charges", 0.95),
    ("Implement refund logic with proper audit trail", 0.85),
    ("Review PCI-DSS compliance for card data handling", 0.95),
    ("Implement idempotency for payment webhook handlers", 0.85),

    # Medical/HIPAA
    ("Implement patient data export with HIPAA-compliant encryption", 0.95),
    ("Review PHI access controls for compliance audit", 0.95),
    ("Implement audit logging for all medical record access", 0.9),

    # Production/Infrastructure
    ("Deploy the new authentication service to production", 0.85),
    ("Implement database failover with zero data loss", 0.9),
    ("Migrate production database schema with zero downtime", 0.9),
    ("Implement circuit breaker pattern for critical service dependencies", 0.8),
    ("Review the load balancer configuration for high availability", 0.8),

    # Data Integrity
    ("Implement database transaction isolation for concurrent order updates", 0.85),
    ("Fix the data corruption bug in the sync service", 0.95),
    ("Implement idempotent message processing for event handlers", 0.8),
    ("Review backup and recovery procedures before production deployment", 0.85),

    # Authorization
    ("Implement role-based access control for admin functions", 0.85),
    ("Review permission checks in the file upload endpoint", 0.85),
    ("Implement OAuth2 scopes for third-party API access", 0.8),
    ("Fix privilege escalation vulnerability in user profile update", 0.95),
]

# Low stakes exemplars (for contrast/calibration)
LOW_STAKES_EXEMPLARS = [
    ("What is JWT authentication?", 0.1),
    ("Explain how SQL injection works", 0.15),
    ("Show me an example of a login form", 0.1),
    ("How do I add a button to this page?", 0.1),
    ("What's the syntax for a Python dictionary?", 0.05),
    ("Write a hello world program", 0.05),
    ("Add a console.log for debugging", 0.1),
    ("Fix the typo in the README", 0.05),
    ("What does this function do?", 0.15),
    ("Explain this error message", 0.2),
]

# Compound patterns: (domain_keywords, action_keywords, weight_modifier)
COMPOUND_PATTERNS = [
    # Security + Implementation = Very High
    (
        {"auth", "authentication", "login", "password", "credential", "token", "session", "jwt", "oauth"},
        {"implement", "create", "build", "add", "write", "deploy", "fix", "modify", "change"},
        1.8
    ),
    # Security + Review = High
    (
        {"auth", "authentication", "security", "vulnerability", "injection", "xss", "csrf", "permission"},
        {"review", "audit", "check", "verify", "analyze", "inspect"},
        1.5
    ),
    # Crypto + Any Action = Very High
    (
        {"encrypt", "decrypt", "hash", "crypto", "aes", "rsa", "key", "salt", "hmac", "signature"},
        {"implement", "create", "fix", "modify", "review", "generate"},
        1.9
    ),
    # Payment + Implementation = Critical
    (
        {"payment", "charge", "billing", "subscription", "stripe", "paypal", "transaction", "refund", "invoice"},
        {"implement", "create", "process", "handle", "fix", "modify"},
        2.0
    ),
    # Database + Destructive = High
    (
        {"database", "db", "sql", "migration", "schema", "table", "index", "query"},
        {"migrate", "delete", "drop", "truncate", "modify", "alter", "update"},
        1.6
    ),
    # Production + Deploy = High
    (
        {"production", "prod", "live", "deploy", "release", "rollout"},
        {"deploy", "push", "release", "update", "migrate", "rollback"},
        1.7
    ),
    # API + Breaking Change = High
    (
        {"api", "endpoint", "route", "interface", "contract", "schema"},
        {"change", "modify", "deprecate", "remove", "break", "migrate"},
        1.4
    ),
]

# Complexity indicators that boost stakes
COMPLEXITY_INDICATORS = [
    (r"\b\d+\s*files?\b", 0.1),  # "5 files", "multiple files"
    (r"\bmultiple\s+(services?|components?|modules?)\b", 0.15),
    (r"\b(across|throughout)\s+the\s+(codebase|project|system)\b", 0.2),
    (r"\barchitectural?\s+(change|decision|refactor)\b", 0.25),
    (r"\bbreaking\s+change\b", 0.3),
    (r"\bbackward\s+compatib", 0.2),
    (r"\b(zero|no)\s+downtime\b", 0.25),
    (r"\bdata\s+(migration|integrity|loss)\b", 0.3),
    (r"\bconcurrency|race\s+condition|deadlock\b", 0.25),
    (r"\bdistributed\s+(system|transaction)\b", 0.2),
]

# Negative patterns that reduce stakes (educational/exploratory)
NEGATIVE_PATTERNS = [
    (r"^(what|how|why|explain|describe|tell me about)\s", -0.4),
    (r"\b(example|tutorial|demo|sample|playground)\b", -0.35),
    (r"\b(learn|understand|curious|wondering)\b", -0.3),
    (r"\b(just|only|simple|basic|quick)\b", -0.2),
    (r"\bfor\s+testing\b", -0.25),
    (r"\blocal(ly|host)?\b", -0.15),
    (r"\bdevelopment\s+(environment|server)\b", -0.2),
    (r"\bsandbox\b", -0.25),
]


@dataclass
class StakesSignal:
    """A single signal contributing to stakes assessment."""
    source: str  # e.g., "semantic", "compound_pattern", "complexity"
    description: str
    weight: float  # Contribution to final score
    confidence: float = 1.0  # How confident we are in this signal


@dataclass
class StakesAssessment:
    """Complete stakes assessment for a task."""
    score: float  # Final score in [0, 1]
    level: str  # "low", "medium", "high", "critical"
    signals: list[StakesSignal] = field(default_factory=list)
    should_use_tot: bool = False
    reasoning: str = ""

    @classmethod
    def from_signals(cls, signals: list[StakesSignal]) -> "StakesAssessment":
        """Aggregate signals into final assessment."""
        if not signals:
            return cls(score=0.0, level="low", reasoning="No stakes signals detected")

        # Weighted average with confidence
        total_weight = sum(s.weight * s.confidence for s in signals)
        total_confidence = sum(s.confidence for s in signals)

        if total_confidence == 0:
            score = 0.0
        else:
            # Normalize to [0, 1]
            score = min(1.0, max(0.0, total_weight / max(1.0, len(signals) * 0.5)))

        # Determine level
        if score >= 0.8:
            level = "critical"
        elif score >= 0.6:
            level = "high"
        elif score >= 0.4:
            level = "medium"
        else:
            level = "low"

        # Should use ToT?
        should_tot = score >= 0.7 or (score >= 0.5 and any(s.source == "semantic" and s.weight >= 0.8 for s in signals))

        # Build reasoning
        top_signals = sorted(signals, key=lambda s: s.weight * s.confidence, reverse=True)[:3]
        reasoning = f"Stakes={score:.2f} ({level}): " + ", ".join(
            f"{s.source}({s.weight:.2f})" for s in top_signals
        )

        return cls(
            score=score,
            level=level,
            signals=signals,
            should_use_tot=should_tot,
            reasoning=reasoning,
        )


class StakesAnalyzer:
    """
    Multi-signal high-stakes detection system.

    Combines semantic matching, compound patterns, complexity signals,
    and context to produce a calibrated stakes assessment.
    """

    def __init__(self):
        self._semantic_embeddings: np.ndarray | None = None
        self._semantic_weights: list[float] = []
        self._initialized = False
        self._embeddings_client = None

    async def _ensure_initialized(self) -> bool:
        """Lazy initialization of semantic embeddings."""
        if self._initialized:
            return True

        try:
            from ..embeddings import get_semantic_router

            router = get_semantic_router()
            if not await router.initialize():
                log.warning("stakes_analyzer_no_embeddings")
                return False

            self._embeddings_client = router.client

            # Pre-compute embeddings for high-stakes exemplars
            embeddings = []
            weights = []

            for text, weight in HIGH_STAKES_EXEMPLARS + LOW_STAKES_EXEMPLARS:
                emb = await self._embeddings_client.embed(text)
                embeddings.append(emb)
                weights.append(weight)

            self._semantic_embeddings = np.array(embeddings)
            self._semantic_weights = weights
            self._initialized = True

            log.info("stakes_analyzer_initialized", exemplars=len(embeddings))
            return True

        except Exception as e:
            log.warning("stakes_analyzer_init_failed", error=str(e))
            return False

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def _analyze_semantic(self, message: str) -> list[StakesSignal]:
        """Layer 1: Semantic similarity to high-stakes exemplars."""
        signals = []

        if not await self._ensure_initialized():
            return signals

        try:
            msg_embedding = await self._embeddings_client.embed(message)

            # Find top matches
            similarities = [
                self._cosine_similarity(msg_embedding, emb)
                for emb in self._semantic_embeddings
            ]

            # Get top 3 matches
            top_indices = np.argsort(similarities)[::-1][:3]

            for idx in top_indices:
                sim = similarities[idx]
                weight = self._semantic_weights[idx]

                if sim >= 0.6:  # Threshold for relevance
                    # Combine similarity with exemplar weight
                    combined_weight = sim * weight

                    exemplar_text = (HIGH_STAKES_EXEMPLARS + LOW_STAKES_EXEMPLARS)[idx][0][:50]
                    signals.append(StakesSignal(
                        source="semantic",
                        description=f"Similar to: {exemplar_text}...",
                        weight=combined_weight,
                        confidence=sim,
                    ))

        except Exception as e:
            log.debug("semantic_stakes_failed", error=str(e))

        return signals

    def _analyze_compound_patterns(self, message: str) -> list[StakesSignal]:
        """Layer 2: Domain + Action compound pattern matching."""
        signals = []
        msg_lower = message.lower()
        words = set(re.findall(r'\b[a-z]+\b', msg_lower))

        for domain_keywords, action_keywords, weight_modifier in COMPOUND_PATTERNS:
            domain_match = words & domain_keywords
            action_match = words & action_keywords

            if domain_match and action_match:
                # Both domain and action present
                signals.append(StakesSignal(
                    source="compound_pattern",
                    description=f"Domain({list(domain_match)[:2]}) + Action({list(action_match)[:2]})",
                    weight=weight_modifier * 0.5,  # Scale to [0, 1] range
                    confidence=0.85,
                ))

        return signals

    def _analyze_complexity(self, message: str) -> list[StakesSignal]:
        """Layer 3: Complexity indicators."""
        signals = []

        for pattern, weight in COMPLEXITY_INDICATORS:
            if re.search(pattern, message, re.IGNORECASE):
                signals.append(StakesSignal(
                    source="complexity",
                    description=f"Complexity indicator: {pattern[:30]}",
                    weight=weight,
                    confidence=0.7,
                ))

        return signals

    def _analyze_negative_patterns(self, message: str) -> list[StakesSignal]:
        """Layer 4: Negative patterns that reduce stakes."""
        signals = []

        for pattern, weight in NEGATIVE_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                signals.append(StakesSignal(
                    source="negative",
                    description=f"Educational/exploratory: {pattern[:30]}",
                    weight=weight,  # Negative weights
                    confidence=0.8,
                ))

        return signals

    def _analyze_context(
        self,
        message: str,
        session_failures: int = 0,
        frustration_level: str = "none",
    ) -> list[StakesSignal]:
        """Layer 5: Context signals from session state."""
        signals = []

        # Previous failures boost stakes
        if session_failures > 0:
            failure_weight = min(0.3, session_failures * 0.1)
            signals.append(StakesSignal(
                source="context",
                description=f"Previous failures: {session_failures}",
                weight=failure_weight,
                confidence=0.9,
            ))

        # Frustration level
        frustration_weights = {
            "none": 0.0,
            "low": 0.1,
            "medium": 0.2,
            "high": 0.35,
            "critical": 0.5,
        }
        if frustration_level in frustration_weights and frustration_weights[frustration_level] > 0:
            signals.append(StakesSignal(
                source="context",
                description=f"User frustration: {frustration_level}",
                weight=frustration_weights[frustration_level],
                confidence=0.85,
            ))

        return signals

    async def analyze(
        self,
        message: str,
        session_failures: int = 0,
        frustration_level: str = "none",
        file_count: int = 0,
    ) -> StakesAssessment:
        """
        Perform complete stakes analysis on a message.

        Args:
            message: The user message to analyze
            session_failures: Number of previous failures in session
            frustration_level: Current frustration level (none/low/medium/high/critical)
            file_count: Number of files being modified

        Returns:
            StakesAssessment with score, level, and should_use_tot flag
        """
        all_signals: list[StakesSignal] = []

        # Layer 1: Semantic similarity (async)
        semantic_signals = await self._analyze_semantic(message)
        all_signals.extend(semantic_signals)

        # Layer 2: Compound patterns
        compound_signals = self._analyze_compound_patterns(message)
        all_signals.extend(compound_signals)

        # Layer 3: Complexity indicators
        complexity_signals = self._analyze_complexity(message)
        all_signals.extend(complexity_signals)

        # Layer 4: Negative patterns
        negative_signals = self._analyze_negative_patterns(message)
        all_signals.extend(negative_signals)

        # Layer 5: Context
        context_signals = self._analyze_context(message, session_failures, frustration_level)
        all_signals.extend(context_signals)

        # Bonus for multi-file operations
        if file_count > 3:
            all_signals.append(StakesSignal(
                source="complexity",
                description=f"Multi-file operation: {file_count} files",
                weight=min(0.3, file_count * 0.05),
                confidence=0.9,
            ))

        # Aggregate into final assessment
        assessment = StakesAssessment.from_signals(all_signals)

        log.debug(
            "stakes_analysis_complete",
            score=assessment.score,
            level=assessment.level,
            signals=len(all_signals),
            should_tot=assessment.should_use_tot,
        )

        return assessment

    def analyze_sync(self, message: str) -> StakesAssessment:
        """
        Synchronous analysis using only non-async layers.

        Use this when you can't await (e.g., in regex detection).
        Skips semantic layer but still provides useful signals.
        """
        all_signals: list[StakesSignal] = []

        # Layers 2-5 (no async)
        all_signals.extend(self._analyze_compound_patterns(message))
        all_signals.extend(self._analyze_complexity(message))
        all_signals.extend(self._analyze_negative_patterns(message))

        return StakesAssessment.from_signals(all_signals)


# Singleton
_stakes_analyzer: StakesAnalyzer | None = None


def get_stakes_analyzer() -> StakesAnalyzer:
    """Get the global StakesAnalyzer instance."""
    global _stakes_analyzer
    if _stakes_analyzer is None:
        _stakes_analyzer = StakesAnalyzer()
    return _stakes_analyzer


async def analyze_stakes(
    message: str,
    session_failures: int = 0,
    frustration_level: str = "none",
    file_count: int = 0,
) -> StakesAssessment:
    """Convenience function for stakes analysis."""
    return await get_stakes_analyzer().analyze(
        message, session_failures, frustration_level, file_count
    )


def analyze_stakes_sync(message: str) -> StakesAssessment:
    """Synchronous stakes analysis (no semantic layer)."""
    return get_stakes_analyzer().analyze_sync(message)
