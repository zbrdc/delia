# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Automatic Context Detection for ACE Framework.

Detects task type from user messages and automatically loads relevant
playbooks and profiles without requiring explicit tool calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import structlog

log = structlog.get_logger()

TaskType = Literal[
    "coding", "testing", "debugging", "architecture",
    "git", "security", "api", "performance", "deployment", "project"
]

# Keyword patterns for task type detection
# Each pattern has a weight: higher weight = more indicative of task type
# Format: (pattern, weight) where weight is 1-3 (3 = highly specific)
TASK_PATTERNS_WEIGHTED: dict[TaskType, list[tuple[str, int]]] = {
    "coding": [
        (r"\b(implement|add|create|build|write|modify|update)\b", 2),
        (r"\b(function|class|method|module|component)\b", 1),
        (r"\b(code|coding|program|develop)\b", 2),
        (r"\b(feature|functionality)\b", 2),
    ],
    "testing": [
        (r"\b(test|tests|testing|pytest|unittest|jest|mocha)\b", 3),
        (r"\b(coverage|mock|fixture|assert|spec)\b", 2),
        (r"\b(unit test|integration test|e2e|end.to.end)\b", 3),
        (r"\b(tdd|test.driven)\b", 3),
    ],
    "debugging": [
        (r"\b(bug|debug|issue|problem)\b", 3),
        (r"\b(error|exception|traceback|stack.?trace)\b", 2),
        (r"\b(broken|failing|crash|crashed|down|outage)\b", 3),
        (r"\b(not working|doesn.?t work|won.?t work|stopped)\b", 3),
        (r"\b(fix|fixing|fixed)\b", 2),  # "fix" often means debugging
        (r"\b(investigate|diagnose|troubleshoot)\b", 3),
    ],
    "architecture": [
        (r"\b(design|architecture|architect)\b", 3),
        (r"\b(pattern|structure|restructure)\b", 2),
        (r"\b(adr|decision|refactor|redesign)\b", 3),
        (r"\b(dependency|interface|abstract|coupling)\b", 2),
        (r"\b(singleton|factory|strategy|observer|repository)\b", 3),
        # Planning/thinking patterns
        (r"\b(plan|planning|planned)\b", 3),
        (r"\b(think|thinking|thought).*(through|about|over)\b", 2),
        (r"\b(approach|strategy|strategize)\b", 2),
        (r"\b(consider|considering|evaluate|evaluating)\b", 2),
        (r"\b(trade.?off|pros?.and.cons?|compare|comparison)\b", 3),
        (r"\b(breakdown|break\s+down|decompose)\b", 2),
        (r"\b(how.should|what.approach|best.way)\b", 3),
    ],
    "git": [
        (r"\b(git|commit|branch|merge|rebase)\b", 3),
        (r"\b(push|pull)\b", 2),  # Separate from PR
        (r"\bPR\b", 3),  # Case-sensitive PR
        (r"\b(pull.?request|merge.?request)\b", 3),
        (r"\b(checkout|stash|diff|cherry.?pick)\b", 2),
        (r"\b(conflict|remote|origin|upstream)\b", 2),
        # Colloquial/conversational git patterns
        (r"\bcheck.?(this.?)?in\b", 3),  # old VCS terminology
        (r"\bland.*(on|to|this|it)\b", 2),  # "land this on main"
        (r"\bsquash\b", 3),  # git squash
        (r"\b(main|master|dev|develop)\s+branch\b", 2),  # branch name mentions
        (r"\bversion.?control\b", 2),  # explicit VCS reference
        (r"\b(amend|revert|reset)\b", 3),  # git commands
    ],
    "security": [
        (r"\b(security|secure|insecure)\b", 3),
        (r"\b(auth|authentication|authorization|oauth|jwt)\b", 3),
        (r"\b(password|token|secret|credential|api.?key)\b", 2),
        (r"\b(vulnerability|exploit|injection|xss|csrf|sql.?injection)\b", 3),
        (r"\b(encrypt|decrypt|hash|sanitize)\b", 2),
    ],
    "api": [
        (r"\b(api|endpoint|route)\b", 3),
        (r"\b(rest|graphql|grpc|websocket)\b", 3),
        (r"\b(request|response|http|status.?code)\b", 2),
        (r"\b(json|payload|body|header)\b", 1),
        (r"\b(GET|POST|PUT|DELETE|PATCH)\b", 2),  # Case-sensitive HTTP methods
    ],
    "performance": [
        (r"\b(performance|optimize|optimization)\b", 3),
        (r"\b(slow|slower|fastest|faster|speed|speed.?up)\b", 3),
        (r"\b(cache|caching|memoize|memoization)\b", 3),
        (r"\b(memory|cpu|gpu|resource)\b", 2),
        (r"\b(profile|profiling|benchmark|benchmarking)\b", 3),
        (r"\b(latency|throughput|bottleneck|timeout)\b", 3),
    ],
    "deployment": [
        (r"\b(deploy|deployment|deploying)\b", 3),
        (r"\b(ci.?cd|pipeline|github.?actions|jenkins)\b", 3),
        (r"\b(docker|container|kubernetes|k8s|helm)\b", 3),
        (r"\b(production|staging|environment)\b", 2),
        (r"\b(release|publish|ship)\b", 2),
    ],
    "project": [
        (r"\b(project|codebase|repository|repo)\b", 2),
        (r"\b(structure|organization|layout|overview)\b", 2),
        (r"\b(setup|configure|install|installation)\b", 2),
        (r"\b(how does|what is|where is|explain|describe)\b", 3),
        (r"\b(documentation|docs|readme)\b", 2),
    ],
}

# Convert to simple pattern dict for backwards compatibility
TASK_PATTERNS: dict[TaskType, list[str]] = {
    task_type: [p[0] for p in patterns]
    for task_type, patterns in TASK_PATTERNS_WEIGHTED.items()
}

# Compile weighted patterns for efficiency
# Format: {task_type: [(compiled_pattern, weight, is_case_sensitive), ...]}
COMPILED_PATTERNS_WEIGHTED: dict[TaskType, list[tuple[re.Pattern, int, bool]]] = {}

for task_type, patterns in TASK_PATTERNS_WEIGHTED.items():
    compiled = []
    for pattern, weight in patterns:
        # Check if pattern should be case-sensitive (uppercase letters in pattern)
        has_uppercase = any(c.isupper() for c in pattern.replace(r"\b", ""))
        if has_uppercase:
            # Case-sensitive patterns (PR, GET, POST, etc.)
            compiled.append((re.compile(pattern), weight, True))
        else:
            # Case-insensitive patterns
            compiled.append((re.compile(pattern, re.IGNORECASE), weight, False))
    COMPILED_PATTERNS_WEIGHTED[task_type] = compiled

# Backwards compatible non-weighted version
COMPILED_PATTERNS: dict[TaskType, list[re.Pattern]] = {
    task_type: [p[0] for p in patterns]
    for task_type, patterns in COMPILED_PATTERNS_WEIGHTED.items()
}


@dataclass
class DetectedContext:
    """Result of context detection."""
    primary_task: TaskType
    secondary_tasks: list[TaskType]
    confidence: float
    matched_keywords: list[str]

    def all_tasks(self) -> list[TaskType]:
        """Return all detected tasks, primary first."""
        return [self.primary_task] + self.secondary_tasks


def detect_task_type(message: str) -> DetectedContext:
    """
    Detect task type(s) from a user message using weighted pattern matching.

    Uses weighted scores where higher weights indicate more specific indicators
    of a task type (weight 3 = highly specific, 1 = general).

    Args:
        message: The user's message or query

    Returns:
        DetectedContext with primary task, secondary tasks, and metadata
    """
    if not message:
        return DetectedContext(
            primary_task="project",
            secondary_tasks=[],
            confidence=0.0,
            matched_keywords=[],
        )

    # Score each task type using weighted patterns
    scores: dict[TaskType, tuple[int, list[str]]] = {}

    for task_type, patterns in COMPILED_PATTERNS_WEIGHTED.items():
        total_weight = 0
        matches: list[str] = []

        for pattern, weight, _is_case_sensitive in patterns:
            found = pattern.findall(message)
            if found:
                # Add weighted score
                total_weight += len(found) * weight
                matches.extend(found)

        if total_weight > 0:
            scores[task_type] = (total_weight, matches)

    if not scores:
        # Default to project for general queries
        return DetectedContext(
            primary_task="project",
            secondary_tasks=[],
            confidence=0.3,
            matched_keywords=[],
        )

    # Sort by weighted score descending
    sorted_tasks = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

    primary_task = sorted_tasks[0][0]
    primary_score = sorted_tasks[0][1][0]
    primary_keywords = sorted_tasks[0][1][1]

    # Secondary tasks (score > 0 and not primary)
    secondary_tasks = [t for t, (s, _) in sorted_tasks[1:] if s > 0][:2]

    # Confidence based on weighted score (6+ weighted points = 100% confidence)
    confidence = min(1.0, primary_score / 6)

    log.debug(
        "context_detected",
        primary=primary_task,
        secondary=secondary_tasks,
        confidence=confidence,
        weighted_score=primary_score,
        keywords=primary_keywords[:5],
    )

    return DetectedContext(
        primary_task=primary_task,
        secondary_tasks=secondary_tasks,
        confidence=confidence,
        matched_keywords=primary_keywords[:10],
    )


def get_relevant_profiles(context: DetectedContext) -> list[str]:
    """
    Get list of profile filenames relevant to the detected context.

    Args:
        context: The detected context

    Returns:
        List of profile filenames (e.g., ["core.md", "coding.md"])
    """
    profiles = ["core.md"]  # Always include core

    # Map task types to profiles
    task_to_profile = {
        "coding": ["coding.md", "python.md"],
        "testing": ["testing.md"],
        "debugging": ["debugging.md"],
        "architecture": ["architecture.md"],
        "git": ["git.md"],
        "security": ["security.md"],
        "api": ["api.md", "fastapi.md"],
        "performance": ["performance.md"],
        "deployment": ["deployment.md"],
        "project": [],  # Just core is enough
    }

    # Add primary task profiles
    profiles.extend(task_to_profile.get(context.primary_task, []))

    # Add secondary task profiles (first one only to avoid overload)
    for task in context.secondary_tasks[:1]:
        for profile in task_to_profile.get(task, []):
            if profile not in profiles:
                profiles.append(profile)

    return profiles


class ContextManager:
    """
    Manages automatic context detection and playbook injection.

    This is a singleton that tracks the current context and provides
    methods for auto-loading relevant playbooks.
    """

    _instance: "ContextManager | None" = None

    def __new__(cls) -> "ContextManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._current_context: DetectedContext | None = None
        self._last_message: str = ""
        self._auto_inject_enabled: bool = True

    @property
    def current_context(self) -> DetectedContext | None:
        return self._current_context

    def detect_and_update(self, message: str) -> DetectedContext:
        """
        Detect context from message and update internal state.

        Args:
            message: User message to analyze

        Returns:
            Detected context
        """
        self._last_message = message
        self._current_context = detect_task_type(message)
        return self._current_context

    def get_auto_context_bullets(self, project_path: str | None = None) -> str:
        """
        Get formatted playbook bullets for current context.

        Returns bullets as a formatted string ready for injection
        into agent responses or tool results.
        """
        if not self._current_context:
            return ""

        from .playbook import get_playbook_manager
        from pathlib import Path

        pm = get_playbook_manager()
        if project_path:
            pm.set_project(Path(project_path))

        parts = []

        # Get bullets for primary task
        bullets = pm.get_top_bullets(self._current_context.primary_task, limit=5)
        if bullets:
            parts.append(f"\n## Auto-loaded Context: {self._current_context.primary_task.title()}\n")
            parts.append("Apply these strategies to your current task:\n")
            for b in bullets:
                parts.append(f"- [{b.id}] {b.content}")

        # Get bullets for secondary tasks (fewer)
        for task in self._current_context.secondary_tasks[:1]:
            bullets = pm.get_top_bullets(task, limit=2)
            if bullets:
                parts.append(f"\n### Also relevant ({task}):\n")
                for b in bullets:
                    parts.append(f"- [{b.id}] {b.content}")

        return "\n".join(parts)

    def enable_auto_inject(self, enabled: bool = True):
        """Enable or disable automatic context injection."""
        self._auto_inject_enabled = enabled
        log.info("auto_inject_toggled", enabled=enabled)

    def is_auto_inject_enabled(self) -> bool:
        return self._auto_inject_enabled


def get_context_manager() -> ContextManager:
    """Get the singleton ContextManager instance."""
    return ContextManager()


# =============================================================================
# DYNAMIC PATTERN LEARNING
# =============================================================================

import json
from pathlib import Path
from datetime import datetime


@dataclass
class LearnedPattern:
    """A pattern learned from user feedback."""
    pattern: str
    task_type: TaskType
    weight: int  # 1-3
    success_count: int = 0
    failure_count: int = 0
    created_at: str = ""
    source: str = "feedback"  # "feedback" or "profile"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def effectiveness(self) -> float:
        """Calculate pattern effectiveness (0-1)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "task_type": self.task_type,
            "weight": self.weight,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedPattern":
        return cls(**data)


class PatternLearner:
    """
    Learns and adapts detection patterns based on feedback.

    Stores learned patterns in .delia/learned_patterns.json
    and updates the detection system dynamically.
    """

    def __init__(self, project_path: Path | None = None):
        self.project_path = project_path or Path.cwd()
        self._patterns: list[LearnedPattern] = []
        self._compiled: dict[TaskType, list[tuple[re.Pattern, int]]] = {}
        self._loaded = False

    def _get_patterns_path(self) -> Path:
        """Get path to learned patterns file."""
        return self.project_path / ".delia" / "learned_patterns.json"

    def load(self) -> None:
        """Load learned patterns from disk."""
        path = self._get_patterns_path()
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._patterns = [LearnedPattern.from_dict(p) for p in data.get("patterns", [])]
                self._compile_patterns()
                log.debug("learned_patterns_loaded", count=len(self._patterns))
            except Exception as e:
                log.warning("learned_patterns_load_failed", error=str(e))
        self._loaded = True

    def save(self) -> None:
        """Persist learned patterns to disk."""
        path = self._get_patterns_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump({
                    "version": 1,
                    "patterns": [p.to_dict() for p in self._patterns],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
            log.debug("learned_patterns_saved", count=len(self._patterns))
        except Exception as e:
            log.warning("learned_patterns_save_failed", error=str(e))

    def _compile_patterns(self) -> None:
        """Compile learned patterns for efficient matching."""
        self._compiled.clear()
        for pattern in self._patterns:
            if pattern.task_type not in self._compiled:
                self._compiled[pattern.task_type] = []
            try:
                compiled = re.compile(pattern.pattern, re.IGNORECASE)
                self._compiled[pattern.task_type].append((compiled, pattern.weight))
            except re.error:
                log.warning("invalid_learned_pattern", pattern=pattern.pattern)

    def add_pattern(
        self,
        pattern: str,
        task_type: TaskType,
        weight: int = 2,
        source: str = "feedback",
    ) -> LearnedPattern:
        """Add a new learned pattern."""
        if not self._loaded:
            self.load()

        # Check for duplicates
        for existing in self._patterns:
            if existing.pattern == pattern and existing.task_type == task_type:
                log.debug("learned_pattern_exists", pattern=pattern)
                return existing

        new_pattern = LearnedPattern(
            pattern=pattern,
            task_type=task_type,
            weight=weight,
            source=source,
        )
        self._patterns.append(new_pattern)
        self._compile_patterns()
        self.save()

        log.info("learned_pattern_added", pattern=pattern, task_type=task_type)
        return new_pattern

    def record_feedback(
        self,
        message: str,
        detected_task: TaskType,
        correct_task: TaskType,
        was_correct: bool,
    ) -> dict:
        """
        Record feedback on a detection to improve future accuracy.

        If detection was wrong, learns new patterns from the message
        that should have indicated the correct task type.

        Args:
            message: The original message
            detected_task: What was detected
            correct_task: What should have been detected
            was_correct: Whether detection was correct

        Returns:
            Dict with learning results
        """
        if not self._loaded:
            self.load()

        result = {
            "was_correct": was_correct,
            "patterns_updated": 0,
            "patterns_added": 0,
        }

        if was_correct:
            # Boost patterns that matched
            for pattern in self._patterns:
                if pattern.task_type == detected_task:
                    try:
                        if re.search(pattern.pattern, message, re.IGNORECASE):
                            pattern.success_count += 1
                            result["patterns_updated"] += 1
                    except re.error:
                        pass
        else:
            # Demote patterns that matched incorrectly
            for pattern in self._patterns:
                if pattern.task_type == detected_task:
                    try:
                        if re.search(pattern.pattern, message, re.IGNORECASE):
                            pattern.failure_count += 1
                            result["patterns_updated"] += 1
                    except re.error:
                        pass

            # Extract words from message that might indicate correct task
            # Simple approach: learn any unique words not in common words
            common_words = {"the", "a", "an", "is", "are", "was", "were", "be",
                          "been", "being", "have", "has", "had", "do", "does",
                          "did", "will", "would", "could", "should", "may",
                          "might", "must", "shall", "can", "to", "of", "in",
                          "for", "on", "with", "at", "by", "from", "as", "into",
                          "through", "during", "before", "after", "above", "below",
                          "between", "under", "again", "further", "then", "once",
                          "here", "there", "when", "where", "why", "how", "all",
                          "each", "few", "more", "most", "other", "some", "such",
                          "no", "nor", "not", "only", "own", "same", "so", "than",
                          "too", "very", "just", "and", "but", "if", "or", "because",
                          "until", "while", "this", "that", "these", "those", "i",
                          "me", "my", "we", "our", "you", "your", "he", "him", "she",
                          "her", "it", "its", "they", "them", "their", "what", "which",
                          "who", "whom", "please", "help", "need", "want", "like"}

            words = re.findall(r'\b[a-zA-Z]{3,}\b', message.lower())
            unique_words = [w for w in words if w not in common_words]

            # Add top unique words as new patterns if we have any
            for word in unique_words[:3]:
                # Check if this word isn't already in base patterns
                already_exists = False
                for patterns in TASK_PATTERNS_WEIGHTED.get(correct_task, []):
                    if word in patterns[0].lower():
                        already_exists = True
                        break

                if not already_exists:
                    self.add_pattern(
                        pattern=rf"\b{word}\b",
                        task_type=correct_task,
                        weight=2,
                        source="feedback",
                    )
                    result["patterns_added"] += 1

        if result["patterns_updated"] > 0 or result["patterns_added"] > 0:
            self.save()

        log.info(
            "detection_feedback_recorded",
            was_correct=was_correct,
            detected=detected_task,
            correct=correct_task,
            updates=result["patterns_updated"],
            added=result["patterns_added"],
        )

        return result

    def get_boosted_patterns(self, task_type: TaskType) -> list[tuple[re.Pattern, int]]:
        """Get learned patterns for a task type, sorted by effectiveness."""
        if not self._loaded:
            self.load()

        patterns = []
        for pattern in self._patterns:
            if pattern.task_type == task_type and pattern.effectiveness >= 0.4:
                try:
                    compiled = re.compile(pattern.pattern, re.IGNORECASE)
                    # Boost weight based on effectiveness
                    boosted_weight = int(pattern.weight * (0.5 + pattern.effectiveness))
                    patterns.append((compiled, boosted_weight))
                except re.error:
                    pass

        return patterns

    def prune_ineffective(self, min_effectiveness: float = 0.3, min_uses: int = 5) -> int:
        """Remove patterns that are consistently wrong."""
        if not self._loaded:
            self.load()

        original_count = len(self._patterns)
        self._patterns = [
            p for p in self._patterns
            if (p.success_count + p.failure_count < min_uses or
                p.effectiveness >= min_effectiveness)
        ]

        removed = original_count - len(self._patterns)
        if removed > 0:
            self._compile_patterns()
            self.save()
            log.info("ineffective_patterns_pruned", count=removed)

        return removed

    def get_stats(self) -> dict:
        """Get statistics about learned patterns."""
        if not self._loaded:
            self.load()

        by_task = {}
        for pattern in self._patterns:
            if pattern.task_type not in by_task:
                by_task[pattern.task_type] = {"count": 0, "avg_effectiveness": 0.0}
            by_task[pattern.task_type]["count"] += 1

        for task_type, stats in by_task.items():
            patterns = [p for p in self._patterns if p.task_type == task_type]
            if patterns:
                stats["avg_effectiveness"] = sum(p.effectiveness for p in patterns) / len(patterns)

        return {
            "total_patterns": len(self._patterns),
            "by_task_type": by_task,
            "top_patterns": [
                {"pattern": p.pattern, "task_type": p.task_type, "effectiveness": p.effectiveness}
                for p in sorted(self._patterns, key=lambda x: x.effectiveness, reverse=True)[:10]
            ],
        }


# Global learner instance
_pattern_learner: PatternLearner | None = None


def get_pattern_learner(project_path: Path | None = None) -> PatternLearner:
    """Get or create the pattern learner for a project."""
    global _pattern_learner
    if _pattern_learner is None or (project_path and _pattern_learner.project_path != project_path):
        _pattern_learner = PatternLearner(project_path)
    return _pattern_learner


def detect_with_learning(message: str, project_path: Path | None = None) -> DetectedContext:
    """
    Detect task type using both base patterns and learned patterns.

    This is the recommended entry point that combines static patterns
    with dynamically learned ones.
    """
    # Get base detection
    context = detect_task_type(message)

    # Enhance with learned patterns
    learner = get_pattern_learner(project_path)
    if not learner._loaded:
        learner.load()

    if learner._patterns:
        # Re-score with learned patterns included
        scores: dict[TaskType, int] = {}

        # Start with base detection score
        scores[context.primary_task] = int(context.confidence * 6)

        # Add learned pattern scores
        for pattern in learner._patterns:
            if pattern.effectiveness >= 0.4:  # Only use effective patterns
                try:
                    if re.search(pattern.pattern, message, re.IGNORECASE):
                        task = pattern.task_type
                        boost = int(pattern.weight * (0.5 + pattern.effectiveness))
                        scores[task] = scores.get(task, 0) + boost
                except re.error:
                    pass

        # Re-determine primary if learned patterns boosted another type
        if scores:
            best_task = max(scores.items(), key=lambda x: x[1])
            if best_task[0] != context.primary_task and best_task[1] > scores.get(context.primary_task, 0):
                log.debug(
                    "learned_pattern_override",
                    original=context.primary_task,
                    new=best_task[0],
                )
                # Update context with learned pattern boost
                return DetectedContext(
                    primary_task=best_task[0],
                    secondary_tasks=[context.primary_task] + [t for t in context.secondary_tasks if t != best_task[0]][:1],
                    confidence=min(1.0, best_task[1] / 6),
                    matched_keywords=context.matched_keywords,
                )

    return context
