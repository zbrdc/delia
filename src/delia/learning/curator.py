# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Curator: Playbook Maintenance via Delta Updates.

Integrates Reflector insights into structured playbook updates.
Uses atomic delta operations (ADD, REMOVE, MERGE) rather than
monolithic rewrites - per Delia Framework principles.

Wraps existing PlaybookManager - no new storage layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from delia.playbook import PlaybookManager, PlaybookBullet

from delia.learning.deduplication import SemanticDeduplicator, get_deduplicator
from delia.learning.reflector import ReflectionResult, ExtractedInsight, InsightType

log = structlog.get_logger()


# =============================================================================
# Quality Gate: Stale Snapshot Detection
# =============================================================================

# Patterns that indicate point-in-time observations (not prescriptive guidance)
STALE_SNAPSHOT_PATTERNS = [
    r"\b\d+\s+files?\s+(?:need|require|should|must|to)\s+(?:migration|update|fix)",  # "71 files need migration"
    r"\bcurrently\s+(?:have|has|using|at)\b",  # "currently using X"
    r"\b(?:today|yesterday|this\s+week|last\s+(?:week|month))\b",  # Time references
    r"\b\d{4}-\d{2}-\d{2}\b",  # ISO dates
    r"\bcommit\s+[a-f0-9]{6,40}\b",  # Git commit hashes
    r"\bversion\s+\d+\.\d+",  # "version 1.2.3" without context
    r"\bwas\s+(?:slow|fast|broken|working|failing)\b",  # Past state observations
    r"\bfound\s+\d+\s+(?:errors?|issues?|bugs?|problems?)\b",  # Discovery counts
]

STALE_SNAPSHOT_COMPILED = [re.compile(p, re.IGNORECASE) for p in STALE_SNAPSHOT_PATTERNS]


def is_stale_snapshot(content: str) -> str | None:
    """
    Detect point-in-time observations that shouldn't become permanent bullets.

    Returns rejection reason if stale, None if OK.

    Examples (REJECT):
    - "71 files need migration" → snapshot of current state
    - "Fixed in commit abc123" → tied to specific commit
    - "Currently using React 18" → version will change

    Examples (ACCEPT):
    - "Prefer styled() over StyleSheet.create" → prescriptive
    - "React 18+ requires concurrent mode" → version-qualified guidance
    """
    for pattern in STALE_SNAPSHOT_COMPILED:
        match = pattern.search(content)
        if match:
            return f"Stale snapshot detected: '{match.group()}'"
    return None


# =============================================================================
# Quality Gate: Generic/Obvious Advice Detection
# =============================================================================

# Patterns for overly vague guidance
GENERIC_ADVICE_PATTERNS = [
    r"^(?:always\s+)?(?:write|use|do|add|follow|ensure)\s+(?:good|proper|correct|best|right)\s+\w+\.?$",
    r"^(?:be|stay)\s+(?:careful|aware|mindful|cautious)\.?$",
    r"^(?:remember|note|don't forget)\s+to\s+\w+\.?$",
    r"^(?:make\s+sure|ensure)\s+(?:to\s+)?(?:test|check|verify)\.?$",
    r"^(?:optimize|improve)\s+(?:for\s+)?performance\.?$",
    r"^(?:handle|catch)\s+(?:all\s+)?(?:errors?|exceptions?)\.?$",
    r"^(?:use|follow)\s+(?:the\s+)?(?:best\s+practices?|standards?)\.?$",
    r"^(?:keep|maintain)\s+(?:code\s+)?(?:clean|readable|simple)\.?$",
    r"^(?:document|comment)\s+(?:your\s+)?(?:code|changes?)\.?$",
]

GENERIC_ADVICE_COMPILED = [re.compile(p, re.IGNORECASE) for p in GENERIC_ADVICE_PATTERNS]

# Words that indicate specific, actionable guidance (should be present)
SPECIFICITY_MARKERS = {
    # Tech terms
    "async", "await", "callback", "promise", "observable",
    "path", "file", "directory", "import", "export", "module",
    "api", "http", "rest", "graphql", "rpc", "endpoint",
    "database", "query", "schema", "migration", "index",
    "cache", "redis", "memory", "storage", "session",
    "auth", "token", "jwt", "oauth", "permission", "role",
    "component", "hook", "state", "props", "context", "reducer",
    "class", "function", "method", "interface", "type", "generic",
    # Framework/tool names
    "react", "vue", "angular", "svelte", "next", "nuxt",
    "fastapi", "django", "flask", "express", "nest",
    "postgres", "mysql", "mongo", "redis", "supabase",
    "jest", "pytest", "vitest", "playwright", "cypress",
    "docker", "kubernetes", "nginx", "aws", "gcp",
    # Specific patterns
    "pathlib", "os.path", "httpx", "requests", "axios",
    "zod", "pydantic", "yup", "joi",
}


def is_generic_advice(content: str) -> str | None:
    """
    Filter obvious or overly vague patterns that don't add value.

    Returns rejection reason if generic, None if OK.

    Examples (REJECT):
    - "Always write good code" → no specific guidance
    - "Handle errors properly" → too vague
    - "Follow best practices" → meaningless

    Examples (ACCEPT):
    - "Use pathlib.Path over os.path" → specific recommendation
    - "Handle async errors with try/catch and user feedback" → actionable
    """
    content_lower = content.lower()

    # Check against generic patterns
    for pattern in GENERIC_ADVICE_COMPILED:
        if pattern.match(content.strip()):
            return "Too generic: lacks specific guidance"

    # Check for specificity markers
    has_specificity = any(marker in content_lower for marker in SPECIFICITY_MARKERS)

    # Short content without specificity markers is likely generic
    word_count = len(content.split())
    if word_count < 10 and not has_specificity:
        # Additional check: does it have actionable structure?
        has_action = any(word in content_lower for word in [
            "use", "avoid", "prefer", "never", "always", "must", "should",
            "instead of", "rather than", "over", "not"
        ])
        if not has_action:
            return "Too generic: short content without specific terms"

    return None


# =============================================================================
# Quality Gate: Contradiction Detection
# =============================================================================

# Simple opposing keyword pairs (no regex backreferences)
OPPOSING_TERMS = [
    ("named export", "default export"),
    ("default export", "named export"),
    ("async", "sync"),
    ("sync", "async"),
    ("class component", "functional component"),
    ("functional component", "class component"),
]


def detect_contradiction(
    new_content: str,
    existing_bullets: list["PlaybookBullet"],
) -> tuple[str, str] | None:
    """
    Detect if new content contradicts existing bullets.

    Returns (conflicting_bullet_id, explanation) if contradiction found, None if OK.

    Examples:
    - New: "Use default exports" vs Existing: "Use named exports" → CONFLICT
    - New: "Always use async" vs Existing: "Prefer sync for simple ops" → CONFLICT
    """
    new_lower = new_content.lower()

    for bullet in existing_bullets:
        existing_lower = bullet.content.lower()

        # Check for opposing term pairs
        for term_a, term_b in OPPOSING_TERMS:
            # New recommends A, existing recommends B
            new_has_a = term_a in new_lower and any(
                w in new_lower for w in ["use", "prefer", "always"]
            )
            existing_has_b = term_b in existing_lower and any(
                w in existing_lower for w in ["use", "prefer", "always"]
            )
            if new_has_a and existing_has_b:
                return (bullet.id, f"Recommends '{term_a}' but existing recommends '{term_b}'")

        # Check for "use X" vs "avoid X" or "never X"
        use_match = re.search(r"\buse\s+(\w+(?:\s+\w+)?)", new_lower)
        if use_match:
            term = use_match.group(1)
            if re.search(rf"\b(?:avoid|never|don't)\s+.*{re.escape(term)}", existing_lower):
                return (bullet.id, f"Says 'use {term}' but existing says avoid it")

        avoid_match = re.search(r"\b(?:avoid|never)\s+(\w+(?:\s+\w+)?)", new_lower)
        if avoid_match:
            term = avoid_match.group(1)
            if re.search(rf"\buse\s+.*{re.escape(term)}", existing_lower):
                return (bullet.id, f"Says 'avoid {term}' but existing says use it")

    return None


# =============================================================================
# Quality Gate: CLAUDE.md Alignment Check
# =============================================================================

# Cache for loaded CLAUDE.md content per project
_claude_md_cache: dict[str, str] = {}


def load_claude_md(project_path: Path | None = None) -> str | None:
    """Load CLAUDE.md content from project, with caching."""
    path = (project_path or Path.cwd()) / "CLAUDE.md"
    path_str = str(path)

    if path_str in _claude_md_cache:
        return _claude_md_cache[path_str]

    if not path.exists():
        return None

    try:
        content = path.read_text(encoding="utf-8")
        _claude_md_cache[path_str] = content
        return content
    except Exception:
        return None


def check_claude_alignment(
    content: str,
    project_path: Path | None = None,
) -> str | None:
    """
    Check if bullet duplicates what's already in CLAUDE.md.

    Returns rejection reason if redundant, None if OK.

    This prevents the learning loop from re-learning what's already
    explicitly documented in project instructions.
    """
    claude_content = load_claude_md(project_path)
    if not claude_content:
        return None

    content_lower = content.lower()
    claude_lower = claude_content.lower()

    # Extract key terms from the new bullet
    # Look for framework/tool names that might already be documented
    key_patterns = [
        r"\b(tamagui|react\s+native|expo)\b",
        r"\b(mobx|mst|mobx-state-tree|swr)\b",
        r"\b(react\s+hook\s+form|zod)\b",
        r"\b(date-fns|moment)\b",
        r"\b(supabase|postgres|rls)\b",
        r"\b(conventional\s+commits?)\b",
        r"\b(jest|playwright|maestro)\b",
        r"\b(pathlib|httpx|pydantic)\b",
    ]

    for pattern in key_patterns:
        match = re.search(pattern, content_lower)
        if match:
            term = match.group(1)
            # Check if CLAUDE.md already mentions this
            if term in claude_lower:
                # Check if CLAUDE.md has a directive about it
                # Look for the term in a prescriptive context
                claude_has_directive = re.search(
                    rf"(?:use|prefer|always|only|must|should).*{re.escape(term)}|{re.escape(term)}.*(?:for|only|must|should)",
                    claude_lower
                )
                if claude_has_directive:
                    return f"Redundant with CLAUDE.md: already documents '{term}'"

    # Check for near-exact content matches (fuzzy)
    # Split into significant phrases
    phrases = re.findall(r'\b\w+(?:\s+\w+){1,3}\b', content_lower)
    for phrase in phrases:
        if len(phrase) > 15 and phrase in claude_lower:
            return f"Redundant with CLAUDE.md: contains '{phrase[:30]}...'"

    return None


class CurationAction(str, Enum):
    """Types of curation operations."""
    ADD = "add"          # Add new bullet
    REMOVE = "remove"    # Remove ineffective bullet
    MERGE = "merge"      # Combine similar bullets
    MODIFY = "modify"    # Update existing bullet
    BOOST = "boost"      # Increase utility (record helpful)
    DEMOTE = "demote"    # Decrease utility (record harmful)


@dataclass
class CurationDelta:
    """A single atomic curation operation."""
    action: CurationAction
    task_type: str
    bullet_id: str | None = None  # For REMOVE, MODIFY, MERGE, BOOST, DEMOTE
    content: str | None = None    # For ADD, MODIFY
    merge_into: str | None = None  # For MERGE - target bullet ID
    section: str = "general_strategies"
    source: Literal["reflector", "manual", "auto_prune"] = "reflector"
    reasoning: str = ""


@dataclass
class CurationResult:
    """Result of curation operations."""
    applied_deltas: list[CurationDelta] = field(default_factory=list)
    skipped_deltas: list[tuple[CurationDelta, str]] = field(default_factory=list)
    bullets_added: int = 0
    bullets_removed: int = 0
    bullets_merged: int = 0
    dedup_prevented: int = 0  # How many adds blocked by dedup
    feedback_recorded: int = 0


class Curator:
    """
    Curator: Maintains playbooks through incremental delta updates.

    Key responsibilities:
    1. Accept insights from Reflector
    2. Check semantic similarity before adding (dedup)
    3. Apply atomic delta operations
    4. Periodic maintenance (prune, merge similar)

    Wraps existing PlaybookManager - no new storage.
    Follows Delia principle: NEVER regenerate entire playbooks.
    """

    def __init__(
        self,
        playbook_manager: "PlaybookManager",
        deduplicator: SemanticDeduplicator | None = None,
    ):
        """
        Initialize Curator.

        Args:
            playbook_manager: Existing PlaybookManager to wrap
            deduplicator: SemanticDeduplicator instance (or uses global)
        """
        self.playbook = playbook_manager
        self.dedup = deduplicator or get_deduplicator()

    async def curate(
        self,
        reflection: ReflectionResult,
        auto_prune: bool = False,
    ) -> CurationResult:
        """
        Process reflection insights into playbook updates.

        This is the main entry point after Reflector.reflect().

        Args:
            reflection: Result from Reflector
            auto_prune: Whether to trigger pruning if threshold reached

        Returns:
            CurationResult with applied changes
        """
        result = CurationResult()

        # 1. Record bullet feedback from reflection
        for bullet_id in reflection.bullets_to_tag_helpful:
            success = self.playbook.record_feedback(
                bullet_id=bullet_id,
                task_type=reflection.task_type,
                helpful=True,
            )
            if success:
                result.feedback_recorded += 1
                result.applied_deltas.append(CurationDelta(
                    action=CurationAction.BOOST,
                    task_type=reflection.task_type,
                    bullet_id=bullet_id,
                    source="reflector",
                    reasoning="Marked helpful based on task success",
                ))

        for bullet_id in reflection.bullets_to_tag_harmful:
            success = self.playbook.record_feedback(
                bullet_id=bullet_id,
                task_type=reflection.task_type,
                helpful=False,
            )
            if success:
                result.feedback_recorded += 1
                result.applied_deltas.append(CurationDelta(
                    action=CurationAction.DEMOTE,
                    task_type=reflection.task_type,
                    bullet_id=bullet_id,
                    source="reflector",
                    reasoning="Marked harmful based on task outcome",
                ))

        # 2. Add new insights as bullets (with dedup check)
        for insight in reflection.insights:
            if insight.confidence < 0.5:
                result.skipped_deltas.append((
                    CurationDelta(
                        action=CurationAction.ADD,
                        task_type=reflection.task_type,
                        content=insight.content,
                    ),
                    f"Low confidence: {insight.confidence:.2f}",
                ))
                continue

            add_result = await self.add_bullet(
                task_type=reflection.task_type,
                content=self._format_insight_content(insight),
                section=self._insight_type_to_section(insight.insight_type),
                source="reflector",
            )

            if add_result.get("added"):
                result.bullets_added += 1
                result.applied_deltas.append(CurationDelta(
                    action=CurationAction.ADD,
                    task_type=reflection.task_type,
                    content=insight.content,
                    source="reflector",
                    reasoning=f"Insight from {reflection.task_type} task",
                ))
            elif add_result.get("deduplicated"):
                result.dedup_prevented += 1
                result.skipped_deltas.append((
                    CurationDelta(
                        action=CurationAction.ADD,
                        task_type=reflection.task_type,
                        content=insight.content,
                    ),
                    add_result.get("reason", "Duplicate detected"),
                ))
            elif add_result.get("quality_rejected"):
                # Quality gate rejection - track separately
                result.skipped_deltas.append((
                    CurationDelta(
                        action=CurationAction.ADD,
                        task_type=reflection.task_type,
                        content=insight.content,
                    ),
                    add_result.get("reason", "Failed quality validation"),
                ))

        log.info(
            "curation_complete",
            task_type=reflection.task_type,
            added=result.bullets_added,
            dedup_prevented=result.dedup_prevented,
            feedback_recorded=result.feedback_recorded,
        )

        return result

    def _format_insight_content(self, insight: ExtractedInsight) -> str:
        """Format insight content with appropriate prefix."""
        content = insight.content

        # Add AVOID prefix for anti-patterns
        if insight.insight_type == InsightType.ANTI_PATTERN:
            if not content.upper().startswith("AVOID"):
                content = f"AVOID: {content}"

        return content

    def _insight_type_to_section(self, insight_type: InsightType) -> str:
        """Map insight type to playbook section."""
        mapping = {
            InsightType.STRATEGY: "learned_strategies",
            InsightType.ANTI_PATTERN: "anti_patterns",
            InsightType.FAILURE_MODE: "failure_modes",
            InsightType.TOOL_USAGE: "tool_usage",
            InsightType.CONTEXT_HINT: "context_hints",
        }
        return mapping.get(insight_type, "general_strategies")

    async def add_bullet(
        self,
        task_type: str,
        content: str,
        section: str = "general_strategies",
        source: str = "reflector",
        skip_dedup: bool = False,
        skip_validation: bool = False,
    ) -> dict[str, Any]:
        """
        Add a new bullet with quality validation and semantic deduplication.

        Args:
            task_type: Playbook type (coding, testing, etc.)
            content: Bullet content
            section: Section within playbook
            source: Origin of bullet (reflector, manual, etc.)
            skip_dedup: Bypass deduplication check
            skip_validation: Bypass quality validation (use sparingly)

        Returns:
            Dict with keys:
            - added: bool - whether bullet was added
            - bullet_id: str | None - ID of added or existing bullet
            - reason: str | None - rejection reason if not added
            - deduplicated: bool - if rejected due to duplicate
            - quality_rejected: bool - if rejected due to quality gate
        """
        result: dict[str, Any] = {
            "added": False,
            "bullet_id": None,
            "reason": None,
            "deduplicated": False,
            "quality_rejected": False,
        }

        existing_bullets = self.playbook.load_playbook(task_type)

        # Get project path for CLAUDE.md check
        project_path = Path(self.playbook.playbook_dir).parent

        # =====================================================================
        # Quality Gate 1: Stale Snapshot Detection
        # =====================================================================
        if not skip_validation:
            stale_reason = is_stale_snapshot(content)
            if stale_reason:
                log.debug("bullet_stale_snapshot", task_type=task_type, reason=stale_reason)
                result["quality_rejected"] = True
                result["reason"] = stale_reason
                return result

        # =====================================================================
        # Quality Gate 2: Generic/Obvious Advice Detection
        # =====================================================================
        if not skip_validation:
            generic_reason = is_generic_advice(content)
            if generic_reason:
                log.debug("bullet_too_generic", task_type=task_type, reason=generic_reason)
                result["quality_rejected"] = True
                result["reason"] = generic_reason
                return result

        # =====================================================================
        # Quality Gate 3: CLAUDE.md Alignment Check
        # =====================================================================
        if not skip_validation:
            claude_reason = check_claude_alignment(content, project_path)
            if claude_reason:
                log.debug("bullet_claude_redundant", task_type=task_type, reason=claude_reason)
                result["quality_rejected"] = True
                result["reason"] = claude_reason
                return result

        # =====================================================================
        # Quality Gate 4: Contradiction Detection
        # =====================================================================
        if not skip_validation and existing_bullets:
            contradiction = detect_contradiction(content, existing_bullets)
            if contradiction:
                bullet_id, conflict_reason = contradiction
                log.debug(
                    "bullet_contradiction",
                    task_type=task_type,
                    conflicts_with=bullet_id,
                    reason=conflict_reason,
                )
                result["quality_rejected"] = True
                result["reason"] = f"Contradiction: {conflict_reason}"
                return result

        # =====================================================================
        # Quality Gate 5: Semantic Deduplication (existing)
        # =====================================================================
        if not skip_dedup and existing_bullets:
            dedup_result = await self.dedup.check_similarity(
                new_content=content,
                existing_bullets=existing_bullets,
            )

            if dedup_result.is_duplicate:
                log.debug(
                    "bullet_dedup_blocked",
                    task_type=task_type,
                    similarity=dedup_result.best_match.similarity if dedup_result.best_match else 0,
                )
                result["deduplicated"] = True
                result["bullet_id"] = dedup_result.best_match.bullet_id if dedup_result.best_match else None
                result["reason"] = f"Duplicate of {result['bullet_id']}"
                return result

            if dedup_result.recommended_action == "merge" and dedup_result.best_match:
                # Boost existing bullet instead of adding
                self.playbook.record_feedback(
                    bullet_id=dedup_result.best_match.bullet_id,
                    task_type=task_type,
                    helpful=True,
                )
                log.debug(
                    "bullet_merged_boost",
                    task_type=task_type,
                    existing_id=dedup_result.best_match.bullet_id,
                )
                result["deduplicated"] = True
                result["bullet_id"] = dedup_result.best_match.bullet_id
                result["reason"] = "Merged with existing (boosted utility)"
                return result

        # =====================================================================
        # Add bullet via PlaybookManager (includes basic quality validation)
        # =====================================================================
        bullet = self.playbook.add_bullet(
            task_type=task_type,
            content=content,
            section=section,
            source=source,
            skip_validation=skip_validation,
        )

        # PlaybookManager returns None if quality gate rejects
        if bullet is None:
            result["quality_rejected"] = True
            result["reason"] = "Failed quality validation (check logs for details)"
            log.debug("bullet_quality_rejected", task_type=task_type, content=content[:50])
            return result

        # Generate embedding for new bullet (if embeddings available)
        try:
            from delia.learning.retrieval import get_retriever
            retriever = get_retriever()
            await retriever.add_bullet_embedding(bullet.id, content, project_path)
        except Exception as e:
            log.debug("bullet_embedding_skipped", error=str(e))

        result["added"] = True
        result["bullet_id"] = bullet.id
        log.debug("bullet_added", task_type=task_type, bullet_id=bullet.id)
        return result

    def apply_feedback(
        self,
        bullet_id: str,
        task_type: str,
        helpful: bool,
    ) -> bool:
        """
        Apply feedback to a bullet.

        Shortcut delegating to PlaybookManager.
        """
        return self.playbook.record_feedback(bullet_id, task_type, helpful)

    async def add_bullets(
        self,
        task_type: str,
        new_insights: list[Any],
        existing_bullets: list[Any] | None = None,
    ) -> dict[str, int]:
        """
        Add multiple insights as bullets with quality validation.

        This is a convenience wrapper for batch addition of insights.

        Args:
            task_type: Playbook type (coding, testing, etc.)
            new_insights: List of ExtractedInsight objects to add
            existing_bullets: Existing bullets for dedup context (optional)

        Returns:
            Dict with counts: added, deduplicated, quality_rejected
        """
        result = {"added": 0, "deduplicated": 0, "quality_rejected": 0}

        for insight in new_insights:
            # Handle both ExtractedInsight objects and raw strings
            if hasattr(insight, "content"):
                content = insight.content
                section = self._insight_type_to_section(insight.insight_type) if hasattr(insight, "insight_type") else "general_strategies"
            else:
                content = str(insight)
                section = "general_strategies"

            add_result = await self.add_bullet(
                task_type=task_type,
                content=content,
                section=section,
                source="reflector",
            )

            if add_result.get("added"):
                result["added"] += 1
            elif add_result.get("quality_rejected"):
                result["quality_rejected"] += 1
            elif add_result.get("deduplicated"):
                result["deduplicated"] += 1

        log.debug(
            "bullets_batch_added",
            task_type=task_type,
            total=len(new_insights),
            **result,
        )
        return result

    async def run_maintenance(
        self,
        task_type: str | None = None,
        max_age_days: int = 90,
        min_utility: float = 0.3,
        merge_threshold: float = 0.85,
    ) -> dict:
        """
        Run periodic playbook maintenance.

        Operations:
        1. Prune stale/low-utility bullets
        2. Merge semantically similar bullets
        3. Report stats

        Args:
            task_type: Specific playbook, or None for all
            max_age_days: Prune bullets older than this without usage
            min_utility: Prune bullets with utility below this
            merge_threshold: Similarity threshold for merging

        Returns:
            Stats on operations performed
        """
        stats = {
            "pruned": 0,
            "merged": 0,
            "task_types_processed": [],
        }

        # Get task types to process
        task_types = [task_type] if task_type else self._get_all_task_types()

        for tt in task_types:
            bullets = self.playbook.load_playbook(tt)
            if not bullets:
                continue

            stats["task_types_processed"].append(tt)

            # 1. Prune low-utility bullets
            pruned = self.playbook.prune_low_utility(
                task_type=tt,
                threshold=min_utility,
                min_uses=3,  # Only prune if used enough to judge
            )
            stats["pruned"] += pruned

            # 2. Find and merge similar bullets
            if len(bullets) > 1:
                clusters = self.dedup.find_clusters(bullets, merge_threshold)
                for cluster in clusters:
                    if len(cluster.bullets) > 1:
                        # Keep the centroid, remove others
                        for bullet in cluster.bullets:
                            if bullet.id != cluster.centroid_id:
                                self._remove_bullet(tt, bullet.id)
                                stats["merged"] += 1

        log.info(
            "maintenance_complete",
            pruned=stats["pruned"],
            merged=stats["merged"],
            task_types=len(stats["task_types_processed"]),
        )

        return stats

    def _get_all_task_types(self) -> list[str]:
        """Get all task types with playbooks."""
        from pathlib import Path

        playbook_dir = Path(self.playbook.playbook_dir)
        if not playbook_dir.exists():
            return []

        return [
            p.stem for p in playbook_dir.glob("*.json")
            if not p.name.startswith("_")
        ]

    def _remove_bullet(self, task_type: str, bullet_id: str) -> bool:
        """Remove a bullet by ID."""
        bullets = self.playbook.load_playbook(task_type)
        original_count = len(bullets)
        bullets = [b for b in bullets if b.id != bullet_id]

        if len(bullets) < original_count:
            self.playbook.save_playbook(task_type, bullets)
            self.dedup.invalidate_cache(bullet_id)
            return True
        return False

    async def merge_similar_bullets(
        self,
        task_type: str,
        similarity_threshold: float = 0.85,
        dry_run: bool = True,
    ) -> list[tuple[str, str, str]]:
        """
        Find and optionally merge semantically similar bullets.

        Args:
            task_type: Playbook to analyze
            similarity_threshold: Minimum similarity to consider
            dry_run: If True, just report without merging

        Returns:
            List of (bullet_id_kept, bullet_id_removed, reason)
        """
        bullets = self.playbook.load_playbook(task_type)
        if len(bullets) < 2:
            return []

        clusters = self.dedup.find_clusters(bullets, similarity_threshold)
        merges = []

        for cluster in clusters:
            if len(cluster.bullets) < 2:
                continue

            # Keep centroid, report/remove others
            for bullet in cluster.bullets:
                if bullet.id != cluster.centroid_id:
                    merges.append((
                        cluster.centroid_id,
                        bullet.id,
                        f"Similarity: {cluster.avg_similarity:.2f}",
                    ))
                    if not dry_run:
                        self._remove_bullet(task_type, bullet.id)

        return merges


# Factory function to get Curator with project-specific manager
def get_curator(project_path: str | None = None) -> Curator:
    """
    Get a Curator instance for a specific project.

    Args:
        project_path: Project path, or None for cwd

    Returns:
        Curator wrapping project's PlaybookManager
    """
    from delia.playbook import get_playbook_manager

    manager = get_playbook_manager()
    if project_path:
        manager.set_project(project_path)

    return Curator(playbook_manager=manager)
