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

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from delia.playbook import PlaybookManager, PlaybookBullet

from delia.learning.deduplication import SemanticDeduplicator, get_deduplicator
from delia.learning.reflector import ReflectionResult, ExtractedInsight, InsightType

log = structlog.get_logger()


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

        # Add the bullet using existing PlaybookManager (now with quality gate)
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
            from pathlib import Path
            project_path = Path(self.playbook.playbook_dir).parent
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
