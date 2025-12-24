# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Learning Coordinator - Manages the Delia Framework learning pipeline.

Coordinates the Reflector→Curator→Playbook pipeline to ensure consistent
learning from task outcomes. Extracted from OrchestrationExecutor (P3).
"""

from __future__ import annotations

from typing import Any

import structlog

from ..learning.curator import Curator
from ..learning.reflector import Reflector
from ..playbook import PlaybookBullet, get_playbook_manager

log = structlog.get_logger()


class LearningCoordinator:
    """
    Coordinates the Delia Framework learning loop.

    Responsibilities:
    - Trigger reflection on task outcomes
    - Curate insights via Curator
    - Update playbooks with learned bullets
    - Track learning metrics
    """

    def __init__(
        self,
        reflector: Reflector | None = None,
        curator: Curator | None = None,
        reflection_model_tier: str = "coder",
    ):
        """
        Initialize the learning coordinator.

        Args:
            reflector: Optional Reflector instance (creates if None)
            curator: Optional Curator instance (creates if None)
            reflection_model_tier: Model tier for reflection (default: "coder" for quality)
                                  P5: Upgraded from "quick" to "coder" for better insights
        """
        self.playbook_manager = get_playbook_manager()
        self.reflector = reflector or Reflector(model_tier=reflection_model_tier)
        self.curator = curator or Curator(playbook_manager=self.playbook_manager)

    async def reflect_on_outcome(
        self,
        task_type: str,
        task_description: str,
        outcome: str,
        success: bool,
        error: str | None = None,
        applied_bullets: list[str] | None = None,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Reflect on a task outcome and learn from it.

        This is the main entry point for the Delia learning loop:
        1. Reflector extracts insights
        2. Curator deduplicates and validates
        3. Playbook manager persists learned bullets

        Args:
            task_type: Type of task (coding, testing, debugging, etc.)
            task_description: What was being attempted
            outcome: Result of the task
            success: Whether the task succeeded
            error: Optional error message if failed
            applied_bullets: Bullet IDs that were applied
            project_path: Optional project path for playbook storage

        Returns:
            Dictionary with reflection results and metrics
        """
        try:
            # Set project context if provided
            if project_path:
                self.playbook_manager.set_project(project_path)

            # Get top bullets for context
            top_bullets = self.playbook_manager.get_top_bullets(task_type, limit=5)
            applied_bullet_ids = applied_bullets or []

            # Extract insights via Reflector
            log.info(
                "triggering_reflection",
                task_type=task_type,
                success=success,
                applied_count=len(applied_bullet_ids),
            )

            reflection = await self.reflector.reflect(
                task_type=task_type,
                task_description=task_description,
                task_outcome=outcome,
                success=success,
                failure_reason=error,
            )

            # Curate insights and add to playbook
            if reflection.insights:
                curation_result = await self.curator.add_bullets(
                    task_type=task_type,
                    new_insights=reflection.insights,
                    existing_bullets=top_bullets,
                )

                log.info(
                    "reflection_completed",
                    insights_extracted=len(reflection.insights),
                    bullets_added=curation_result.get("added", 0),
                    bullets_deduped=curation_result.get("deduplicated", 0),
                )

                return {
                    "success": True,
                    "insights_extracted": len(reflection.insights),
                    "bullets_added": curation_result.get("added", 0),
                    "bullets_deduplicated": curation_result.get("deduplicated", 0),
                    "reflection": reflection,
                    "curation": curation_result,
                }
            else:
                log.info("reflection_no_insights", task_type=task_type)
                return {
                    "success": True,
                    "insights_extracted": 0,
                    "bullets_added": 0,
                    "message": "No actionable insights extracted",
                }

        except Exception as e:
            log.error(
                "reflection_failed",
                task_type=task_type,
                error=str(e),
                exc_info=True,
            )
            return {
                "success": False,
                "error": str(e),
            }

    async def record_bullet_feedback(
        self,
        task_type: str,
        bullet_ids: list[str],
        quality_score: float,
        project_path: str | None = None,
    ) -> None:
        """
        Record feedback on bullet effectiveness.

        Maps quality scores to helpful/harmful feedback:
        - quality >= 0.7: helpful
        - quality < 0.4: harmful

        Args:
            task_type: Task type the bullets were used for
            bullet_ids: List of bullet IDs that were applied
            quality_score: Task outcome quality (0.0-1.0)
            project_path: Optional project path
        """
        if not bullet_ids:
            return

        if project_path:
            self.playbook_manager.set_project(project_path)

        # Load bullets for this task type
        bullets = self.playbook_manager.load_playbook(task_type)

        # Map quality score to helpful/harmful
        helpful = quality_score >= 0.7

        # Find top 3 bullets to give credit to
        top_bullets = self.playbook_manager.get_top_bullets(task_type, limit=3)

        for bullet in top_bullets:
            self.playbook_manager.record_feedback(
                bullet_id=bullet.id,
                task_type=task_type,
                helpful=helpful,
            )

        log.info(
            "recorded_bullet_feedback",
            task_type=task_type,
            quality_score=quality_score,
            helpful=helpful,
            bullets_updated=len(top_bullets),
        )


# Global instance
_learning_coordinator: LearningCoordinator | None = None


def get_learning_coordinator(
    reflection_model_tier: str = "coder",
) -> LearningCoordinator:
    """
    Get the global LearningCoordinator instance.

    Args:
        reflection_model_tier: Model tier for reflection (default: "coder")
                              P5: Upgraded from "quick" for better reflection quality

    Returns:
        Global LearningCoordinator instance
    """
    global _learning_coordinator
    if _learning_coordinator is None:
        _learning_coordinator = LearningCoordinator(
            reflection_model_tier=reflection_model_tier
        )
    return _learning_coordinator
