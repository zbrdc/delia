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

"""
ACE-inspired Strategic Playbook Management.

This module implements the "Playbook" system for Delia, allowing it to
accumulate, refine, and organize strategies through incremental delta updates.

The Playbook consists of itemized 'bullets' (lessons, strategies, failure modes)
stored as JSON to prevent 'Context Collapse' caused by monolithic prompt rewriting.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from . import paths

log = structlog.get_logger()

@dataclass
class PlaybookBullet:
    """A single unit of strategic knowledge."""
    content: str
    id: str = field(default_factory=lambda: f"strat-{uuid.uuid4().hex[:8]}")
    section: str = "general_strategies"  # e.g., 'coding', 'api_usage', 'safety'
    helpful_count: int = 0
    harmful_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str | None = None
    source_task: str | None = None  # Reference to the task that generated this
    
    @property
    def utility_score(self) -> float:
        """Calculate a utility score based on usage history."""
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5  # Neutral for new bullets
        return self.helpful_count / total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlaybookBullet":
        return cls(**data)


class PlaybookManager:
    """Manages the lifecycle of strategic playbooks."""
    
    def __init__(self, playbook_dir: Path | None = None):
        self.playbook_dir = playbook_dir or paths.DATA_DIR / "playbooks"
        self._ensure_dir()
        # Cache for loaded bullets {task_type: [bullets]}
        self._cache: dict[str, list[PlaybookBullet]] = {}

    def _ensure_dir(self):
        """Ensure the playbooks directory exists."""
        self.playbook_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, task_type: str) -> Path:
        """Get the file path for a specific task type playbook."""
        return self.playbook_dir / f"{task_type}.json"

    def load_playbook(self, task_type: str = "general") -> list[PlaybookBullet]:
        """Load bullets for a specific task type."""
        if task_type in self._cache:
            return self._cache[task_type]
            
        path = self._get_path(task_type)
        if not path.exists():
            return []
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
                bullets = [PlaybookBullet.from_dict(b) for b in data]
                self._cache[task_type] = bullets
                return bullets
        except Exception as e:
            log.error("playbook_load_failed", task_type=task_type, error=str(e))
            return []

    def save_playbook(self, task_type: str, bullets: list[PlaybookBullet]):
        """Persist a playbook to disk."""
        path = self._get_path(task_type)
        try:
            with open(path, "w") as f:
                json.dump([b.to_dict() for b in bullets], f, indent=2)
            self._cache[task_type] = bullets
            log.info("playbook_saved", task_type=task_type, count=len(bullets))
        except Exception as e:
            log.error("playbook_save_failed", task_type=task_type, error=str(e))

    def add_bullet(self, task_type: str, content: str, section: str = "general_strategies") -> PlaybookBullet:
        """Add a new strategic bullet to a playbook."""
        bullets = self.load_playbook(task_type)
        
        # Simple deduplication check
        for b in bullets:
            if b.content.strip().lower() == content.strip().lower():
                log.debug("playbook_duplicate_skipped", content=content[:30])
                return b
                
        new_bullet = PlaybookBullet(content=content, section=section, source_task=task_type)
        bullets.append(new_bullet)
        self.save_playbook(task_type, bullets)
        return new_bullet

    def record_feedback(self, bullet_id: str, task_type: str, helpful: bool):
        """Update the helpful/harmful count for a bullet."""
        bullets = self.load_playbook(task_type)
        for b in bullets:
            if b.id == bullet_id:
                if helpful:
                    b.helpful_count += 1
                else:
                    b.harmful_count += 1
                b.last_used = datetime.now().isoformat()
                self.save_playbook(task_type, bullets)
                return True
        return False

    def get_top_bullets(self, task_type: str, limit: int = 5) -> list[PlaybookBullet]:
        """Retrieve the most relevant/helpful bullets for a task."""
        bullets = self.load_playbook(task_type)
        # Sort by utility score descending, then by last_used
        sorted_bullets = sorted(
            bullets, 
            key=lambda x: (x.utility_score, x.last_used or ""), 
            reverse=True
        )
        return sorted_bullets[:limit]

    def format_for_prompt(self, task_type: str, limit: int = 5) -> str:
        """Format the top bullets into a string for LLM injection."""
        bullets = self.get_top_bullets(task_type, limit)
        if not bullets:
            return ""
            
        formatted = "### STRATEGIC PLAYBOOK (Learned Lessons)\n"
        for b in bullets:
            # ACE-style bullet format: [ID] content
            formatted += f"- [{b.id}] {b.content}\n"
        return formatted

# Global manager instance
playbook_manager = PlaybookManager()
