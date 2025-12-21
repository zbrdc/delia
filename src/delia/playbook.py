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
    """Manages the lifecycle of strategic playbooks.

    Playbooks are ALWAYS project-specific, stored in .delia/playbooks/
    relative to the current working directory. No global fallback.
    """

    def __init__(self, playbook_dir: Path | None = None):
        # Force project-specific: .delia/playbooks/ in CWD
        self.playbook_dir = playbook_dir or (Path.cwd() / ".delia" / "playbooks")
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

    def get_all_bullets(self) -> dict[str, list[PlaybookBullet]]:
        """Get all bullets across all task types for MCP exposure."""
        result = {}
        for path in self.playbook_dir.glob("*.json"):
            task_type = path.stem
            result[task_type] = self.load_playbook(task_type)
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get playbook statistics for monitoring."""
        all_bullets = self.get_all_bullets()
        total = sum(len(b) for b in all_bullets.values())
        by_utility = []
        for task_type, bullets in all_bullets.items():
            for b in bullets:
                by_utility.append({
                    "id": b.id,
                    "task_type": task_type,
                    "utility": b.utility_score,
                    "helpful": b.helpful_count,
                    "harmful": b.harmful_count,
                })
        by_utility.sort(key=lambda x: x["utility"], reverse=True)
        return {
            "total_bullets": total,
            "task_types": list(all_bullets.keys()),
            "top_bullets": by_utility[:10],
            "low_utility": [b for b in by_utility if b["utility"] < 0.3],
        }

    def prune_low_utility(self, task_type: str, threshold: float = 0.3, min_uses: int = 5) -> int:
        """Remove bullets with low utility scores (if used enough to be confident)."""
        bullets = self.load_playbook(task_type)
        original_count = len(bullets)

        # Only prune if we have enough usage data
        pruned = [
            b for b in bullets
            if (b.helpful_count + b.harmful_count) < min_uses or b.utility_score >= threshold
        ]

        if len(pruned) < original_count:
            self.save_playbook(task_type, pruned)
            log.info("playbook_pruned", task_type=task_type, removed=original_count - len(pruned))

        return original_count - len(pruned)


# Global manager instance
playbook_manager = PlaybookManager()


# =============================================================================
# Project Playbook Generation (Called by `delia index`)
# =============================================================================

def detect_tech_stack(file_paths: list[str], dependencies: list[str]) -> dict[str, Any]:
    """Detect the project's tech stack from file extensions and dependencies."""
    extensions = {}
    for path in file_paths:
        ext = Path(path).suffix.lower()
        if ext:
            extensions[ext] = extensions.get(ext, 0) + 1

    # Detect primary language
    lang_map = {
        ".py": "Python",
        ".ts": "TypeScript", ".tsx": "TypeScript",
        ".js": "JavaScript", ".jsx": "JavaScript",
        ".rs": "Rust",
        ".go": "Go",
        ".java": "Java",
    }

    primary_lang = None
    max_count = 0
    for ext, count in extensions.items():
        if ext in lang_map and count > max_count:
            max_count = count
            primary_lang = lang_map[ext]

    # Detect frameworks from dependencies
    frameworks = []
    dep_set = {d.lower() for d in dependencies}

    framework_patterns = {
        "fastapi": "FastAPI",
        "flask": "Flask",
        "django": "Django",
        "pydantic": "Pydantic",
        "sqlalchemy": "SQLAlchemy",
        "react": "React",
        "vue": "Vue",
        "express": "Express",
        "mcp": "MCP (Model Context Protocol)",
        "structlog": "structlog",
        "httpx": "httpx",
        "asyncio": "asyncio",
    }

    for pattern, name in framework_patterns.items():
        if any(pattern in d for d in dep_set):
            frameworks.append(name)

    return {
        "primary_language": primary_lang,
        "extensions": extensions,
        "frameworks": frameworks,
        "is_async": "asyncio" in dep_set or "httpx" in dep_set,
    }


def generate_project_bullets(
    tech_stack: dict[str, Any],
    file_summaries: dict[str, Any],
    project_root: str,
) -> list[PlaybookBullet]:
    """Generate initial playbook bullets from project analysis."""
    bullets = []

    # 1. Tech stack bullets
    if tech_stack.get("primary_language"):
        bullets.append(PlaybookBullet(
            content=f"This project uses {tech_stack['primary_language']} as its primary language.",
            section="project_context",
            source_task="auto_generated",
        ))

    if tech_stack.get("frameworks"):
        frameworks = ", ".join(tech_stack["frameworks"])
        bullets.append(PlaybookBullet(
            content=f"Key frameworks/libraries: {frameworks}",
            section="project_context",
            source_task="auto_generated",
        ))

    if tech_stack.get("is_async"):
        bullets.append(PlaybookBullet(
            content="This project uses async/await patterns. Prefer async def for I/O operations.",
            section="coding_patterns",
            source_task="auto_generated",
        ))

    # 2. Project structure bullets
    if file_summaries:
        # Group by directory to understand structure
        dirs = {}
        for path in file_summaries.keys():
            parent = str(Path(path).parent)
            if parent not in dirs:
                dirs[parent] = []
            dirs[parent].append(Path(path).name)

        # Add key directory hints
        key_dirs = ["src", "tests", "lib", "api", "models", "utils", "services"]
        for key_dir in key_dirs:
            for dir_path, files in dirs.items():
                if key_dir in dir_path.lower():
                    bullets.append(PlaybookBullet(
                        content=f"The `{dir_path}/` directory contains {len(files)} files related to {key_dir}.",
                        section="project_structure",
                        source_task="auto_generated",
                    ))
                    break

    # 3. Pydantic pattern detection
    if "Pydantic" in tech_stack.get("frameworks", []):
        bullets.append(PlaybookBullet(
            content="Use Pydantic models for data validation and configuration. Check existing models before creating new ones.",
            section="coding_patterns",
            source_task="auto_generated",
        ))

    # 4. Testing patterns
    if any("test" in str(p).lower() for p in file_summaries.keys()):
        bullets.append(PlaybookBullet(
            content="This project has tests. Run `uv run pytest` or equivalent before committing changes.",
            section="verification",
            source_task="auto_generated",
        ))

    return bullets


async def generate_project_playbook(summarizer: Any = None) -> int:
    """
    Generate a project-specific playbook from codebase analysis.

    Called by `delia index` after summarization completes.
    Returns the number of bullets generated.
    """
    from .orchestration.summarizer import get_summarizer

    if summarizer is None:
        summarizer = get_summarizer()

    if not summarizer.summaries:
        log.warning("no_summaries_for_playbook", msg="Run delia index --summarize first")
        return 0

    # Collect data for analysis
    file_paths = list(summarizer.summaries.keys())
    all_deps = []
    for s in summarizer.summaries.values():
        all_deps.extend(s.dependencies)

    # Detect tech stack
    tech_stack = detect_tech_stack(file_paths, all_deps)

    # Generate bullets
    bullets = generate_project_bullets(
        tech_stack=tech_stack,
        file_summaries=summarizer.summaries,
        project_root=str(Path.cwd()),
    )

    # Store in playbook manager under "project" task type
    for bullet in bullets:
        playbook_manager.add_bullet("project", bullet.content, bullet.section)

    log.info("project_playbook_generated", count=len(bullets), tech_stack=tech_stack)
    return len(bullets)
