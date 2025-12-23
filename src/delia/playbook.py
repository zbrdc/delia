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
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field, computed_field

from . import paths

log = structlog.get_logger()


def _generate_bullet_id() -> str:
    return f"strat-{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now().isoformat()


class PlaybookBullet(BaseModel):
    """A single unit of strategic knowledge."""
    content: str
    id: str = Field(default_factory=_generate_bullet_id)
    section: str = Field(default="general_strategies", description="e.g., 'coding', 'api_usage', 'safety'")
    helpful_count: int = Field(default=0, ge=0)
    harmful_count: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=_now_iso)
    last_used: str | None = None
    source_task: str | None = Field(default=None, description="Reference to the task that generated this")
    source: Literal["seed", "learned", "manual", "reflector", "curator"] = Field(
        default="learned",
        description="Origin: 'seed' from profiles, 'learned' from feedback, 'manual' from user, 'reflector'/'curator' from ACE"
    )

    @computed_field
    @property
    def utility_score(self) -> float:
        """Calculate a utility score based on usage history."""
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5  # Neutral for new bullets
        return self.helpful_count / total

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlaybookBullet":
        # Map alternative field names from agent-written files
        mapped = {}
        field_mapping = {
            "created": "created_at",
            "use_count": "helpful_count",  # Approximate mapping
        }
        
        # Fields that are properties/computed, not stored
        skip_fields = {"utility_score"}
        
        for key, value in data.items():
            if key in skip_fields:
                continue
            mapped_key = field_mapping.get(key, key)
            mapped[mapped_key] = value
        
        return cls(**mapped)


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
    
    def set_project(self, project_path: Path | str):
        """Switch to a different project's playbooks.
        
        Call this when the working project changes (e.g., MCP request from different dir).
        Clears the cache and updates the playbook directory.
        """
        project_path = Path(project_path) if isinstance(project_path, str) else project_path
        new_dir = project_path / ".delia" / "playbooks"
        if new_dir != self.playbook_dir:
            self.playbook_dir = new_dir
            self._cache.clear()
            self._ensure_dir()
            log.info("playbook_project_switched", path=str(new_dir))

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
                
                # Handle both formats:
                # 1. Flat array: [{"content": ...}, ...]
                # 2. Wrapped object: {"bullets": [...], "playbook": ...}
                if isinstance(data, dict) and "bullets" in data:
                    bullet_list = data["bullets"]
                elif isinstance(data, list):
                    bullet_list = data
                else:
                    log.warning("playbook_unknown_format", task_type=task_type)
                    return []
                
                bullets = [PlaybookBullet.from_dict(b) for b in bullet_list]
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

    def add_bullet(
        self,
        task_type: str,
        content: str,
        section: str = "general_strategies",
        source: Literal["seed", "learned", "manual", "reflector", "curator"] = "learned",
    ) -> PlaybookBullet:
        """Add a new strategic bullet to a playbook."""
        bullets = self.load_playbook(task_type)

        # Simple deduplication check
        for b in bullets:
            if b.content.strip().lower() == content.strip().lower():
                log.debug("playbook_duplicate_skipped", content=content[:30])
                return b

        new_bullet = PlaybookBullet(
            content=content,
            section=section,
            source_task=task_type,
            source=source,
        )
        bullets.append(new_bullet)
        self.save_playbook(task_type, bullets)
        return new_bullet

    def delete_bullet(self, task_type: str, bullet_id: str) -> bool:
        """Delete a bullet from a playbook by its ID.

        Returns True if the bullet was found and deleted, False otherwise.
        """
        bullets = self.load_playbook(task_type)
        original_count = len(bullets)

        # Filter out the bullet with matching ID
        bullets = [b for b in bullets if b.id != bullet_id]

        if len(bullets) < original_count:
            self.save_playbook(task_type, bullets)
            log.info("playbook_bullet_deleted", task_type=task_type, bullet_id=bullet_id)
            return True

        log.warning("playbook_bullet_not_found", task_type=task_type, bullet_id=bullet_id)
        return False

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

    def detect_task_type(self, content: str) -> str:
        """Detect the task type from content for automatic playbook selection.
        
        Uses keyword matching to determine which playbook is most relevant.
        """
        content_lower = content.lower()
        
        # Keywords for each task type (ordered by specificity)
        patterns = [
            ("testing", ["test", "pytest", "coverage", "assert", "mock", "fixture", "spec"]),
            ("debugging", ["error", "bug", "fix", "stack trace", "exception", "crash", "fail"]),
            ("security", ["auth", "security", "password", "token", "encrypt", "vulnerability", "xss", "sql injection"]),
            ("deployment", ["deploy", "ci/cd", "docker", "kubernetes", "pipeline", "release", "production"]),
            ("performance", ["optimize", "performance", "slow", "latency", "cache", "profil", "benchmark"]),
            ("architecture", ["design", "architect", "adr", "refactor", "pattern", "structure", "module"]),
            ("api", ["api", "endpoint", "rest", "graphql", "request", "response", "route"]),
            ("git", ["commit", "branch", "merge", "pull request", "pr", "rebase", "git"]),
            ("coding", ["implement", "add", "create", "write", "function", "class", "method", "code"]),
        ]
        
        for task_type, keywords in patterns:
            for kw in keywords:
                if kw in content_lower:
                    return task_type
        
        # Default to coding if no pattern matches
        return "coding"

    def get_auto_injection_bullets(
        self, 
        content: str, 
        include_project: bool = True,
        limit: int = 5,
    ) -> str:
        """Get automatically-selected playbook bullets for prompt injection.
        
        This is the primary entry point for ACE Framework enforcement.
        Called by ContextEngine.prepare_content() to ensure playbook guidance
        is ALWAYS included in prompts.
        
        Args:
            content: The task/prompt to analyze
            include_project: Also include project-specific bullets
            limit: Max bullets per category
        
        Returns:
            Formatted string for injection, or empty string if no bullets
        """
        task_type = self.detect_task_type(content)
        
        parts = []
        
        # Task-specific bullets
        task_bullets = self.get_top_bullets(task_type, limit)
        if task_bullets:
            parts.append(f"#### {task_type.title()} Guidance")
            for b in task_bullets:
                parts.append(f"- [{b.id}] {b.content}")
        
        # Always include project bullets for context
        if include_project:
            project_bullets = self.get_top_bullets("project", limit=3)
            if project_bullets:
                parts.append("#### Project Context")
                for b in project_bullets:
                    parts.append(f"- [{b.id}] {b.content}")
        
        if not parts:
            return ""
        
        header = "### ACE PLAYBOOK (Apply These Learned Strategies)"
        footer = "_Report feedback with: report_feedback(bullet_id, helpful=True/False)_"
        
        return f"{header}\n" + "\n".join(parts) + f"\n\n{footer}"

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


def get_playbook_manager() -> PlaybookManager:
    """Get the global playbook manager instance."""
    return playbook_manager


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
    # P4: Use curator to ensure semantic deduplication
    from .ace.curator import Curator
    curator = Curator(playbook_manager=playbook_manager)

    added_count = 0
    dedup_count = 0

    for bullet in bullets:
        try:
            result = await curator.add_bullet(
                task_type="project",
                content=bullet.content,
                section=bullet.section,
            )
            if result.get("added"):
                added_count += 1
            elif result.get("deduplicated"):
                dedup_count += 1
        except Exception as e:
            log.warning("curator_add_failed_project_bullet", error=str(e), content=bullet.content[:50])

    log.info(
        "project_playbook_generated",
        total=len(bullets),
        added=added_count,
        deduplicated=dedup_count,
        tech_stack=tech_stack
    )
    return added_count


# =============================================================================
# PROFILE RECOMMENDATIONS
# =============================================================================

# Mapping of tech stack patterns to relevant profiles
PROFILE_MAPPINGS: dict[str, list[str]] = {
    # Languages - ONLY the base language profile, not framework profiles
    "Python": ["python.md"],
    "TypeScript": ["typescript.md"],
    "JavaScript": ["typescript.md"],  # Use TS profile for JS too
    "Go": ["golang.md"],
    "Rust": ["rust.md"],
    "Swift": ["ios.md"],
    "Kotlin": ["android.md"],
    "Dart": ["flutter.md"],
    "PHP": ["laravel.md"],
    "Solidity": ["solidity.md"],
    "C": ["c.md"],
    "C++": ["cpp.md"],
    # Frameworks - detected from dependencies, add specific profiles
    "FastAPI": ["fastapi.md", "api.md"],
    "Django": ["django.md", "api.md"],
    "Flask": ["api.md"],
    "React": ["react.md"],
    "Vue": ["vue.md"],
    "Next.js": ["nextjs.md", "react.md"],
    "NestJS": ["nestjs.md", "api.md"],
    "Express": ["api.md"],
    "Svelte": ["svelte.md"],
    "Angular": ["angular.md"],
    "Pydantic": ["fastapi.md"],
    "SQLAlchemy": ["api.md"],
    "PyTorch": ["deeplearning.md", "ml.md"],
    "TensorFlow": ["deeplearning.md", "ml.md"],
    "scikit-learn": ["ml.md"],
    "LangChain": ["llm.md"],
    "Playwright": ["testing.md"],
    "pytest": ["testing.md"],
    "Jest": ["testing.md"],
}

# Best practice patterns to check for
BEST_PRACTICE_PATTERNS: dict[str, dict[str, Any]] = {
    "python": {
        "type_hints": {
            "pattern": r"def \w+\([^)]*\)\s*->",
            "description": "Function type hints",
            "recommendation": "Add return type hints to functions for better code documentation and IDE support",
        },
        "async_patterns": {
            "pattern": r"async def",
            "description": "Async/await usage",
            "recommendation": "Consider using async/await for I/O-bound operations",
        },
        "pydantic_models": {
            "pattern": r"class \w+\(BaseModel\)",
            "description": "Pydantic models",
            "recommendation": "Use Pydantic models for data validation and configuration",
        },
        "structured_logging": {
            "pattern": r"structlog|log\.\w+\([\"']\w+[\"'],",
            "description": "Structured logging",
            "recommendation": "Use structured logging (structlog) with event names and context",
        },
    },
    "typescript": {
        "strict_types": {
            "pattern": r'"strict":\s*true',
            "description": "Strict TypeScript mode",
            "recommendation": "Enable strict mode in tsconfig.json for better type safety",
        },
        "interface_usage": {
            "pattern": r"interface \w+",
            "description": "Interface definitions",
            "recommendation": "Use interfaces for object shapes and contracts",
        },
    },
    "react": {
        "memo_usage": {
            "pattern": r"React\.memo|useMemo|useCallback",
            "description": "Memoization patterns",
            "recommendation": "Use React.memo, useMemo, and useCallback for performance optimization",
        },
        "error_boundaries": {
            "pattern": r"ErrorBoundary|componentDidCatch",
            "description": "Error boundaries",
            "recommendation": "Implement error boundaries to catch and handle component errors",
        },
    },
    "testing": {
        "test_coverage": {
            "pattern": r"@pytest\.mark|describe\(|it\(|test\(",
            "description": "Test definitions",
            "recommendation": "Ensure adequate test coverage for critical paths",
        },
    },
}


class ProfileRecommendation(BaseModel):
    """A profile recommendation with rationale."""
    profile: str
    reason: str
    priority: Literal["high", "medium", "low"]
    detected_by: str = Field(description="What triggered this recommendation")


class PatternGap(BaseModel):
    """A gap between current code and best practices."""
    category: str
    pattern_name: str
    description: str
    recommendation: str
    current_usage: int = Field(ge=0, description="How many times pattern is found")
    severity: Literal["info", "warning", "suggestion"]


def recommend_profiles(
    tech_stack: dict[str, Any],
    project_root: Path | None = None,
) -> list[ProfileRecommendation]:
    """
    Recommend relevant profiles based on detected tech stack.

    Args:
        tech_stack: Output from detect_tech_stack()
        project_root: Project root for checking existing profiles

    Returns:
        List of profile recommendations with rationale
    """
    recommendations: list[ProfileRecommendation] = []
    seen_profiles: set[str] = set()

    # Universal profiles - essential for ALL projects (cross-cutting concerns)
    universal_profiles = [
        ("core.md", "Essential patterns for all projects"),
        ("coding.md", "Code quality standards"),
        ("git.md", "Version control best practices"),
        ("architecture.md", "Architectural patterns"),
        ("security.md", "Security is always relevant"),
        ("testing.md", "Testing is always relevant"),
        ("performance.md", "Performance considerations"),
        ("deployment.md", "Deployment patterns"),
    ]
    for profile, reason in universal_profiles:
        if profile not in seen_profiles:
            recommendations.append(ProfileRecommendation(
                profile=profile,
                reason=reason,
                priority="high",
                detected_by="universal",
            ))
            seen_profiles.add(profile)

    # Recommend based on primary language
    primary_lang = tech_stack.get("primary_language")
    if primary_lang and primary_lang in PROFILE_MAPPINGS:
        for profile in PROFILE_MAPPINGS[primary_lang]:
            if profile not in seen_profiles:
                recommendations.append(ProfileRecommendation(
                    profile=profile,
                    reason=f"Primary language is {primary_lang}",
                    priority="high",
                    detected_by=f"language:{primary_lang}",
                ))
                seen_profiles.add(profile)

    # Recommend based on detected frameworks
    frameworks = tech_stack.get("frameworks", [])
    for framework in frameworks:
        if framework in PROFILE_MAPPINGS:
            for profile in PROFILE_MAPPINGS[framework]:
                if profile not in seen_profiles:
                    recommendations.append(ProfileRecommendation(
                        profile=profile,
                        reason=f"Project uses {framework}",
                        priority="medium",
                        detected_by=f"framework:{framework}",
                    ))
                    seen_profiles.add(profile)

    # Add API profile for backend frameworks
    api_frameworks = {"FastAPI", "Django", "Flask", "Express", "NestJS"}
    if any(f in frameworks for f in api_frameworks) and "api.md" not in seen_profiles:
        recommendations.append(ProfileRecommendation(
            profile="api.md",
            reason="API framework detected",
            priority="medium",
            detected_by="api_framework",
        ))
        seen_profiles.add("api.md")

    log.info("profiles_recommended", count=len(recommendations), tech_stack=tech_stack)
    return recommendations


def analyze_pattern_gaps(
    project_root: Path,
    tech_stack: dict[str, Any],
) -> list[PatternGap]:
    """
    Analyze project code for best practice pattern gaps.

    Scans the codebase for presence of recommended patterns
    and identifies areas for improvement.

    Args:
        project_root: Root directory of the project
        tech_stack: Detected tech stack

    Returns:
        List of pattern gaps with recommendations
    """
    import re
    from .orchestration.constants import CODE_EXTENSIONS, IGNORE_DIRS

    gaps: list[PatternGap] = []
    primary_lang = tech_stack.get("primary_language", "").lower()

    # Determine which pattern categories to check
    categories_to_check = ["testing"]  # Always check testing
    if primary_lang == "python":
        categories_to_check.append("python")
    elif primary_lang == "typescript":
        categories_to_check.extend(["typescript", "react"])

    # Collect all code content for scanning
    code_content = ""
    code_files = 0
    for file_path in project_root.rglob("*"):
        if any(part in IGNORE_DIRS for part in file_path.parts):
            continue
        if file_path.suffix in CODE_EXTENSIONS and file_path.is_file():
            try:
                code_content += file_path.read_text(errors="ignore") + "\n"
                code_files += 1
            except Exception:
                continue

    if not code_content:
        return gaps

    # Check each pattern category
    for category in categories_to_check:
        if category not in BEST_PRACTICE_PATTERNS:
            continue

        for pattern_name, pattern_info in BEST_PRACTICE_PATTERNS[category].items():
            try:
                matches = len(re.findall(pattern_info["pattern"], code_content))
            except re.error:
                continue

            # Determine if this is a gap
            # Low usage relative to code size might indicate a gap
            expected_min = code_files // 5  # Expect at least 1 per 5 files

            if matches < expected_min:
                severity = "suggestion" if matches > 0 else "warning"
                gaps.append(PatternGap(
                    category=category,
                    pattern_name=pattern_name,
                    description=pattern_info["description"],
                    recommendation=pattern_info["recommendation"],
                    current_usage=matches,
                    severity=severity,
                ))

    log.info("pattern_gaps_analyzed", gaps=len(gaps), files_scanned=code_files)
    return gaps


def format_recommendations(
    recommendations: list[ProfileRecommendation],
    gaps: list[PatternGap] | None = None,
) -> dict[str, Any]:
    """
    Format recommendations and gaps for display.

    Returns:
        Dict with recommendations grouped by priority and gaps
    """
    result: dict[str, Any] = {
        "high_priority": [],
        "medium_priority": [],
        "low_priority": [],
        "gaps": [],
    }

    for rec in recommendations:
        entry = {
            "profile": rec.profile,
            "reason": rec.reason,
            "detected_by": rec.detected_by,
        }
        if rec.priority == "high":
            result["high_priority"].append(entry)
        elif rec.priority == "medium":
            result["medium_priority"].append(entry)
        else:
            result["low_priority"].append(entry)

    if gaps:
        for gap in gaps:
            result["gaps"].append({
                "category": gap.category,
                "pattern": gap.pattern_name,
                "description": gap.description,
                "recommendation": gap.recommendation,
                "current_usage": gap.current_usage,
                "severity": gap.severity,
            })

    result["summary"] = {
        "total_recommendations": len(recommendations),
        "high_priority_count": len(result["high_priority"]),
        "gaps_found": len(result["gaps"]) if gaps else 0,
    }

    return result


# =============================================================================
# AUTOMATIC RE-EVALUATION SYSTEM
# =============================================================================

class EvaluationState(BaseModel):
    """Tracks the state of pattern/profile evaluations."""
    last_evaluation: str = Field(description="ISO timestamp")
    lines_at_evaluation: int = Field(ge=0, description="Total lines of code at last evaluation")
    commit_hash: str | None = Field(default=None, description="Git commit hash at evaluation")
    recommendations_count: int = Field(default=0, ge=0)
    gaps_count: int = Field(default=0, ge=0)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationState":
        return cls.model_validate(data)


# Thresholds for triggering re-evaluation
REEVALUATION_THRESHOLDS = {
    "lines_changed": 3000,  # Re-evaluate after 3k lines changed
    "days_inactive": 30,  # Re-evaluate if not checked in 30 days
    "commits_since": 50,  # Re-evaluate after 50 commits
}


def get_evaluation_state_path(project_root: Path) -> Path:
    """Get path to evaluation state file."""
    return project_root / ".delia" / "evaluation_state.json"


def load_evaluation_state(project_root: Path) -> EvaluationState | None:
    """Load the last evaluation state from disk."""
    state_path = get_evaluation_state_path(project_root)
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            data = json.load(f)
        return EvaluationState.from_dict(data)
    except Exception as e:
        log.warning("evaluation_state_load_failed", error=str(e))
        return None


def save_evaluation_state(project_root: Path, state: EvaluationState) -> None:
    """Save evaluation state to disk."""
    state_path = get_evaluation_state_path(project_root)
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        log.debug("evaluation_state_saved", path=str(state_path))
    except Exception as e:
        log.warning("evaluation_state_save_failed", error=str(e))


def count_code_lines(project_root: Path) -> int:
    """Count total lines of code in the project (excluding tests)."""
    from .orchestration.constants import CODE_EXTENSIONS, should_ignore_file

    total_lines = 0
    for file_path in project_root.rglob("*"):
        if should_ignore_file(file_path):
            continue
        if file_path.suffix in CODE_EXTENSIONS and file_path.is_file():
            try:
                total_lines += len(file_path.read_text(errors="ignore").splitlines())
            except Exception:
                continue
    return total_lines


def get_current_commit(project_root: Path) -> str | None:
    """Get current git commit hash if in a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def count_commits_since(project_root: Path, since_commit: str) -> int:
    """Count commits since a given commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", f"{since_commit}..HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


def check_reevaluation_needed(project_root: Path) -> dict[str, Any]:
    """
    Check if pattern/profile re-evaluation is needed.

    Triggers re-evaluation based on:
    - Lines of code changed (threshold: 15,000)
    - Days since last evaluation (threshold: 30 days)
    - Commits since last evaluation (threshold: 50)

    Returns:
        Dict with needs_reevaluation bool and reasons
    """
    state = load_evaluation_state(project_root)

    result = {
        "needs_reevaluation": False,
        "reasons": [],
        "state": None,
        "current": {},
    }

    # If no previous evaluation, definitely need one
    if state is None:
        result["needs_reevaluation"] = True
        result["reasons"].append("No previous evaluation found")
        return result

    result["state"] = state.to_dict()

    # Check lines changed
    current_lines = count_code_lines(project_root)
    result["current"]["lines"] = current_lines
    lines_changed = abs(current_lines - state.lines_at_evaluation)

    if lines_changed >= REEVALUATION_THRESHOLDS["lines_changed"]:
        result["needs_reevaluation"] = True
        result["reasons"].append(
            f"Code changed by {lines_changed:,} lines (threshold: {REEVALUATION_THRESHOLDS['lines_changed']:,})"
        )

    # Check days since last evaluation
    try:
        last_eval = datetime.fromisoformat(state.last_evaluation)
        days_since = (datetime.now() - last_eval).days
        result["current"]["days_since_evaluation"] = days_since

        if days_since >= REEVALUATION_THRESHOLDS["days_inactive"]:
            result["needs_reevaluation"] = True
            result["reasons"].append(
                f"Last evaluation was {days_since} days ago (threshold: {REEVALUATION_THRESHOLDS['days_inactive']})"
            )
    except Exception:
        pass

    # Check commits since last evaluation
    if state.commit_hash:
        commits_since = count_commits_since(project_root, state.commit_hash)
        result["current"]["commits_since"] = commits_since

        if commits_since >= REEVALUATION_THRESHOLDS["commits_since"]:
            result["needs_reevaluation"] = True
            result["reasons"].append(
                f"{commits_since} commits since last evaluation (threshold: {REEVALUATION_THRESHOLDS['commits_since']})"
            )

    log.info(
        "reevaluation_check",
        needs_reevaluation=result["needs_reevaluation"],
        reasons=result["reasons"],
    )
    return result


def run_reevaluation(
    project_root: Path,
    tech_stack: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Run pattern/profile re-evaluation if needed.

    Args:
        project_root: Project root path
        tech_stack: Pre-detected tech stack (optional)
        force: Force re-evaluation even if not needed

    Returns:
        Dict with evaluation results
    """
    result: dict[str, Any] = {
        "evaluated": False,
        "reason": None,
        "recommendations": [],
        "gaps": [],
    }

    # Check if re-evaluation is needed
    check = check_reevaluation_needed(project_root)
    if not force and not check["needs_reevaluation"]:
        result["reason"] = "Re-evaluation not needed"
        result["next_check"] = check
        return result

    result["evaluated"] = True
    result["trigger_reasons"] = check.get("reasons", ["forced"])

    # Detect tech stack if not provided
    if tech_stack is None:
        from .orchestration.constants import CODE_EXTENSIONS, IGNORE_DIRS

        # Collect file paths and dependencies
        code_files = []
        for file_path in project_root.rglob("*"):
            if any(part in IGNORE_DIRS for part in file_path.parts):
                continue
            if file_path.suffix in CODE_EXTENSIONS and file_path.is_file():
                code_files.append(str(file_path.relative_to(project_root)))

        # Extract dependencies
        dependencies: list[str] = []
        pyproject = project_root / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                deps = data.get("project", {}).get("dependencies", [])
                dependencies.extend(deps)
            except Exception:
                pass

        package_json = project_root / "package.json"
        if package_json.exists():
            try:
                pkg = json.loads(package_json.read_text())
                dependencies.extend(pkg.get("dependencies", {}).keys())
            except Exception:
                pass

        tech_stack = detect_tech_stack(code_files, dependencies)

    # Run recommendations
    recommendations = recommend_profiles(tech_stack, project_root)
    result["recommendations"] = format_recommendations(recommendations)["high_priority"]

    # Run gap analysis
    gaps = analyze_pattern_gaps(project_root, tech_stack)
    result["gaps"] = [
        {
            "pattern": g.pattern_name,
            "category": g.category,
            "recommendation": g.recommendation,
            "severity": g.severity,
        }
        for g in gaps
    ]

    # Clean up unnecessary profiles (report only, don't auto-remove)
    cleanup_result = cleanup_unnecessary_profiles(
        project_root, 
        tech_stack=tech_stack,
        auto_remove=False,  # Prompt user, don't auto-delete
    )
    if cleanup_result.obsolete:
        result["obsolete_profiles"] = cleanup_result.obsolete
        result["obsolete_reasons"] = cleanup_result.reason
        result["cleanup_action"] = (
            "The following profiles are no longer relevant to your tech stack. "
            "Consider removing them from .delia/profiles/ or run cleanup with auto_remove=True."
        )
    result["current_profiles"] = cleanup_result.kept

    # Save new evaluation state
    new_state = EvaluationState(
        last_evaluation=datetime.now().isoformat(),
        lines_at_evaluation=count_code_lines(project_root),
        commit_hash=get_current_commit(project_root),
        recommendations_count=len(recommendations),
        gaps_count=len(gaps),
    )
    save_evaluation_state(project_root, new_state)
    result["state_saved"] = True

    log.info(
        "reevaluation_complete",
        recommendations=len(recommendations),
        gaps=len(gaps),
        triggers=check.get("reasons", []),
    )

    return result


def prune_stale_bullets(
    project_root: Path,
    max_age_days: int = 90,
    min_utility: float = 0.3,
) -> dict[str, Any]:
    """
    Prune stale or low-utility bullets from playbooks.

    Removes bullets that:
    - Haven't been used in max_age_days
    - Have utility score below min_utility with sufficient usage

    Args:
        project_root: Project root path
        max_age_days: Maximum days without use before pruning
        min_utility: Minimum utility score to keep

    Returns:
        Dict with pruning statistics
    """
    manager = get_playbook_manager()
    manager.set_project(project_root)

    result = {
        "pruned": [],
        "kept": [],
        "by_task_type": {},
    }

    task_types = ["coding", "testing", "architecture", "debugging", "project",
                  "git", "security", "deployment", "api", "performance"]

    cutoff_date = datetime.now().isoformat()[:10]  # YYYY-MM-DD

    for task_type in task_types:
        bullets = manager.load_playbook(task_type)
        kept = []
        pruned = []

        for bullet in bullets:
            # Check age
            last_used = bullet.last_used or bullet.created_at
            try:
                last_used_date = datetime.fromisoformat(last_used[:10])
                age_days = (datetime.now() - last_used_date).days
            except Exception:
                age_days = 0

            # Check utility
            total_feedback = bullet.helpful_count + bullet.harmful_count
            utility = bullet.utility_score

            # Prune if: too old with no recent use, OR low utility with enough data
            should_prune = False
            prune_reason = None

            if age_days > max_age_days and bullet.helpful_count == 0:
                should_prune = True
                prune_reason = f"Unused for {age_days} days"
            elif total_feedback >= 5 and utility < min_utility:
                should_prune = True
                prune_reason = f"Low utility ({utility:.2f}) after {total_feedback} feedbacks"

            if should_prune:
                pruned.append({
                    "id": bullet.id,
                    "content": bullet.content[:50] + "..." if len(bullet.content) > 50 else bullet.content,
                    "reason": prune_reason,
                })
            else:
                kept.append(bullet)

        # Save pruned playbook
        if len(pruned) > 0:
            manager.save_playbook(task_type, kept)
            result["by_task_type"][task_type] = {
                "pruned": len(pruned),
                "kept": len(kept),
            }
            result["pruned"].extend(pruned)

    result["total_pruned"] = len(result["pruned"])
    log.info("bullets_pruned", total=len(result["pruned"]))

    return result


class ProfileCleanupResult(BaseModel):
    """Result of profile cleanup operation."""
    removed: list[str] = Field(default_factory=list, description="Profiles that were removed")
    kept: list[str] = Field(default_factory=list, description="Profiles that were kept")
    obsolete: list[str] = Field(default_factory=list, description="Profiles marked as obsolete (not auto-removed)")
    reason: dict[str, str] = Field(default_factory=dict, description="Reason for each removal/obsolete marking")


def cleanup_unnecessary_profiles(
    project_root: Path,
    tech_stack: dict[str, Any] | None = None,
    auto_remove: bool = False,
) -> ProfileCleanupResult:
    """
    Identify and optionally remove profiles that are no longer relevant.
    
    Compares existing profiles in .delia/profiles/ with currently recommended
    profiles based on tech stack. Marks obsolete ones for removal.
    
    Args:
        project_root: Project root path
        tech_stack: Pre-detected tech stack (optional, will detect if None)
        auto_remove: If True, automatically delete obsolete profiles.
                    If False (default), just report them for user decision.
    
    Returns:
        ProfileCleanupResult with removed/kept/obsolete profiles
    """
    result = ProfileCleanupResult()
    
    profiles_dir = project_root / ".delia" / "profiles"
    if not profiles_dir.exists():
        return result
    
    # Detect tech stack if not provided
    if tech_stack is None:
        from .orchestration.constants import CODE_EXTENSIONS, IGNORE_DIRS
        
        code_files = []
        for file_path in project_root.rglob("*"):
            if any(part in IGNORE_DIRS for part in file_path.parts):
                continue
            if file_path.suffix in CODE_EXTENSIONS and file_path.is_file():
                code_files.append(str(file_path.relative_to(project_root)))
        
        # Extract dependencies
        dependencies: list[str] = []
        pyproject = project_root / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                deps = data.get("project", {}).get("dependencies", [])
                dependencies.extend(deps)
            except Exception:
                pass
        
        package_json = project_root / "package.json"
        if package_json.exists():
            try:
                pkg = json.loads(package_json.read_text())
                dependencies.extend(pkg.get("dependencies", {}).keys())
            except Exception:
                pass
        
        tech_stack = detect_tech_stack(code_files, dependencies)
    
    # Get current recommendations
    recommendations = recommend_profiles(tech_stack, project_root)
    recommended_files = {r.profile for r in recommendations}
    
    # Check each existing profile
    for profile_file in profiles_dir.glob("*.md"):
        profile_name = profile_file.name
        
        if profile_name in recommended_files:
            result.kept.append(profile_name)
        else:
            # Profile is no longer recommended
            reason = _get_obsolete_reason(profile_name, tech_stack)
            result.reason[profile_name] = reason
            
            if auto_remove:
                try:
                    profile_file.unlink()
                    result.removed.append(profile_name)
                    log.info("profile_removed", profile=profile_name, reason=reason)
                except Exception as e:
                    log.warning("profile_remove_failed", profile=profile_name, error=str(e))
                    result.obsolete.append(profile_name)
            else:
                result.obsolete.append(profile_name)
    
    if result.removed or result.obsolete:
        log.info(
            "profile_cleanup_complete",
            removed=len(result.removed),
            obsolete=len(result.obsolete),
            kept=len(result.kept),
        )
    
    return result


def _get_obsolete_reason(profile_name: str, tech_stack: dict[str, Any]) -> str:
    """Get human-readable reason why a profile is obsolete."""
    # Map profile names to their tech stack requirements
    profile_requirements = {
        "react.md": "React",
        "vue.md": "Vue",
        "angular.md": "Angular",
        "nextjs.md": "Next.js",
        "nestjs.md": "NestJS",
        "fastapi.md": "FastAPI",
        "django.md": "Django",
        "flask.md": "Flask",
        "python.md": "Python",
        "typescript.md": "TypeScript",
        "rust.md": "Rust",
        "go.md": "Go",
        "java.md": "Java",
        "mcp.md": "MCP",
        "ai-ml.md": "AI/ML",
        "data-engineering.md": "Data Engineering",
    }
    
    if profile_name in profile_requirements:
        required = profile_requirements[profile_name]
        detected_langs = tech_stack.get("languages", [])
        detected_frameworks = tech_stack.get("frameworks", [])
        all_detected = detected_langs + detected_frameworks
        
        if required not in all_detected:
            return f"{required} not detected in project (found: {', '.join(all_detected[:3]) or 'none'})"
    
    return "No longer matches project tech stack"


# =============================================================================
# Profile to Playbook Seed Conversion (ACE Framework)
# =============================================================================

import re
from dataclasses import dataclass
from typing import Iterator


@dataclass
class ExtractedBullet:
    """A bullet extracted from a profile markdown file."""
    content: str
    section: str
    profile_source: str


def parse_profile_to_bullets(profile_content: str, profile_name: str) -> list[ExtractedBullet]:
    """
    Parse a profile markdown file and extract atomic bullet strategies.

    Extraction rules:
    1. Lines in ALWAYS/NEVER/AVOID blocks -> individual bullets
    2. Bullet points (- or *) that are actionable strategies
    3. Skip code examples, headings, and prose descriptions

    Args:
        profile_content: The markdown content of the profile
        profile_name: Name of the profile (e.g., "python.md")

    Returns:
        List of extracted bullets
    """
    bullets: list[ExtractedBullet] = []
    lines = profile_content.split("\n")

    current_section = "general"
    in_code_block = False
    in_rules_block = False  # ALWAYS/NEVER/AVOID block

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track code blocks to skip
        if stripped.startswith("```"):
            # Check if this is the start of an ALWAYS/NEVER block
            if not in_code_block:
                # Look ahead for rule patterns
                next_lines = "\n".join(lines[i+1:i+5])
                if re.search(r"^(ALWAYS|NEVER|AVOID|DO NOT):", next_lines, re.MULTILINE):
                    in_rules_block = True
            else:
                in_rules_block = False
            in_code_block = not in_code_block
            continue

        # Inside code block - check for rule patterns
        if in_code_block and in_rules_block:
            # Extract ALWAYS/NEVER/AVOID items
            if stripped.startswith("- "):
                content = stripped[2:].strip()
                if len(content) > 10 and not content.startswith("#"):
                    bullets.append(ExtractedBullet(
                        content=content,
                        section=current_section,
                        profile_source=profile_name,
                    ))
            continue

        # Skip regular code blocks
        if in_code_block:
            continue

        # Track section from headings
        if stripped.startswith("## "):
            section_name = stripped[3:].strip().lower()
            # Normalize section names
            section_map = {
                "core principles": "principles",
                "best practices": "best_practices",
                "file organization": "structure",
                "naming conventions": "naming",
                "error handling": "error_handling",
                "async patterns": "async",
                "hooks best practices": "hooks",
                "performance optimization": "performance",
                "state management": "state",
                "accessibility": "accessibility",
            }
            current_section = section_map.get(section_name, section_name.replace(" ", "_"))
            continue

        # Extract bullet points that are strategies (not in code blocks)
        if stripped.startswith("- ") and not in_code_block:
            content = stripped[2:].strip()
            # Filter for actionable content
            if _is_actionable_bullet(content):
                bullets.append(ExtractedBullet(
                    content=content,
                    section=current_section,
                    profile_source=profile_name,
                ))

    return bullets


def _is_actionable_bullet(content: str) -> bool:
    """Check if a bullet is an actionable strategy worth keeping."""
    # Too short
    if len(content) < 15:
        return False

    # Skip file paths, imports, etc.
    if content.startswith(("├", "│", "└", "import ", "from ")):
        return False

    # Skip pure examples/references
    if content.startswith(("Example:", "See:", "Note:", "http", "`")):
        return False

    # Prefer bullets that start with action words or have clear directives
    action_patterns = [
        r"^(Use|Always|Never|Avoid|Prefer|Add|Create|Define|Implement|Handle)",
        r"^(Ensure|Maintain|Follow|Check|Validate|Test|Run|Keep)",
        r"should|must|always|never|avoid|prefer",
    ]

    for pattern in action_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True

    # Accept bullets with code patterns (likely technical guidance)
    if "`" in content or "()" in content:
        return True

    return False


def seed_playbooks_from_profiles(
    project_path: Path,
    detected_tech: dict[str, Any],
    force: bool = False,
) -> dict[str, int]:
    """
    Seed project playbooks from profile templates based on detected tech stack.

    This is called during project initialization to provide baseline strategies
    that can then be refined through feedback.

    Args:
        project_path: Path to the project
        detected_tech: Tech stack detection results (languages, frameworks)
        force: If True, re-seed even if playbooks exist

    Returns:
        Dict mapping task_type to number of seeds added
    """
    from . import paths

    templates_dir = Path(__file__).parent / "templates" / "profiles"
    playbooks_dir = project_path / ".delia" / "playbooks"
    playbooks_dir.mkdir(parents=True, exist_ok=True)

    # Determine which profiles to seed from
    profiles_to_load = _select_profiles_for_tech(detected_tech, templates_dir)

    log.info("seeding_playbooks", profiles=profiles_to_load)

    # Map profile sections to playbook task types
    section_to_task = {
        "principles": "coding",
        "best_practices": "coding",
        "naming": "coding",
        "async": "coding",
        "performance": "performance",
        "error_handling": "debugging",
        "hooks": "coding",
        "state": "coding",
        "accessibility": "coding",
        "structure": "project",
        "security": "security",
        "testing": "testing",
        "deployment": "deployment",
        "git": "git",
    }

    results: dict[str, int] = {}
    pm = PlaybookManager(playbooks_dir)

    for profile_name in profiles_to_load:
        profile_path = templates_dir / profile_name
        if not profile_path.exists():
            continue

        try:
            content = profile_path.read_text()
            extracted = parse_profile_to_bullets(content, profile_name)

            for bullet in extracted:
                # Map section to task type
                task_type = section_to_task.get(bullet.section, "coding")

                # Check if playbook already has content (skip unless force)
                if not force:
                    existing = pm.load_playbook(task_type)
                    # Only seed if playbook is empty or has only seeds
                    has_learned = any(b.source == "learned" for b in existing)
                    if has_learned:
                        continue

                # Add as seed bullet
                pm.add_bullet(
                    task_type=task_type,
                    content=bullet.content,
                    section=bullet.section,
                    source="seed",
                )
                results[task_type] = results.get(task_type, 0) + 1

        except Exception as e:
            log.warning("profile_seed_failed", profile=profile_name, error=str(e))

    log.info("playbooks_seeded", results=results)
    return results


def _select_profiles_for_tech(tech_stack: dict[str, Any], templates_dir: Path) -> list[str]:
    """Select which profiles to load based on detected tech stack."""
    profiles = ["core.md"]  # Always include core

    # Map languages/frameworks to profiles
    lang = tech_stack.get("primary_language", "").lower()
    frameworks = [f.lower() for f in tech_stack.get("frameworks", [])]

    # Language profiles
    lang_profiles = {
        "python": "python.md",
        "typescript": "typescript.md",
        "javascript": "typescript.md",  # Use TS profile for JS too
        "rust": "rust.md",
        "go": "golang.md",
        "java": "java.md",
        "c": "c.md",
        "cpp": "cpp.md",
        "c++": "cpp.md",
    }

    if lang in lang_profiles:
        profiles.append(lang_profiles[lang])

    # Framework profiles
    framework_profiles = {
        "fastapi": "fastapi.md",
        "django": "django.md",
        "flask": "python.md",  # No flask.md, use python
        "react": "react.md",
        "vue": "vue.md",
        "angular": "angular.md",
        "svelte": "svelte.md",
        "nextjs": "nextjs.md",
        "next.js": "nextjs.md",
        "nestjs": "nestjs.md",
        "nest.js": "nestjs.md",
        "laravel": "laravel.md",
        "flutter": "flutter.md",
    }

    for fw in frameworks:
        fw_lower = fw.lower()
        for pattern, profile in framework_profiles.items():
            if pattern in fw_lower:
                if profile not in profiles:
                    profiles.append(profile)

    # Special detection
    if tech_stack.get("is_async"):
        # Async projects benefit from API patterns
        if "api.md" not in profiles:
            profiles.append("api.md")

    # Only include profiles that exist
    return [p for p in profiles if (templates_dir / p).exists()]


def get_seed_stats(project_path: Path) -> dict[str, Any]:
    """Get statistics about seeded vs learned bullets in a project."""
    playbooks_dir = project_path / ".delia" / "playbooks"
    if not playbooks_dir.exists():
        return {"seeded": 0, "learned": 0, "manual": 0, "total": 0}

    pm = PlaybookManager(playbooks_dir)
    all_bullets = pm.get_all_bullets()

    stats = {"seeded": 0, "learned": 0, "manual": 0, "total": 0}

    for task_type, bullets in all_bullets.items():
        for b in bullets:
            source = getattr(b, "source", "learned")  # Default for old bullets
            if source == "seed":
                stats["seeded"] += 1
            elif source == "manual":
                stats["manual"] += 1
            else:
                stats["learned"] += 1
            stats["total"] += 1

    return stats
