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
Delia Model Profile Management (inspired by Stanford ACE research).

This module implements specialized model profiles for task-specific routing.
Instead of always using large general-purpose models, profiles allow routing
to smaller, fine-tuned specialists that excel at specific tasks.

Example: A 3B SQL-specialist may outperform a 14B generalist for SQL queries.

Design follows the PlaybookManager pattern - profiles learn from feedback
to improve routing decisions over time.
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
class ModelProfile:
    """
    A specialized model profile for task-specific routing.

    Profiles map models to their specializations and track performance
    to enable intelligent routing decisions.

    Example profile:
        {
            "id": "sql-expert",
            "model": "sqlcoder:7b",
            "specializations": ["sql", "database", "query", "postgres"],
            "tasks": ["generate", "review"],
            "parent": "codellama:7b",
            "vram_gb": 5.0
        }
    """

    id: str  # Unique profile identifier
    model: str  # Actual model name (e.g., "sqlcoder:7b")
    specializations: list[str] = field(default_factory=list)  # Keywords it's trained for
    tasks: list[str] = field(default_factory=list)  # Task types it handles well
    parent: str | None = None  # Base model it was fine-tuned from
    vram_gb: float = -1.0  # VRAM requirement (-1 = auto-detect)
    context_tokens: int = -1  # Context window (-1 = auto-detect)
    quantization: str | None = None  # e.g., "Q4_K_M", "Q8_0", None for full
    adapter_path: str | None = None  # Path to LoRA adapter if applicable

    # Learning metrics (updated by feedback)
    helpful_count: int = 0
    harmful_count: int = 0
    total_calls: int = 0
    avg_latency_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str | None = None

    @property
    def utility_score(self) -> float:
        """Calculate utility score based on feedback history."""
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5  # Neutral for new profiles
        return self.helpful_count / total

    @property
    def confidence(self) -> float:
        """Confidence based on usage volume."""
        # Low confidence until we have enough data
        if self.total_calls < 5:
            return 0.3
        if self.total_calls < 20:
            return 0.6
        return 0.9

    def matches(self, task_type: str, content: str) -> tuple[bool, float]:
        """
        Check if this profile matches a task.

        Returns (matches, score) where score indicates match quality.
        Higher scores mean better fit.
        """
        score = 0.0
        content_lower = content.lower()

        # Task type match (0.4 weight)
        if task_type in self.tasks:
            score += 0.4

        # Specialization keyword match (0.3 weight)
        if self.specializations:
            matched = sum(1 for spec in self.specializations if spec.lower() in content_lower)
            match_ratio = matched / len(self.specializations)
            score += 0.3 * min(match_ratio * 2, 1.0)  # Boost for partial matches

        # Utility score from feedback (0.2 weight)
        score += 0.2 * self.utility_score

        # Confidence adjustment (0.1 weight)
        score += 0.1 * self.confidence

        # Threshold: need at least 30% match to be considered
        return score >= 0.3, score

    def record_usage(self, latency_ms: float):
        """Record a usage event."""
        self.total_calls += 1
        # Running average for latency
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = self.avg_latency_ms * 0.9 + latency_ms * 0.1
        self.last_used = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        # Handle missing fields gracefully
        return cls(
            id=data.get("id", f"profile-{uuid.uuid4().hex[:8]}"),
            model=data.get("model", ""),
            specializations=data.get("specializations", []),
            tasks=data.get("tasks", []),
            parent=data.get("parent"),
            vram_gb=data.get("vram_gb", -1.0),
            context_tokens=data.get("context_tokens", -1),
            quantization=data.get("quantization"),
            adapter_path=data.get("adapter_path"),
            helpful_count=data.get("helpful_count", 0),
            harmful_count=data.get("harmful_count", 0),
            total_calls=data.get("total_calls", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_used=data.get("last_used"),
        )


class ProfileManager:
    """
    Manages model profiles for intelligent task routing.

    Profiles are stored per-task-type, similar to PlaybookManager.
    Learning from feedback improves routing over time.
    """

    PROFILES_FILE = "model_profiles.json"

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or paths.DATA_DIR
        self._profiles: dict[str, ModelProfile] = {}
        self._loaded = False

    def _get_path(self) -> Path:
        return self.data_dir / self.PROFILES_FILE

    def _ensure_loaded(self):
        """Lazy load profiles from disk."""
        if self._loaded:
            return

        path = self._get_path()
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    for profile_id, profile_data in data.items():
                        self._profiles[profile_id] = ModelProfile.from_dict(profile_data)
                log.info("profiles_loaded", count=len(self._profiles))
            except Exception as e:
                log.error("profiles_load_failed", error=str(e))

        self._loaded = True

    def _save(self):
        """Persist profiles to disk."""
        path = self._get_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump({pid: p.to_dict() for pid, p in self._profiles.items()}, f, indent=2)
            log.debug("profiles_saved", count=len(self._profiles))
        except Exception as e:
            log.error("profiles_save_failed", error=str(e))

    def register_profile(self, profile: ModelProfile) -> ModelProfile:
        """Register or update a profile."""
        self._ensure_loaded()

        # Check for existing profile with same ID
        if profile.id in self._profiles:
            existing = self._profiles[profile.id]
            # Preserve learning metrics
            profile.helpful_count = existing.helpful_count
            profile.harmful_count = existing.harmful_count
            profile.total_calls = existing.total_calls
            profile.avg_latency_ms = existing.avg_latency_ms

        self._profiles[profile.id] = profile
        self._save()
        log.info("profile_registered", id=profile.id, model=profile.model, specializations=profile.specializations)
        return profile

    def get_profile(self, profile_id: str) -> ModelProfile | None:
        """Get a profile by ID."""
        self._ensure_loaded()
        return self._profiles.get(profile_id)

    def list_profiles(self) -> list[ModelProfile]:
        """List all registered profiles."""
        self._ensure_loaded()
        return list(self._profiles.values())

    def find_best_profile(self, task_type: str, content: str) -> tuple[ModelProfile | None, float]:
        """
        Find the best matching profile for a task.

        Returns (profile, score) or (None, 0) if no match.
        """
        self._ensure_loaded()

        best_profile = None
        best_score = 0.0

        for profile in self._profiles.values():
            matches, score = profile.matches(task_type, content)
            if matches and score > best_score:
                best_profile = profile
                best_score = score

        if best_profile:
            log.debug(
                "profile_matched",
                profile=best_profile.id,
                model=best_profile.model,
                score=f"{best_score:.2f}",
                task=task_type,
            )

        return best_profile, best_score

    def record_feedback(self, profile_id: str, helpful: bool, latency_ms: float = 0):
        """Record feedback for a profile to improve routing."""
        self._ensure_loaded()

        profile = self._profiles.get(profile_id)
        if not profile:
            log.warning("profile_feedback_not_found", id=profile_id)
            return

        if helpful:
            profile.helpful_count += 1
        else:
            profile.harmful_count += 1

        if latency_ms > 0:
            profile.record_usage(latency_ms)

        self._save()
        log.debug(
            "profile_feedback_recorded",
            id=profile_id,
            helpful=helpful,
            utility=f"{profile.utility_score:.2f}",
        )

    def record_usage(self, profile_id: str, latency_ms: float):
        """Record usage without explicit feedback."""
        self._ensure_loaded()

        profile = self._profiles.get(profile_id)
        if profile:
            profile.record_usage(latency_ms)
            self._save()

    def load_from_settings(self, specialists: dict[str, Any]):
        """
        Load profiles from settings.json 'specialists' section.

        Example settings.json:
            "specialists": {
                "sql-expert": {
                    "model": "sqlcoder:7b",
                    "specializations": ["sql", "database", "query"],
                    "tasks": ["generate", "review"]
                }
            }
        """
        for profile_id, config in specialists.items():
            config["id"] = profile_id
            profile = ModelProfile.from_dict(config)
            self.register_profile(profile)

        log.info("profiles_loaded_from_settings", count=len(specialists))


# =============================================================================
# SINGLETON + COMMON PROFILES
# =============================================================================

_profile_manager: ProfileManager | None = None


def get_profile_manager() -> ProfileManager:
    """Get the global ProfileManager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager


# Pre-defined common specialist profiles
COMMON_PROFILES = [
    ModelProfile(
        id="sql-specialist",
        model="sqlcoder:7b",
        specializations=["sql", "database", "query", "postgres", "mysql", "sqlite", "select", "join"],
        tasks=["generate", "review"],
        parent="codellama:7b",
        vram_gb=5.0,
    ),
    ModelProfile(
        id="python-specialist",
        model="codestral:7b",
        specializations=["python", "django", "flask", "fastapi", "pandas", "numpy"],
        tasks=["generate", "review", "analyze"],
        vram_gb=5.0,
    ),
    ModelProfile(
        id="typescript-specialist",
        model="qwen2.5-coder:7b",
        specializations=["typescript", "javascript", "react", "vue", "node", "npm"],
        tasks=["generate", "review", "analyze"],
        vram_gb=5.0,
    ),
    ModelProfile(
        id="shell-specialist",
        model="granite-code:3b",
        specializations=["bash", "shell", "linux", "docker", "kubernetes", "devops"],
        tasks=["generate", "quick"],
        vram_gb=2.5,
    ),
    ModelProfile(
        id="math-specialist",
        model="qwen2.5-math:7b",
        specializations=["math", "calculation", "statistics", "probability", "algebra"],
        tasks=["analyze", "quick"],
        vram_gb=5.0,
    ),
]


def register_common_profiles():
    """Register common specialist profiles if their models are available."""
    manager = get_profile_manager()
    for profile in COMMON_PROFILES:
        # Only register if not already present (preserves learned metrics)
        if not manager.get_profile(profile.id):
            manager.register_profile(profile)
