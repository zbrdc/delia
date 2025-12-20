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

"""Type definitions for Delia.

This module provides typed enums for task types and model tiers,
improving type safety and self-documentation.

Usage:
    from delia.types import TaskType, ModelTier

    task = TaskType.REVIEW
    tier = ModelTier.CODER

    # String conversion
    str(TaskType.REVIEW)  # "review"
    TaskType("review")    # TaskType.REVIEW
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class TaskType(StrEnum):
    """Task types for LLM delegation.

    Each task type influences model selection and prompt optimization.
    """

    # Quick tasks - use fast models
    QUICK = "quick"
    SUMMARIZE = "summarize"

    # Code tasks - use code-specialized models
    REVIEW = "review"
    ANALYZE = "analyze"
    GENERATE = "generate"

    # Complex tasks - use large reasoning models
    PLAN = "plan"
    CRITIQUE = "critique"
    THINK = "think"


class ModelTier(StrEnum):
    """Model tiers for routing.

    Tiers map to specific models in backend configuration.
    """

    QUICK = "quick"  # Fast, small models (7B-14B)
    CODER = "coder"  # Code-specialized models (14B-30B)
    MOE = "moe"  # Large MoE models (30B+)
    THINKING = "thinking"  # Extended reasoning models


class BackendType(StrEnum):
    """Backend types for routing decisions."""

    LOCAL = "local"  # Local GPU inference
    REMOTE = "remote"  # Remote API service


class Provider(StrEnum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    GEMINI = "gemini"
    OPENAI = "openai"
    VLLM = "vllm"
    LMSTUDIO = "lmstudio"
    CUSTOM = "custom"


# Convenience sets for validation (backwards compatible with existing code)
VALID_TASKS = frozenset(t.value for t in TaskType)
VALID_MODELS = frozenset(t.value for t in ModelTier)
VALID_BACKEND_TYPES = frozenset(t.value for t in BackendType)
VALID_PROVIDERS = frozenset(p.value for p in Provider)


@dataclass
class Workspace:
    """Defines a workspace boundary for agent file operations.

    All file operations (read, list, search) are confined to paths within
    the workspace root. This prevents agents from accessing files outside
    the intended project directory.

    Attributes:
        root: Absolute path to the workspace root directory
        allow_parent_traversal: If True, allows .. in paths (but still confined to root)
        additional_allowed: Extra paths outside root that are allowed (e.g., shared libs)

    Usage:
        workspace = Workspace(root=Path("/home/user/project"))

        # These would be allowed:
        validate_path_in_workspace("src/main.py", workspace)  # relative
        validate_path_in_workspace("/home/user/project/src/main.py", workspace)  # absolute

        # These would be blocked:
        validate_path_in_workspace("/etc/passwd", workspace)
        validate_path_in_workspace("../other-project/secrets.py", workspace)
    """

    root: Path
    allow_parent_traversal: bool = False
    additional_allowed: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure root is an absolute resolved path."""
        if isinstance(self.root, str):
            self.root = Path(self.root)
        self.root = self.root.expanduser().resolve()

        # Resolve additional allowed paths too
        self.additional_allowed = [
            Path(p).expanduser().resolve() for p in self.additional_allowed
        ]

    def contains(self, path: Path | str) -> bool:
        """Check if a path is within this workspace.

        Args:
            path: Path to check (absolute or relative to cwd)

        Returns:
            True if path is within workspace boundaries
        """
        try:
            resolved = Path(path).expanduser().resolve()
        except (ValueError, OSError):
            return False

        # Check if under workspace root
        try:
            resolved.relative_to(self.root)
            return True
        except ValueError:
            pass

        # Check additional allowed paths
        for allowed in self.additional_allowed:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue

        return False

    def resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace root.

        Args:
            path: Path (absolute or relative)

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path is outside workspace
        """
        # If relative, make it relative to workspace root
        p = Path(path)
        if not p.is_absolute():
            resolved = (self.root / p).resolve()
        else:
            resolved = p.expanduser().resolve()

        if not self.contains(resolved):
            raise ValueError(f"Path '{path}' is outside workspace '{self.root}'")

        return resolved
