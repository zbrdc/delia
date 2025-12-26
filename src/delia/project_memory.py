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
Project Memory System - Automatic Instruction Loading.

Automatically discovers and loads project instructions from:
1. ./DELIA.md (project root)
2. ./.delia/DELIA.md (project config directory)
3. ./.delia/rules/*.md (modular rules)
4. ~/.delia/DELIA.md (user defaults)
5. ./DELIA.local.md (local overrides, git-ignored)

Supports import syntax: @path/to/file.md to include other files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from . import paths
from .context import get_project_path

log = structlog.get_logger()

# Memory file names to discover
MEMORY_FILES = [
    "DELIA.md",
    "delia.md",
    ".delia/DELIA.md",
    ".delia/delia.md",
]

# Local override (should be git-ignored)
LOCAL_MEMORY_FILE = "DELIA.local.md"

# Rules directory for modular instructions
RULES_DIR = ".delia/rules"

# Import pattern: @path/to/file.md or @./relative/path.md
IMPORT_PATTERN = re.compile(r"^@(.+\.md)\s*$", re.MULTILINE)

# Maximum depth for recursive imports (prevent infinite loops)
MAX_IMPORT_DEPTH = 5

# Maximum total size for loaded memories (prevent memory issues)
MAX_TOTAL_SIZE = 500_000  # 500KB


@dataclass
class LoadedMemory:
    """Represents a loaded memory file."""

    path: Path
    content: str
    source: str  # "project", "user", "local", "rules", "import"
    size: int = 0

    def __post_init__(self):
        self.size = len(self.content)


@dataclass
class ProjectMemoryState:
    """State of loaded project memories."""

    memories: list[LoadedMemory] = field(default_factory=list)
    total_size: int = 0
    load_errors: list[str] = field(default_factory=list)

    @property
    def combined_content(self) -> str:
        """Get all memories combined into a single string."""
        if not self.memories:
            return ""

        sections = []
        for mem in self.memories:
            header = f"<!-- Memory: {mem.path.name} ({mem.source}) -->"
            sections.append(f"{header}\n{mem.content}")

        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memories": [
                {
                    "path": str(m.path),
                    "source": m.source,
                    "size": m.size,
                }
                for m in self.memories
            ],
            "total_size": self.total_size,
            "load_errors": self.load_errors,
        }


class ProjectMemory:
    """
    Discovers and loads project instructions automatically.

    Finds and loads project-specific instructions from multiple locations,
    supporting all AI coding assistants.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        user_dir: Path | None = None,
    ):
        """
        Initialize the project memory loader.

        Args:
            project_root: Project root directory (default: cwd)
            user_dir: User config directory (default: ~/.delia)
        """
        self.project_root = get_project_path(project_root)
        self.user_dir = user_dir or paths.USER_DELIA_DIR
        self._state: ProjectMemoryState | None = None
        self._loaded_paths: set[Path] = set()  # Track loaded files to prevent duplicates
        self._loaded_dirs: set[Path] = set()  # Track dirs that have had a memory file loaded

    def discover(self) -> ProjectMemoryState:
        """
        Discover and load all project memories.

        Returns:
            ProjectMemoryState with all loaded memories
        """
        self._state = ProjectMemoryState()
        self._loaded_paths = set()
        self._loaded_dirs = set()

        # 1. Load user-level defaults first (lowest priority)
        self._load_user_memories()

        # 2. Load project-level memories
        self._load_project_memories()

        # 3. Load modular rules from .delia/rules/
        self._load_rules()

        # 4. Load local overrides last (highest priority)
        self._load_local_override()

        # Calculate total size
        self._state.total_size = sum(m.size for m in self._state.memories)

        log.info(
            "project_memory_loaded",
            memories=len(self._state.memories),
            total_size=self._state.total_size,
            errors=len(self._state.load_errors),
        )

        return self._state

    def _load_user_memories(self) -> None:
        """Load user-level default memories from ~/.delia/."""
        for filename in MEMORY_FILES[:2]:  # Only DELIA.md variants
            path = self.user_dir / filename
            self._load_file(path, source="user")

    def _load_project_memories(self) -> None:
        """Load project-level memories from project root."""
        for filename in MEMORY_FILES:
            path = self.project_root / filename
            self._load_file(path, source="project")

    def _load_rules(self) -> None:
        """Load modular rules from .delia/rules/*.md."""
        rules_dir = self.project_root / RULES_DIR
        if not rules_dir.is_dir():
            return

        # Sort for consistent ordering
        rule_files = sorted(rules_dir.glob("*.md"))

        for path in rule_files:
            self._load_file(path, source="rules")

    def _load_local_override(self) -> None:
        """Load local override file (git-ignored)."""
        path = self.project_root / LOCAL_MEMORY_FILE
        self._load_file(path, source="local")

    def _load_file(
        self,
        path: Path,
        source: str,
        depth: int = 0,
    ) -> bool:
        """
        Load a single file with import resolution.

        Args:
            path: Path to the file
            source: Source type (project, user, local, rules, import)
            depth: Current import depth (for recursion limit)

        Returns:
            True if file was loaded successfully
        """
        # Resolve to absolute path
        try:
            abs_path = path.resolve()
        except (OSError, ValueError):
            return False

        # Skip if already loaded or doesn't exist
        if abs_path in self._loaded_paths:
            return False
        if not abs_path.is_file():
            return False

        # Skip case variants (e.g., skip delia.md if DELIA.md already loaded from same dir)
        dir_key = (abs_path.parent, abs_path.name.lower())
        if dir_key in self._loaded_dirs:
            return False

        # Check size limit
        try:
            file_size = abs_path.stat().st_size
            if self._state.total_size + file_size > MAX_TOTAL_SIZE:
                self._state.load_errors.append(
                    f"Skipped {path}: would exceed size limit"
                )
                return False
        except OSError:
            return False

        # Read content
        try:
            content = abs_path.read_text(encoding="utf-8")
        except Exception as e:
            self._state.load_errors.append(f"Failed to read {path}: {e}")
            return False

        # Mark as loaded before processing imports (prevents circular imports)
        self._loaded_paths.add(abs_path)
        self._loaded_dirs.add((abs_path.parent, abs_path.name.lower()))

        # Process imports if not at max depth
        if depth < MAX_IMPORT_DEPTH:
            content = self._process_imports(content, abs_path.parent, depth)

        # Add to state
        memory = LoadedMemory(
            path=abs_path,
            content=content,
            source=source,
        )
        self._state.memories.append(memory)
        self._state.total_size += memory.size

        log.debug(
            "memory_file_loaded",
            path=str(path),
            source=source,
            size=memory.size,
        )

        return True

    def _process_imports(
        self,
        content: str,
        base_dir: Path,
        depth: int,
    ) -> str:
        """
        Process @import syntax in content.

        Replaces @path/to/file.md with the file's contents.

        Args:
            content: Content to process
            base_dir: Base directory for relative imports
            depth: Current import depth

        Returns:
            Content with imports resolved
        """
        def replace_import(match: re.Match) -> str:
            import_path = match.group(1).strip()

            # Resolve path (relative to base_dir or absolute)
            if import_path.startswith("~"):
                full_path = Path(import_path).expanduser()
            elif import_path.startswith("/"):
                full_path = Path(import_path)
            else:
                full_path = base_dir / import_path

            # Load the imported file
            try:
                abs_path = full_path.resolve()
                if abs_path in self._loaded_paths:
                    return f"<!-- Already imported: {import_path} -->"

                if not abs_path.is_file():
                    return f"<!-- Import not found: {import_path} -->"

                # Check depth
                if depth >= MAX_IMPORT_DEPTH:
                    return f"<!-- Import depth exceeded: {import_path} -->"

                # Read and recursively process
                imported_content = abs_path.read_text(encoding="utf-8")
                self._loaded_paths.add(abs_path)

                # Process nested imports
                imported_content = self._process_imports(
                    imported_content,
                    abs_path.parent,
                    depth + 1,
                )

                return f"<!-- Imported: {import_path} -->\n{imported_content}"

            except Exception as e:
                return f"<!-- Import error ({import_path}): {e} -->"

        return IMPORT_PATTERN.sub(replace_import, content)

    def get_context_injection(self) -> str:
        """
        Get the memory content formatted for context injection.

        Returns:
            Formatted string ready to inject into LLM context
        """
        if not self._state or not self._state.memories:
            return ""

        return f"""
## Project Instructions

The following project-specific instructions were automatically loaded:

{self._state.combined_content}

---
"""

    def list_memories(self) -> list[dict[str, Any]]:
        """
        List all loaded memories.

        Returns:
            List of memory info dicts
        """
        if not self._state:
            return []

        return [
            {
                "path": str(m.path),
                "name": m.path.name,
                "source": m.source,
                "size": m.size,
                "size_kb": round(m.size / 1024, 1),
            }
            for m in self._state.memories
        ]


# Module-level singleton
_project_memory: ProjectMemory | None = None


def get_project_memory(force_reload: bool = False) -> ProjectMemory:
    """
    Get the global ProjectMemory instance.

    Args:
        force_reload: Force re-discovery of memories

    Returns:
        ProjectMemory instance with loaded state
    """
    global _project_memory

    if _project_memory is None or force_reload:
        _project_memory = ProjectMemory()
        _project_memory.discover()

    return _project_memory


def get_project_context() -> str:
    """
    Get project memory content for context injection.

    Convenience function for use in ContextEngine.

    Returns:
        Formatted project instructions string
    """
    pm = get_project_memory()
    return pm.get_context_injection()


def list_project_memories() -> list[dict[str, Any]]:
    """
    List all loaded project memories.

    Returns:
        List of memory info dicts
    """
    pm = get_project_memory()
    return pm.list_memories()


def reload_project_memories() -> ProjectMemoryState:
    """
    Force reload of all project memories.

    Returns:
        Fresh ProjectMemoryState
    """
    global _project_memory
    _project_memory = ProjectMemory()
    return _project_memory.discover()
