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
File helper functions for reading files and Delia's memory system.

This module provides utilities for safely reading files from disk,
including size limits, error handling, and Delia's native memory integration.
"""

import re
from pathlib import Path

import structlog

from .config import config
from .context import get_project_path

log = structlog.get_logger()


def get_memory_dir(project_path: Path | None = None) -> Path:
    """Get project-specific memory directory (.delia/memories/).

    Args:
        project_path: Project root. Defaults to project context.
    """
    return get_project_path(project_path) / ".delia" / "memories"


# NOTE: MEMORY_DIR is kept for backwards compatibility
# Callers should prefer get_memory_dir() for proper project isolation
@property
def _memory_dir_property():
    return get_memory_dir()


class _LazyMemoryDir:
    """Lazy MEMORY_DIR that respects project context."""
    def __truediv__(self, other):
        return get_memory_dir() / other

    def exists(self):
        return get_memory_dir().exists()

    def glob(self, pattern):
        return get_memory_dir().glob(pattern)


MEMORY_DIR = _LazyMemoryDir()


def read_file_safe(file_path: str, max_size: int | None = None) -> tuple[str | None, str | None]:
    """
    Safely read file with size limit.

    Args:
        file_path: Path to file (absolute or relative, ~ expanded)
        max_size: Maximum file size in bytes (defaults to config.max_file_size)

    Returns:
        Tuple of (content, error) - one will be None
    """
    if max_size is None:
        max_size = config.max_file_size
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return None, f"File not found: {file_path}"
        if path.stat().st_size > max_size:
            return None, f"File too large: {path.stat().st_size} > {max_size}"
        return path.read_text(encoding="utf-8", errors="replace"), None
    except Exception as e:
        return None, f"Error reading file: {e}"


def read_memory(name: str) -> str | None:
    """
    Read a Delia memory file.

    Memory files are markdown files stored in the memories directory.
    Names are sanitized to prevent path traversal.

    Args:
        name: Memory name (will be sanitized)

    Returns:
        Memory content or None if not found
    """
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    memory_path = MEMORY_DIR / f"{safe_name}.md"

    if memory_path.exists():
        return memory_path.read_text()
    return None


def list_memories() -> list[str]:
    """
    List all available Delia memories.

    Returns:
        List of memory names (without extension) sorted alphabetically.
    """
    if not MEMORY_DIR.exists():
        return []

    memories = [f.stem for f in MEMORY_DIR.glob("*.md")]
    return sorted(memories)


def read_files(file_paths: str, max_size_bytes: int = 500_000) -> list[tuple[str, str]]:
    """
    Read multiple files from disk efficiently.

    Args:
        file_paths: Comma-separated file paths (absolute or relative to cwd)
        max_size_bytes: Maximum file size to read (default 500KB)

    Returns:
        List of (path, content) tuples for successfully read files.
        Files that don't exist or are too large are skipped with a warning.
    """
    results = []
    paths_list = [p.strip() for p in file_paths.split(",") if p.strip()]

    for path_str in paths_list:
        try:
            file_path = Path(path_str)

            # Try relative to project path if not absolute
            if not file_path.is_absolute():
                file_path = get_project_path() / file_path

            # Fallback: try relative to project root (for cross-directory contexts)
            if not file_path.exists():
                project_relative = PROJECT_ROOT / path_str
                if project_relative.exists():
                    file_path = project_relative
                    log.debug("file_resolved_via_project_root", path=path_str)

            if not file_path.exists():
                log.warning("file_read_skipped", path=path_str, reason="not_found")
                continue

            if not file_path.is_file():
                log.warning("file_read_skipped", path=path_str, reason="not_a_file")
                continue

            size = file_path.stat().st_size
            if size > max_size_bytes:
                log.warning(
                    "file_read_skipped",
                    path=path_str,
                    reason="too_large",
                    size_kb=size // 1024,
                    max_kb=max_size_bytes // 1024,
                )
                continue

            content = file_path.read_text(encoding="utf-8")
            results.append((path_str, content))
            log.info("file_read_success", path=path_str, size_kb=size // 1024)
        except (OSError, ValueError) as e:
            log.warning("file_read_skipped", path=path_str[:100], reason="invalid_path", error=str(e))
        except Exception as e:
            log.warning("file_read_failed", path=path_str[:100], error=str(e))

    return results
