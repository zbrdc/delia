# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Delia Cleanup Utilities

Handles cleanup of legacy global data and per-project resets.
"""

from pathlib import Path
import shutil
import structlog

log = structlog.get_logger()


def cleanup_legacy_global_data(dry_run: bool = True) -> dict:
    """
    Remove legacy global data directories that are now per-project.

    This removes:
    - ~/.delia/data/sessions/ (now <project>/.delia/sessions/)
    - ~/.delia/data/memories/ (now <project>/.delia/memories/)
    - ~/.delia/data/playbooks/ (now <project>/.delia/playbooks/)

    Args:
        dry_run: If True, only report what would be deleted

    Returns:
        Dictionary with cleanup results
    """
    from .paths import get_data_dir

    data_dir = get_data_dir()
    legacy_dirs = [
        data_dir / "sessions",
        data_dir / "memories",
        data_dir / "playbooks",
    ]

    results = {
        "dry_run": dry_run,
        "removed": [],
        "not_found": [],
        "errors": [],
    }

    for dir_path in legacy_dirs:
        if not dir_path.exists():
            results["not_found"].append(str(dir_path))
            continue

        # Count files
        try:
            file_count = len(list(dir_path.rglob("*")))
            size_bytes = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())

            if dry_run:
                log.info(
                    "would_remove_legacy_dir",
                    path=str(dir_path),
                    files=file_count,
                    size_mb=round(size_bytes / 1024 / 1024, 2),
                )
                results["removed"].append({
                    "path": str(dir_path),
                    "files": file_count,
                    "size_bytes": size_bytes,
                    "action": "would_remove",
                })
            else:
                shutil.rmtree(dir_path)
                log.info(
                    "removed_legacy_dir",
                    path=str(dir_path),
                    files=file_count,
                    size_mb=round(size_bytes / 1024 / 1024, 2),
                )
                results["removed"].append({
                    "path": str(dir_path),
                    "files": file_count,
                    "size_bytes": size_bytes,
                    "action": "removed",
                })
        except Exception as e:
            log.error("cleanup_error", path=str(dir_path), error=str(e))
            results["errors"].append({"path": str(dir_path), "error": str(e)})

    return results


def cleanup_project_delia_dir(project_path: Path, dry_run: bool = True) -> dict:
    """
    Clean up a project's .delia/ directory for re-initialization.

    WARNING: This deletes ALL project-specific data including:
    - Sessions
    - Playbooks
    - Memories
    - Profiles
    - Project summaries
    - Evaluation state

    Args:
        project_path: Path to the project directory
        dry_run: If True, only report what would be deleted

    Returns:
        Dictionary with cleanup results
    """
    project_path = Path(project_path).resolve()
    delia_dir = project_path / ".delia"

    results = {
        "dry_run": dry_run,
        "project": str(project_path),
        "removed": [],
        "not_found": False,
    }

    if not delia_dir.exists():
        results["not_found"] = True
        log.info("delia_dir_not_found", path=str(delia_dir))
        return results

    # Count what would be removed
    try:
        file_count = len(list(delia_dir.rglob("*")))
        size_bytes = sum(f.stat().st_size for f in delia_dir.rglob("*") if f.is_file())

        if dry_run:
            log.warning(
                "would_remove_project_delia_dir",
                path=str(delia_dir),
                files=file_count,
                size_mb=round(size_bytes / 1024 / 1024, 2),
            )
            results["removed"].append({
                "path": str(delia_dir),
                "files": file_count,
                "size_bytes": size_bytes,
                "action": "would_remove",
            })
        else:
            shutil.rmtree(delia_dir)
            log.warning(
                "removed_project_delia_dir",
                path=str(delia_dir),
                files=file_count,
                size_mb=round(size_bytes / 1024 / 1024, 2),
            )
            results["removed"].append({
                "path": str(delia_dir),
                "files": file_count,
                "size_bytes": size_bytes,
                "action": "removed",
            })
    except Exception as e:
        log.error("cleanup_error", path=str(delia_dir), error=str(e))
        results["error"] = str(e)

    return results


def migrate_global_sessions_to_project(
    project_path: Path,
    session_filter: str | None = None,
    dry_run: bool = True,
) -> dict:
    """
    Migrate old global sessions to a specific project (optional).

    This is OPTIONAL - most users should just start fresh.
    Only use if you have important session history you want to preserve.

    Args:
        project_path: Target project to migrate sessions to
        session_filter: Optional filter (e.g., client_id) to select sessions
        dry_run: If True, only report what would be migrated

    Returns:
        Dictionary with migration results
    """
    from .paths import get_data_dir
    import json

    project_path = Path(project_path).resolve()
    legacy_sessions_dir = get_data_dir() / "sessions"
    target_sessions_dir = project_path / ".delia" / "sessions"

    results = {
        "dry_run": dry_run,
        "project": str(project_path),
        "migrated": [],
        "skipped": [],
        "errors": [],
    }

    if not legacy_sessions_dir.exists():
        results["note"] = "No legacy sessions directory found"
        return results

    # Create target directory
    if not dry_run:
        target_sessions_dir.mkdir(parents=True, exist_ok=True)

    # Scan legacy sessions
    for session_file in legacy_sessions_dir.glob("*.json"):
        try:
            # Read session to check filter
            with open(session_file) as f:
                session_data = json.load(f)

            # Apply filter if provided
            if session_filter:
                client_id = session_data.get("client_id", "")
                if session_filter not in client_id:
                    results["skipped"].append(str(session_file.name))
                    continue

            # Migrate
            if dry_run:
                results["migrated"].append({
                    "file": session_file.name,
                    "action": "would_migrate",
                })
            else:
                target_file = target_sessions_dir / session_file.name
                shutil.copy2(session_file, target_file)
                results["migrated"].append({
                    "file": session_file.name,
                    "action": "migrated",
                })
        except Exception as e:
            log.error("migration_error", file=str(session_file), error=str(e))
            results["errors"].append({"file": str(session_file), "error": str(e)})

    return results


def cleanup_all(dry_run: bool = True) -> dict:
    """
    Full cleanup: Remove all legacy global data.

    This removes:
    - ~/.delia/data/sessions/
    - ~/.delia/data/memories/
    - ~/.delia/data/playbooks/

    Does NOT touch:
    - ~/.delia/settings.json (backend configs)
    - ~/.delia/data/cache/ (system metrics)
    - ~/.delia/data/users/ (auth)
    - Per-project .delia/ directories

    Args:
        dry_run: If True, only report what would be deleted

    Returns:
        Comprehensive cleanup results
    """
    results = {
        "dry_run": dry_run,
        "legacy_global": cleanup_legacy_global_data(dry_run=dry_run),
    }

    total_size = sum(
        item["size_bytes"]
        for item in results["legacy_global"]["removed"]
        if isinstance(item, dict)
    )
    total_files = sum(
        item["files"]
        for item in results["legacy_global"]["removed"]
        if isinstance(item, dict)
    )

    results["summary"] = {
        "total_files": total_files,
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "action": "would_remove" if dry_run else "removed",
    }

    return results
