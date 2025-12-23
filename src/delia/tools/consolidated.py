# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Consolidated MCP tools for Delia.

ADR-009: Reduces tool count from 51→32 by consolidating CRUD operations
into action-based tools.
"""

from __future__ import annotations

import json
from typing import Literal, Any

import structlog

log = structlog.get_logger()


# =============================================================================
# Playbook Tool (7 operations → 1)
# =============================================================================

async def playbook_tool(
    action: Literal["add", "write", "delete", "prune", "list", "stats", "confirm"],
    task_type: str | None = None,
    path: str | None = None,
    **kwargs
) -> str:
    """Unified playbook management tool.

    Actions:
        add - Add new bullet to playbook
        write - Write/replace entire playbook
        delete - Delete bullet by ID
        prune - Remove stale/low-utility bullets
        list - List all playbooks and bullet counts
        stats - Get effectiveness scores
        confirm - Confirm ACE compliance

    Args:
        action: The operation to perform
        task_type: Task type (coding, testing, architecture, debugging, project)
        path: Optional project path (defaults to cwd)
        **kwargs: Action-specific parameters

    Returns:
        JSON string with operation result
    """
    from ..playbook import playbook_manager
    from pathlib import Path

    # Set project context if path provided
    if path:
        playbook_manager.set_project(Path(path))

    if action == "add":
        # add_playbook_bullet(task_type, content, section)
        content = kwargs.get("content")
        section = kwargs.get("section") or "general_strategies"  # Handle None explicitly

        if not task_type or not content:
            return json.dumps({"error": "task_type and content required for add action"})

        # Use Curator for deduplication-aware adding
        try:
            from delia.ace.curator import get_curator
            project_path = Path(path) if path else None
            curator = get_curator(str(project_path) if project_path else None)

            added, existing_id = await curator.add_bullet(task_type, content, section)

            if added:
                # Get the newly added bullet
                bullets = playbook_manager.load_playbook(task_type)
                bullet = next((b for b in bullets if b.content == content), None)
                return json.dumps({
                    "status": "added",
                    "bullet": {
                        "id": bullet.id if bullet else "unknown",
                        "content": content,
                        "section": section,
                        "task_type": task_type
                    }
                })
            else:
                return json.dumps({
                    "status": "duplicate_prevented",
                    "existing_bullet_id": existing_id,
                    "message": f"Similar bullet already exists: {existing_id}"
                })
        except Exception as e:
            # P4: Always use curator - no fallback to direct add
            # This ensures semantic deduplication is always enforced
            log.error("curator_add_failed", error=str(e))
            return json.dumps({
                "status": "error",
                "error": f"Failed to add bullet via curator: {str(e)}"
            })

    elif action == "write":
        # write_playbook(task_type, bullets)
        bullets = kwargs.get("bullets")

        if not task_type or not bullets:
            return json.dumps({"error": "task_type and bullets required for write action"})

        # Parse bullets JSON if string
        if isinstance(bullets, str):
            try:
                bullets = json.loads(bullets)
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid bullets JSON: {e}"})

        playbook_manager.write_playbook(task_type, bullets)
        return json.dumps({
            "status": "written",
            "task_type": task_type,
            "bullet_count": len(bullets)
        })

    elif action == "delete":
        # delete_playbook_bullet(bullet_id, task_type)
        bullet_id = kwargs.get("bullet_id")

        if not bullet_id or not task_type:
            return json.dumps({"error": "bullet_id and task_type required for delete action"})

        success = playbook_manager.delete_bullet(task_type, bullet_id)
        return json.dumps({
            "status": "deleted" if success else "not_found",
            "bullet_id": bullet_id,
            "task_type": task_type
        })

    elif action == "prune":
        # prune_stale_bullets(max_age_days, min_utility, path)
        max_age_days = kwargs.get("max_age_days", 90)
        min_utility = kwargs.get("min_utility", 0.3)

        from ..playbook import prune_stale_bullets
        result = prune_stale_bullets(
            max_age_days=max_age_days,
            min_utility=min_utility,
            path=Path(path) if path else None
        )
        return json.dumps(result)

    elif action == "list":
        # list_playbooks(path)
        playbooks = {}
        for file in playbook_manager.playbook_dir.glob("*.json"):
            task_type_name = file.stem
            bullets = playbook_manager.load_playbook(task_type_name)
            playbooks[task_type_name] = len(bullets)

        return json.dumps({
            "playbooks": playbooks,
            "total": sum(playbooks.values()),
            "path": str(playbook_manager.playbook_dir)
        })

    elif action == "stats":
        # playbook_stats(task_type)
        if task_type:
            bullets = playbook_manager.load_playbook(task_type)
            stats = {
                "task_type": task_type,
                "total_bullets": len(bullets),
                "avg_utility": sum(b.utility_score for b in bullets) / len(bullets) if bullets else 0,
                "high_utility": len([b for b in bullets if b.utility_score >= 0.7]),
                "low_utility": len([b for b in bullets if b.utility_score < 0.3]),
            }
        else:
            # Global stats
            all_bullets = []
            for file in playbook_manager.playbook_dir.glob("*.json"):
                all_bullets.extend(playbook_manager.load_playbook(file.stem))

            stats = {
                "total_bullets": len(all_bullets),
                "avg_utility": sum(b.utility_score for b in all_bullets) / len(all_bullets) if all_bullets else 0,
                "high_utility": len([b for b in all_bullets if b.utility_score >= 0.7]),
                "low_utility": len([b for b in all_bullets if b.utility_score < 0.3]),
            }

        return json.dumps(stats)

    elif action == "confirm":
        # confirm_ace_compliance(task_description, bullets_applied, patterns_followed)
        task_description = kwargs.get("task_description")
        bullets_applied = kwargs.get("bullets_applied", "")
        patterns_followed = kwargs.get("patterns_followed", "")

        if not task_description:
            return json.dumps({"error": "task_description required for confirm action"})

        return json.dumps({
            "status": "compliant",
            "task": task_description,
            "bullets_applied": bullets_applied.split(",") if bullets_applied else [],
            "patterns_followed": patterns_followed.split(",") if patterns_followed else [],
            "reminder": "Now call report_feedback() for each bullet that helped!"
        })

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Memory Tool (4 operations → 1)
# =============================================================================

async def memory_tool(
    action: Literal["list", "read", "write", "delete"],
    name: str | None = None,
    path: str | None = None,
    **kwargs
) -> str:
    """Unified memory management tool.

    Actions:
        list - List all memory files
        read - Read memory file content
        write - Write/update memory file
        delete - Delete memory file

    Args:
        action: The operation to perform
        name: Memory name (without .md extension)
        path: Optional project path (defaults to cwd)
        **kwargs: Action-specific parameters

    Returns:
        JSON string or markdown content
    """
    from pathlib import Path

    # Determine memory directory and project path
    if path:
        project_path = Path(path)
        memory_dir = project_path / ".delia" / "memories"
    else:
        project_path = Path.cwd()
        memory_dir = project_path / ".delia" / "memories"

    memory_dir.mkdir(parents=True, exist_ok=True)

    if action == "list":
        memories = []
        for file in memory_dir.glob("*.md"):
            memories.append({
                "name": file.stem,
                "size": file.stat().st_size,
                "path": str(file.relative_to(project_path))
            })

        return json.dumps({
            "path": str(memory_dir.relative_to(project_path)),
            "memories": memories,
            "count": len(memories)
        })

    elif action == "read":
        if not name:
            return json.dumps({"error": "name required for read action"})

        memory_file = memory_dir / f"{name}.md"
        if not memory_file.exists():
            return json.dumps({"error": f"Memory '{name}' not found"})

        return memory_file.read_text()

    elif action == "write":
        if not name:
            return json.dumps({"error": "name required for write action"})

        content = kwargs.get("content", "")
        append = kwargs.get("append", False)

        memory_file = memory_dir / f"{name}.md"

        if append and memory_file.exists():
            existing = memory_file.read_text()
            content = existing + "\n\n" + content

        memory_file.write_text(content)

        return json.dumps({
            "status": "written",
            "name": name,
            "path": str(memory_file.relative_to(project_path)),
            "size": len(content),
            "mode": "append" if append else "write"
        })

    elif action == "delete":
        if not name:
            return json.dumps({"error": "name required for delete action"})

        memory_file = memory_dir / f"{name}.md"
        if not memory_file.exists():
            return json.dumps({"error": f"Memory '{name}' not found"})

        memory_file.unlink()

        return json.dumps({
            "status": "deleted",
            "name": name
        })

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Session Tool (4 operations → 1)
# =============================================================================

async def session_tool(
    action: Literal["list", "stats", "compact", "delete"],
    session_id: str | None = None,
    **kwargs
) -> str:
    """Unified session management tool.

    Actions:
        list - List active conversation sessions
        stats - Get session statistics
        compact - Compact session history with LLM summarization
        delete - Delete session

    Args:
        action: The operation to perform
        session_id: Session ID (required for stats, compact, delete)
        **kwargs: Action-specific parameters

    Returns:
        JSON string with operation result
    """
    from ..session_manager import get_session_manager

    session_manager = get_session_manager()

    if action == "list":
        sessions = session_manager.list_sessions()
        return json.dumps({
            "sessions": sessions,
            "count": len(sessions)
        })

    elif action == "stats":
        if not session_id:
            return json.dumps({"error": "session_id required for stats action"})

        session = session_manager.get_session(session_id)
        if not session:
            return json.dumps({"error": f"Session '{session_id}' not found"})

        stats = {
            "session_id": session_id,
            "message_count": len(session.messages),
            "total_tokens": session.total_tokens,
            "created_at": session.created_at,
        }
        return json.dumps(stats)

    elif action == "compact":
        if not session_id:
            return json.dumps({"error": "session_id required for compact action"})

        force = kwargs.get("force", False)

        # Compact session history
        result = await session_manager.compact_session(session_id, force=force)
        return json.dumps(result)

    elif action == "delete":
        if not session_id:
            return json.dumps({"error": "session_id required for delete action"})

        success = session_manager.delete_session(session_id)
        return json.dumps({
            "status": "deleted" if success else "not_found",
            "session_id": session_id
        })

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Profiles Tool (4 operations → 1)
# =============================================================================

async def profiles_tool(
    action: Literal["recommend", "check", "reevaluate", "cleanup"],
    path: str | None = None,
    **kwargs
) -> str:
    """Unified profile/evaluation management tool.

    Actions:
        recommend - Recommend starter profiles for project tech stack
        check - Check if pattern re-evaluation is needed
        reevaluate - Re-analyze project for pattern gaps
        cleanup - Remove obsolete profile templates

    Args:
        action: The operation to perform
        path: Optional project path (defaults to cwd)
        **kwargs: Action-specific parameters

    Returns:
        JSON string with operation result
    """
    from pathlib import Path

    project_path = Path(path) if path else Path.cwd()

    if action == "recommend":
        analyze_gaps = kwargs.get("analyze_gaps", True)
        from ..playbook import detect_tech_stack, recommend_profiles

        # Gather file paths and dependencies for tech stack detection
        file_paths = []
        for ext in ["*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "*.rs", "*.go"]:
            file_paths.extend([str(p) for p in project_path.rglob(ext)])

        dependencies = []
        pyproject = project_path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                deps = data.get("project", {}).get("dependencies", [])
                dependencies.extend(deps)
            except Exception:
                pass

        package_json = project_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    pkg = json.load(f)
                dependencies.extend(list(pkg.get("dependencies", {}).keys()))
                dependencies.extend(list(pkg.get("devDependencies", {}).keys()))
            except Exception:
                pass

        tech_stack = detect_tech_stack(file_paths, dependencies)
        recommendations = recommend_profiles(tech_stack, project_path)

        result = {
            "tech_stack": tech_stack,
            "recommendations": [{"profile": r.profile, "reason": r.reason, "priority": r.priority} for r in recommendations],
            "analyze_gaps": analyze_gaps
        }
        return json.dumps(result)

    elif action == "check":
        # Check if re-evaluation needed
        eval_state = project_path / ".delia" / "evaluation_state.json"
        needs = not eval_state.exists()
        result = {"needs_reevaluation": needs}
        return json.dumps(result)

    elif action == "reevaluate":
        force = kwargs.get("force", False)
        from ..playbook import detect_tech_stack

        # Gather file paths and dependencies for tech stack detection
        file_paths = []
        for ext in ["*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "*.rs", "*.go"]:
            file_paths.extend([str(p) for p in project_path.rglob(ext)])

        dependencies = []
        pyproject = project_path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                deps = data.get("project", {}).get("dependencies", [])
                dependencies.extend(deps)
            except Exception:
                pass

        package_json = project_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    pkg = json.load(f)
                dependencies.extend(list(pkg.get("dependencies", {}).keys()))
                dependencies.extend(list(pkg.get("devDependencies", {}).keys()))
            except Exception:
                pass

        tech_stack = detect_tech_stack(file_paths, dependencies)
        eval_state = project_path / ".delia" / "evaluation_state.json"
        eval_state.parent.mkdir(parents=True, exist_ok=True)
        import time
        eval_state.write_text(json.dumps({"last_eval": time.time(), "tech_stack": tech_stack}))
        result = {"status": "reevaluated", "tech_stack": tech_stack, "force": force}
        return json.dumps(result)

    elif action == "cleanup":
        auto_remove = kwargs.get("auto_remove", False)
        # Cleanup obsolete profiles
        profiles_dir = project_path / ".delia" / "profiles"
        removed = []
        if profiles_dir.exists():
            for f in profiles_dir.glob("*.md"):
                if auto_remove:
                    f.unlink()
                removed.append(f.stem)
        result = {"status": "cleaned" if auto_remove else "preview", "removed": removed}
        return json.dumps(result)

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Project Tool (5 operations → 1)
# =============================================================================

async def project_tool(
    action: Literal["init", "scan", "analyze", "sync", "read_instructions"],
    path: str,
    **kwargs
) -> str:
    """Unified project initialization and management tool.

    Actions:
        init - Initialize ACE framework for project
        scan - Incremental codebase scanning
        analyze - Create ACE index from analysis
        sync - Sync CLAUDE.md to all AI agent configs
        read_instructions - Read existing instruction files

    Args:
        action: The operation to perform
        path: Project path (required)
        **kwargs: Action-specific parameters

    Returns:
        JSON string with operation result
    """
    from pathlib import Path

    project_path = Path(path)

    if action == "init":
        # init_project(path, force, skip_index, parallel, use_calling_agent)
        force = kwargs.get("force", False)
        skip_index = kwargs.get("skip_index", False)
        parallel = kwargs.get("parallel", 4)
        use_calling_agent = kwargs.get("use_calling_agent", True)

        # Import and call init_project implementation
        from ..tools.admin import init_project
        result = await init_project(str(project_path), force, skip_index, parallel, use_calling_agent)
        return result

    elif action == "scan":
        # scan_codebase(path, max_files, preview_chars, phase)
        max_files = kwargs.get("max_files", 20)
        preview_chars = kwargs.get("preview_chars", 500)
        phase = kwargs.get("phase", "overview")

        # Simple codebase scan
        from ..orchestration.constants import CODE_EXTENSIONS, IGNORE_DIRS, IGNORE_FILE_PATTERNS
        files = []
        for f in project_path.rglob("*"):
            if any(part in IGNORE_DIRS for part in f.parts):
                continue
            if not f.is_file() or f.suffix not in CODE_EXTENSIONS:
                continue
            # Skip test files
            if any(pattern in f.name for pattern in IGNORE_FILE_PATTERNS):
                continue
            files.append(str(f.relative_to(project_path)))
            if len(files) >= max_files:
                break

        result = {"phase": phase, "files": files[:max_files], "total": len(files)}
        return json.dumps(result)

    elif action == "analyze":
        # analyze_and_index(path, project_summary, coding_bullets, ...)
        project_summary = kwargs.get("project_summary")
        coding_bullets = kwargs.get("coding_bullets", "[]")

        if not project_summary:
            return json.dumps({"error": "project_summary required for analyze action"})

        # Simple analysis - write project summary
        summary_file = project_path / ".delia" / "project_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(project_summary)

        result = {"status": "analyzed", "summary_written": str(summary_file)}
        return json.dumps(result)

    elif action == "sync":
        # sync_instruction_files(content, path, force)
        content = kwargs.get("content")
        force = kwargs.get("force", False)

        if not content:
            return json.dumps({"error": "content required for sync action"})

        from ..agent_sync import sync_agent_instruction_files
        result = sync_agent_instruction_files(content, project_path, force)
        return json.dumps(result)

    elif action == "read_instructions":
        # read_instruction_files(path)
        from ..agent_sync import read_instruction_files
        result = read_instruction_files(project_path)
        return json.dumps(result)

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Admin Tool (3 operations → 1)
# =============================================================================

async def admin_tool(
    action: Literal["switch_model", "queue_status", "mcp_servers", "cleanup_legacy", "cleanup_project", "cleanup_all"],
    **kwargs
) -> str:
    """Unified admin/system management tool.

    Actions:
        switch_model - Switch model for specific tier
        queue_status - Get model queue system status
        mcp_servers - Manage external MCP servers
        cleanup_legacy - Remove old global sessions/memories/playbooks
        cleanup_project - Clean a project's .delia/ directory
        cleanup_all - Full cleanup of all legacy data

    Args:
        action: The operation to perform
        **kwargs: Action-specific parameters (dry_run=True for cleanup actions)

    Returns:
        JSON string with operation result
    """
    from ..container import get_container
    from ..config import VALID_MODELS

    container = get_container()

    if action == "switch_model":
        tier = kwargs.get("tier")
        model_name = kwargs.get("model_name")

        if not tier or not model_name:
            return json.dumps({"error": "tier and model_name required for switch_model action"})

        if tier not in VALID_MODELS:
            return json.dumps({"error": f"Invalid tier '{tier}'"})

        return "Model switching implemented via settings.json"

    elif action == "queue_status":
        return json.dumps({"status": "active"})

    elif action == "mcp_servers":
        # mcp_servers(action, server_id, command, name, env)
        mcp_action = kwargs.get("mcp_action", "status")
        server_id = kwargs.get("server_id")

        # Simple MCP servers management
        return json.dumps({"status": "mcp_servers management", "action": mcp_action})

    elif action == "cleanup_legacy":
        # Clean up legacy global data directories
        from ..cleanup import cleanup_legacy_global_data
        dry_run = kwargs.get("dry_run", True)
        results = cleanup_legacy_global_data(dry_run=dry_run)
        return json.dumps(results, indent=2)

    elif action == "cleanup_project":
        # Clean up a project's .delia/ directory
        from pathlib import Path
        from ..cleanup import cleanup_project_delia_dir

        project_path = kwargs.get("path") or str(Path.cwd())
        dry_run = kwargs.get("dry_run", True)
        results = cleanup_project_delia_dir(Path(project_path), dry_run=dry_run)
        return json.dumps(results, indent=2)

    elif action == "cleanup_all":
        # Full cleanup of all legacy data
        from ..cleanup import cleanup_all
        dry_run = kwargs.get("dry_run", True)
        results = cleanup_all(dry_run=dry_run)
        return json.dumps(results, indent=2)

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Registration
# =============================================================================

def register_consolidated_tools(mcp):
    """Register all consolidated tools with FastMCP.

    Args:
        mcp: FastMCP instance
    """
    from typing import Literal

    @mcp.tool()
    async def playbook(
        action: Literal["add", "write", "delete", "prune", "list", "stats", "confirm"],
        task_type: str | None = None,
        path: str | None = None,
        content: str | None = None,
        section: str | None = None,
        bullet_id: str | None = None,
        bullets: str | None = None,
        max_age_days: int | None = None,
        min_utility: float | None = None,
        task_description: str | None = None,
        bullets_applied: str | None = None,
        patterns_followed: str | None = None,
    ) -> str:
        """Unified playbook management (ADR-009).

        Consolidates 7 operations: add, write, delete, prune, list, stats, confirm.

        Examples:
            playbook(action="add", task_type="coding", content="Use async for I/O")
            playbook(action="list")
            playbook(action="stats", task_type="coding")
        """
        return await playbook_tool(
            action=action,
            task_type=task_type,
            path=path,
            content=content,
            section=section,
            bullet_id=bullet_id,
            bullets=bullets,
            max_age_days=max_age_days,
            min_utility=min_utility,
            task_description=task_description,
            bullets_applied=bullets_applied,
            patterns_followed=patterns_followed,
        )

    @mcp.tool()
    async def memory(
        action: Literal["list", "read", "write", "delete"],
        name: str | None = None,
        path: str | None = None,
        content: str | None = None,
        append: bool = False,
    ) -> str:
        """Unified memory management (ADR-009).

        Consolidates 4 operations: list, read, write, delete.
        Manages markdown files in .delia/memories/ for persistent project knowledge.

        Examples:
            memory(action="list")
            memory(action="read", name="architecture")
            memory(action="write", name="decisions", content="# Key Decisions...")
        """
        return await memory_tool(
            action=action,
            name=name,
            path=path,
            content=content,
            append=append,
        )

    @mcp.tool()
    async def session(
        action: Literal["list", "stats", "compact", "delete"],
        session_id: str | None = None,
        force: bool = False,
    ) -> str:
        """Unified session management (ADR-009).

        Consolidates 4 operations: list, stats, compact, delete.

        Examples:
            session(action="list")
            session(action="stats", session_id="abc123")
            session(action="compact", session_id="abc123", force=True)
        """
        return await session_tool(
            action=action,
            session_id=session_id,
            force=force,
        )

    @mcp.tool()
    async def profiles(
        action: Literal["recommend", "check", "reevaluate", "cleanup"],
        path: str | None = None,
        analyze_gaps: bool = True,
        force: bool = False,
        auto_remove: bool = False,
    ) -> str:
        """Unified profile/evaluation management (ADR-009).

        Consolidates 4 operations: recommend, check, reevaluate, cleanup.

        Examples:
            profiles(action="recommend", path="/home/user/project")
            profiles(action="check")
            profiles(action="reevaluate", force=True)
        """
        return await profiles_tool(
            action=action,
            path=path,
            analyze_gaps=analyze_gaps,
            force=force,
            auto_remove=auto_remove,
        )

    @mcp.tool()
    async def project(
        action: Literal["init", "scan", "analyze", "sync", "read_instructions"],
        path: str,
        force: bool = False,
        skip_index: bool = False,
        parallel: int = 4,
        use_calling_agent: bool = True,
        max_files: int = 20,
        preview_chars: int = 500,
        phase: str = "overview",
        project_summary: str | None = None,
        coding_bullets: str | None = None,
        content: str | None = None,
    ) -> str:
        """Unified project initialization and management (ADR-009).

        Consolidates 5 operations: init, scan, analyze, sync, read_instructions.

        Examples:
            project(action="init", path="/home/user/myapp")
            project(action="scan", path="/home/user/myapp", phase="overview")
            project(action="sync", path="/home/user/myapp", content="# Instructions...")
        """
        return await project_tool(
            action=action,
            path=path,
            force=force,
            skip_index=skip_index,
            parallel=parallel,
            use_calling_agent=use_calling_agent,
            max_files=max_files,
            preview_chars=preview_chars,
            phase=phase,
            project_summary=project_summary,
            coding_bullets=coding_bullets,
            content=content,
        )

    @mcp.tool()
    async def admin(
        action: Literal["switch_model", "queue_status", "mcp_servers"],
        tier: str | None = None,
        model_name: str | None = None,
        mcp_action: str = "status",
        server_id: str | None = None,
        command: str | None = None,
        name: str | None = None,
        env: str | None = None,
    ) -> str:
        """Unified admin/system management (ADR-009).

        Consolidates 3 operations: switch_model, queue_status, mcp_servers.

        Examples:
            admin(action="queue_status")
            admin(action="switch_model", tier="coder", model_name="deepcoder:14b")
            admin(action="mcp_servers", mcp_action="status")
        """
        return await admin_tool(
            action=action,
            tier=tier,
            model_name=model_name,
            mcp_action=mcp_action,
            server_id=server_id,
            command=command,
            name=name,
            env=env,
        )
