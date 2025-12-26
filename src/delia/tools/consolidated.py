# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Consolidated MCP tools for Delia.

ADR-009: Reduces tool count from 51→32 by consolidating CRUD operations
into action-based tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Any

import structlog

from ..context import get_project_path

log = structlog.get_logger()


# Use canonical get_project_path from context module
_resolve_project_path = get_project_path


# =============================================================================
# Playbook Tool (7 operations → 1)
# =============================================================================

async def playbook_tool(
    action: Literal["add", "write", "delete", "prune", "list", "stats", "confirm", "search", "index", "feedback", "learning_stats", "prune_patterns"],
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
        confirm - Confirm framework compliance
        search - Semantic search for relevant bullets via ChromaDB
        index - Index all playbook bullets to ChromaDB
        feedback - Record detection feedback to improve auto_context accuracy
        learning_stats - Get statistics about learned detection patterns
        prune_patterns - Remove learned patterns with low effectiveness

    Args:
        action: The operation to perform
        task_type: Task type (coding, testing, architecture, debugging, project)
        path: Optional project path (defaults to cwd)
        **kwargs: Action-specific parameters (query for search, limit for search)

    Returns:
        JSON string with operation result
    """
    from ..playbook import playbook_manager

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
            from delia.learning.curator import get_curator
            project_path = Path(path) if path else None
            curator = get_curator(str(project_path) if project_path else None)

            add_result = await curator.add_bullet(task_type, content, section)

            if add_result.get("added"):
                bullet_id = add_result.get("bullet_id")

                # Sync to ChromaDB for semantic search
                try:
                    from delia.learning.retrieval import get_retriever
                    retriever = get_retriever()
                    from ..playbook import PlaybookBullet
                    bullet = PlaybookBullet(id=bullet_id, content=content, section=section)
                    await retriever.index_bullets_to_chromadb(
                        bullets=[bullet],
                        task_type=task_type,
                        project=project_path.name if project_path else "global",
                        project_path=project_path,
                    )
                except Exception as e:
                    log.debug("chromadb_sync_failed", error=str(e))

                return json.dumps({
                    "status": "added",
                    "bullet": {
                        "id": bullet_id,
                        "content": content,
                        "section": section,
                        "task_type": task_type
                    }
                })
            elif add_result.get("quality_rejected"):
                return json.dumps({
                    "status": "quality_rejected",
                    "reason": add_result.get("reason"),
                    "message": "Content failed quality validation. Bullets must be 15-300 chars, actionable, and not vague."
                })
            else:
                return json.dumps({
                    "status": "duplicate_prevented",
                    "existing_bullet_id": add_result.get("bullet_id"),
                    "message": f"Similar bullet already exists: {add_result.get('bullet_id')}"
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

        # Convert list of strings to PlaybookBullet objects
        from ..playbook import PlaybookBullet
        bullet_objects = []
        for content in bullets:
            bullet_objects.append(PlaybookBullet(content=content))
        playbook_manager.save_playbook(task_type, bullet_objects)

        # Sync to ChromaDB - re-index this task_type
        project_path = _resolve_project_path(path)
        try:
            from delia.learning.retrieval import get_retriever
            retriever = get_retriever()
            await retriever.index_bullets_to_chromadb(
                bullets=bullet_objects,
                task_type=task_type,
                project=project_path.name,
                project_path=project_path,
            )
        except Exception as e:
            log.debug("chromadb_sync_failed", error=str(e))

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

        # Also remove from ChromaDB
        if success:
            project_path = _resolve_project_path(path)
            try:
                from delia.orchestration.vector_store import get_vector_store
                store = get_vector_store(project_path)
                store.get_collection(store.COLLECTION_PLAYBOOK).delete(ids=[bullet_id])
            except Exception as e:
                log.debug("chromadb_delete_failed", error=str(e))

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
        # confirm_compliance(task_description, bullets_applied, patterns_followed)
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

    elif action == "search":
        # Semantic search for playbook bullets via ChromaDB
        query = kwargs.get("query", "")
        if not query:
            return json.dumps({"error": "query required for search action"})

        limit = kwargs.get("limit", 5)
        project_path = _resolve_project_path(path)

        try:
            from delia.learning.retrieval import get_retriever
            retriever = get_retriever()

            # Use project name (basename) for filtering, not full path
            project_name = project_path.name
            scored_bullets = await retriever.retrieve_from_chromadb(
                query=query,
                task_type=task_type,
                project=project_name,
                project_path=project_path,
                limit=limit,
            )

            return json.dumps({
                "query": query,
                "task_type": task_type,
                "results": [
                    {
                        "id": sb.bullet.id,
                        "content": sb.bullet.content,
                        "section": sb.bullet.section,
                        "score": round(sb.final_score, 3),
                        "relevance": round(sb.relevance_score, 3),
                        "utility": round(sb.utility_score, 3),
                    }
                    for sb in scored_bullets
                ],
                "count": len(scored_bullets),
            })

        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})

    elif action == "index":
        # Index all playbook bullets to ChromaDB
        project_path = _resolve_project_path(path)

        try:
            from delia.learning.retrieval import get_retriever
            retriever = get_retriever()

            task_types = ["coding", "testing", "debugging", "security", "architecture",
                         "deployment", "performance", "api", "git", "project"]

            if task_type:
                task_types = [task_type]

            indexed_counts = {}
            project_name = project_path.name  # Use basename for consistent filtering
            for tt in task_types:
                bullets = playbook_manager.load_playbook(tt)
                if not bullets:
                    continue

                count = await retriever.index_bullets_to_chromadb(
                    bullets=bullets,
                    task_type=tt,
                    project=project_name,
                    project_path=project_path,
                )
                indexed_counts[tt] = count

            return json.dumps({
                "status": "indexed",
                "by_task_type": indexed_counts,
                "total": sum(indexed_counts.values()),
                "project_path": str(project_path),
            })

        except Exception as e:
            return json.dumps({"error": f"Indexing failed: {str(e)}"})

    elif action == "feedback":
        # Record detection feedback to improve auto_context accuracy
        from ..context_detector import get_pattern_learner

        message = kwargs.get("message")
        detected_task = kwargs.get("detected_task")
        correct_task = kwargs.get("correct_task")

        if not message or not detected_task or not correct_task:
            return json.dumps({"error": "message, detected_task, and correct_task required for feedback action"})

        project_path = _resolve_project_path(path)
        learner = get_pattern_learner(project_path)

        was_correct = detected_task == correct_task
        result = learner.record_feedback(
            message=message,
            detected_task=detected_task,
            correct_task=correct_task,
            was_correct=was_correct,
        )

        return json.dumps({
            "success": True,
            "was_correct": was_correct,
            "patterns_updated": result["patterns_updated"],
            "patterns_added": result["patterns_added"],
            "message": (
                "Detection was correct, patterns reinforced."
                if was_correct else
                f"Learning from feedback: {result['patterns_added']} new patterns added."
            ),
        })

    elif action == "learning_stats":
        # Get statistics about learned detection patterns
        from ..context_detector import get_pattern_learner

        project_path = _resolve_project_path(path)
        learner = get_pattern_learner(project_path)
        stats = learner.get_stats()

        return json.dumps({
            "project": str(project_path),
            "learned_patterns": stats,
            "suggestions": {
                "prune_command": "Use playbook(action='prune_patterns') to remove ineffective patterns",
                "effectiveness_threshold": 0.4,
            },
        }, indent=2)

    elif action == "prune_patterns":
        # Remove learned patterns with low effectiveness
        from ..context_detector import get_pattern_learner

        min_effectiveness = kwargs.get("min_effectiveness", 0.3)
        min_uses = kwargs.get("min_uses", 5)

        project_path = _resolve_project_path(path)
        learner = get_pattern_learner(project_path)
        removed = learner.prune_ineffective(min_effectiveness, min_uses)

        return json.dumps({
            "success": True,
            "patterns_removed": removed,
            "message": f"Pruned {removed} ineffective patterns." if removed > 0 else "No patterns needed pruning.",
        }, indent=2)

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Memory Tool (4 operations → 1)
# =============================================================================

async def memory_tool(
    action: Literal["list", "read", "write", "delete", "search", "index"],
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
        search - Semantic search across memories (query param)
        index - Re-index all memories to ChromaDB

    Args:
        action: The operation to perform
        name: Memory name (without .md extension)
        path: Optional project path (defaults to cwd)
        **kwargs: Action-specific parameters (query for search)

    Returns:
        JSON string or markdown content
    """
    project_path = _resolve_project_path(path)
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

        # Sync to ChromaDB for semantic search
        try:
            from delia.orchestration.vector_store import get_vector_store
            from delia.embeddings import get_embeddings_client

            client = await get_embeddings_client()
            embedding = await client.embed(content)

            if embedding is not None:
                store = get_vector_store(project_path)
                store.add_memory(
                    memory_id=name,
                    content=content,
                    embedding=embedding.tolist(),
                    name=name,
                    project=project_path.name,
                )
        except Exception as e:
            log.debug("chromadb_memory_sync_failed", error=str(e))

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

        # Also remove from ChromaDB
        try:
            from delia.orchestration.vector_store import get_vector_store
            store = get_vector_store(project_path)
            store.delete_by_filter(
                store.COLLECTION_MEMORIES,
                where={"name": name}
            )
        except Exception:
            pass  # ChromaDB cleanup is best-effort

        return json.dumps({
            "status": "deleted",
            "name": name
        })

    elif action == "search":
        query = kwargs.get("query", "")
        if not query:
            return json.dumps({"error": "query required for search action"})

        try:
            from delia.orchestration.vector_store import get_vector_store
            from delia.embeddings import get_embeddings_client

            # Get query embedding (uses singleton with query caching)
            client = await get_embeddings_client()
            query_embedding = await client.embed_query(query)

            if query_embedding is None:
                return json.dumps({"error": "Failed to generate query embedding"})

            store = get_vector_store(project_path)
            results = store.search_memories(
                query_embedding=query_embedding.tolist(),
                n_results=kwargs.get("limit", 5),
            )

            return json.dumps({
                "query": query,
                "results": [
                    {
                        "name": r["metadata"].get("name", r["id"]),
                        "score": round(r["score"], 3),
                        "preview": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                    }
                    for r in results
                ],
                "count": len(results),
            })

        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})

    elif action == "index":
        # Re-index all memories to ChromaDB using batch embeddings
        try:
            from delia.orchestration.vector_store import get_vector_store
            from delia.embeddings import get_embeddings_client

            client = await get_embeddings_client()
            store = get_vector_store(project_path)

            # Collect all memories for batch processing
            memories = []
            for file in memory_dir.glob("*.md"):
                content = file.read_text()
                memories.append((file.stem, content))

            if not memories:
                return json.dumps({
                    "status": "indexed",
                    "count": 0,
                    "project": str(project_path),
                })

            # Batch embed all memories at once
            contents = [m[1] for m in memories]
            embeddings = await client.embed_batch(contents, input_type="document")

            indexed = 0
            for (name, content), embedding in zip(memories, embeddings):
                # Skip zero vectors (failed embeddings)
                if embedding is None or (hasattr(embedding, 'sum') and embedding.sum() == 0):
                    continue

                store.add_memory(
                    memory_id=name,
                    content=content,
                    embedding=embedding.tolist(),
                    name=name,
                    project=project_path.name,
                )
                indexed += 1

            return json.dumps({
                "status": "indexed",
                "count": indexed,
                "project": str(project_path),
            })

        except Exception as e:
            return json.dumps({"error": f"Indexing failed: {str(e)}"})

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Session Tool (4 operations → 1)
# =============================================================================

async def session_tool(
    action: Literal["list", "stats", "compact", "delete", "snapshot"],
    session_id: str | None = None,
    **kwargs
) -> str:
    """Unified session management tool.

    Actions:
        list - List active conversation sessions
        stats - Get session statistics
        compact - Compact session history with LLM summarization
        delete - Delete session
        snapshot - Save task state for continuation (requires task_summary, pending_items)

    Args:
        action: The operation to perform
        session_id: Session ID (required for stats, compact, delete)
        **kwargs: Action-specific parameters (task_summary, pending_items, key_decisions, files_modified, next_steps for snapshot)

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

    elif action == "snapshot":
        # Save task state for continuation in a new conversation
        from datetime import datetime
        from ..playbook import get_playbook_manager
        from ..context import get_project_path

        task_summary = kwargs.get("task_summary")
        pending_items = kwargs.get("pending_items")

        if not task_summary or not pending_items:
            return json.dumps({"error": "task_summary and pending_items required for snapshot action"})

        key_decisions = kwargs.get("key_decisions")
        files_modified = kwargs.get("files_modified")
        next_steps = kwargs.get("next_steps")
        path = kwargs.get("path")

        pm = get_playbook_manager()
        if path:
            project_path = Path(path).resolve()
        elif pm.playbook_dir.exists():
            project_path = pm.playbook_dir.parent.parent
        else:
            project_path = get_project_path()

        memories_dir = project_path / ".delia" / "memories"
        memories_dir.mkdir(parents=True, exist_ok=True)

        try:
            pending_list = json.loads(pending_items) if pending_items else []
        except json.JSONDecodeError:
            pending_list = [pending_items]

        try:
            decisions_dict = json.loads(key_decisions) if key_decisions else {}
        except json.JSONDecodeError:
            decisions_dict = {"note": key_decisions} if key_decisions else {}

        try:
            files_list = json.loads(files_modified) if files_modified else []
        except json.JSONDecodeError:
            files_list = [files_modified] if files_modified else []

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Task Snapshot",
            "",
            f"*Captured: {timestamp}*",
            "",
            "## Summary",
            task_summary,
            "",
            "## Pending Items",
        ]

        for item in pending_list:
            lines.append(f"- [ ] {item}")
        lines.append("")

        if decisions_dict:
            lines.append("## Key Decisions")
            for key, value in decisions_dict.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if files_list:
            lines.append("## Files Modified")
            for f in files_list:
                lines.append(f"- `{f}`")
            lines.append("")

        if next_steps:
            lines.append("## Next Steps")
            lines.append(next_steps)
            lines.append("")

        lines.extend([
            "---",
            "*To continue: Read this file at the start of your next session.*",
        ])

        content = "\n".join(lines)
        snapshot_path = memories_dir / "task_snapshot.md"
        snapshot_path.write_text(content)

        return json.dumps({
            "status": "snapshot_saved",
            "path": str(snapshot_path.relative_to(project_path)),
            "message": (
                "Task state captured. Suggest user start a new conversation. "
                "The next agent should call memory(action='read', name='task_snapshot') to resume."
            ),
            "pending_count": len(pending_list),
        }, indent=2)

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Profiles Tool (4 operations → 1)
# =============================================================================

async def profiles_tool(
    action: Literal["recommend", "check", "reevaluate", "cleanup", "index", "search"],
    path: str | None = None,
    **kwargs
) -> str:
    """Unified profile/evaluation management tool.

    Actions:
        recommend - Recommend starter profiles for project tech stack
        check - Check if pattern re-evaluation is needed
        reevaluate - Re-analyze project for pattern gaps
        cleanup - Remove obsolete profile templates
        index - Index profiles to ChromaDB for semantic search
        search - Search profiles semantically by query

    Args:
        action: The operation to perform
        path: Optional project path (defaults to cwd)
        **kwargs: Action-specific parameters (query, limit for search)

    Returns:
        JSON string with operation result
    """

    project_path = _resolve_project_path(path)

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

    elif action == "index":
        # Index all profiles to ChromaDB for semantic search
        from ..embeddings import get_embeddings_client
        from ..orchestration.vector_store import get_vector_store

        profiles_dir = project_path / ".delia" / "profiles"
        if not profiles_dir.exists():
            return json.dumps({"error": "No profiles directory found", "indexed": 0})

        store = get_vector_store(project_path)
        client = await get_embeddings_client()
        project_name = project_path.name

        indexed = 0
        profiles_found = list(profiles_dir.glob("*.md"))

        for profile_file in profiles_found:
            try:
                content = profile_file.read_text()
                if not content.strip():
                    continue

                # Generate embedding
                embedding = await client.embed(content[:2000])  # Limit for embedding
                if embedding is None:
                    continue

                # Convert numpy array to list if needed
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()

                store.add_profile(
                    profile_id=f"{project_name}:{profile_file.stem}",
                    content=content,
                    embedding=embedding,
                    name=profile_file.stem,
                    project=project_name,
                )
                indexed += 1
            except Exception as e:
                log.debug("profile_index_error", profile=profile_file.stem, error=str(e))

        log.info("profiles_indexed_to_chromadb", count=indexed, project=project_name)
        return json.dumps({"indexed": indexed, "total": len(profiles_found)})

    elif action == "search":
        # Search profiles semantically
        query = kwargs.get("query")
        limit = kwargs.get("limit", 5)

        if not query:
            return json.dumps({"error": "query parameter required for search"})

        from ..embeddings import get_embeddings_client
        from ..orchestration.vector_store import get_vector_store

        client = await get_embeddings_client()
        store = get_vector_store(project_path)
        project_name = project_path.name

        # Embed query
        query_embedding = await client.embed_query(query)
        if query_embedding is None:
            return json.dumps({"error": "Failed to embed query", "results": []})

        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        results = store.search_profiles(
            query_embedding=query_embedding,
            project=project_name,
            n_results=limit,
        )

        return json.dumps({
            "query": query,
            "count": len(results),
            "results": results,
        })

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Project Tool (5 operations → 1)
# =============================================================================

async def project_tool(
    action: Literal["init", "scan", "analyze", "sync", "read_instructions", "profile", "overview"],
    path: str,
    **kwargs
) -> str:
    """Unified project initialization and management tool.

    Actions:
        init - Initialize Delia framework for project
        scan - Incremental codebase scanning
        analyze - Create Delia index from analysis
        sync - Sync CLAUDE.md to all AI agent configs
        read_instructions - Read existing instruction files
        profile - Load a specific profile by name
        overview - Get hierarchical project structure summary

    Args:
        action: The operation to perform
        path: Project path (required)
        **kwargs: Action-specific parameters

    Returns:
        JSON string with operation result
    """

    project_path = Path(path)

    if action == "init":
        # init_project(path, force, skip_index, parallel, use_calling_agent)
        force = kwargs.get("force", False)
        skip_index = kwargs.get("skip_index", False)
        parallel = kwargs.get("parallel", 4)
        use_calling_agent = kwargs.get("use_calling_agent", False)

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
        # List DELIA.md instruction files loaded into context
        from ..project_memory import get_project_memory, list_project_memories

        memories = list_project_memories()
        pm = get_project_memory()
        state = pm._state

        result = {
            "status": "success",
            "instruction_files": memories,
            "total_size": state.total_size if state else 0,
            "load_hierarchy": [
                "1. ~/.delia/DELIA.md (user defaults)",
                "2. ./DELIA.md (project instructions)",
                "3. ./.delia/DELIA.md (project config)",
                "4. ./.delia/rules/*.md (modular rules)",
                "5. ./DELIA.local.md (local overrides)",
            ],
            "message": "Use project(action='sync', content='...') to update instruction files.",
        }
        return json.dumps(result)

    elif action == "profile":
        # Load a specific profile by name
        name = kwargs.get("name")
        if not name:
            return json.dumps({"error": "name required for profile action"})

        # Ensure .md extension
        if not name.endswith(".md"):
            name = f"{name}.md"

        profiles_dir = project_path / ".delia" / "profiles"
        templates_dir = Path(__file__).parent.parent / "templates" / "profiles"

        content = None
        source = None

        # Check project profiles first
        profile_path = profiles_dir / name
        if profile_path.exists():
            try:
                content = profile_path.read_text()
                source = "project"
            except Exception as e:
                return json.dumps({"error": f"Failed to read profile: {e}"})

        # Fall back to templates
        if content is None:
            template_path = templates_dir / name
            if template_path.exists():
                try:
                    content = template_path.read_text()
                    source = "template"
                except Exception as e:
                    return json.dumps({"error": f"Failed to read template: {e}"})

        # Profile not found - list available
        if content is None:
            available = []
            if profiles_dir.exists():
                available.extend(p.name for p in profiles_dir.glob("*.md"))
            return json.dumps({
                "error": f"Profile '{name}' not found",
                "available_profiles": sorted(available),
            }, indent=2)

        return json.dumps({
            "name": name,
            "source": source,
            "content": content,
        }, indent=2)

    elif action == "overview":
        # Get hierarchical project structure summary (was project_overview)
        from ..orchestration.summarizer import get_summarizer

        summarizer = get_summarizer()
        await summarizer.initialize()
        overview = summarizer.get_project_overview()

        if not overview or overview == "Project overview not yet generated.":
            return json.dumps({"error": "Project not yet indexed. Run project(action='scan') first."}, indent=2)

        return overview

    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# =============================================================================
# Admin Tool (3 operations → 1)
# =============================================================================

async def admin_tool(
    action: Literal[
        "switch_model", "queue_status", "mcp_servers", "vector_store",
        "cleanup_legacy", "cleanup_project", "cleanup_all",
        "models", "dashboard", "model_info", "tools", "describe",
        "framework_stats", "framework_cleanup"
    ],
    **kwargs
) -> str:
    """Unified admin/system management tool (ADR-010).

    Actions:
        switch_model - Switch model for specific tier
        queue_status - Get model queue system status
        mcp_servers - Manage external MCP servers
        vector_store - Get ChromaDB vector store stats
        cleanup_legacy - Remove old global sessions/memories/playbooks
        cleanup_project - Clean a project's .delia/ directory
        cleanup_all - Full cleanup of all legacy data
        models - List all configured models
        dashboard - Get dashboard URL
        model_info - Get info about specific model (requires model_name)
        tools - List available tools (optional category)
        describe - Get tool details (requires tool_name)
        framework_stats - Get framework enforcement stats
        framework_cleanup - Clean stale trackers (optional max_age_hours)

    Args:
        action: The operation to perform
        **kwargs: Action-specific parameters

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
        from ..cleanup import cleanup_project_delia_dir

        project_path = _resolve_project_path(kwargs.get("path"))
        dry_run = kwargs.get("dry_run", True)
        results = cleanup_project_delia_dir(project_path, dry_run=dry_run)
        return json.dumps(results, indent=2)

    elif action == "cleanup_all":
        # Full cleanup of all legacy data
        from ..cleanup import cleanup_all
        dry_run = kwargs.get("dry_run", True)
        results = cleanup_all(dry_run=dry_run)
        return json.dumps(results, indent=2)

    elif action == "vector_store":
        # Get ChromaDB vector store statistics for current project
        try:
            from ..orchestration.vector_store import get_vector_store
            project_path = _resolve_project_path(kwargs.get("path"))
            store = get_vector_store(project_path)
            stats = store.get_stats()
            stats["project"] = str(project_path)
            return json.dumps(stats, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Vector store unavailable: {str(e)}"})

    # ADR-010: New actions absorbed from standalone tools
    elif action == "models":
        from .admin import models_impl
        return await models_impl()

    elif action == "dashboard":
        from .admin import dashboard_url_impl
        return await dashboard_url_impl()

    elif action == "model_info":
        from .admin import get_model_info_impl
        model_name = kwargs.get("model_name")
        if not model_name:
            return json.dumps({"error": "model_info requires model_name parameter"})
        return await get_model_info_impl(model_name)

    elif action == "tools":
        # List available tools by category
        from .registry import TOOL_CATEGORIES
        category = kwargs.get("category")

        # Simplified tool listing
        return json.dumps({
            "hint": "Use the MCP tools/list endpoint for full tool listing",
            "categories": list(TOOL_CATEGORIES.keys()),
            "filter": category,
        }, indent=2)

    elif action == "describe":
        tool_name = kwargs.get("tool_name")
        if not tool_name:
            return json.dumps({"error": "describe requires tool_name parameter"})
        return json.dumps({
            "hint": "Use the MCP tools/get endpoint for tool details",
            "tool": tool_name,
        }, indent=2)

    elif action == "framework_stats":
        from .handlers_enforcement import get_manager
        manager = get_manager()
        stats = manager.get_stats()

        project_details = {}
        for project in stats["projects"]:
            tracker = manager.get_tracker(project)
            project_details[project] = {
                "context_started": tracker.is_context_started(project),
                "playbook_queried": tracker.was_playbook_queried(project),
                "last_activity": tracker.get_last_activity(),
            }

        return json.dumps({
            "result": {
                "active_projects": stats["active_projects"],
                "projects": project_details,
            }
        }, indent=2)

    elif action == "framework_cleanup":
        from .handlers_enforcement import get_manager
        manager = get_manager()
        max_age_hours = kwargs.get("max_age_hours", 1.0)
        max_age_seconds = int(max_age_hours * 3600)
        removed = manager.cleanup_stale(max_age_seconds)

        return json.dumps({
            "result": {
                "trackers_removed": removed,
                "remaining_projects": len(manager.list_projects()),
            }
        }, indent=2)

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
        action: Literal["add", "write", "delete", "prune", "list", "stats", "confirm", "search", "index", "feedback", "learning_stats", "prune_patterns"],
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
        query: str | None = None,
        limit: int = 5,
        message: str | None = None,
        detected_task: str | None = None,
        correct_task: str | None = None,
        min_effectiveness: float = 0.3,
        min_uses: int = 5,
    ) -> str:
        """Manage playbooks: add, write, delete, prune, list, stats, confirm, search, index, feedback, learning_stats, prune_patterns."""
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
            query=query,
            limit=limit,
            message=message,
            detected_task=detected_task,
            correct_task=correct_task,
            min_effectiveness=min_effectiveness,
            min_uses=min_uses,
        )

    @mcp.tool()
    async def memory(
        action: Literal["list", "read", "write", "delete", "search", "index"],
        name: str | None = None,
        path: str | None = None,
        content: str | None = None,
        append: bool = False,
        query: str | None = None,
        limit: int = 5,
    ) -> str:
        """Manage memories (.delia/memories/): list, read, write, delete, search, index."""
        return await memory_tool(
            action=action,
            name=name,
            path=path,
            content=content,
            append=append,
            query=query,
            limit=limit,
        )

    @mcp.tool()
    async def session(
        action: Literal["list", "stats", "compact", "delete", "snapshot"],
        session_id: str | None = None,
        force: bool = False,
        task_summary: str | None = None,
        pending_items: str | None = None,
        key_decisions: str | None = None,
        files_modified: str | None = None,
        next_steps: str | None = None,
        path: str | None = None,
    ) -> str:
        """Unified session management (ADR-010). Actions: list, stats, compact, delete, snapshot."""
        return await session_tool(
            action=action,
            session_id=session_id,
            force=force,
            task_summary=task_summary,
            pending_items=pending_items,
            key_decisions=key_decisions,
            files_modified=files_modified,
            next_steps=next_steps,
            path=path,
        )

    @mcp.tool()
    async def profiles(
        action: Literal["recommend", "check", "reevaluate", "cleanup", "index", "search"],
        path: str | None = None,
        analyze_gaps: bool = True,
        force: bool = False,
        auto_remove: bool = False,
        query: str | None = None,
        limit: int = 5,
    ) -> str:
        """Unified profile/evaluation management (ADR-009). Actions: recommend, check, reevaluate, cleanup, index, search."""
        return await profiles_tool(
            action=action,
            path=path,
            analyze_gaps=analyze_gaps,
            force=force,
            auto_remove=auto_remove,
            query=query,
            limit=limit,
        )

    @mcp.tool()
    async def project(
        action: Literal["init", "scan", "analyze", "sync", "read_instructions", "profile", "overview"],
        path: str,
        force: bool = False,
        skip_index: bool = False,
        parallel: int = 4,
        use_calling_agent: bool = False,
        max_files: int = 20,
        preview_chars: int = 500,
        phase: str = "overview",
        project_summary: str | None = None,
        coding_bullets: str | None = None,
        content: str | None = None,
        name: str | None = None,
    ) -> str:
        """Unified project initialization and management (ADR-009). Actions: init, scan, analyze, sync, read_instructions, profile, overview."""
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
            name=name,
        )

    @mcp.tool()
    async def admin(
        action: Literal[
            "switch_model", "queue_status", "mcp_servers", "vector_store",
            "models", "dashboard", "model_info", "tools", "describe",
            "framework_stats", "framework_cleanup"
        ],
        tier: str | None = None,
        model_name: str | None = None,
        mcp_action: str = "status",
        server_id: str | None = None,
        command: str | None = None,
        name: str | None = None,
        env: str | None = None,
        tool_name: str | None = None,
        category: str | None = None,
        max_age_hours: float = 1.0,
    ) -> str:
        """Unified admin/system management (ADR-010). Actions: switch_model, queue_status, mcp_servers, vector_store, models, dashboard, model_info, tools, describe, framework_stats, framework_cleanup."""
        return await admin_tool(
            action=action,
            tier=tier,
            model_name=model_name,
            mcp_action=mcp_action,
            server_id=server_id,
            tool_name=tool_name,
            category=category,
            max_age_hours=max_age_hours,
            command=command,
            name=name,
            env=env,
        )
