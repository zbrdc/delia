# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Administrative tools for Delia.

This module provides:
1. Module-level implementation functions (importable by consolidated.py)
2. MCP tool registration for non-consolidated tools only

Per ADR-009, the following are ONLY exposed via consolidated.py:
- init_project -> project(action="init")
- scan_codebase -> project(action="scan")
- analyze_and_index -> project(action="analyze")
- cleanup_profiles -> profiles(action="cleanup")
- switch_model -> admin(action="switch_model")
- queue_status -> admin(action="queue_status")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import humanize
import structlog
from fastmcp import FastMCP

from ..container import get_container
from ..config import get_backend_health, VALID_MODELS
from ..routing import BackendScorer
from ..agent_sync import sync_agent_instruction_files

log = structlog.get_logger()


# =============================================================================
# Implementation Functions (module-level, importable)
# =============================================================================

async def health_impl() -> str:
    """Check health status of Delia and all configured GPU backends."""
    container = get_container()
    health_status = await container.backend_manager.get_health_status()
    weights = container.backend_manager.get_scoring_weights()
    scorer = BackendScorer(weights=weights)
    backend_lookup = {b.id: b for b in container.backend_manager.backends.values()}

    for backend_info in health_status["backends"]:
        backend_obj = backend_lookup.get(backend_info["id"])
        if backend_obj and backend_info["enabled"]:
            backend_info["score"] = round(scorer.score(backend_obj), 3)
            from ..config import get_backend_metrics
            metrics = get_backend_metrics(backend_info["id"])
            if metrics.total_requests > 0:
                backend_info["metrics"] = {
                    "success_rate": f"{metrics.success_rate * 100:.1f}%",
                    "latency_p50_ms": round(metrics.latency_p50, 1),
                    "throughput_tps": round(metrics.tokens_per_second, 1),
                    "total_requests": metrics.total_requests,
                }

    model_usage, _, _, _ = container.stats_service.get_snapshot()
    total_quick_tokens = model_usage["quick"]["tokens"]
    total_coder_tokens = model_usage["coder"]["tokens"]
    total_moe_tokens = model_usage["moe"]["tokens"]
    total_thinking_tokens = model_usage["thinking"]["tokens"]
    local_tokens = total_quick_tokens + total_coder_tokens + total_moe_tokens + total_thinking_tokens
    local_calls = sum(model_usage[t]["calls"] for t in ("quick", "coder", "moe", "thinking"))

    local_savings = (local_tokens / 1000) * container.config.gpt4_cost_per_1k_tokens
    from ..voting_stats import get_voting_stats_tracker
    voting_stats = get_voting_stats_tracker().get_stats()

    status = {
        "status": health_status["status"],
        "active_backend": health_status["active_backend"],
        "backends": health_status["backends"],
        "routing": health_status["routing"],
        "usage": {
            "total_calls": humanize.intcomma(local_calls),
            "total_tokens": humanize.intword(local_tokens),
            "estimated_savings": f"${local_savings:,.2f}",
        },
        "voting": voting_stats,
    }
    return json.dumps(status, indent=2)


async def dashboard_url_impl() -> str:
    """Get the URL of the running Delia dashboard."""
    from ..mcp_server import _dashboard_port, _dashboard_process

    if _dashboard_process is None or _dashboard_port is None:
        return json.dumps({
            "status": "not_running",
            "message": "Dashboard not running. Ensure dashboard is built: cd dashboard && npm run build",
        })

    if _dashboard_process.poll() is not None:
        return json.dumps({
            "status": "stopped",
            "message": "Dashboard process exited unexpectedly",
        })

    return json.dumps({
        "status": "running",
        "url": f"http://localhost:{_dashboard_port}",
        "port": _dashboard_port,
    })


async def models_impl() -> str:
    """List all configured models across all GPU backends."""
    container = get_container()
    backends = container.backend_manager.get_enabled_backends()
    from ..mcp_server import get_loaded_models
    loaded = await get_loaded_models()

    result = {"backends": [], "currently_loaded": loaded}
    for b in backends:
        result["backends"].append({"id": b.id, "name": b.name, "provider": b.provider, "models": b.models})
    return json.dumps(result, indent=2)


async def switch_backend_impl(backend_id: str) -> str:
    """Switch the active LLM backend."""
    container = get_container()
    if container.backend_manager.set_active_backend(backend_id):
        return f"Switched to backend: {backend_id}"
    return f"Error: Backend '{backend_id}' not found or disabled."


async def get_model_info_impl(model_name: str) -> str:
    """Get detailed information about a specific model."""
    from ..mcp_server import get_model_info
    info = get_model_info(model_name)
    return json.dumps(info, indent=2)


async def init_project(
    path: str,
    force: bool = False,
    skip_index: bool = False,
    parallel: int = 4,
    use_calling_agent: bool = True,
) -> str:
    """
    Initialize a project with Delia's ACE Framework.

    IMPORTANT: You MUST provide the 'path' parameter with the absolute path
    to the project you want to initialize. Do NOT omit this parameter.

    This generates per-project instruction files customized to the detected tech stack:
    - CLAUDE.md, .gemini/instructions.md, .github/copilot-instructions.md
    - .delia/playbooks/*.json (strategic bullets)
    - .delia/profiles/*.md (starter templates)

    Args:
        path: REQUIRED. Absolute path to the project to initialize.
        force: Overwrite existing framework files
        skip_index: Skip indexing (use existing analysis)
        parallel: Number of parallel summarization tasks
        use_calling_agent: If True (default), calling agent does summarization

    Returns:
        JSON with initialization results.
    """
    from ..playbook import detect_tech_stack, playbook_manager
    from ..orchestration.summarizer import get_summarizer
    from ..orchestration.graph import get_symbol_graph
    from ..llm import init_llm_module
    from ..queue import ModelQueue

    container = get_container()

    if not path:
        return json.dumps({
            "error": "path parameter is REQUIRED",
            "message": "You must specify the absolute path to the project to initialize.",
            "example": "init_project(path='/home/user/projects/my-app')",
        })

    project_root = Path(path).resolve()
    if not project_root.exists():
        return json.dumps({
            "error": f"Path does not exist: {path}",
            "message": "Provide a valid absolute path to an existing project directory.",
        })
    project_name = project_root.name
    results: dict[str, Any] = {"project": project_name, "path": str(project_root), "steps": []}

    health_status = await container.backend_manager.get_health_status()
    available_backends = [
        b for b in health_status["backends"]
        if b.get("enabled") and b.get("available")
    ]
    has_local_backends = len(available_backends) > 0

    try:
        from ..orchestration.constants import CODE_EXTENSIONS, should_ignore_file
        code_files = []
        for file_path in project_root.rglob("*"):
            if should_ignore_file(file_path):
                continue
            if file_path.suffix in CODE_EXTENSIONS and file_path.is_file():
                code_files.append(str(file_path.relative_to(project_root)))
        code_files = sorted(code_files)

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
        requirements = project_root / "requirements.txt"
        if requirements.exists():
            try:
                for line in requirements.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        dependencies.append(line.split("==")[0].split(">=")[0].split("<")[0])
            except Exception:
                pass
        package_json = project_root / "package.json"
        if package_json.exists():
            try:
                pkg = json.loads(package_json.read_text())
                dependencies.extend(pkg.get("dependencies", {}).keys())
                dependencies.extend(pkg.get("devDependencies", {}).keys())
            except Exception:
                pass

        tech_stack = detect_tech_stack(code_files, dependencies)
        results["tech_stack"] = {
            "primary_language": tech_stack.get("primary_language"),
            "frameworks": tech_stack.get("frameworks", []),
        }

        use_manual_mode = use_calling_agent or not has_local_backends
        reason = "use_calling_agent=True" if use_calling_agent else "no_local_backends"

        if use_manual_mode:
            log.info("init_project_manual_mode", project=project_name, reason=reason)
            results["manual_mode"] = True
            results["status"] = "INCOMPLETE - TOOL CALLS REQUIRED"

            results["WHAT_IS_DELIA"] = (
                "ACE Framework: Project-specific playbooks with learned strategies. "
                "Call get_playbook() before coding, report_feedback() after."
            )

            results["STOP_AND_READ"] = (
                "DO NOT just describe what you will do. "
                "You MUST call the tools below RIGHT NOW to complete initialization."
            )
            results["REQUIRED_TOOL_CALLS"] = [
                {
                    "order": 1,
                    "tool": "mcp__delia__scan_codebase",
                    "purpose": "Read key files automatically",
                    "args": {"path": str(project_root)},
                },
                {
                    "order": 2,
                    "tool": "mcp__delia__analyze_and_index",
                    "purpose": "Submit YOUR analysis - generates summaries, playbooks, and syncs files in ONE call",
                    "args": {
                        "path": str(project_root),
                        "project_summary": "<YOUR analysis as JSON>",
                        "coding_bullets": "<YOUR bullets as JSON array>",
                    },
                },
            ]
            results["FLOW"] = "scan_codebase → (you analyze) → analyze_and_index → DONE"

            results["total_files"] = len(code_files)

            delia_dir = project_root / ".delia"
            delia_dir.mkdir(parents=True, exist_ok=True)
            playbooks_dir = delia_dir / "playbooks"
            playbooks_dir.mkdir(exist_ok=True)
            profiles_dir = delia_dir / "profiles"
            profiles_dir.mkdir(exist_ok=True)

            from ..agent_sync import detect_ai_agents
            detected_agents = detect_ai_agents(project_root)
            has_existing_instructions = (project_root / "CLAUDE.md").exists()

            results["detected_agents"] = [
                info["description"] for info in detected_agents.values() if info["exists"]
            ]
            results["has_existing_claude_md"] = has_existing_instructions

            from ..playbook import recommend_profiles, format_recommendations
            recommendations = recommend_profiles(tech_stack, project_root)
            formatted = format_recommendations(recommendations)

            template_dir = Path(__file__).parent.parent / "templates" / "profiles"
            starter_profiles = []
            recommended_files = {r.profile for r in recommendations}

            if template_dir.exists():
                for template_file in template_dir.glob("*.md"):
                    if template_file.name in recommended_files:
                        dest = profiles_dir / template_file.name
                        if not dest.exists() or force:
                            dest.write_text(template_file.read_text())
                            starter_profiles.append(template_file.name)

            results["starter_profiles_copied"] = starter_profiles

            if has_existing_instructions:
                results["UPDATE_INSTRUCTIONS"] = (
                    "Project has existing CLAUDE.md. Review the profiles in .delia/profiles/ "
                    "and UPDATE the existing instructions with any missing patterns or best practices."
                )

            results["profile_recommendations"] = {
                "high_priority": formatted["high_priority"][:5],
                "total": formatted["summary"]["total_recommendations"],
            }

            return json.dumps(results, indent=2)

        # Standard path: local backends available
        if not skip_index:
            model_queue = ModelQueue()
            init_llm_module(
                stats_callback=lambda *a, **k: None,
                save_stats_callback=lambda: None,
                model_queue=model_queue,
            )

            summarizer = get_summarizer()
            graph = get_symbol_graph()

            graph_count = await graph.sync(force=force)
            results["steps"].append({"step": "symbol_graph", "files": graph_count})

            summary_count = await summarizer.sync_project(force=force, summarize=True, parallel=parallel)
            results["steps"].append({"step": "summaries", "files": summary_count})
        else:
            results["steps"].append({"step": "indexing", "skipped": True})

        from ..agent_sync import detect_ai_agents
        detected_agents = detect_ai_agents(project_root)

        files_written = []
        claude_md_path = project_root / "CLAUDE.md"

        if force or not claude_md_path.exists():
            from ..cli import _generate_claude_md
            claude_md_content = _generate_claude_md(project_name, tech_stack, project_root)
            files_written, detected_agents = sync_agent_instruction_files(
                project_root, claude_md_content, force=force
            )
        else:
            log.info("skipping_instruction_files", reason="CLAUDE.md exists, use force=True to overwrite")
            results["instruction_files_skipped"] = "Existing files preserved. Use force=True to overwrite."

        results["detected_agents"] = {
            k: {"description": v["description"], "exists": v["exists"], "updated": v.get("updated", False)}
            for k, v in detected_agents.items()
        }

        if files_written:
            results["files_written"] = files_written

        from ..playbook import generate_project_playbook
        playbook_count = await generate_project_playbook(summarizer if not skip_index else get_summarizer())
        results["steps"].append({"step": "playbook", "bullets": playbook_count})

        results["status"] = "success"
        results["manual_mode"] = False
        log.info("init_project_complete", project=project_name, files=len(files_written))

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        log.error("init_project_failed", project=project_name, error=str(e))

    return json.dumps(results, indent=2)


async def scan_codebase(
    path: str,
    max_files: int = 20,
    preview_chars: int = 500,
    phase: str = "overview",
) -> str:
    """
    Scan a codebase in manageable chunks for ACE initialization.

    Phases:
    - "overview": Quick scan returning structure and file list (default)
    - "manifests": Read package.json, pyproject.toml, etc.
    - "entry_points": Read main entry files (index.ts, main.py, etc.)
    - "samples": Read sample source files to understand patterns
    """
    from .models import (
        DirectoryInfo,
        FilePreview,
        ScanStructure,
        ScanOverviewResponse,
        ScanManifestsResponse,
        ScanEntryPointsResponse,
        ScanSamplesResponse,
    )

    project_root = Path(path).resolve()
    if not project_root.exists():
        return json.dumps({"error": f"Path does not exist: {path}"})

    def read_file_preview(file_path: Path, chars: int = 500) -> FilePreview | None:
        try:
            content = file_path.read_text(errors='ignore')
            preview = content[:chars] + ("..." if len(content) > chars else "")
            return FilePreview(
                path=str(file_path.relative_to(project_root)),
                size=file_path.stat().st_size,
                lines=content.count('\n'),
                preview=preview,
            )
        except Exception as e:
            log.debug("scan_file_error", file=str(file_path), error=str(e))
            return None

    if phase == "overview":
        dirs: list[DirectoryInfo] = []
        root_files: list[str] = []

        for item in sorted(project_root.iterdir()):
            if item.name.startswith('.') and item.name not in ['.github', '.delia']:
                continue
            if item.is_dir():
                file_count = 0
                try:
                    file_count = len(list(item.rglob("*")))
                except PermissionError:
                    pass
                dirs.append(DirectoryInfo(name=item.name, file_count=file_count))
            elif item.is_file():
                root_files.append(item.name)

        extensions: dict[str, int] = {}
        for f in project_root.rglob("*"):
            if f.is_file() and f.suffix:
                extensions[f.suffix] = extensions.get(f.suffix, 0) + 1

        top_extensions = dict(sorted(extensions.items(), key=lambda x: -x[1])[:5])

        response = ScanOverviewResponse(
            project=project_root.name,
            path=str(project_root),
            structure=ScanStructure(dirs=dirs, root_files=root_files),
            top_extensions=top_extensions,
            total_files=sum(extensions.values()),
            has_package_json=(project_root / "package.json").exists(),
            has_pyproject=(project_root / "pyproject.toml").exists(),
            has_src=(project_root / "src").exists(),
            has_tests=(project_root / "tests").exists() or (project_root / "test").exists(),
            NEXT_PHASES=[
                "Call scan_codebase with phase='manifests' to read package files",
                "Call scan_codebase with phase='entry_points' to read main files",
                "Call scan_codebase with phase='samples' to read source samples",
            ],
        )
        return response.model_dump_json(indent=2)

    elif phase == "manifests":
        manifests = ["package.json", "pyproject.toml", "Cargo.toml", "go.mod",
                    "pom.xml", "build.gradle", "Gemfile", "tsconfig.json",
                    "vite.config.ts", "next.config.js"]
        files_read: list[FilePreview] = []
        for m in manifests:
            mp = project_root / m
            if mp.exists() and len(files_read) < max_files:
                if info := read_file_preview(mp, preview_chars):
                    files_read.append(info)

        response = ScanManifestsResponse(
            project=project_root.name,
            path=str(project_root),
            manifest_files=files_read,
            NEXT="Now you understand dependencies. Call phase='entry_points' next.",
        )
        return response.model_dump_json(indent=2)

    elif phase == "entry_points":
        entry_patterns = [
            "src/index.ts", "src/index.tsx", "src/App.tsx", "src/main.ts",
            "src/main.py", "app/main.py", "main.py", "app.py",
            "index.js", "src/index.js", "README.md", "CLAUDE.md",
        ]
        files_read: list[FilePreview] = []
        for pattern in entry_patterns:
            fp = project_root / pattern
            if fp.exists() and len(files_read) < max_files:
                if info := read_file_preview(fp, preview_chars):
                    files_read.append(info)

        response = ScanEntryPointsResponse(
            project=project_root.name,
            path=str(project_root),
            entry_files=files_read,
            NEXT="Now you understand entry points. Call phase='samples' for code patterns.",
        )
        return response.model_dump_json(indent=2)

    elif phase == "samples":
        src_dirs = ["src", "app", "lib", "components", "pages", "hooks"]
        code_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}
        files_read: list[FilePreview] = []

        for src_dir in src_dirs:
            src_path = project_root / src_dir
            if src_path.exists() and src_path.is_dir():
                for code_file in sorted(src_path.rglob("*"))[:50]:
                    if code_file.is_file() and code_file.suffix in code_extensions:
                        if len(files_read) < max_files:
                            if info := read_file_preview(code_file, preview_chars):
                                files_read.append(info)
                break

        all_previews = " ".join(f.preview for f in files_read)
        patterns: list[str] = []
        pattern_signals = {
            "React": ["useState", "useEffect", "jsx"],
            "Async": ["async ", "await ", "Promise"],
            "TypeScript": ["interface ", "type ", ": string"],
            "Testing": ["describe(", "it(", "test(", "expect("],
        }
        for name, signals in pattern_signals.items():
            if any(s in all_previews for s in signals):
                patterns.append(name)

        response = ScanSamplesResponse(
            project=project_root.name,
            path=str(project_root),
            sample_files=files_read,
            detected_patterns=patterns,
            NEXT="Now call analyze_and_index with your analysis to complete initialization.",
        )
        return response.model_dump_json(indent=2)

    else:
        return json.dumps({"error": f"Unknown phase: {phase}. Use: overview, manifests, entry_points, samples"})


async def analyze_and_index(
    path: str,
    project_summary: str,
    coding_bullets: str,
    testing_bullets: str = "[]",
    architecture_bullets: str = "[]",
    debugging_bullets: str = "[]",
    project_bullets: str = "[]",
    git_bullets: str = "[]",
    security_bullets: str = "[]",
    deployment_bullets: str = "[]",
    api_bullets: str = "[]",
    performance_bullets: str = "[]",
) -> str:
    """
    Submit your analysis of the codebase to create the ACE Framework index.

    After calling scan_codebase, analyze the code and call THIS tool with:
    - A structured project summary
    - Playbook bullets for each category

    This tool stores everything and syncs instruction files in one call.
    """
    from ..playbook import get_playbook_manager

    project_root = Path(path).resolve()
    if not project_root.exists():
        return json.dumps({"error": f"Path does not exist: {path}"})

    results = {"path": str(project_root), "indexed": []}

    delia_dir = project_root / ".delia"
    delia_dir.mkdir(parents=True, exist_ok=True)
    playbooks_dir = delia_dir / "playbooks"
    playbooks_dir.mkdir(exist_ok=True)

    try:
        summary_data = json.loads(project_summary) if isinstance(project_summary, str) else project_summary
        summary_path = delia_dir / "project_summary.json"
        summary_path.write_text(json.dumps(summary_data, indent=2))
        results["indexed"].append("project_summary.json")
    except Exception as e:
        results["summary_error"] = str(e)
        summary_data = {}

    pm = get_playbook_manager()
    pm.set_project(project_root)

    playbook_data = {
        "coding": coding_bullets,
        "testing": testing_bullets,
        "architecture": architecture_bullets,
        "debugging": debugging_bullets,
        "project": project_bullets,
        "git": git_bullets,
        "security": security_bullets,
        "deployment": deployment_bullets,
        "api": api_bullets,
        "performance": performance_bullets,
    }

    for task_type, bullets_json in playbook_data.items():
        try:
            bullets = json.loads(bullets_json) if isinstance(bullets_json, str) else bullets_json
            if bullets:
                playbook_path = playbooks_dir / f"{task_type}.json"

                existing_bullets = []
                existing_contents = set()
                if playbook_path.exists():
                    try:
                        existing_bullets = json.loads(playbook_path.read_text())
                        existing_contents = {b.get("content", "").strip().lower() for b in existing_bullets if isinstance(b, dict)}
                    except Exception:
                        pass

                new_count = 0
                for b in bullets:
                    if isinstance(b, str):
                        content = b.strip()
                        if content.lower() not in existing_contents:
                            existing_bullets.append({
                                "content": content,
                                "section": "general_strategies",
                                "source": "learned",
                            })
                            existing_contents.add(content.lower())
                            new_count += 1
                    elif isinstance(b, dict):
                        content = b.get("content", str(b)).strip()
                        if content.lower() not in existing_contents:
                            existing_bullets.append({
                                "content": content,
                                "section": b.get("section", "general_strategies"),
                                "source": "learned",
                            })
                            existing_contents.add(content.lower())
                            new_count += 1

                playbook_path.write_text(json.dumps(existing_bullets, indent=2))
                seed_count = sum(1 for b in existing_bullets if isinstance(b, dict) and b.get("source") == "seed")
                results["indexed"].append(f"playbooks/{task_type}.json ({new_count} new + {seed_count} seeds = {len(existing_bullets)} total)")
        except Exception as e:
            results[f"{task_type}_error"] = str(e)

    try:
        from ..cli import _generate_claude_md

        tech_stack = summary_data.get("tech_stack", {}) if isinstance(summary_data, dict) else {}

        claude_md = project_root / "CLAUDE.md"
        if not claude_md.exists():
            content = _generate_claude_md(project_root.name, tech_stack, project_root)
            files_written, _ = sync_agent_instruction_files(project_root, content, force=False)
            results["instruction_files"] = files_written
        else:
            results["instruction_files"] = "CLAUDE.md exists, preserved (use sync_instruction_files to update)"
    except Exception as e:
        results["sync_error"] = str(e)

    results["status"] = "indexed"
    log.info("project_analyzed_and_indexed", path=str(project_root), items=len(results["indexed"]))

    return json.dumps(results, indent=2)


async def cleanup_profiles_impl(
    path: str | None = None,
    auto_remove: bool = False,
) -> str:
    """
    Identify and optionally remove obsolete profile templates.
    """
    from ..playbook import cleanup_unnecessary_profiles as do_cleanup

    project_root = Path(path or ".").resolve()
    if not project_root.exists():
        return json.dumps({"error": f"Path does not exist: {path}"})

    result = do_cleanup(project_root, auto_remove=auto_remove)

    output = {
        "project": str(project_root),
        "kept": result.kept,
        "obsolete": result.obsolete,
        "removed": result.removed,
        "reasons": result.reason,
    }

    if result.obsolete and not auto_remove:
        output["action_required"] = (
            f"{len(result.obsolete)} obsolete profile(s) found. "
            "Call cleanup_profiles with auto_remove=True to remove them."
        )

    return json.dumps(output, indent=2)


# =============================================================================
# MCP Tool Registration (non-consolidated tools only)
# =============================================================================

def register_admin_tools(mcp: FastMCP):
    """Register administrative tools with FastMCP.

    NOTE: Per ADR-009, only non-consolidated tools are registered here.
    The following are exposed via consolidated.py instead:
    - init_project, scan_codebase, analyze_and_index -> project()
    - cleanup_profiles -> profiles()
    - switch_model, queue_status -> admin()
    """

    @mcp.tool()
    async def health() -> str:
        """Check health status of Delia and all configured GPU backends."""
        return await health_impl()

    @mcp.tool()
    async def dashboard_url() -> str:
        """
        Get the URL of the running Delia dashboard.

        The dashboard auto-launches with the MCP server and provides:
        - System health and backend status
        - Tool usage metrics and analytics
        - Session browser
        - ACE Framework editor (playbooks, memories)
        - Dependency graph visualization

        Returns:
            JSON with dashboard URL and status.
        """
        return await dashboard_url_impl()

    @mcp.tool()
    async def models() -> str:
        """List all configured models across all GPU backends."""
        return await models_impl()

    @mcp.tool()
    async def switch_backend(backend_id: str) -> str:
        """Switch the active LLM backend."""
        return await switch_backend_impl(backend_id)

    @mcp.tool()
    async def get_model_info_tool(model_name: str) -> str:
        """Get detailed information about a specific model."""
        return await get_model_info_impl(model_name)
