# SPDX-License-Identifier: GPL-3.0-or-later

"""
AI Coding Agent Detection and Instruction File Synchronization.

This module provides utilities to detect which AI coding assistants are configured
in a project and keep their instruction files synchronized.

ALL major AI coding agents now support MCP:
- Claude Code, Gemini, GitHub Copilot, VS Code, Cursor, Windsurf

Each agent gets its own instruction file with agent-specific guidance for using
Delia's MCP tools (get_playbook, report_feedback, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


# Agent-specific metadata
AGENT_INFO = {
    "claude_root": {
        "name": "Claude Code",
        "mcp_config": "Add to .claude/settings.json or use `claude mcp add`",
    },
    "claude_dir": {
        "name": "Claude Code",
        "mcp_config": "Add to .claude/settings.json or use `claude mcp add`",
    },
    "gemini": {
        "name": "Google Gemini",
        "mcp_config": "Configure in Gemini Code Assist settings",
    },
    "copilot": {
        "name": "GitHub Copilot",
        "mcp_config": "Configure in VS Code settings.json under github.copilot",
    },
    "cursor": {
        "name": "Cursor",
        "mcp_config": "Add to .cursor/mcp.json",
    },
    "windsurf": {
        "name": "Windsurf",
        "mcp_config": "Add to ~/.codeium/windsurf/mcp_config.json",
    },
}


def generate_agent_instructions(
    agent_id: str,
    project_name: str,
    project_rules: str,
    project_overview: str = "",
) -> str:
    """
    Generate agent-specific instruction content.

    Each agent gets its own instructions, NOT a copy of CLAUDE.md.
    All agents support MCP and can use Delia's tools.

    Args:
        agent_id: The agent identifier (e.g., 'gemini', 'copilot')
        project_name: Name of the project
        project_rules: Project-specific rules to include
        project_overview: Optional project overview section

    Returns:
        Agent-specific instruction content
    """
    agent = AGENT_INFO.get(agent_id, {"name": "AI Assistant", "mcp_config": "Configure MCP server"})

    content = f"""# {project_name} Development Instructions

This file provides guidance to {agent['name']} when working with this repository.

---

## Delia ACE Framework

This project uses Delia for AI-assisted coding. Delia provides project-specific
playbooks with learned strategies, patterns, and guidance.

### MCP Configuration

To enable Delia tools, configure the MCP server:
- {agent['mcp_config']}
- Server: `delia serve` or `uvx delia serve`

### Using Playbook Guidance

**Before starting a task**, query for relevant guidance:

```
mcp__delia__get_playbook(task_type="coding")
mcp__delia__get_project_context()
```

**After completing a task**, report feedback:

```
mcp__delia__report_feedback(bullet_id="strat-xxx", task_type="coding", helpful=True)
```

---

{project_rules}
"""

    if project_overview:
        content += f"\n---\n\n## Project Overview\n\n{project_overview}\n"

    return content


def extract_project_rules(claude_md_content: str) -> tuple[str, str, str]:
    """
    Extract project-specific rules from CLAUDE.md content.

    Separates:
    - Project name (from first heading)
    - Critical rules section
    - Project overview section

    Returns:
        Tuple of (project_name, rules_section, overview_section)
    """
    lines = claude_md_content.split('\n')

    # Extract project name from first heading
    project_name = "Project"
    for line in lines:
        if line.startswith('# '):
            # Extract name, removing " Development Instructions" suffix if present
            project_name = line[2:].replace(' Development Instructions', '').strip()
            break

    # Find key sections
    in_rules = False
    in_overview = False
    rules_lines = []
    overview_lines = []

    for i, line in enumerate(lines):
        # Start of critical rules section
        if 'CRITICAL RULES' in line.upper() or 'INLINED CRITICAL RULES' in line.upper():
            in_rules = True
            in_overview = False
            continue

        # Start of project overview section
        if line.startswith('## Project Overview') or line.startswith('## Overview'):
            in_overview = True
            in_rules = False
            continue

        # End of section on next major heading
        if line.startswith('## ') and (in_rules or in_overview):
            if in_rules:
                in_rules = False
            if in_overview:
                in_overview = False
            continue

        # Collect lines
        if in_rules:
            rules_lines.append(line)
        if in_overview:
            overview_lines.append(line)

    rules = '\n'.join(rules_lines).strip()
    overview = '\n'.join(overview_lines).strip()

    # If no rules found, use a default
    if not rules:
        rules = "## Critical Rules\n\n- Follow project coding standards\n- Write tests for new functionality\n- No hardcoded secrets"
    else:
        rules = "## Critical Rules\n\n" + rules

    return project_name, rules, overview


def detect_ai_agents(project_root: Path) -> dict[str, dict[str, Any]]:
    """
    Detect which AI coding assistants are configured in the project.

    Returns a dict mapping agent name to its instruction file info:
    {
        "claude_root": {"dir": Path, "file": "CLAUDE.md", "exists": True, ...},
        "gemini": {"dir": Path, "file": "instructions.md", "exists": True, ...},
        ...
    }
    """
    # Define all known AI assistant locations
    agent_configs = {
        "claude_root": {
            "dir": project_root,
            "file": "CLAUDE.md",
            "description": "Claude Code (root)",
        },
        "claude_dir": {
            "dir": project_root / ".claude",
            "file": "INSTRUCTIONS.md",
            "description": "Claude Code (.claude/)",
        },
        "gemini": {
            "dir": project_root / ".gemini",
            "file": "instructions.md",
            "description": "Google Gemini",
        },
        "copilot": {
            "dir": project_root / ".github",
            "file": "copilot-instructions.md",
            "description": "GitHub Copilot",
        },
        "cursor": {
            "dir": project_root / ".cursor",
            "file": "rules",
            "description": "Cursor AI",
        },
        "windsurf": {
            "dir": project_root / ".windsurf",
            "file": "rules",
            "description": "Windsurf/Codeium",
        },
        "aider": {
            "dir": project_root,
            "file": ".aider.conf.yml",
            "description": "Aider",
        },
        "continue": {
            "dir": project_root / ".continue",
            "file": "config.json",
            "description": "Continue.dev",
        },
    }

    detected = {}
    for agent_id, config in agent_configs.items():
        file_path = config["dir"] / config["file"]
        detected[agent_id] = {
            "dir": config["dir"],
            "file": config["file"],
            "path": file_path,
            "exists": file_path.exists(),
            "dir_exists": config["dir"].exists(),
            "description": config["description"],
        }

    return detected


def sync_agent_instruction_files(
    project_root: Path,
    content: str,
    force: bool = False,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """
    Sync instruction content to all detected AI coding assistant files.

    Strategy:
    - CLAUDE.md at root gets the full content (primary source of truth)
    - Other agents get AGENT-SPECIFIC instructions generated from CLAUDE.md
    - Each agent's file references Delia MCP with agent-specific config
    - Returns list of files written and detection info

    Args:
        project_root: Path to the project root directory
        content: The CLAUDE.md content (primary source of truth)
        force: If True, create all agent directories and files

    Returns:
        Tuple of (files_written, detected_agents) where:
        - files_written: List of relative paths that were written
        - detected_agents: Dict with detection info and update status
    """
    files_written = []
    detected_agents = detect_ai_agents(project_root)

    # Extract project info from CLAUDE.md content for generating agent-specific files
    project_name, project_rules, project_overview = extract_project_rules(content)

    # Agents to sync
    primary_agents = ["claude_root", "claude_dir", "gemini", "copilot", "cursor", "windsurf"]

    # Check if Claude is configured (either CLAUDE.md exists or .claude/ dir exists)
    claude_configured = (
        detected_agents["claude_root"]["exists"] or
        detected_agents["claude_dir"]["dir_exists"]
    )

    for agent_id in primary_agents:
        agent = detected_agents.get(agent_id)
        if not agent:
            continue

        file_path = agent["path"]

        # Decision logic - all agents treated equally:
        # Write if file exists (update) OR if dir/indicator exists (agent is configured)
        should_write = False
        if agent_id == "claude_root":
            # CLAUDE.md: only if it exists OR .claude/ dir exists (Claude is configured)
            if agent["exists"] or claude_configured:
                should_write = True
        elif agent["exists"]:
            # File exists - update it to keep in sync
            should_write = True
        elif agent["dir_exists"]:
            # Directory exists but file doesn't - user has this agent configured
            should_write = True

        # Force mode - create everything
        if force and not should_write:
            agent["dir"].mkdir(parents=True, exist_ok=True)
            should_write = True

        if should_write:
            if not agent["dir"].exists():
                agent["dir"].mkdir(parents=True, exist_ok=True)

            # All agents get the same comprehensive content
            # Since all major agents now support MCP, they should have identical instructions
            # This ensures consistent behavior across Claude, Gemini, Copilot, Cursor, etc.
            agent_content = content

            file_path.write_text(agent_content)
            files_written.append(str(file_path.relative_to(project_root)))
            detected_agents[agent_id]["updated"] = True
            log.info("agent_file_synced", agent=agent_id, path=str(file_path.relative_to(project_root)))

    return files_written, detected_agents


def get_agent_summary(detected_agents: dict[str, dict[str, Any]]) -> str:
    """
    Generate a human-readable summary of detected agents.

    Args:
        detected_agents: Dict from detect_ai_agents or sync_agent_instruction_files

    Returns:
        Formatted string listing agents and their status
    """
    lines = []
    for agent_id, info in detected_agents.items():
        status = ""
        if info.get("updated"):
            status = " [updated]"
        elif info.get("exists"):
            status = " [exists]"
        elif info.get("dir_exists"):
            status = " [dir only]"

        if info.get("updated") or info.get("exists"):
            lines.append(f"  - {info['description']}: {info['file']}{status}")

    if not lines:
        return "  No AI coding assistants detected"

    return "\n".join(lines)
