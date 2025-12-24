# SPDX-License-Identifier: GPL-3.0-or-later

"""
AI Coding Agent Detection and Instruction File Synchronization.

This module provides utilities to detect which AI coding assistants are configured
in a project and keep their instruction files synchronized.

Architecture:
- mcp_instructions.md (in src/delia/) is the CANONICAL source for Delia tool usage
- Agent-specific files (CLAUDE.md, .gemini/instructions.md, etc.) extend the canonical
  content with agent-specific guidance
- Playbook bullets are embedded for subagents that lack MCP access

ALL major AI coding agents now support MCP:
- Claude Code, Gemini, GitHub Copilot, VS Code, Cursor, Windsurf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


# Agent-specific metadata and customizations
AGENT_INFO = {
    "claude_root": {
        "name": "Claude Code",
        "mcp_config": "Add to .claude/settings.json or use `claude mcp add`",
        "specific_guidance": """
## Claude Code Specific Features

### Hooks Support
Claude Code supports hooks that can inject context automatically:
- Pre-tool hooks can add playbook bullets before file operations
- Post-tool hooks can trigger feedback collection

### Task Subagents
When spawning Task subagents, they may not have MCP access. The playbook bullets
embedded below ensure subagents still benefit from Delia guidance.

### Slash Commands
Use `/delia` commands if configured in your Claude Code setup.
""",
    },
    "claude_dir": {
        "name": "Claude Code",
        "mcp_config": "Add to .claude/settings.json or use `claude mcp add`",
        "specific_guidance": "",  # Same as claude_root, shares the guidance
    },
    "gemini": {
        "name": "Google Gemini",
        "mcp_config": "Configure in Gemini Code Assist settings",
        "specific_guidance": """
## Gemini Specific Notes

### Context Window
Gemini has a large context window. However, still prefer LSP tools over reading
entire files to maintain efficiency and reduce noise.

### MCP Integration
Ensure the Delia MCP server is configured in your Gemini Code Assist settings.
""",
    },
    "copilot": {
        "name": "GitHub Copilot",
        "mcp_config": "Configure in VS Code settings.json under github.copilot",
        "specific_guidance": """
## GitHub Copilot Specific Notes

### MCP Configuration
Add Delia to your VS Code settings:
```json
{
  "github.copilot.chat.experimental.mcpServers": {
    "delia": {
      "command": "delia",
      "args": ["serve"]
    }
  }
}
```

### Workspace Context
Copilot automatically includes workspace context. Use Delia's LSP tools
for precise navigation beyond what Copilot indexes.
""",
    },
    "cursor": {
        "name": "Cursor",
        "mcp_config": "Add to .cursor/mcp.json",
        "specific_guidance": """
## Cursor Specific Notes

### MCP Configuration
Add to `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "delia": {
      "command": "delia",
      "args": ["serve"]
    }
  }
}
```

### Composer Integration
When using Composer for multi-file edits, call `auto_context` first to
ensure you have the right playbook bullets for the task.
""",
    },
    "windsurf": {
        "name": "Windsurf",
        "mcp_config": "Add to ~/.codeium/windsurf/mcp_config.json",
        "specific_guidance": """
## Windsurf Specific Notes

### Cascade Flows
When using Cascade for multi-step workflows, ensure each step calls
the appropriate Delia checkpoint tools.

### MCP Configuration
Configure in `~/.codeium/windsurf/mcp_config.json`.
""",
    },
}


def get_mcp_instructions_path() -> Path:
    """Get the path to the canonical mcp_instructions.md file."""
    # This file lives in the delia package
    return Path(__file__).parent / "mcp_instructions.md"


def load_mcp_instructions() -> str:
    """
    Load the canonical Delia MCP instructions.

    This is the single source of truth for how to use Delia tools.
    All agent instruction files should include this content.
    """
    path = get_mcp_instructions_path()
    if path.exists():
        return path.read_text()
    else:
        log.warning("mcp_instructions_not_found", path=str(path))
        return ""


def load_embedded_playbook_bullets(project_root: Path) -> str:
    """
    Load and format playbook bullets for embedding into instruction files.

    This enables subagents (which don't have MCP access) to still benefit
    from Delia Framework guidance by having the bullets embedded directly.

    Returns:
        Formatted markdown string with embedded bullets
    """
    delia_dir = project_root / ".delia"
    if not delia_dir.exists():
        return ""

    playbooks_dir = delia_dir / "playbooks"
    if not playbooks_dir.exists():
        return ""

    lines = [
        "",
        "---",
        "",
        "## PROJECT PLAYBOOK (Auto-embedded)",
        "",
        "These are learned strategies from this project. Apply them to relevant tasks.",
        "For the latest bullets, use `auto_context()` or read `.delia/playbooks/*.json` directly.",
        "",
    ]

    # Priority order for task types
    task_types = ["coding", "testing", "architecture", "debugging", "project", "git", "security", "deployment", "api", "performance"]

    for task_type in task_types:
        playbook_path = playbooks_dir / f"{task_type}.json"
        if not playbook_path.exists():
            continue

        try:
            with open(playbook_path) as f:
                data = json.load(f)

            # Handle both formats: array of bullets or object with "bullets" key
            if isinstance(data, list):
                bullets = data
            else:
                bullets = data.get("bullets", [])

            if not bullets:
                continue

            # Sort by utility score, take top 5
            sorted_bullets = sorted(
                bullets,
                key=lambda b: b.get("utility_score", 0.5) if isinstance(b, dict) else 0.5,
                reverse=True
            )[:5]

            if sorted_bullets:
                lines.append(f"### {task_type.title()}")
                for bullet in sorted_bullets:
                    content = bullet.get("content", "")
                    if content:
                        lines.append(f"- {content}")
                lines.append("")

        except (json.JSONDecodeError, IOError):
            continue

    if len(lines) <= 8:  # Only header lines, no bullets
        return ""

    return "\n".join(lines)


def generate_agent_instructions(
    agent_id: str,
    project_name: str,
    mcp_instructions: str,
    embedded_bullets: str = "",
) -> str:
    """
    Generate agent-specific instruction content.

    Each agent gets:
    1. The canonical mcp_instructions.md content (Delia tool usage)
    2. Agent-specific guidance (hooks, MCP config, etc.)
    3. Embedded playbook bullets (for subagent fallback)

    Args:
        agent_id: The agent identifier (e.g., 'gemini', 'copilot')
        project_name: Name of the project
        mcp_instructions: The canonical Delia instructions content
        embedded_bullets: Pre-loaded playbook bullets for subagent access

    Returns:
        Complete agent-specific instruction content
    """
    agent = AGENT_INFO.get(agent_id, {"name": "AI Assistant", "mcp_config": "Configure MCP server", "specific_guidance": ""})

    # Start with project header
    content = f"""# {project_name} - AI Coding Instructions

This file provides guidance for {agent['name']} when working with this repository.

**MCP Configuration**: {agent['mcp_config']}

---

"""

    # Add the canonical Delia instructions
    content += mcp_instructions

    # Add agent-specific guidance if any
    specific = agent.get("specific_guidance", "").strip()
    if specific:
        content += f"\n\n---\n\n{specific}"

    # Add subagent fallback section
    content += """

---

## Subagent Fallback (No MCP Access)

If running as a subagent without MCP tool access, read `.delia/` files directly:
- `.delia/playbooks/*.json` - Task-specific bullets (coding, testing, etc.)
- `.delia/memories/*.md` - Persistent project knowledge
- `.delia/project_summary.json` - Project overview

The playbook bullets below are auto-embedded for convenience.
"""

    # Add embedded bullets
    if embedded_bullets:
        content += embedded_bullets

    return content


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


def extract_project_name(project_root: Path) -> str:
    """
    Extract project name from the project root.

    Tries:
    1. pyproject.toml [project].name
    2. package.json name
    3. Directory name
    """
    # Try pyproject.toml
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
                name = data.get("project", {}).get("name")
                if name:
                    return name.replace("-", " ").replace("_", " ").title()
        except Exception:
            pass

    # Try package.json
    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            with open(package_json) as f:
                data = json.load(f)
                name = data.get("name")
                if name:
                    return name.replace("-", " ").replace("_", " ").title()
        except Exception:
            pass

    # Fall back to directory name
    return project_root.name.replace("-", " ").replace("_", " ").title()


def sync_agent_instruction_files(
    project_root: Path,
    content: str | None = None,
    force: bool = False,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """
    Sync instruction content to all detected AI coding assistant files.

    Strategy:
    - Load canonical content from mcp_instructions.md
    - Generate agent-specific files with embedded playbook bullets
    - Embedded bullets enable subagents (which lack MCP access) to benefit from Delia
    - Returns list of files written and detection info

    Args:
        project_root: Path to the project root directory
        content: Optional override content (if None, loads from mcp_instructions.md)
        force: If True, create all agent directories and files

    Returns:
        Tuple of (files_written, detected_agents) where:
        - files_written: List of relative paths that were written
        - detected_agents: Dict with detection info and update status
    """
    files_written = []
    detected_agents = detect_ai_agents(project_root)

    # Load canonical instructions
    mcp_instructions = content if content else load_mcp_instructions()
    if not mcp_instructions:
        log.warning("no_mcp_instructions", msg="Cannot sync without mcp_instructions.md")
        return files_written, detected_agents

    # Extract project name
    project_name = extract_project_name(project_root)

    # Load embedded playbook bullets for subagent access
    embedded_bullets = load_embedded_playbook_bullets(project_root)

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

        # Decision logic:
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

            # Generate agent-specific content
            agent_content = generate_agent_instructions(
                agent_id=agent_id,
                project_name=project_name,
                mcp_instructions=mcp_instructions,
                embedded_bullets=embedded_bullets,
            )

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
