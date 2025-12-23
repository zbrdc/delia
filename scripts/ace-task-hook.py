#!/usr/bin/env python3
"""
ACE Framework Task Hook for Claude Code

This hook intercepts Task tool calls and injects ACE context (playbook bullets,
profiles) into the subagent prompt. This enables subagents to benefit from
project-specific guidance even though they don't have MCP tool access.

Install by adding to ~/.claude/settings.json:
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Task",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/ace-task-hook.py"
          }
        ]
      }
    ]
  }
}
"""

import json
import os
import sys
from pathlib import Path


def find_delia_dir() -> Path | None:
    """Find .delia directory starting from cwd and walking up."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        delia_dir = parent / ".delia"
        if delia_dir.is_dir():
            return delia_dir
    return None


def detect_task_type(prompt: str) -> str:
    """Detect task type from prompt keywords."""
    prompt_lower = prompt.lower()

    keywords = {
        "coding": ["implement", "add", "create", "build", "write", "refactor", "code"],
        "testing": ["test", "pytest", "coverage", "mock", "assert", "spec"],
        "debugging": ["bug", "error", "fix", "debug", "broken", "failing", "issue"],
        "architecture": ["design", "architecture", "pattern", "adr", "plan", "approach"],
        "git": ["git", "commit", "branch", "merge", "pr", "push", "pull"],
        "security": ["security", "auth", "password", "token", "vulnerability"],
        "api": ["api", "endpoint", "rest", "graphql", "route"],
        "deployment": ["deploy", "docker", "ci/cd", "production", "ship"],
    }

    for task_type, words in keywords.items():
        if any(word in prompt_lower for word in words):
            return task_type

    return "project"


def load_playbook(delia_dir: Path, task_type: str) -> list[dict]:
    """Load playbook bullets for a task type."""
    playbook_path = delia_dir / "playbooks" / f"{task_type}.json"
    if not playbook_path.exists():
        return []

    try:
        with open(playbook_path) as f:
            data = json.load(f)
            return data.get("bullets", [])[:5]  # Top 5 bullets
    except (json.JSONDecodeError, IOError):
        return []


def load_project_summary(delia_dir: Path) -> dict:
    """Load project summary."""
    summary_path = delia_dir / "project_summary.json"
    if not summary_path.exists():
        return {}

    try:
        with open(summary_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def format_ace_context(task_type: str, bullets: list[dict], summary: dict) -> str:
    """Format ACE context for injection into prompt."""
    lines = [
        "## ACE Framework Context (Auto-Injected)",
        "",
        f"**Detected Task Type:** {task_type}",
        "",
    ]

    # Add project info
    if summary:
        if overview := summary.get("overview"):
            lines.append(f"**Project:** {overview[:200]}...")
            lines.append("")

    # Add playbook bullets
    if bullets:
        lines.append("**Playbook Guidance:**")
        for bullet in bullets:
            content = bullet.get("content", "")
            if content:
                lines.append(f"- {content}")
        lines.append("")

    # Add fallback instructions
    lines.extend([
        "**Note:** If you need more context, read `.delia/` files directly:",
        "- `.delia/playbooks/*.json` - Task-specific guidance",
        "- `.delia/memories/*.md` - Persistent project knowledge",
        "",
        "---",
        "",
    ])

    return "\n".join(lines)


def main():
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Invalid input, allow without modification
        sys.exit(0)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Only process Task tool calls
    if tool_name != "Task":
        sys.exit(0)

    # Get the prompt
    prompt = tool_input.get("prompt", "")
    if not prompt:
        sys.exit(0)

    # Find .delia directory
    delia_dir = find_delia_dir()
    if not delia_dir:
        # No .delia directory, allow without modification
        sys.exit(0)

    # Detect task type and load context
    task_type = detect_task_type(prompt)
    bullets = load_playbook(delia_dir, task_type)

    # Also load coding bullets as fallback
    if task_type != "coding":
        bullets.extend(load_playbook(delia_dir, "coding")[:3])

    summary = load_project_summary(delia_dir)

    # Format ACE context
    ace_context = format_ace_context(task_type, bullets, summary)

    # Prepend ACE context to prompt
    modified_prompt = ace_context + prompt

    # Output hook response with modified input
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": f"ACE context injected for {task_type} task",
            "updatedInput": {
                **tool_input,
                "prompt": modified_prompt,
            },
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
