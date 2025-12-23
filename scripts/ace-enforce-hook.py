#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
ACE Framework Enforcement Hook for Claude Code.

This PreToolUse hook ensures the ACE workflow is followed by:
1. Checking if auto_context() was called before other Delia tools
2. Warning/blocking if ACE workflow not followed
3. Allowing auto_context, check_ace_status, and read_initial_instructions to pass

Install by adding to ~/.claude/settings.json:
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "mcp__delia__*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/ace-enforce-hook.py"
          }
        ]
      }
    ]
  }
}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Tools that are allowed without prior auto_context call
ACE_BOOTSTRAP_TOOLS = {
    "mcp__delia__auto_context",
    "mcp__delia__check_ace_status",
    "mcp__delia__read_initial_instructions",
    "mcp__delia__health",
    "mcp__delia__models",
    "mcp__delia__set_project",
    "mcp__delia__get_project_context",
    "mcp__delia__get_playbook",
    # Admin tools that don't need ACE
    "mcp__delia__dashboard_url",
    "mcp__delia__queue_status",
    "mcp__delia__mcp_servers",
}

# Session marker file to track if auto_context was called
SESSION_MARKER_DIR = Path("/tmp/delia-ace-sessions")


def get_session_marker_path() -> Path:
    """Get the session marker file path for current Claude session."""
    # Use parent PID to group tool calls from same conversation
    ppid = os.getppid()
    SESSION_MARKER_DIR.mkdir(exist_ok=True)
    return SESSION_MARKER_DIR / f"session-{ppid}.marker"


def is_ace_started() -> bool:
    """Check if auto_context was called in this session."""
    marker = get_session_marker_path()
    return marker.exists()


def mark_ace_started() -> None:
    """Mark that auto_context was called."""
    marker = get_session_marker_path()
    marker.write_text("ace_started")


def main() -> None:
    """Process hook input and enforce ACE workflow."""
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Invalid input, allow without modification
        sys.exit(0)

    tool_name = input_data.get("tool_name", "")

    # Not a Delia tool, allow
    if not tool_name.startswith("mcp__delia__"):
        sys.exit(0)

    # Bootstrap tools are always allowed
    if tool_name in ACE_BOOTSTRAP_TOOLS:
        # If this is auto_context, mark session as ACE-started
        if tool_name == "mcp__delia__auto_context":
            mark_ace_started()
        sys.exit(0)

    # Check if ACE was started
    if not is_ace_started():
        # ACE not started - warn and block
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "ACE WORKFLOW REQUIRED: Call auto_context() first before using other Delia tools. "
                    "This ensures you get project-specific guidance before making changes. "
                    "Example: mcp__delia__auto_context(message=\"your task description\")"
                ),
            }
        }
        print(json.dumps(output))
        sys.exit(0)

    # ACE was started, allow the tool
    sys.exit(0)


if __name__ == "__main__":
    main()
