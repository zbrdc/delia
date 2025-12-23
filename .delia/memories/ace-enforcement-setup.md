# ACE Framework Enforcement Setup

This guide explains how to set up ACE Framework enforcement to ensure AI agents follow the ACE workflow.

## Two-Layer Enforcement

ACE enforcement works at two levels:

1. **Hook Layer (Claude Code)**: PreToolUse hook intercepts tool calls and blocks if ACE not started
2. **Tool Layer (All MCP clients)**: Tools check `ACEEnforcementTracker` and return error if ACE not started

## Setup: Claude Code Hook

### Install the Enforcement Hook

1. Copy hook script to a stable location:
```bash
cp scripts/ace-enforce-hook.py ~/.delia/hooks/
chmod +x ~/.delia/hooks/ace-enforce-hook.py
```

2. Add to `~/.claude/settings.json`:
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "mcp__delia__*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.delia/hooks/ace-enforce-hook.py"
          }
        ]
      }
    ]
  }
}
```

### What the Hook Does

- Intercepts all `mcp__delia__*` tool calls
- Checks if `auto_context` was called in this session
- Blocks non-exempt tools with clear error message
- Exempt tools (auto_context, health, etc.) always allowed

### Session Tracking

The hook creates session markers in `/tmp/delia-ace-sessions/`:
- Marker created when `auto_context` is called
- Uses parent PID to group tool calls from same conversation
- Markers auto-cleaned on system restart

## Setup: Tool-Level Gating

Tool gating is built into Delia and requires no setup. It works automatically:

### Gated Tools

These tools require `auto_context()` first:
- `delegate` - LLM task delegation
- `write_file` - File creation
- `edit_file` - File modification
- (other code-modifying tools)

### Exempt Tools

These tools work without prior `auto_context()`:
- `auto_context` - The start of ACE workflow
- `check_ace_status` - Check project status
- `read_initial_instructions` - Read ACE manual
- `health` - Check system status
- `models` - List available models
- `get_playbook` - Manual playbook retrieval
- `get_project_context` - Get project info
- `init_project` - Initialize ACE for project
- `scan_codebase` - Scan project files

### How Gating Works

1. Tool checks `ACEEnforcementTracker.require_ace_started()`
2. If ACE not started, returns JSON error:
```json
{
  "result": {
    "error": "ACE_WORKFLOW_REQUIRED",
    "message": "You must call auto_context() before using this tool...",
    "action": "auto_context(message=\"your task\", path=\"/project\")",
    "tool_blocked": "delegate"
  }
}
```
3. Agent sees error and knows to call `auto_context()` first

### Session Window

ACE "started" status lasts 30 minutes from last `auto_context()` call.
After 30 minutes, agent must call `auto_context()` again.

## Disabling Enforcement

For testing, you can disable gating:

```python
from delia.tools.handlers import get_ace_tracker

tracker = get_ace_tracker()
tracker._gating_enabled = False  # Disable tool gating
```

The hook must be removed from settings to disable hook-level enforcement.

## Troubleshooting

### "ACE_WORKFLOW_REQUIRED" Error

**Cause**: Tool called before `auto_context()`
**Fix**: Call `auto_context(message="your task")` first

### Hook Not Blocking

**Check**:
1. Hook script path is correct in settings.json
2. Script is executable (`chmod +x`)
3. Python3 is in PATH

### Tool Gating Not Working

**Check**:
1. Delia MCP server is updated to latest version
2. `_gating_enabled` is True (default)
