# ACE Framework Multi-Platform Setup

This guide explains how to enable ACE Framework context for different AI coding assistants.

## Overview

The ACE Framework provides project-specific guidance through playbook bullets. Different AI agents access this in different ways:

| Agent | Method | Hook Support |
|-------|--------|--------------|
| Claude Code | Hook + Embedded bullets | ✅ Full |
| Cursor | Hook + Embedded bullets | ✅ Full (v1.7+) |
| Windsurf | Enterprise hooks + Embedded | ⚠️ Limited |
| VS Code Copilot | Embedded bullets only | ❌ None |
| Google Gemini | Embedded bullets only | ❌ None |

## Setup by Platform

### Claude Code

**Automatic (Recommended):** Run `delia init` in your project. This creates CLAUDE.md with embedded playbook bullets.

**Hook Setup (Dynamic injection for Task subagents):**

1. Copy `scripts/ace-task-hook.py` to a stable location
2. Add to `~/.claude/settings.json`:

```json
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
```

The hook intercepts Task tool calls and injects ACE context into subagent prompts.

### Cursor

**Embedded Bullets:** Run `delia init --force` to create `.cursor/rules` with embedded bullets.

**Hook Setup (v1.7+):**

1. Copy `scripts/ace-cursor-hook.sh` to a stable location
2. Configure in Cursor settings (hook API varies by version)

### VS Code / GitHub Copilot

**Embedded Bullets Only:** Copilot has no hooks API.

1. Run `delia init` or `delia init --force`
2. This creates `.github/copilot-instructions.md` with:
   - Project-specific rules
   - Embedded playbook bullets (top 5 per task type)
   - Fallback instructions to read `.delia/` directly

### Google Gemini Code Assist

**Embedded Bullets Only:** Gemini has no hooks API.

1. Run `delia init` or `delia init --force`
2. This creates `.gemini/instructions.md` with embedded bullets

### Windsurf

**Embedded Bullets:** Run `delia init --force` to create `.windsurf/rules`.

**Enterprise Hooks:** Windsurf has "Cascade Hooks" for enterprise customers (logging/policy).

## How It Works

### Embedded Bullets

When you run `delia init` or sync instruction files:

1. Delia reads `.delia/playbooks/*.json`
2. Sorts bullets by `utility_score`
3. Takes top 5 bullets per task type
4. Embeds them in each agent's instruction file

This ensures subagents (which can't call MCP tools) still get project guidance.

### Dynamic Hook Injection

For Claude Code and Cursor:

1. Hook intercepts tool calls (e.g., Task)
2. Detects task type from prompt keywords
3. Loads relevant playbook bullets
4. Prepends ACE context to the prompt
5. Subagent receives enriched prompt

## Keeping Bullets Fresh

Embedded bullets are snapshots. To update them:

```bash
# Re-sync instruction files with latest playbook bullets
delia sync
```

Hook-based injection always loads fresh bullets from `.delia/playbooks/`.

## Fallback for Subagents

All instruction files include a fallback section:

```markdown
## Subagent Fallback (No MCP Access)

If running as a subagent without MCP tool access, read `.delia/` files directly:
- `.delia/playbooks/*.json` - Task-specific bullets
- `.delia/memories/*.md` - Persistent project knowledge
- `.delia/project_summary.json` - Project overview
```

This enables any agent to benefit from ACE guidance.
