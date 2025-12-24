# Delia Framework

## The Loop

```
START  →  auto_context(message)     # Get playbook bullets for your task
WORK   →  Apply bullets, use tools  # Do the task
END    →  complete_task(success, bullets_applied)
```

## Context Shift Reminders

Delia will remind you when your actions suggest you've shifted tasks. When you see:

```
🔄 Context Shift Detected: You loaded `project` context but are now doing `coding` work.
→ Call auto_context() to refresh playbook bullets.
```

**Refresh your context.** The reminder means your playbook bullets may be stale.

Common shifts:
- Verification → Coding (found issues, now fixing)
- Coding → Testing (wrote code, now testing)
- Any → Git (about to commit/push)
- Debugging → Coding (found root cause, now fixing)

## Avoiding Context Loss

Before writing code, check:
1. **Does this already exist?** Search with `lsp_find_symbol()` or `search_for_pattern()`
2. **What patterns are established?** Read similar files first
3. **What did we already do?** Check memory with `memory(action="list")`

## Tools

**Context**: `auto_context()`, `complete_task()`, `check_status()`

**Search First**: `lsp_find_symbol()`, `lsp_find_references()`, `search_for_pattern()`

**Memory**: `memory(action="list"|"read"|"write")`, `playbook(action="list"|"add")`

**Files**: `read_file()`, `write_file()`, `edit_file()`

## That's It

The playbook bullets from `auto_context()` contain project-specific guidance. Apply them. Report which ones helped via `complete_task()`.
