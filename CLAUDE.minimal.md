# Delia Framework - Quick Reference

## Workflow

```
START  →  auto_context(message)              # Load playbook for task
WORK   →  Apply bullets, use tools           # Do the task
END    →  complete_task(success, bullets)    # Record feedback
```

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
