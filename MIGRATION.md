# Migration Guide: Delia 2.0 Tool Consolidation

**Version**: 1.x → 2.0
**Date**: 2025-12-21
**Breaking Change**: Yes

## Overview

Delia 2.0 consolidates 51 MCP tools down to 27 tools using action-based parameters. This reduces cognitive overhead while preserving all functionality.

**Impact**: 30 tools have been removed and replaced with 6 consolidated tools.

## Quick Migration Examples

### Before (v1.x)
```python
# Old API - multiple tools
session_list(client_id="user123")
list_memories(path="/project")
add_playbook_bullet(task_type="coding", content="Use async/await")
```

### After (v2.0)
```python
# New API - action-based parameters
session(action="list", client_id="user123")
memory(action="list", path="/project")
playbook(action="add", task_type="coding", content="Use async/await")
```

## Consolidated Tools

### 1. `playbook(action, ...)`

Consolidates 7 playbook operations.

| Old Tool | New Call |
|----------|----------|
| `add_playbook_bullet(...)` | `playbook(action="add", ...)` |
| `write_playbook(...)` | `playbook(action="write", ...)` |
| `delete_playbook_bullet(...)` | `playbook(action="delete", ...)` |
| `list_playbooks(...)` | `playbook(action="list", ...)` |
| `playbook_stats(...)` | `playbook(action="stats", ...)` |
| `prune_stale_bullets(...)` | `playbook(action="prune", ...)` |
| `confirm_ace_compliance(...)` | `playbook(action="confirm", ...)` |

**Actions**: `add`, `write`, `delete`, `list`, `stats`, `prune`, `confirm`

**Examples**:
```python
# Add a bullet
playbook(action="add", task_type="coding", content="Use jq for JSON parsing", section="shell_patterns")

# List all playbooks
playbook(action="list", path="/project")

# Get statistics
playbook(action="stats", task_type="testing")

# Prune stale bullets
playbook(action="prune", max_age_days=90, min_utility=0.3)

# Confirm ACE compliance
playbook(action="confirm", task_description="Added auth", bullets_applied="strat-001,strat-002", patterns_followed="JWT,bcrypt")
```

### 2. `memory(action, ...)`

Consolidates 4 memory operations.

| Old Tool | New Call |
|----------|----------|
| `list_memories(...)` | `memory(action="list", ...)` |
| `read_memory(...)` | `memory(action="read", ...)` |
| `write_memory(...)` | `memory(action="write", ...)` |
| `delete_memory(...)` | `memory(action="delete", ...)` |

**Actions**: `list`, `read`, `write`, `delete`

**Examples**:
```python
# List all memories
memory(action="list", path="/project")

# Read a memory
memory(action="read", name="architecture-decisions", path="/project")

# Write/update a memory
memory(action="write", name="debugging-notes", content="# Auth Bug\n\nFixed JWT expiration...", append=True)

# Delete a memory
memory(action="delete", name="old-notes")
```

### 3. `session(action, ...)`

Consolidates 4 session operations.

| Old Tool | New Call |
|----------|----------|
| `session_list(...)` | `session(action="list", ...)` |
| `session_stats(...)` | `session(action="stats", ...)` |
| `session_compact(...)` | `session(action="compact", ...)` |
| `session_delete(...)` | `session(action="delete", ...)` |

**Actions**: `list`, `stats`, `compact`, `delete`

**Examples**:
```python
# List all sessions
session(action="list", client_id="user123")

# Get session statistics
session(action="stats", session_id="sess_abc123")

# Compact session history
session(action="compact", session_id="sess_abc123", force=True)

# Delete a session
session(action="delete", session_id="sess_abc123")
```

### 4. `profiles(action, ...)`

Consolidates 4 profile operations.

| Old Tool | New Call |
|----------|----------|
| `recommend_profiles(...)` | `profiles(action="recommend", ...)` |
| `check_reevaluation(...)` | `profiles(action="check", ...)` |
| `run_reevaluation(...)` | `profiles(action="reevaluate", ...)` |
| `cleanup_profiles(...)` | `profiles(action="cleanup", ...)` |

**Actions**: `recommend`, `check`, `reevaluate`, `cleanup`

**Examples**:
```python
# Recommend profiles for a project
profiles(action="recommend", path="/project", analyze_gaps=True)

# Check if re-evaluation is needed
profiles(action="check", path="/project")

# Run re-evaluation
profiles(action="reevaluate", path="/project", force=False)

# Clean up obsolete profiles
profiles(action="cleanup", path="/project", auto_remove=False)
```

### 5. `project(action, ...)`

Consolidates 5 project operations.

| Old Tool | New Call |
|----------|----------|
| `init_project(...)` | `project(action="init", ...)` |
| `scan_codebase(...)` | `project(action="scan", ...)` |
| `analyze_and_index(...)` | `project(action="analyze", ...)` |
| `sync_instruction_files(...)` | `project(action="sync", ...)` |
| `read_instruction_files(...)` | `project(action="read_instructions", ...)` |

**Actions**: `init`, `scan`, `analyze`, `sync`, `read_instructions`

**Examples**:
```python
# Initialize ACE framework for a project
project(action="init", path="/new-project", force=False, skip_index=False)

# Scan codebase
project(action="scan", path="/project", phase="overview", max_files=20)

# Analyze and index
project(action="analyze", path="/project", project_summary="{...}", coding_bullets="[...]")

# Sync instruction files to all AI agents
project(action="sync", content="# Project Instructions\n...", path="/project", force=False)

# Read existing instruction files
project(action="read_instructions", path="/project")
```

### 6. `admin(action, ...)`

Consolidates 3+ admin operations.

| Old Tool | New Call |
|----------|----------|
| `switch_model(...)` | `admin(action="switch_model", ...)` |
| `queue_status()` | `admin(action="queue_status")` |
| `mcp_servers(...)` | `admin(action="mcp_servers", ...)` |

**Actions**: `switch_model`, `queue_status`, `mcp_servers`, `health`, `models`, `switch_backend`, `model_info`

**Examples**:
```python
# Switch model for a tier
admin(action="switch_model", tier="quick", model_name="qwen3:14b")

# Check queue status
admin(action="queue_status")

# Manage MCP servers
admin(action="mcp_servers", command="status")

# Health check
admin(action="health")

# List available models
admin(action="models")
```

## Preserved Tools (No Changes)

These tools remain unchanged and work exactly as before:

### Core ACE Framework (4 tools)
- `get_playbook(task_type, limit, path)` - Get strategic bullets
- `report_feedback(bullet_id, task_type, helpful, path)` - Report bullet effectiveness
- `get_project_context(path)` - Get project overview
- `set_project(path)` - Switch project context

### LSP Code Intelligence (9 tools)
- `lsp_goto_definition(path, line, character)`
- `lsp_find_references(path, line, character)`
- `lsp_hover(path, line, character)`
- `lsp_get_symbols(path)`
- `lsp_find_symbol(name, path, kind)`
- `lsp_rename_symbol(path, line, character, new_name, apply)`
- `lsp_replace_symbol_body(path, symbol_name, new_body)`
- `lsp_insert_before_symbol(path, symbol_name, content)`
- `lsp_insert_after_symbol(path, symbol_name, content)`

### LLM Delegation (6 tools)
- `delegate(task, content, ...)`
- `think(problem, context, depth, session_id)`
- `batch(tasks, include_metadata, max_tokens, session_id)`
- `chain(steps, session_id, continue_on_error)`
- `workflow(definition, session_id, max_retries)`
- `agent(prompt, system_prompt, model, max_iterations, tools, backend_type, workspace)`

## Common Migration Patterns

### Pattern 1: List → Read → Write

**Before**:
```python
list_memories(path="/project")
content = read_memory(name="notes", path="/project")
write_memory(name="notes", content=updated_content, path="/project")
```

**After**:
```python
memory(action="list", path="/project")
content = memory(action="read", name="notes", path="/project")
memory(action="write", name="notes", content=updated_content, path="/project")
```

### Pattern 2: Session Management

**Before**:
```python
sessions = session_list(client_id="user123")
stats = session_stats(session_id="sess_abc")
session_compact(session_id="sess_abc", force=True)
session_delete(session_id="sess_abc")
```

**After**:
```python
sessions = session(action="list", client_id="user123")
stats = session(action="stats", session_id="sess_abc")
session(action="compact", session_id="sess_abc", force=True)
session(action="delete", session_id="sess_abc")
```

### Pattern 3: Playbook CRUD

**Before**:
```python
add_playbook_bullet(task_type="coding", content="Use async", section="patterns")
list_playbooks(path="/project")
delete_playbook_bullet(bullet_id="strat-001", task_type="coding")
playbook_stats(task_type="coding")
```

**After**:
```python
playbook(action="add", task_type="coding", content="Use async", section="patterns")
playbook(action="list", path="/project")
playbook(action="delete", bullet_id="strat-001", task_type="coding")
playbook(action="stats", task_type="coding")
```

## Automated Migration Script

Use this regex find-and-replace to assist migration:

```bash
# Session tools
session_list\( → session(action="list",
session_stats\( → session(action="stats",
session_compact\( → session(action="compact",
session_delete\( → session(action="delete",

# Memory tools
list_memories\( → memory(action="list",
read_memory\( → memory(action="read",
write_memory\( → memory(action="write",
delete_memory\( → memory(action="delete",

# Playbook tools
add_playbook_bullet\( → playbook(action="add",
write_playbook\( → playbook(action="write",
delete_playbook_bullet\( → playbook(action="delete",
list_playbooks\( → playbook(action="list",
playbook_stats\( → playbook(action="stats",
prune_stale_bullets\( → playbook(action="prune",
confirm_ace_compliance\( → playbook(action="confirm",

# Profile tools
recommend_profiles\( → profiles(action="recommend",
check_reevaluation\( → profiles(action="check",
run_reevaluation\( → profiles(action="reevaluate",
cleanup_profiles\( → profiles(action="cleanup",

# Project tools
init_project\( → project(action="init",
scan_codebase\( → project(action="scan",
analyze_and_index\( → project(action="analyze",
sync_instruction_files\( → project(action="sync",
read_instruction_files\( → project(action="read_instructions",

# Admin tools
switch_model\( → admin(action="switch_model",
queue_status\( → admin(action="queue_status"
mcp_servers\( → admin(action="mcp_servers",
```

## Error Handling

### Old Error (v1.x)
```
Error: Tool 'session_compact' not found
```

### Solution (v2.0)
```python
# Use consolidated API
session(action="compact", session_id="...")
```

## FAQ

### Q: Can I still use the old tools?
**A**: No. This is a clean break with no backward compatibility. Old tools have been completely removed.

### Q: Why the breaking change?
**A**: Consolidation reduces cognitive overhead from 51 tools to 27 tools (-47%), making Delia easier to learn and use while preserving all functionality.

### Q: How do I know which action to use?
**A**: Each consolidated tool has a clear set of actions documented in `mcp_instructions.md`. Actions are verbs that describe the operation (list, read, write, delete, etc.).

### Q: What if I have code using old tools?
**A**: Update your code using the migration patterns above. The functionality is identical, only the API has changed.

### Q: Are there any performance differences?
**A**: No. Consolidated tools have the same performance characteristics as the old tools.

### Q: Will there be a deprecation period?
**A**: No. This is a one-time clean break. Version 2.0 removes old tools completely.

## Version Compatibility

| Delia Version | Tool Count | API Style |
|--------------|------------|-----------|
| v1.x | 51 tools | Individual tools |
| v2.0 | 27 tools | Action-based consolidated tools |

## Support

- Documentation: `mcp_instructions.md`
- Examples: See `CONSOLIDATION_PLAN.md`
- Issues: https://github.com/delia/issues

---

**Need Help?** Refer to `mcp_instructions.md` for complete API documentation of all consolidated tools.
