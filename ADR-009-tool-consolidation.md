# ADR-009: Tool API Consolidation

**Status**: Approved
**Date**: 2024-12-21
**Decision**: Consolidate Delia's 51 MCP tools to 32 using action-based parameters

## Context

Delia currently exposes 51 MCP tools across 8 categories. While comprehensive, this creates:
- Cognitive overhead for users learning the API
- Documentation complexity
- Maintenance burden for similar CRUD operations

## Decision

Consolidate CRUD-like operations into action-based tools while preserving semantically distinct operations.

### Consolidated Tools (30 → 6)

1. **playbook(action, task_type, ...)**
   - Actions: add, write, delete, prune, list, stats, confirm
   - Replaces: add_playbook_bullet, write_playbook, delete_playbook_bullet, prune_stale_bullets, list_playbooks, playbook_stats, confirm_ace_compliance

2. **session(action, session_id, ...)**
   - Actions: list, stats, compact, delete
   - Replaces: session_list, session_stats, session_compact, session_delete

3. **memory(action, name, ...)**
   - Actions: list, read, write, delete
   - Replaces: list_memories, read_memory, write_memory, delete_memory

4. **profiles(action, ...)**
   - Actions: recommend, check, reevaluate, cleanup
   - Replaces: recommend_profiles, check_reevaluation, run_reevaluation, cleanup_profiles

5. **project(action, path, ...)**
   - Actions: init, scan, analyze, sync, read_instructions
   - Replaces: init_project, scan_codebase, analyze_and_index, sync_instruction_files, read_instruction_files

6. **admin(action, ...)**
   - Actions: switch_model, queue_status, mcp_servers
   - Replaces: switch_model, queue_status, mcp_servers

### Preserved Tools (21)

**LSP Code Intelligence (9)**: Each has unique parameters/semantics
- lsp_goto_definition, lsp_find_references, lsp_hover
- lsp_get_symbols, lsp_find_symbol, lsp_rename_symbol
- lsp_replace_symbol_body, lsp_insert_before_symbol, lsp_insert_after_symbol

**Core ACE (3)**: Critical workflow tools
- get_playbook, get_project_context, report_feedback

**Delegation (6)**: Distinct orchestration patterns
- delegate, batch, chain, workflow, think, agent

**Admin (3)**: Core system tools
- health, models, switch_backend

**Special (1)**: Critical for context switching
- set_project

## Consequences

**Positive:**
- 37% reduction in tool count (51 → 32)
- Cleaner, more consistent API
- Easier to learn and document
- Pattern established for future CRUD tools

**Negative:**
- Requires migration for existing integrations
- Action parameter adds one level of indirection
- Breaking change (requires version bump)

## Implementation Plan

1. Create new consolidated tool functions in `tools/consolidated.py`
2. Implement action dispatch with validation
3. Add deprecation warnings to old tools
4. Update MCP instructions and playbooks
5. Migrate internal usage to new API
6. Version bump: 1.0.0 → 2.0.0
7. Remove deprecated tools after transition period

## Migration Example

```python
# OLD
add_playbook_bullet(task_type="coding", content="Use async", section="patterns")
list_playbooks()
delete_playbook_bullet(bullet_id="strat-123", task_type="coding")

# NEW
playbook(action="add", task_type="coding", content="Use async", section="patterns")
playbook(action="list")
playbook(action="delete", bullet_id="strat-123", task_type="coding")
```
