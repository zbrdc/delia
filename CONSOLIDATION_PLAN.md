# Tool Consolidation Implementation Plan

**Goal**: Reduce Delia's tool count from 51 → 32 tools (-37%)
**Status**: Planning
**Target**: Version 2.0.0

## Phase 1: Create Consolidated Tools

### 1.1 Create `src/delia/tools/consolidated.py`

```python
# New consolidated tool implementations
async def playbook_tool(action: str, task_type: str | None = None, **kwargs) -> str:
    """Unified playbook management tool.

    Actions:
        add - Add new bullet
        write - Write/replace entire playbook
        delete - Delete bullet by ID
        prune - Remove stale bullets
        list - List all playbooks
        stats - Get effectiveness scores
        confirm - Confirm ACE compliance
    """
    if action == "add":
        return await _playbook_add(task_type, kwargs.get("content"), kwargs.get("section"))
    elif action == "write":
        return await _playbook_write(task_type, kwargs.get("bullets"))
    # ... etc
```

Similar pattern for:
- `session_tool(action, session_id, **kwargs)`
- `memory_tool(action, name, **kwargs)`
- `profiles_tool(action, **kwargs)`
- `project_tool(action, path, **kwargs)`
- `admin_tool(action, **kwargs)`

### 1.2 Register in `mcp_server.py`

```python
@mcp.tool()
async def playbook(
    action: Literal["add", "write", "delete", "prune", "list", "stats", "confirm"],
    task_type: str | None = None,
    **kwargs
) -> str:
    """Unified playbook management."""
    from .tools.consolidated import playbook_tool
    return await playbook_tool(action, task_type, **kwargs)
```

## Phase 2: Add Deprecation Warnings

### 2.1 Wrap old tools with deprecation decorator

```python
@deprecated(replacement="playbook(action='add', ...)", version="2.0.0")
async def add_playbook_bullet(...) -> str:
    log.warning("add_playbook_bullet is deprecated, use playbook(action='add') instead")
    return await playbook_tool("add", ...)
```

### 2.2 Update tool docstrings

Add deprecation notices to all old tools.

## Phase 3: Update Documentation

### 3.1 MCP Instructions (`mcp_instructions.md`)

Replace tool lists with consolidated versions:

```markdown
### ACE Framework Tools
- **playbook(action, task_type, ...)** - Unified playbook management
  - Actions: add, write, delete, prune, list, stats, confirm
- **get_playbook(task_type, limit?, path?)** - Get strategic bullets (unchanged)
- **get_project_context(path?)** - Get project overview (unchanged)
- **report_feedback(bullet_id, task_type, helpful)** - Report feedback (unchanged)
```

### 3.2 Playbook Bullets

Update coding/architecture playbooks with new tool patterns.

### 3.3 CLAUDE.md and agent instructions

Sync updated documentation to all agents.

## Phase 4: Internal Migration

### 4.1 Update internal callers

Find all internal uses of old tools:
```bash
grep -r "add_playbook_bullet\|write_playbook" src/delia/
```

Replace with new consolidated API.

### 4.2 Update tests

Migrate test suite to use new tools:
- `tests/test_playbook.py`
- `tests/test_session.py`
- `tests/test_memory.py`

## Phase 5: Version Bump & Release

### 5.1 Version 2.0.0

Update:
- `pyproject.toml`: version = "2.0.0"
- `CHANGELOG.md`: Document breaking changes
- Migration guide for users

### 5.2 Deprecation Timeline

- **v2.0.0**: Old tools deprecated with warnings
- **v2.1.0**: (3 months later) Remove deprecated tools
- **v3.0.0**: Clean break, only consolidated tools

## File Checklist

New files:
- [ ] `src/delia/tools/consolidated.py` - Consolidated tool implementations
- [ ] `MIGRATION.md` - User migration guide
- [ ] `ADR-009-tool-consolidation.md` - Architecture decision record ✓

Modified files:
- [ ] `src/delia/mcp_server.py` - Register new tools, deprecate old
- [ ] `src/delia/tools/handlers.py` - Add deprecation wrappers
- [ ] `src/delia/mcp_instructions.md` - Updated tool reference
- [ ] `CLAUDE.md` - Updated for consolidation
- [ ] `.gemini/instructions.md` - Synced
- [ ] `.github/copilot-instructions.md` - Synced
- [ ] `pyproject.toml` - Version bump
- [ ] `CHANGELOG.md` - Release notes

Test files:
- [ ] `tests/test_consolidated_tools.py` - New test suite
- [ ] `tests/test_playbook.py` - Update to new API
- [ ] `tests/test_session.py` - Update to new API
- [ ] `tests/test_memory.py` - Update to new API

## Testing Strategy

1. **Unit tests**: Each action in consolidated tools
2. **Integration tests**: Full workflow with new API
3. **Deprecation tests**: Verify warnings are emitted
4. **Migration tests**: Verify old→new equivalence
5. **Backward compat**: Old tools work until removal

## Rollout

1. Create feature branch: `feat/consolidate-tools-adr009`
2. Implement Phase 1-2 (new tools + deprecation)
3. Internal testing
4. Update documentation (Phase 3)
5. Migrate internals (Phase 4)
6. PR review
7. Merge to main
8. Release v2.0.0
9. Monitor adoption
10. Remove deprecated tools in v2.1.0
