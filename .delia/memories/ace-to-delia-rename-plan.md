# ACE → Delia Framework Rename Plan

## Goal
Remove "ACE" branding so agents only need to know about "Delia" and "the framework" - not a separate "ACE" concept.

## Naming Strategy

| Old Term | New Term |
|----------|----------|
| ACE Framework | Delia Framework / "the framework" |
| ACE workflow | Context workflow / Delia workflow |
| ACE playbook | Playbook (unchanged) |
| ACE compliance | Framework compliance |

## Phase 1: Directory & File Renames

### 1.1 Directory: `src/delia/ace/` → `src/delia/learning/`
Files inside:
- `reflector.py` (keep name)
- `curator.py` (keep name)
- `deduplication.py` (keep name)
- `retrieval.py` (keep name)
- `__init__.py` (update imports)

### 1.2 File: `handlers_ace.py` → `handlers_enforcement.py`
Update all imports in:
- `handlers.py`
- `handlers_orchestration.py`
- `mcp_server.py`

## Phase 2: Symbol Renames

### 2.1 Classes
| Old | New |
|-----|-----|
| `ACEEnforcementTracker` | `EnforcementTracker` |
| `ACEEnforcementManager` | `EnforcementManager` |

### 2.2 Functions
| Old | New |
|-----|-----|
| `get_ace_tracker()` | `get_tracker()` |
| `get_ace_manager()` | `get_manager()` |
| `check_ace_gate()` | `check_context_gate()` |
| `inject_ace_reminder()` | `inject_reminder()` |
| `record_ace_started()` | `record_context_started()` |
| `is_ace_started()` | `is_context_started()` |
| `require_ace_started()` | `require_context_started()` |
| `auto_trigger_reflection()` | (keep) |

### 2.3 Constants
| Old | New |
|-----|-----|
| `ACE_EXEMPT_TOOLS` | `EXEMPT_TOOLS` |
| `ACE_REFLECTOR_PROMPT` | `REFLECTOR_PROMPT` |
| `ACE_CURATOR_PROMPT` | `CURATOR_PROMPT` |

### 2.4 Variables
| Old | New |
|-----|-----|
| `_ace_manager` | `_enforcement_manager` |
| `_ace_started` | `_context_started` |

## Phase 3: Tool Names (MCP Interface)

| Old | New |
|-----|-----|
| `check_ace_status` | `check_status` |
| `ace_manager_stats` | `framework_stats` |
| `ace_manager_cleanup` | `framework_cleanup` |

**Keep unchanged:**
- `auto_context()` - already good
- `complete_task()` - already good
- `get_playbook()` - already good
- `get_profile()` - already good

## Phase 4: Error Messages & Log Events

### 4.1 Error Codes
| Old | New |
|-----|-----|
| `ACE_WORKFLOW_REQUIRED` | `CONTEXT_WORKFLOW_REQUIRED` |
| `ACE_PLAYBOOK_NOT_QUERIED` | `PLAYBOOK_NOT_QUERIED` |

### 4.2 Log Events (structlog)
```python
# Old → New
"ace_workflow_started" → "context_workflow_started"
"ace_playbook_queried" → "playbook_queried"
"ace_checkpoint_called" → "checkpoint_called"
"ace_tracker_created" → "tracker_created"
"ace_tracker_cleaned" → "tracker_cleaned"
"ace_task_completed" → "task_completed"
"ace_reflection_complete" → "reflection_complete"
"ace_reflection_failed" → "reflection_failed"
"ace_meta_learning_complete" → "meta_learning_complete"
"ace_metrics_error" → "framework_metrics_error"
```

## Phase 5: API Routes

| Old | New |
|-----|-----|
| `/api/ace/metrics` | `/api/framework/metrics` |

## Phase 6: Documentation

### 6.1 Files to Update
- `CLAUDE.md` - Replace "ACE Framework" with "Delia Framework"
- `src/delia/mcp_instructions.md` - Same
- `.delia/profiles/core.md` - Title says "ACE Framework"

### 6.2 Memories to Update
| Memory | Action |
|--------|--------|
| `ace_framework.md` | Rename to `framework-research.md` (keep as reference) |
| `ace-framework-philosophy.md` | Rename to `framework-philosophy.md` |
| `ace-implementation-principles.md` | Rename to `implementation-principles.md` |
| `ace-component-design.md` | Rename to `component-design.md` |
| `ace-multi-platform-setup.md` | Rename to `multi-platform-setup.md` |
| `ace-enforcement-setup.md` | Rename to `enforcement-setup.md` |

### 6.3 Scripts to Update
- `scripts/ace-enforce-hook.py` → `scripts/delia-enforce-hook.py`
- `scripts/ace-task-hook.py` → `scripts/delia-task-hook.py`

## Phase 7: Import Updates

After directory rename, update all imports:
```python
# Old
from delia.ace import Reflector, Curator
from delia.tools.handlers_ace import get_ace_tracker

# New
from delia.learning import Reflector, Curator
from delia.tools.handlers_enforcement import get_tracker
```

Files with imports to update:
- `src/delia/tools/handlers.py`
- `src/delia/tools/handlers_orchestration.py`
- `src/delia/tools/handlers_playbook.py`
- `src/delia/orchestration/executor.py`
- `src/delia/orchestration/context.py`
- `src/delia/mcp_server.py`
- `src/delia/api.py`

## Execution Order

1. **Create git branch** `refactor/ace-to-delia-rename`
2. **Phase 1**: Directory/file renames (git mv)
3. **Phase 2**: Symbol renames (LSP rename when possible)
4. **Phase 3**: Tool names (requires MCP interface update)
5. **Phase 4**: Error/log strings (search & replace)
6. **Phase 5**: API routes
7. **Phase 6**: Documentation
8. **Phase 7**: Fix all imports
9. **Run tests**: `uv run pytest`
10. **Update playbooks**: Remove ACE references from bullet content

## Risks

1. **Breaking MCP clients** - Tool name changes affect existing users
   - Mitigation: Keep old tool names as aliases during transition
   
2. **Import errors** - Many files import from `ace/` and `handlers_ace`
   - Mitigation: Run full test suite after each phase
   
3. **Documentation drift** - External docs may reference ACE
   - Mitigation: Search GitHub issues/PRs for ACE mentions

## Success Criteria

- [ ] No `ACE` or `ace_` in tool names
- [ ] No `ACE` in error messages shown to users
- [ ] CLAUDE.md refers only to "Delia Framework"
- [ ] All tests pass
- [ ] No `from delia.ace` imports (now `from delia.learning`)
