# Per-Project Isolation Analysis

**Date**: 2025-12-21
**Issue**: Delia must isolate ALL project-specific data to prevent confusion between projects

## Current Storage Structure

### ✅ Already Per-Project (Correct)

Located in `<project>/.delia/`:

- **Playbooks**: `<project>/.delia/playbooks/*.json`
  - Managed by: `playbook_manager.py`
  - Set via: `playbook_manager.set_project(path)`

- **Memories**: `<project>/.delia/memories/*.md`
  - Managed by: `memory_tool()` in consolidated.py
  - Auto-detects from `path` parameter or `cwd`

- **Profiles**: `<project>/.delia/profiles/*.md`
  - Managed by: `profiles_tool()`
  - Auto-detects from `path` parameter

- **Project Summary**: `<project>/.delia/project_summary.json`
  - Managed by: `project_tool(action="write_summary")`

- **Evaluation State**: `<project>/.delia/evaluation_state.json`
  - Managed by: `profiles_tool(action="check")`

### ❌ Currently Global (VIOLATIONS - Must Fix)

Located in `~/.delia/data/` or `DELIA_DATA_DIR`:

1. **Sessions** ⚠️ **CRITICAL**
   - Current: `~/.delia/data/sessions/`
   - Defined: `paths.py:126` - `SESSIONS_DIR`
   - Used by: `session_manager.py:338`
   - **Problem**: Sessions from different projects get mixed together
   - **Fix**: Move to `<project>/.delia/sessions/`

2. **Memories** (legacy)
   - Current: `~/.delia/data/memories/`
   - Defined: `paths.py:124` - `MEMORIES_DIR`
   - Status: Appears unused (tools use `.delia/memories`)
   - **Fix**: Remove from paths.py

3. **Playbooks** (legacy)
   - Current: `~/.delia/data/playbooks/`
   - Defined: `paths.py:144` - `PLAYBOOKS_DIR`
   - Status: Appears unused (playbook_manager uses `.delia/playbooks`)
   - **Fix**: Remove from paths.py

4. **Orchestration** (unclear)
   - Current: `~/.delia/data/orchestration/`
   - Defined: `paths.py:146` - `ORCHESTRATION_DIR`
   - Status: Need to investigate usage
   - **Decision**: TBD - might be OK if truly system-wide

### ✅ Correctly Global (System-Wide Data)

These should remain in `~/.delia/data/`:

- **Cache files**: Stats, metrics, affinity, circuit breaker, prewarm
  - `cache/usage_stats.json`
  - `cache/enhanced_stats.json`
  - `cache/backend_metrics.json`
  - `cache/affinity.json`
  - `cache/circuit_breaker.json`
  - `cache/prewarm.json`
  - `cache/live_logs.json`
  - **Rationale**: System-wide performance metrics across all projects

- **User database**: `users/users.db`
  - **Rationale**: Global user authentication

- **Settings**: `~/.delia/settings.json`
  - **Rationale**: Global backend configuration

## Required Changes

### 1. SessionManager Per-Project Storage

**File**: `src/delia/session_manager.py`

**Current Problem**:
```python
# Line 338
self.session_dir = session_dir or paths.SESSIONS_DIR  # Global!
```

**Solution**:
```python
# Auto-detect project from cwd if not provided
if session_dir is None:
    project_path = Path.cwd()
    session_dir = project_path / ".delia" / "sessions"
else:
    session_dir = Path(session_dir)

session_dir.mkdir(parents=True, exist_ok=True)
self.session_dir = session_dir
```

**Impact**:
- Each project gets its own isolated session storage
- No cross-project session contamination
- Backward compatible with explicit `session_dir` parameter

### 2. Remove Legacy Global Dirs from paths.py

**File**: `src/delia/paths.py`

**Lines to Remove** (from `__getattr__`):
```python
# Line 124 - REMOVE
elif name == "MEMORIES_DIR":
    return get_data_dir() / "memories"

# Line 126 - REMOVE
elif name == "SESSIONS_DIR":
    return get_data_dir() / "sessions"

# Line 144 - REMOVE
elif name == "PLAYBOOKS_DIR":
    return get_data_dir() / "playbooks"
```

**Lines to Remove** (from `ensure_directories()`):
```python
# Lines 155-159, 167-169 - REMOVE
memories_dir = data_dir / "memories"
sessions_dir = data_dir / "sessions"
playbooks_dir = data_dir / "playbooks"
# ... corresponding .mkdir() calls
```

### 3. Auto-Detect Project Context

**File**: `src/delia/mcp_server.py`

Add project context detection on startup:
```python
# At server initialization
def auto_detect_project():
    """Auto-detect project from cwd and set context."""
    cwd = Path.cwd()

    # Check if we're in a project (has .delia/, .git/, or pyproject.toml)
    if (cwd / ".delia").exists() or (cwd / ".git").exists() or (cwd / "pyproject.toml").exists():
        set_project_context(str(cwd))
        log.info("auto_detected_project", path=str(cwd))
        return cwd

    # Walk up to find project root
    for parent in cwd.parents:
        if (parent / ".delia").exists() or (parent / ".git").exists():
            set_project_context(str(parent))
            log.info("auto_detected_project", path=str(parent))
            return parent

    # Default to cwd
    set_project_context(str(cwd))
    return cwd

# Call on server start
project_path = auto_detect_project()
```

### 4. Update Session Tools

**File**: `src/delia/tools/consolidated.py`

Update `session_tool()` to use per-project storage:
```python
async def session_tool(action: str, session_id: str | None = None, ...):
    from ..session_manager import get_session_manager
    from pathlib import Path

    # Get session manager with per-project storage
    sm = get_session_manager()

    # Ensure using project-specific storage
    project_path = Path.cwd()
    sm.session_dir = project_path / ".delia" / "sessions"
    sm.session_dir.mkdir(parents=True, exist_ok=True)

    # ... rest of implementation
```

## Final Per-Project Structure

```
<project>/
├── .delia/                         # ALL project-specific data here
│   ├── sessions/                   # ✅ Session history (NEW)
│   │   └── *.json                  # Session files
│   ├── playbooks/                  # ✅ Already per-project
│   │   ├── coding.json
│   │   ├── testing.json
│   │   └── ...
│   ├── memories/                   # ✅ Already per-project
│   │   ├── architecture.md
│   │   └── debugging.md
│   ├── profiles/                   # ✅ Already per-project
│   │   └── *.md
│   ├── project_summary.json        # ✅ Already per-project
│   ├── evaluation_state.json       # ✅ Already per-project
│   ├── symbol_graph.json           # ✅ Already per-project
│   └── compliance/                 # ✅ Already per-project
│       └── ace_log.jsonl
│
~/.delia/                           # Global system-wide data ONLY
├── settings.json                   # Backend configs
└── data/
    ├── cache/                      # System performance metrics
    │   ├── usage_stats.json
    │   ├── backend_metrics.json
    │   ├── affinity.json
    │   └── circuit_breaker.json
    └── users/                      # Global auth
        └── users.db
```

## Testing Plan

1. **Test Session Isolation**:
   ```bash
   cd /project1
   delia chat  # Creates .delia/sessions/

   cd /project2
   delia chat  # Creates separate .delia/sessions/

   # Verify: Sessions don't mix between projects
   ```

2. **Test Playbook Isolation** (already working):
   ```bash
   cd /project1
   get_playbook(task_type="coding")  # Loads from /project1/.delia/playbooks/

   cd /project2
   get_playbook(task_type="coding")  # Loads from /project2/.delia/playbooks/
   ```

3. **Test Memory Isolation** (already working):
   ```bash
   memory(action="list", path="/project1")  # Shows project1 memories only
   memory(action="list", path="/project2")  # Shows project2 memories only
   ```

## Migration Notes

### For Existing Users

**Sessions**: Existing global sessions in `~/.delia/data/sessions/` will remain there but won't be accessible to new per-project sessions. Users can:
- Option 1: Ignore old sessions (they'll age out)
- Option 2: Manually move relevant sessions to project `.delia/sessions/`
- Option 3: We provide a migration script

**Playbooks/Memories**: Already per-project, no migration needed.

### Backward Compatibility

**Breaking Change**: Sessions are now per-project. This is intentional to fix the isolation violation.

**Mitigation**: Document clearly in MIGRATION.md and CHANGELOG.md.

## Implementation Checklist

- [ ] Update SessionManager to use per-project storage
- [ ] Remove MEMORIES_DIR, SESSIONS_DIR, PLAYBOOKS_DIR from paths.py
- [ ] Add auto-detect project context on server start
- [ ] Update session_tool() to enforce per-project storage
- [ ] Test with multiple projects
- [ ] Document in MIGRATION.md
- [ ] Update CHANGELOG.md

---

**Priority**: HIGH - This fixes a fundamental isolation violation that could cause data leakage between projects.
