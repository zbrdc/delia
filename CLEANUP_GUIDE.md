# Delia Cleanup Guide

**Version**: 2.0+
**Purpose**: Clean up legacy global data after per-project isolation fixes

## Why Cleanup is Needed

Delia v2.0 moved from global data storage to per-project isolation:

| Before (v1.x) | After (v2.0) |
|---------------|--------------|
| `~/.delia/data/sessions/` | `<project>/.delia/sessions/` |
| `~/.delia/data/memories/` | `<project>/.delia/memories/` |
| `~/.delia/data/playbooks/` | `<project>/.delia/playbooks/` |

The old global directories are now obsolete and can be safely removed.

## Quick Start

### 1. Check What Would Be Removed (Dry Run)

```python
# See what cleanup would remove (safe - doesn't delete anything)
admin(action="cleanup_all", dry_run=True)
```

**Example Output**:
```json
{
  "dry_run": true,
  "legacy_global": {
    "removed": [
      {
        "path": "/home/user/.delia/data/sessions",
        "files": 40,
        "size_bytes": 62000,
        "action": "would_remove"
      }
    ]
  },
  "summary": {
    "total_files": 51,
    "total_size_mb": 0.06,
    "action": "would_remove"
  }
}
```

### 2. Actually Remove Legacy Data

```python
# WARNING: This DELETES old global data!
admin(action="cleanup_all", dry_run=False)
```

## Cleanup Options

### Option A: Full Cleanup (Recommended)

Remove ALL legacy global data in one operation:

```python
# Dry run first (always recommended)
admin(action="cleanup_all", dry_run=True)

# If happy with what will be removed, run for real
admin(action="cleanup_all", dry_run=False)
```

**What it removes**:
- `~/.delia/data/sessions/` (old global sessions)
- `~/.delia/data/memories/` (old global memories, if any)
- `~/.delia/data/playbooks/` (old global playbooks, if any)

**What it KEEPS** (correctly global):
- `~/.delia/settings.json` ✅
- `~/.delia/data/cache/` ✅
- `~/.delia/data/users/` ✅

### Option B: Selective Cleanup

Remove only specific legacy directories:

```python
# Just legacy sessions/memories/playbooks
admin(action="cleanup_legacy", dry_run=False)
```

### Option C: Per-Project Reset

Clean a specific project's `.delia/` directory for re-initialization:

```python
# WARNING: This deletes ALL project data!
admin(action="cleanup_project", path="/path/to/project", dry_run=False)
```

**What it removes**:
- `<project>/.delia/sessions/`
- `<project>/.delia/playbooks/`
- `<project>/.delia/memories/`
- `<project>/.delia/profiles/`
- `<project>/.delia/project_summary.json`
- Everything in `<project>/.delia/`

**Use case**: Starting fresh on a project that was initialized incorrectly.

## Safe Workflow

**Always use this workflow**:

1. **Dry run first** to see what would be removed:
   ```python
   admin(action="cleanup_all", dry_run=True)
   ```

2. **Review the output** - check files/sizes

3. **Run for real** if you're happy:
   ```python
   admin(action="cleanup_all", dry_run=False)
   ```

## Migration (Optional)

If you have important old sessions you want to keep, you can migrate them to a specific project:

```python
from src.delia.cleanup import migrate_global_sessions_to_project
from pathlib import Path

# Dry run
results = migrate_global_sessions_to_project(
    project_path=Path("/path/to/project"),
    session_filter="client_id_filter",  # optional
    dry_run=True
)

# Run for real
results = migrate_global_sessions_to_project(
    project_path=Path("/path/to/project"),
    dry_run=False
)
```

**NOTE**: Most users should just start fresh and ignore old sessions.

## CLI Usage (Future)

A `delia cleanup` command will be added:

```bash
# Dry run
delia cleanup --dry-run

# Full cleanup
delia cleanup --all

# Legacy only
delia cleanup --legacy

# Per-project
delia cleanup --project /path/to/project

# Interactive
delia cleanup --interactive
```

## What Stays vs What Goes

### ✅ Stays (Correctly Global)

These are system-wide and should NEVER be cleaned:

| Path | Purpose | Keep? |
|------|---------|-------|
| `~/.delia/settings.json` | Backend configs | ✅ YES |
| `~/.delia/data/cache/*.json` | System metrics | ✅ YES |
| `~/.delia/data/users/users.db` | User auth | ✅ YES |
| `~/.delia/data/orchestration/` | System-wide | ✅ YES |

### ❌ Goes (Legacy/Obsolete)

These are now per-project and should be cleaned:

| Path | New Location | Clean? |
|------|--------------|--------|
| `~/.delia/data/sessions/` | `<project>/.delia/sessions/` | ❌ YES |
| `~/.delia/data/memories/` | `<project>/.delia/memories/` | ❌ YES |
| `~/.delia/data/playbooks/` | `<project>/.delia/playbooks/` | ❌ YES |

## Cleanup Checklist

Use this checklist when cleaning up multiple projects:

- [ ] Backup any important data (optional, but safe)
- [ ] Run dry run: `admin(action="cleanup_all", dry_run=True)`
- [ ] Review what would be removed
- [ ] Actually cleanup: `admin(action="cleanup_all", dry_run=False)`
- [ ] For each project directory:
  - [ ] `cd /path/to/project`
  - [ ] Run `delia init` (or call `init_project()`)
  - [ ] Verify `.delia/` directory created
  - [ ] Test: create a session, verify it's in `.delia/sessions/`

## Example: Clean Re-Init All Projects

```bash
#!/bin/bash
# cleanup_all_projects.sh

echo "=== Cleaning up Delia for fresh start ==="

# 1. Cleanup legacy global data
echo "Step 1: Cleaning legacy global data..."
python -c "
import asyncio
from src.delia.tools.consolidated import admin_tool

async def cleanup():
    result = await admin_tool(action='cleanup_all', dry_run=False)
    print(result)

asyncio.run(cleanup())
"

# 2. Re-init all your projects
PROJECTS=(
    "/home/user/projects/project1"
    "/home/user/projects/project2"
    "/home/user/projects/delia"
)

for project in "${PROJECTS[@]}"; do
    echo "Step 2: Re-initializing $project..."
    cd "$project" || continue

    # Optional: Clean existing .delia/ if you want fresh start
    # rm -rf .delia/

    # Re-init
    delia init --force

    echo "  ✓ $project initialized"
done

echo "=== Cleanup complete! ==="
```

## Troubleshooting

### "Directory not found" errors

**Cause**: Legacy directories already don't exist (good!)

**Solution**: Nothing to do - already clean.

### "Permission denied" errors

**Cause**: Files owned by different user or protected

**Solution**: Use `sudo` or change ownership:
```bash
sudo chown -R $USER ~/.delia/data/sessions/
```

### Want to undo cleanup

**Cause**: Accidentally deleted something

**Solution**: Old sessions are gone. Start fresh with new per-project sessions. This is the intended behavior - old global sessions are obsolete.

## FAQ

### Q: Will this delete my backend configs?

**A**: No! `~/.delia/settings.json` is preserved. Only obsolete project data is removed.

### Q: Do I need to cleanup before re-init?

**A**: No, but recommended. Old global data wastes disk space and won't be used anyway.

### Q: Can I get old sessions back?

**A**: Not after cleanup. Use dry run first to review. But old sessions were global and mixed projects - you want fresh per-project sessions anyway.

### Q: What if I have hundreds of old sessions?

**A**: Cleanup is fast. Typical cleanup removes <1MB in <1 second.

### Q: Should I migrate old sessions or start fresh?

**A**: **Start fresh** (recommended). Old global sessions mixed different projects and are unreliable.

## Summary

**TL;DR**: Run this once after upgrading to v2.0:

```python
# 1. Check what will be removed
admin(action="cleanup_all", dry_run=True)

# 2. Remove legacy data
admin(action="cleanup_all", dry_run=False)

# 3. Re-init each project
# cd /project && delia init
```

Then you're ready to use Delia v2.0 with proper per-project isolation! 🎯

---

**Related**:
- `PER_PROJECT_FIXES_COMPLETE.md` - Why cleanup is needed
- `MIGRATION.md` - Tool consolidation changes
- `ADR-009` - Consolidation architecture decision
