# Delia Tooling Improvement Roadmap

**Date:** 2024-12-23
**Status:** P0/P1/P2 IMPLEMENTED

## Implementation Summary

### Completed (2024-12-23)

| Priority | Item | Status | Details |
|----------|------|--------|---------|
| P0 | Bulk File Operations | ✅ Done | `read_files`, `edit_files` |
| P1 | Tool Discoverability | ✅ Done | `list_tools`, `describe_tool`, categories |
| P2 | Per-Project ACE Isolation | ✅ Done | See below |

---

## P2 Per-Project ACE Isolation

**Problem Solved:** Previously, a global `_ace_tracker` singleton was shared across all projects. If agents work on multiple projects concurrently, ACE state could leak between them.

**Solution:** Created `ACEEnforcementManager` that manages per-project trackers.

### Files Modified
- `src/delia/tools/handlers.py` - Added ACEEnforcementManager, updated get_ace_tracker

### New Components

**ACEEnforcementManager (lines 197-258)**
```python
class ACEEnforcementManager:
    """Manages per-project ACE enforcement trackers."""
    
    def get_tracker(self, project_path: str) -> ACEEnforcementTracker
    def cleanup_stale(self, max_age_seconds: int = 3600) -> int
    def list_projects(self) -> list[str]
    def get_stats(self) -> dict
```

**Updated get_ace_tracker (lines 265-275)**
```python
def get_ace_tracker(project_path: str | None = None) -> ACEEnforcementTracker:
    """Get tracker for a project. Defaults to cwd if path not provided."""
    path = project_path or str(Path.cwd())
    return _ace_manager.get_tracker(path)
```

**New Admin Tools**
- `ace_manager_stats()` - Get stats on active project trackers
- `ace_manager_cleanup(max_age_hours)` - Clean up stale trackers

### Key Behavior
- Each project path gets its own isolated ACEEnforcementTracker
- Paths are normalized via `Path.resolve()` for consistency
- Thread-safe via `threading.Lock()`
- Automatic cleanup of trackers inactive > 1 hour
- Backwards compatible - existing calls without project_path use cwd

---

## Remaining Work

| Priority | Item | Effort | Description |
|----------|------|--------|-------------|
| **P2** | handlers.py Modularization | Medium | Split 2400+ line file into modules |
| **P2** | Type/signature search | Medium | Find functions by return/param types |
| **P2** | AST pattern search | Medium | Find code by structural patterns |
| **P3** | LSP Refactoring Tools | High | Code actions (extract, organize imports) |
| **P3** | Cross-file data flow | High | Track data between files |
| **P3** | Change impact analysis | High | "What breaks if I change this?" |

---

## Code Locations Reference

| Component | File | Lines |
|-----------|------|-------|
| ACEEnforcementTracker | `handlers.py` | 71-194 |
| ACEEnforcementManager | `handlers.py` | 197-258 |
| get_ace_tracker | `handlers.py` | 265-275 |
| ace_manager_stats | `handlers.py` | 2348-2376 |
| ace_manager_cleanup | `handlers.py` | 2378-2399 |
