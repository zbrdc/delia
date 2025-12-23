# Cleanup Backlog

Audit performed: 2024-12-22

## Dead Code (Never Imported)

### `src/delia/eval_harness.py` (23 lines)
- **Status:** DEAD
- **Reason:** Broken stub for `lm-evaluation-harness` integration
- **Issues:**
  - Imports `lm_eval` which is not a project dependency
  - Has syntax/indentation errors
- **Action:** Delete

### `src/delia/personas.py` (261 lines)
- **Status:** DEAD
- **Reason:** Dynamic persona system that was never wired up
- **Issues:**
  - Not imported anywhere in the codebase
  - Defines PersonaType, Persona classes but nothing uses them
- **Action:** Delete

### `src/delia/sandbox.py` (374 lines)
- **Status:** DEAD
- **Reason:** Sandboxed code execution using `llm-sandbox`
- **Issues:**
  - Not imported anywhere in the codebase
  - Optional dependency `llm-sandbox` not in pyproject.toml extras
- **Action:** Delete

## Deprecated Code (Marked for Removal)

### `src/delia/frustration.py` (376 lines)
- **Status:** DEPRECATED
- **Reason:** Explicitly marked deprecated in docstring
- **Consolidated into:** `orchestration/intrinsics.py` (IntrinsicsEngine.check_user_state())
- **Issues:**
  - Only imports itself in docstring example
  - Functionality moved elsewhere
- **Action:** Delete after verifying intrinsics.py has the functionality

### `src/delia/messages.py` (partial)
- **Status:** DEPRECATED ALIASES
- **Reason:** Old "garden-themed" naming
- **Deprecated aliases:**
  - `GardenEvent` → `StatusEvent`
  - `MESSAGES` → `STATUS_MESSAGES`
  - `VINE_MESSAGES` → `TIER_MESSAGES`
  - `get_message` → `get_status_message`
- **Action:** Search for usages, then remove aliases

## Questionable (Review Before Removing)

### `src/delia/melons.py` (436 lines)
- **Status:** QUESTIONABLE
- **Reason:** Gamification/reward system ("melon tracker")
- **Used by:** Tests and benchmarks only (not production code paths)
- **Action:** Decide if gamification is a wanted feature

### `src/delia/multi_user_tracking.py` (581 lines)
- **Status:** QUESTIONABLE
- **Reason:** Multi-tenant/rate limiting support
- **Used by:** Tests only
- **Action:** Decide if multi-tenant is a wanted feature

## Other Cleanup

### `build/` directory
- **Status:** SHOULD NOT EXIST
- **Reason:** Build artifacts in repo, contains duplicate .py files
- **Action:** Add to .gitignore, delete from repo

### `.mcp.json`
- **Status:** CLEANED (2024-12-22)
- **Action taken:** Removed `coderag` entry (external project, not part of Delia)

### `src/delia/tools/handlers.py`
- **Status:** CLEANED (2024-12-22)
- **Action taken:** Removed `coderag_search` tool recommendations (5 occurrences)
