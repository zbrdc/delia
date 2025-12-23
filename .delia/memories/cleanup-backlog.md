# Cleanup Backlog

Dead code and deprecated modules to remove in future cleanup sessions.

## Dead Code (Never Imported)

| File | Reason |
|------|--------|
| `src/delia/eval_harness.py` | Broken stub, never imported |
| `src/delia/personas.py` | Never imported |
| `src/delia/sandbox.py` | Never imported |
| `data/constitution.md` | Exists but never loaded by code |

## Deprecated (Superseded)

| File | Reason |
|------|--------|
| `src/delia/frustration.py` | Old frustration detection, superseded by ACE |

## Questionable (Needs Review)

| File | Reason |
|------|--------|
| `src/delia/melons.py` | Performance economy system - check if actively used |
| `src/delia/multi_user_tracking.py` | Check if session system uses this |

## Action Items

1. Verify each file has zero imports before removal
2. Remove from `__init__.py` exports if present
3. Update any documentation referencing removed modules
