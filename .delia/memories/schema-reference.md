# Delia Framework Schema Reference

This document defines the data formats used by the Delia Framework.

## Schema Versioning

Delia uses semantic versioning for its data schemas. The current schema version is **v2**.

### Version History

| Version | Description |
|---------|-------------|
| v1 | Original flat array format (legacy) |
| v2 | Wrapped format with metadata (current) |

### Migration

- Migration is **automatic on read** - old formats are understood
- Migration is **applied on save** - all saves use current version
- Use `delia validate --fix` to migrate all files explicitly

## Validation

```bash
delia validate           # Validate current project
delia validate /path     # Validate specific project
delia validate --fix     # Migrate to current schema version
```

Checks:
- JSON structure in playbooks
- Required fields (id, content) in bullets
- Schema version compatibility
- Content length (10-500 chars recommended)
- Profiles/memories exist

## Directory Structure

```
.delia/
├── playbooks/           # Per-task-type learned patterns
│   ├── coding.json
│   ├── testing.json
│   └── ...
├── profiles/            # Framework/domain guidance (markdown)
│   ├── core.md          # Always loaded
│   └── ...
├── memories/            # Persistent project knowledge (markdown)
│   └── ...
└── project_summary.json # Project metadata
```

## Playbook Schema (v2)

```json
{
  "schema_version": 2,
  "task_type": "coding",
  "updated_at": "2025-12-23T12:00:00.000000",
  "bullets": [
    {
      "id": "strat-{hash}",
      "content": "Use pathlib.Path over os.path",
      "section": "code_standards",
      "helpful_count": 5,
      "harmful_count": 0,
      "created_at": "2025-12-22T...",
      "last_used": "2025-12-23T...",
      "source_task": "coding",
      "source": "learned"
    }
  ]
}
```

### Top-Level Fields (v2)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | int | Yes | Schema version (currently 2) |
| `task_type` | string | Yes | Task category (coding, testing, etc.) |
| `updated_at` | ISO8601 | Yes | Last modification timestamp |
| `bullets` | array | Yes | Array of bullet objects |

### Bullet Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier (`strat-{hash}`) |
| `content` | string | Yes | The guidance (10-500 chars) |
| `section` | string | No | Category within playbook |
| `helpful_count` | int | Auto | Times marked helpful |
| `harmful_count` | int | Auto | Times marked harmful |
| `created_at` | ISO8601 | Auto | When added |
| `last_used` | ISO8601 | Auto | Last retrieval |
| `source_task` | string | Auto | Task that generated this |
| `source` | enum | Auto | Origin: seed, learned, manual, reflector, curator |

### Computed Fields

| Field | Formula | Description |
|-------|---------|-------------|
| `utility_score` | `helpful / (helpful + harmful + 1)` | Effectiveness (0.0-1.0) |

### Content Quality Rules

Good bullets are:
- **Actionable**: "Use X instead of Y" not "X is better"
- **Specific**: "Use pathlib.Path" not "use good libraries"
- **Concise**: 10-150 chars ideal, 500 max
- **Verifiable**: Can be checked by grep/test

Bad bullets (should be rejected):
- Vague: "Write good code"
- Too long: Multi-paragraph essays
- Duplicates: Same advice worded differently
- Obvious: "Use type hints" (already in core profile)

### Utility Score Thresholds

| Range | Interpretation |
|-------|----------------|
| 0.0-0.3 | Low utility - candidate for pruning |
| 0.3-0.7 | Moderate utility |
| 0.7-1.0 | High utility - proven effective |

## Task Types

| Type | Keywords |
|------|----------|
| coding | implement, add, create, build, write, refactor |
| testing | test, pytest, coverage, mock, assert |
| debugging | bug, error, fix, debug, broken, failing |
| architecture | design, pattern, ADR, plan, think through |
| git | commit, branch, merge, PR, push, pull |
| deployment | docker, CI/CD, production, ship |
| security | auth, password, token, injection, XSS |
| project | how, what, where, explain |
| api | endpoint, REST, GraphQL, request |
| performance | optimize, cache, slow, fast |

## Profiles and Memories

### Profiles (.delia/profiles/*.md)

Markdown files with framework-specific guidance. Loaded based on project detection.

### Memories (.delia/memories/*.md)

Markdown files for persistent project knowledge. Free-form content.

---

*Schema version: 2 | Last updated: 2025-12-23*
