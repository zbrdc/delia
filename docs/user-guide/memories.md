# Memories

Memories are persistent knowledge stored as Markdown files. Unlike playbooks (atomic patterns), memories hold rich, contextual information about your project.

## What Are Memories?

Markdown files in `.delia/memories/`:

```
.delia/memories/
├── architecture.md       # System design notes
├── conventions.md        # Code conventions
├── delia-workflow-guide.md  # How to use Delia
└── decisions.md          # Key decisions made
```

## When to Use Memories

| Use Case | Example |
|----------|---------|
| Architecture notes | "Auth uses JWT with refresh tokens" |
| API contracts | "POST /users returns 201 with Location header" |
| Decisions made | "Chose PostgreSQL over MongoDB because..." |
| Lessons learned | "Don't use X library, it has memory leaks" |
| Context for AI | "This is a React Native app for iOS/Android" |

## Managing Memories

### List Memories

```python
memory(action="list")
```

### Read a Memory

```python
memory(action="read", name="architecture")
```

### Write a Memory

```python
memory(action="write", name="api-contracts", content="""
# API Contracts

## Users Endpoint
- GET /users - List all users
- POST /users - Create user (returns 201)
- GET /users/:id - Get user by ID
""")
```

### Delete a Memory

```python
memory(action="delete", name="old-notes")
```

## Memory Discovery

Delia discovers memories from multiple locations (in priority order):

1. **User defaults**: `~/.delia/DELIA.md`
2. **Project root**: `./DELIA.md`, `./.delia/DELIA.md`
3. **Modular rules**: `./.delia/rules/*.md`
4. **Local overrides**: `./DELIA.local.md` (git-ignored)

## Import Syntax

Memories can include other files:

```markdown
# Project Instructions

@path/to/other-file.md

## Additional Notes
...
```

## Memories vs Playbooks

| Aspect | Memories | Playbooks |
|--------|----------|-----------|
| Format | Markdown | JSON |
| Content | Rich context | Atomic patterns |
| Updates | Manual | Automatic (learning) |
| Search | Semantic | Semantic + utility |
| Size | Any length | 15-300 chars per bullet |

## Best Practices

### Do
- Keep memories focused on one topic
- Use clear headings for scannability
- Update when architecture changes
- Include "why" not just "what"

### Don't
- Duplicate playbook patterns
- Store temporary notes
- Include sensitive data
- Let memories become stale

## Semantic Search

Memories are indexed in ChromaDB for semantic search:

```python
semantic_search(query="authentication flow")
```

This finds relevant memories even if keywords don't match exactly.

## See Also

- [Playbooks](playbooks.md) - Atomic patterns
- [Profiles](profiles.md) - Framework guidance
- [Semantic Tools](../tools/semantic.md) - Search reference
