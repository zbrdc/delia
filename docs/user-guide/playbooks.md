# Playbooks

Playbooks store learned patterns as atomic, actionable bullets. They're the core of Delia's learning system.

## What Are Playbooks?

Playbooks are JSON files in `.delia/playbooks/` organized by task type:

```
.delia/playbooks/
├── coding.json      # Code patterns
├── testing.json     # Test patterns
├── debugging.json   # Debug patterns
├── git.json         # Git conventions
├── architecture.json # Design patterns
└── security.json    # Security practices
```

## Bullet Structure

Each bullet is a single, actionable lesson:

```json
{
  "id": "strat-pathlib",
  "content": "Use pathlib.Path over os.path for file operations",
  "section": "code_standards",
  "helpful_count": 19,
  "harmful_count": 0,
  "utility_score": 1.0,
  "source": "learned"
}
```

### Fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (prefix: strat-, anti-, etc.) |
| `content` | The actual guidance (15-300 chars) |
| `section` | Category within task type |
| `helpful_count` | Times marked helpful |
| `harmful_count` | Times marked harmful |
| `utility_score` | helpful / (helpful + harmful) |
| `source` | Origin: seed, learned, manual, reflector |

## How Bullets Are Retrieved

When you call `auto_context()`, Delia:

1. **Embeds** your task description
2. **Searches** ChromaDB for similar bullets
3. **Scores** results: `relevance × utility × recency`
4. **Returns** top-ranked bullets

### Scoring Formula

```
final_score = relevance^1.0 × utility^0.5 × recency^0.3
```

- **Relevance**: Semantic similarity (0-1)
- **Utility**: helpful / (helpful + harmful)
- **Recency**: Time decay (30-day half-life)

## Managing Playbooks

### Add a Bullet

```python
playbook(
    action="add",
    task_type="coding",
    content="Always validate file paths before operations",
    section="security"
)
```

### List Bullets

```python
playbook(action="list", task_type="coding")
```

### Search Bullets

```python
playbook(action="search", query="error handling")
```

### Delete a Bullet

```python
playbook(action="delete", bullet_id="strat-old-pattern")
```

### View Statistics

```python
playbook(action="stats")
```

## Quality Gates

Delia enforces quality on bullets:

- **Length**: 15-300 characters
- **Actionable**: Must be specific guidance
- **No vagueness**: Rejects "be careful" or "consider..."
- **No placeholders**: Rejects TODO, FIXME, etc.
- **Deduplication**: Checks semantic similarity before adding

## How Learning Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ complete_   │ ──▶ │   Curator    │ ──▶ │  Playbook   │
│   task()    │     │ (delta ops)  │     │   updated   │
└─────────────┘     └──────────────┘     └─────────────┘
```

1. You call `complete_task(bullets_applied=[...])`
2. Curator increments `helpful_count` on those bullets
3. Utility scores recalculate
4. Next retrieval ranks them higher

### Delta Operations

The Curator applies atomic changes:

| Operation | Effect |
|-----------|--------|
| ADD | New bullet from reflection |
| BOOST | Increment helpful_count |
| DEMOTE | Increment harmful_count |
| REMOVE | Delete low-utility bullet |
| MERGE | Combine similar bullets |

## Playbook vs Profiles vs Memories

| Type | Format | Purpose | Updates |
|------|--------|---------|---------|
| Playbooks | JSON | Atomic patterns | Auto (learning loop) |
| Profiles | Markdown | Framework guidance | Manual |
| Memories | Markdown | Project knowledge | Manual |

## Manual Editing

You can edit playbooks directly:

```bash
# Edit coding playbook
vim .delia/playbooks/coding.json
```

After editing, reindex:

```python
playbook(action="index")
```

## See Also

- [Workflow](workflow.md) - The learning loop
- [Memories](memories.md) - Persistent knowledge
- [Profiles](profiles.md) - Framework guidance
