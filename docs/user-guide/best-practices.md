# Best Practices

Tips for getting the most out of Delia.

## For AI Assistants

### Always Use the Learning Loop

```python
# START every task
auto_context(message="<describe the task>")

# WORK with loaded patterns
# ...

# END every task
complete_task(success=True, bullets_applied='["id1", "id2"]')
```

**Why**: Skipping `complete_task()` means Delia can't learn. The user will be upset.

### Track Bullet IDs

When `auto_context()` returns bullets, note their IDs. Report which ones you actually used in `complete_task()`.

```python
# auto_context returned: ["strat-pathlib", "strat-httpx", "strat-async"]
# I used pathlib and httpx patterns
complete_task(bullets_applied='["strat-pathlib", "strat-httpx"]')
```

### Use Checkpoints

Before making file changes:

```python
think(about="adherence")  # Am I following loaded patterns?
```

### Prefer Delia Tools

Use Delia tools over built-in agent tools:

| Instead of | Use |
|------------|-----|
| Agent's file read | `read_file()` |
| Agent's grep | `search_for_pattern()` |
| Agent's find | `lsp_find_symbol()` |

**Why**: Delia tools integrate with the learning loop and work cross-agent.

### Progressive Disclosure

Don't read entire files. Navigate progressively:

```python
# 1. Get file symbols
lsp_get_symbols(path="auth.py")

# 2. Find specific symbol
lsp_find_symbol(name="authenticate")

# 3. Read just that function
read_file(path="auth.py", start_line=45, end_line=80)
```

## For Developers

### Initialize Every Project

```python
project(action="init")
```

This creates `.delia/` with:
- Playbooks seeded from templates
- Profiles matched to your tech stack
- ChromaDB index for semantic search

### Configure Embeddings

For best semantic search, use Voyage AI:

```bash
# ~/.delia/.env
DELIA_VOYAGE_API_KEY=your-key-here
```

Fallbacks work but Voyage provides highest quality.

### Commit Playbooks

Playbooks and profiles should be committed:

```bash
git add .delia/playbooks/ .delia/profiles/
git commit -m "Update learned patterns"
```

This shares learning across team members.

### Don't Commit

Never commit:
- `.delia/chroma/` - Regenerated from source
- `.delia/data/` - Session data
- `settings.json` - User-specific config

### Run Health Checks

```bash
uv run delia doctor
```

Checks:
- Embedding service availability
- ChromaDB status
- Playbook integrity
- LSP server status

## Common Patterns

### Search Before Creating

```python
# Before creating a new utility
lsp_find_symbol(name="utils")
search_for_pattern(pattern="def format_date")
```

### Semantic for Concepts

```python
# Find code by meaning
semantic_search(query="user authentication")
```

### Grep for Exact Strings

```python
# Find exact text
search_for_pattern(pattern="API_KEY")
```

### Graph for Dependencies

```python
# Understand file relationships
codebase_graph(file_path="auth.py", depth=2)
```

## Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Skip auto_context | No patterns loaded | Always call first |
| Skip complete_task | No learning | Always call last |
| Read entire files | Wastes context | Use line ranges |
| Ignore bullets | Repeats mistakes | Apply loaded patterns |
| Use agent builtins | No learning integration | Use Delia tools |

## See Also

- [Workflow](workflow.md) - The learning loop
- [Tools Overview](../tools/README.md) - All available tools
- [Troubleshooting](../getting-started/troubleshooting.md) - Common issues
