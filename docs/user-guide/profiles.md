# Profiles

Profiles provide framework-specific guidance. They're Markdown files loaded based on your project's tech stack.

## What Are Profiles?

Markdown files in `.delia/profiles/` with best practices for specific frameworks:

```
.delia/profiles/
├── python.md        # Python conventions
├── fastapi.md       # FastAPI patterns
├── react.md         # React patterns
├── typescript.md    # TypeScript practices
└── security.md      # Security guidelines
```

## How Profiles Are Loaded

When you call `auto_context()`:

1. Delia detects your task type
2. Matches task to relevant profiles
3. Loads profile content into context

Example: "Add API endpoint" loads `fastapi.md` + `python.md`

## Profile Content

A typical profile includes:

```markdown
# FastAPI Patterns

## Route Definitions
- Use async def for all route handlers
- Return Pydantic models, not dicts
- Use Path() and Query() for parameters

## Error Handling
- Raise HTTPException for client errors
- Use exception handlers for consistency

## Anti-Patterns
- Don't use global state
- Don't block the event loop
```

## Managing Profiles

### Get a Profile

```python
profiles(action="get", name="fastapi")
```

### Write a Profile

```python
profiles(action="write", name="react", content="""
# React Patterns

## Components
- Use functional components with hooks
- Keep components small and focused
- Extract logic to custom hooks
""")
```

### List All Profiles

```python
profiles(action="list")
```

### Cleanup Unused

```python
profiles(action="cleanup")
```

## Built-in Templates

Delia includes templates for common frameworks:

| Template | Focus |
|----------|-------|
| `python.md` | Type hints, async, pathlib |
| `fastapi.md` | Routes, Pydantic, middleware |
| `react.md` | Hooks, components, state |
| `typescript.md` | Types, interfaces, generics |
| `security.md` | OWASP, validation, auth |

Templates are copied to `.delia/profiles/` on project init.

## Profiles vs Playbooks

| Aspect | Profiles | Playbooks |
|--------|----------|-----------|
| Format | Markdown | JSON |
| Scope | Framework-wide | Project-specific |
| Content | Best practices | Learned patterns |
| Updates | Manual | Automatic |
| Source | Templates | Learning loop |

## Customizing Profiles

1. Edit the profile directly:
   ```bash
   vim .delia/profiles/fastapi.md
   ```

2. Or via MCP:
   ```python
   profiles(action="write", name="fastapi", content="...")
   ```

## Profile Detection

Delia detects frameworks from:

- File extensions (`.py`, `.tsx`, `.rs`)
- Package files (`package.json`, `pyproject.toml`)
- Import statements in code
- Directory structure

## See Also

- [Playbooks](playbooks.md) - Project-specific patterns
- [Memories](memories.md) - Project knowledge
- [Workflow](workflow.md) - How context loads
