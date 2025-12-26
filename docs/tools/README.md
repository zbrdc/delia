# Tool Reference

Complete reference for all Delia MCP tools.

## Tool Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| [Files](files.md) | 6 | Read, write, edit, search files |
| [LSP](lsp.md) | 13 | Code navigation and refactoring |
| [Framework](framework.md) | 7 | Learning loop and context |
| [Semantic](semantic.md) | 2 | Embeddings-based search |
| [Consolidated](consolidated.md) | 6 | Playbook, project, session, git |

## Quick Reference

### Most Used Tools

```python
# Load context at task start
auto_context(message="implement feature X")

# Find symbols by name
lsp_find_symbol(name="UserService")

# Read specific lines
read_file(path="auth.py", start_line=10, end_line=50)

# Search by meaning
semantic_search(query="authentication logic")

# Edit file
edit_file(path="auth.py", old_text="old code", new_text="new code")

# Record feedback at task end
complete_task(success=True, bullets_applied='["id1"]')
```

### Navigation Pattern

```python
# 1. Overview
lsp_get_symbols(path="module.py")

# 2. Find specific
lsp_find_symbol(name="ClassName")

# 3. Read section
read_file(path="module.py", start_line=45, end_line=80)

# 4. Find usages
lsp_find_references(path="module.py", line=50, character=10)
```

## Tool Profiles

Tools are organized into profiles:

| Profile | Tool Count | Includes |
|---------|------------|----------|
| `light` | ~23 | Files, LSP, Framework, Semantic |
| `standard` | ~31 | Light + Consolidated + Git |
| `full` | ~40+ | Standard + Resources + MCP Management |

Set via environment:

```bash
DELIA_TOOLS=standard uv run delia serve
```

## Common Patterns

### Search Strategy

| Need | Tool |
|------|------|
| Find by name | `lsp_find_symbol(name="...")` |
| Find by meaning | `semantic_search(query="...")` |
| Find exact text | `search_for_pattern(pattern="...")` |
| Find by file pattern | `find_file(pattern="*.py")` |

### Edit Strategy

| Need | Tool |
|------|------|
| Replace text | `edit_file(old_text="...", new_text="...")` |
| Rename symbol | `lsp_edit(action="rename", ...)` |
| Extract function | `lsp_refactor(action="extract", ...)` |
| Write new file | `write_file(path="...", content="...")` |

## See Also

- [Files](files.md) - File operations
- [LSP](lsp.md) - Code navigation
- [Framework](framework.md) - Learning loop
- [Semantic](semantic.md) - Embeddings search
- [Consolidated](consolidated.md) - Management tools
