# File Tools

Tools for reading, writing, editing, and searching files.

## read_file

Read file contents with optional line range.

```python
read_file(
    path="src/auth.py",        # Required: file path
    start_line=10,             # Optional: start line (1-indexed)
    end_line=50                # Optional: end line
)
```

**Returns**: File contents with line numbers

**Tips**:
- Use line ranges to save context
- Prefer over reading entire files

## write_file

Write content to a file. Creates directories if needed.

```python
write_file(
    path="src/new_module.py",  # Required: file path
    content="...",             # Required: file content
    create_dirs=True           # Optional: create parent dirs (default: True)
)
```

**Returns**: Success confirmation

**Tips**:
- Check if file exists first
- Use `edit_file` for modifications

## edit_file

Replace text in an existing file.

```python
edit_file(
    path="src/auth.py",        # Required: file path
    old_text="def old():",     # Required: text to find
    new_text="def new():"      # Required: replacement text
)
```

**Returns**: Success confirmation with diff

**Tips**:
- `old_text` must match exactly (including whitespace)
- For complex edits, use `lsp_edit`

## list_dir

List directory contents.

```python
list_dir(
    path="src/",               # Required: directory path
    recursive=False,           # Optional: include subdirs
    pattern="*.py"             # Optional: filter by glob
)
```

**Returns**: List of files/directories

**Tips**:
- Use `pattern` to filter results
- Set `recursive=True` for deep listing

## find_file

Find files by glob pattern.

```python
find_file(
    pattern="**/*.py",         # Required: glob pattern
    path="src/"                # Optional: search root
)
```

**Returns**: List of matching file paths

**Tips**:
- Use `**` for recursive matching
- Faster than `list_dir` with recursive

## search_for_pattern

Search file contents with regex (grep-like).

```python
search_for_pattern(
    pattern="def authenticate",  # Required: regex pattern
    path="src/",                 # Optional: search root
    file_pattern="*.py",         # Optional: filter files
    context_lines=2              # Optional: lines before/after match
)
```

**Returns**: Matching lines with file paths and line numbers

**Tips**:
- Use for exact text search
- For concept search, use `semantic_search`

## Comparison

| Tool | Use Case |
|------|----------|
| `read_file` | Read known file |
| `find_file` | Find files by name pattern |
| `search_for_pattern` | Find files by content |
| `semantic_search` | Find files by meaning |

## Examples

### Read a Function

```python
# First find it
lsp_find_symbol(name="authenticate")
# Returns: auth.py, lines 45-80

# Then read it
read_file(path="auth.py", start_line=45, end_line=80)
```

### Create a New Module

```python
write_file(
    path="src/utils/helpers.py",
    content='''"""Helper utilities."""

def format_date(dt):
    return dt.strftime("%Y-%m-%d")
'''
)
```

### Find All TODOs

```python
search_for_pattern(
    pattern="# TODO",
    path="src/",
    file_pattern="*.py"
)
```

## See Also

- [LSP Tools](lsp.md) - Symbol-aware navigation
- [Semantic Tools](semantic.md) - Search by meaning
