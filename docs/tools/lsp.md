# LSP Tools

Language Server Protocol tools for semantic code navigation and refactoring.

## Navigation

### lsp_get_symbols

List all symbols in a file.

```python
lsp_get_symbols(path="src/auth.py")
```

**Returns**: List of symbols with names, kinds, and line ranges

**Use**: Get overview of file structure

### lsp_find_symbol

Find symbols by name. Supports path syntax for nested symbols.

```python
# Simple search
lsp_find_symbol(name="UserService")

# Path syntax: class.method
lsp_find_symbol(name="UserService.authenticate")

# With filters
lsp_find_symbol(
    name="authenticate",
    path="src/",              # Search scope
    kind="function",          # Symbol kind
    depth=2,                  # Max nesting depth
    include_body=True         # Include source code
)
```

**Returns**: Matching symbols with locations

### lsp_goto_definition

Find where a symbol is defined.

```python
lsp_goto_definition(
    path="src/auth.py",       # File containing reference
    line=25,                  # Line number (1-indexed)
    character=10              # Column position
)
```

**Returns**: Definition location (file, line, column)

### lsp_find_references

Find all usages of a symbol.

```python
lsp_find_references(
    path="src/auth.py",
    line=25,
    character=10,
    include_symbols=True,     # Include containing symbols
    include_body=False        # Include source code
)
```

**Returns**: List of reference locations

### lsp_hover

Get documentation and type info for a symbol.

```python
lsp_hover(
    path="src/auth.py",
    line=25,
    character=10
)
```

**Returns**: Docstring, type signature, etc.

### lsp_find_referencing_symbols

Find functions/classes that reference a symbol.

```python
lsp_find_referencing_symbols(
    path="src/auth.py",
    line=25,
    character=10,
    kinds=["function", "method"],
    include_body=True
)
```

**Returns**: Containing symbols that use the target

## Semantic Search

### lsp_find_symbol_semantic

Find symbols by natural language query.

```python
lsp_find_symbol_semantic(
    query="user authentication logic",
    top_k=5,                  # Number of results
    kinds=["function", "class"],
    include_body=True,
    boost_recent=True         # Prefer recently modified
)
```

**Returns**: Semantically similar symbols

## Editing

### lsp_edit

Consolidated edit operations.

```python
# Rename symbol
lsp_edit(
    action="rename",
    path="src/auth.py",
    line=25,
    character=10,
    new_name="authenticateUser"
)

# Replace symbol body
lsp_edit(
    action="replace",
    path="src/auth.py",
    symbol="UserService.authenticate",
    new_body="def authenticate(self, user):\n    return True"
)

# Insert before symbol
lsp_edit(
    action="insert_before",
    path="src/auth.py",
    symbol="UserService",
    content="# Authentication service\n"
)

# Insert after symbol
lsp_edit(
    action="insert_after",
    path="src/auth.py",
    symbol="UserService.authenticate",
    content="\ndef logout(self):\n    pass"
)
```

### lsp_refactor

Refactoring operations.

```python
# Organize imports
lsp_refactor(
    action="organize",
    path="src/auth.py"
)

# Move symbol (planned)
lsp_refactor(
    action="move",
    path="src/auth.py",
    symbol="helper_func",
    target="src/utils.py"
)
```

### lsp_batch

Batch operations with undo support.

```python
# Execute batch
lsp_batch(
    action="execute",
    operations=[
        {"type": "rename", "path": "...", "old": "foo", "new": "bar"},
        {"type": "replace", "path": "...", "symbol": "...", "body": "..."}
    ]
)

# View history
lsp_batch(action="history")

# Undo last batch
lsp_batch(action="undo")
```

## Analysis

### lsp_get_hot_files

Recently modified files.

```python
lsp_get_hot_files(
    limit=10,                 # Max results
    since_hours=24            # Time window
)
```

### lsp_get_dependencies

File dependency analysis.

```python
lsp_get_dependencies(
    path="src/auth.py",
    include_symbols=True,     # Show imported symbols
    max_depth=2               # Transitive depth
)
```

**Returns**: Imports, exports, and dependents

### lsp_organize_imports

Clean up Python imports.

```python
lsp_organize_imports(
    path="src/auth.py",
    remove_unused=True,
    sort_imports=True
)
```

## Navigation Pattern

Progressive disclosure workflow:

```python
# 1. Get file overview
lsp_get_symbols(path="auth.py")

# 2. Find specific symbol
lsp_find_symbol(name="authenticate")

# 3. Read the code
read_file(path="auth.py", start_line=45, end_line=80)

# 4. Find all usages
lsp_find_references(path="auth.py", line=50, character=10)

# 5. Understand callers
lsp_find_referencing_symbols(path="auth.py", line=50, character=10)
```

## See Also

- [File Tools](files.md) - Basic file operations
- [Semantic Tools](semantic.md) - Embeddings search
