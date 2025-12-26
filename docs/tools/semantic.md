# Semantic Tools

Tools for embeddings-based search and codebase understanding.

## semantic_search

Find code by meaning using embeddings.

```python
semantic_search(
    query="user authentication logic",  # Required: natural language query
    top_k=5,                             # Optional: number of results
    file_pattern="*.py"                  # Optional: filter files
)
```

**Returns**: Code snippets ranked by semantic similarity

**Example**:
```python
semantic_search(query="error handling patterns")
# Returns code related to error handling, even if it doesn't
# contain the words "error" or "handling"
```

### When to Use

| Scenario | Tool |
|----------|------|
| Find by meaning | `semantic_search` |
| Find exact text | `search_for_pattern` |
| Find by symbol name | `lsp_find_symbol` |

### Tips

- Be descriptive: "user session management" > "sessions"
- Include context: "React component for user profile" > "profile"
- Combine with file pattern for precision

## codebase_graph

Query codebase dependency structure.

```python
# Mode 1: Overview (no args)
codebase_graph()
# Returns: Summary of files, imports, exports

# Mode 2: File info
codebase_graph(file_path="src/auth.py")
# Returns: File's imports, exports, dependencies

# Mode 3: Related files
codebase_graph(
    file_path="src/auth.py",
    depth=2                    # Follow dependencies N levels deep
)
# Returns: Files related to auth.py within 2 hops

# Mode 4: Explain relationship
codebase_graph(
    explain_source="src/auth.py",
    explain_target="src/middleware.py"
)
# Returns: How these files are connected
```

### Use Cases

**Understanding structure**:
```python
# What does this codebase look like?
codebase_graph()
```

**Impact analysis**:
```python
# What depends on auth.py?
codebase_graph(file_path="src/auth.py", depth=2)
```

**Connection discovery**:
```python
# How are these files related?
codebase_graph(
    explain_source="src/api/routes.py",
    explain_target="src/db/models.py"
)
```

## How Embeddings Work

1. **Indexing**: Code is chunked and embedded during project init
2. **Query**: Your search query is embedded
3. **Search**: ChromaDB finds similar embeddings
4. **Rank**: Results sorted by cosine similarity

### Embedding Providers

| Provider | Quality | Speed | Cost |
|----------|---------|-------|------|
| Voyage AI | Highest | Fast | API key |
| Ollama | High | Medium | Local |
| Sentence Transformers | Good | Slow | Local/Free |

Configure in `~/.delia/.env`:
```bash
DELIA_VOYAGE_API_KEY=your-key
```

## Comparison with Other Search

| Tool | Finds | How |
|------|-------|-----|
| `semantic_search` | Code by meaning | Embeddings |
| `search_for_pattern` | Exact text | Regex |
| `lsp_find_symbol` | Symbols by name | AST |
| `lsp_find_symbol_semantic` | Symbols by meaning | Embeddings |

## Examples

### Find Authentication Code

```python
# Semantic approach (recommended)
semantic_search(query="user login and session management")

# Grep approach (fallback)
search_for_pattern(pattern="def login|def authenticate")
```

### Understand Module Purpose

```python
# What's in auth.py?
codebase_graph(file_path="src/auth.py")

# What uses auth.py?
codebase_graph(file_path="src/auth.py", depth=2)
```

### Find Similar Code

```python
# Find code similar to error handling
semantic_search(query="exception handling with retry logic")
```

## See Also

- [LSP Tools](lsp.md) - Symbol-based navigation
- [File Tools](files.md) - Text-based search
- [Configuration](../getting-started/configuration.md) - Embedding setup
