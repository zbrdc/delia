# ADR-009: AST-Based Code Indexing

## Status
Accepted

## Date
2024-12-20

## Context

The CodeSummarizer was failing to index large files (>50KB) because:

1. Raw content truncation (`content[:2000]`) lost semantic meaning
2. Embedding models (mxbai-embed-large) have ~512 token context limits
3. A 130KB file's first 2000 chars = mostly imports, no actual code

Example failure:
```
File: src/delia/mcp_server.py (130KB)
Error: 'the input length exceeds the context length'
Result: NOT INDEXED
```

## Decision

Use AST (Abstract Syntax Tree) extraction for large files to create semantically rich embeddings.

### Implementation

```python
def _extract_key_sections(self, rel_path: str, content: str) -> str:
    """Extract key sections from large files for better embeddings."""
    if rel_path.endswith('.py'):
        return self._extract_python_signatures(content)
    elif rel_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
        return self._extract_js_signatures(content)
    else:
        return content[:1500]  # Fallback
```

### Python AST Extraction

For Python files, extract:
- Module docstring (first 300 chars)
- Import list (first 10)
- Class definitions with bases
- Method signatures (first 10 per class)
- Function signatures with docstrings

```python
def _extract_python_signatures(self, content: str) -> str:
    import ast
    tree = ast.parse(content)

    # Module docstring
    docstring = ast.get_docstring(tree)

    # Class/function signatures
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            # Extract: class Name(Base): """doc"""
            # Extract methods: def method(args): ...
        elif isinstance(node, ast.FunctionDef):
            # Extract: def name(args): """doc"""
```

### JS/TS Regex Extraction

For JavaScript/TypeScript, use regex patterns:
- `import ... from '...'`
- `class Name extends Base { }`
- `function name(args) { }`
- `const name = () => { }`

### Embedding Pipeline

```python
# In _index_file():
if len(content) > 5000:
    extracted = self._extract_key_sections(rel_path, content)
    embed_content = f"File: {rel_path}\n{extracted[:1500]}"
else:
    embed_content = f"File: {rel_path}\n{content[:1500]}"
```

## Results

Before:
```
src/delia/mcp_server.py (130KB) → FAILED (context length exceeded)
```

After:
```
src/delia/mcp_server.py (130KB) →
  Extracted: 9911 bytes (docstring + 50+ function signatures)
  Embedded: 1024 dimensions
  Status: SUCCESS
```

### Extracted Content Example

```python
"""Delia — Multi-Model LLM Delegation Server

A pure MCP server that intelligently routes tasks to optimal models..."""
# Imports: asyncio, contextvars, json, logging, os, re, threading, time, uuid
def _early_configure_silent_logging():  """Configure structlog..."""
def _schedule_background_task(coro: Any):  """Schedule a fire-and-forget..."""
async def _prewarm_check_loop():  """Background task that periodically..."""
class OrchestrationService():
    def __init__(self, executor, affinity, prewarm, melons): ...
    async def process(self, message, session_id, ...): ...
    async def _process_impl(self, message, ...): ...
```

## Consequences

### Positive
- Large files now index successfully
- Better semantic representation (signatures > raw code)
- Faster indexing (less content to embed)
- More relevant search results

### Negative
- AST parsing adds ~10ms per Python file
- Regex fallback for non-Python is less accurate
- Some implementation details lost (only signatures, not bodies)

### Future Improvements

| Feature | Complexity | Value |
|---------|------------|-------|
| Symbol-level embeddings | Medium | Index individual functions |
| Call graph extraction | Medium | Track function dependencies |
| Docstring-only index | Easy | Dedicated "how do I" search |
| Type annotation index | Hard | Follow types through codebase |

## Files Changed

| File | Change |
|------|--------|
| `orchestration/summarizer.py` | Added `_extract_key_sections()`, `_extract_python_signatures()`, `_extract_js_signatures()`, `_extract_signatures_regex()` |

## Code Statistics

After fix:
```
Files indexed: 317 (was 314)
Files with embeddings: 317 (was 311)
Large files now indexed: mcp_server.py, melons.py, prompts.py
```
