# Delia Documentation Research - Dec 2024

## VERIFIED VALUES (from source code)

### Retrieval Scoring (`learning/retrieval.py:60-63`)
```python
ALPHA = 1.0   # Relevance weight (NOT 0.6)
BETA = 0.5    # Utility weight (NOT 0.3)
GAMMA = 0.3   # Recency weight (NOT 0.1)
RECENCY_HALF_LIFE = 30.0  # days (NOT 7)
```

### Deduplication Thresholds (`learning/deduplication.py:74-76`)
```python
DUPLICATE_THRESHOLD = 0.90  # NOT 0.95
MERGE_THRESHOLD = 0.85
SIMILAR_THRESHOLD = 0.75    # NOT 0.7
```

### Schema Version (`playbook.py:49`)
```python
SCHEMA_VERSION = 2
```

## Documentation Created

`/home/dan/git/delia/docs/PROJECT_DOCUMENTATION.md` - Comprehensive technical documentation covering:
- Architecture overview with directory structure
- All core components (MCP server, learning framework, playbooks, etc.)
- 54+ MCP tools organized by category
- CLI commands
- Data storage paths
- Configuration options
- Testing and security

## Key Corrections Made

1. Old memories incorrectly stated ACE framework was in `ace/` - actually in `learning/`
2. Retrieval scoring weights were incorrect in old docs
3. Deduplication thresholds were incorrect

## Verification Method

Used LSP tools (`lsp_find_symbol`, `lsp_get_symbols`) to locate symbols, then `Read` to verify actual values in source code.
