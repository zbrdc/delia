# Delia LSP Enhancement Plan

## Goal
Upgrade Delia's LSP tooling to be more powerful than Claude's standard tools. Learn from Serena's patterns but build Delia's own implementation with unique features.

## Implementation Progress

### ✅ Completed

#### 1. workspace/symbol Support (2024-12-24)
- Added `workspace_symbol` method to `DeliaLSPClient`
- Added `_format_workspace_symbols` helper
- Updated `lsp_find_symbol_impl` to use workspace/symbol for global searches
- Added 1s warmup delay after LSP server initialization for indexing

#### 2. Name Path Syntax (2024-12-24)
- Added `parse_name_path` helper to parse `Foo.bar.baz` into (leaf, container_path)
- Added `matches_container_path` helper for container filtering
- Supports both Python-style (Foo.bar) and Rust-style (Foo::bar) paths
- `lsp_find_symbol` now accepts name paths like `DeliaLSPClient.get_client`

#### 3. include_body Option (2024-12-24)
- Added `include_body: bool = False` parameter to `lsp_find_symbol`
- Added `find_symbol_end` helper for Python indentation-based end detection
- Updated `read_symbol_body` to auto-detect symbol end when not provided
- Returns full source code of matched symbols inline with results
- Limits to 5 results when include_body=True (50 lines max per symbol)

#### 4. depth and kinds Filters (2024-12-24)
- Added `depth: int | None = None` - filter by symbol nesting level (0=top-level)
- Added `kinds: List[str] | None = None` - filter by multiple kinds at once
- Added `matches_filters` helper that combines all filter checks

#### 5. lsp_find_referencing_symbols (2024-12-24)
- Returns containing symbols that reference a target, not just raw locations
- Uses `find_references` + `document_symbols` to map locations → symbols
- Fixed Pyright Issue #10086: Added `_ensure_workspace_indexed()` to open all files via didOpen
- Fixed `_format_locations` to handle tuples and LocationLink types
- Added dict-to-Workspace conversion for MCP compatibility

#### 6. lsp_find_symbol_semantic (2024-12-24)
- CodeRAG + LSP fusion for meaning-based symbol search
- Uses semantic embeddings to find relevant files, then LSP for precise symbols
- Parameters: `query`, `top_k`, `kinds`, `include_body`, `boost_recent`
- Integrates recency scoring when `boost_recent=True` (70% relevance + 30% recency)

#### 7. Hot File Awareness (2024-12-24)
- Added `get_hot_files()` method to SymbolGraph - returns recently modified files
- Added `get_file_recency_score()` - exponential decay based on modification time
- New MCP tool `lsp_get_hot_files` - lists files modified in last N hours
- `lsp_find_symbol_semantic` boosts recent files in ranking by default

#### 8. Profile-Aware Warnings (2024-12-24)
- Added pattern matching for security, testing, and API files/symbols
- `get_profile_warnings()` - checks paths and symbol names against patterns
- `get_profile_context_for_warnings()` - formats warning messages
- Integrated into `lsp_replace_symbol_body`, `lsp_insert_before_symbol`, `lsp_insert_after_symbol`
- Warns when modifying auth, security, test, or API code

### 🔄 In Progress

None currently.

### 📋 Pending (Advanced Features)

#### Refactoring (Delia-Enhanced)
- [ ] `lsp_move_symbol` with import convention learning
- [ ] `lsp_extract_method` with LLM-assisted naming
- [ ] `lsp_batch` with sequence learning

---

## Key Files

- `src/delia/lsp_client.py` - Core LSP client methods + workspace indexing workaround
- `src/delia/tools/lsp.py` - MCP tool wrappers with all enhancements
- `src/delia/orchestration/graph.py` - Hot file awareness methods
- `tests/test_lsp.py` - LSP tests including dict workspace compatibility

## Usage Examples

```python
# Semantic symbol search (CodeRAG + LSP fusion)
lsp_find_symbol_semantic(query="authentication logic", top_k=10)

# With recency boost disabled
lsp_find_symbol_semantic(query="database connection", boost_recent=False)

# Get hot files
lsp_get_hot_files(limit=10, since_hours=24)

# Editing with profile warnings (automatic)
lsp_replace_symbol_body(path="src/auth.py", symbol_name="login", new_body="...")
# Returns: "Replaced function 'login'...
#          ⚠️ Profile-aware context:
#            • security.md: File path contains 'auth'"
```

## Current Tool List

| Tool | Description |
|------|-------------|
| `lsp_goto_definition` | Jump to symbol definition |
| `lsp_find_references` | Find all usages of symbol |
| `lsp_hover` | Get documentation/type info |
| `lsp_get_symbols` | List symbols in file |
| `lsp_find_symbol` | Search symbols with filters |
| `lsp_find_referencing_symbols` | Find symbols that reference target |
| `lsp_find_symbol_semantic` | Semantic search with LSP |
| `lsp_get_hot_files` | List recently modified files |
| `lsp_rename_symbol` | Rename across codebase |
| `lsp_replace_symbol_body` | Replace symbol code |
| `lsp_insert_before_symbol` | Insert code before symbol |
| `lsp_insert_after_symbol` | Insert code after symbol |
