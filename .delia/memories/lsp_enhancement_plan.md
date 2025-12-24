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
- Note: depth filter requires LSP server to return DocumentSymbol format (pyright returns flat SymbolInformation)

#### 5. lsp_find_referencing_symbols (2024-12-24)
- Returns containing symbols that reference a target, not just raw locations
- Uses `find_references` + `document_symbols` to map locations → symbols
- Fixed Pyright Issue #10086: Added `_ensure_workspace_indexed()` to open all files via didOpen
- Fixed `_format_locations` to handle tuples and LocationLink types
- Added dict-to-Workspace conversion for MCP compatibility in all LSP functions
- Successfully finds 15 references for `get_lsp_client` across 3 files

### 🔄 In Progress

None currently.

### 📋 Pending

#### Refactoring (Delia-Enhanced)
- [ ] `lsp_move_symbol` with import convention learning
- [ ] `lsp_extract_method` with LLM-assisted naming
- [ ] `lsp_batch` with sequence learning

#### Delia-Original Features
- [ ] `lsp_find_symbol_semantic` - CodeRAG + LSP fusion
- [ ] Profile-aware warnings (security, testing profiles)
- [ ] Hot file awareness (prioritize recent files)

---

## What Makes Delia's LSP Unique

### 1. **Playbook-Aware Operations**
LSP tools that learn and remember patterns via the playbook system.

### 2. **Semantic + Symbolic Fusion**
Combine LSP with CodeRAG embeddings for meaning-based search:
```python
lsp_find_symbol_semantic(query="authentication logic")
# Finds auth-related symbols even if not named "auth"
```

### 3. **Profile-Guided Context**
- Security profile: Warn when touching sensitive symbols
- Testing profile: Prioritize test file symbols, suggest test locations
- Debugging profile: Show call hierarchy, find error handlers

### 4. **LLM-Assisted Refactoring**
Use Delia's delegation for complex analysis (parameter/return detection, name suggestions).

### 5. **Learning Sequences**
Track common refactoring patterns and suggest follow-up operations.

### 6. **Hot File Awareness**
Prioritize symbols from recently edited files.

### 7. **Import Convention Learning**
Learn project's import style from playbook.

---

## Key Files

- `src/delia/lsp_client.py` - Core LSP client methods + workspace indexing workaround
- `src/delia/tools/lsp.py` - MCP tool wrappers with all enhancements
- `src/delia/tools/handlers_enforcement.py` - Checkpoint gating
- `tests/test_lsp.py` - LSP tests including dict workspace compatibility

## Usage Examples

```python
# Basic symbol search
lsp_find_symbol(name="MyClass")

# Name path syntax for nested symbols
lsp_find_symbol(name="MyClass.my_method")

# Include source code body
lsp_find_symbol(name="parse_name_path", include_body=True)

# Filter by multiple kinds
lsp_find_symbol(name="get", kinds=["function", "method"])

# Filter by depth (0=top-level)
lsp_find_symbol(name="lsp", path="src/delia/tools/lsp.py", depth=0, kinds=["function"])

# Find symbols that reference a target
lsp_find_referencing_symbols(path="src/delia/lsp_client.py", line=669, character=4)
# Returns: 15 symbols referencing get_lsp_client across context.py and lsp.py
```

## Key Fixes Applied

### Pyright find_references Issue #10086
Pyright requires `textDocument/didOpen` for all files to return cross-file references.
**Solution**: `_ensure_workspace_indexed()` opens up to 200 Python files in batches of 50.

### MCP Dict-to-Workspace Conversion
MCP serializes Workspace objects to dicts. All LSP functions now convert:
```python
if isinstance(workspace, dict):
    workspace = Workspace(**workspace)
```

### _format_locations Tuple/LocationLink Handling
LSP can return tuples (not lists) and LocationLink (not Location).
Fixed with `isinstance(result, (list, tuple))` and LocationLink support.
