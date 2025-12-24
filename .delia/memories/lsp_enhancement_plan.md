# Delia LSP Enhancement Plan

## Status: ✅ COMPLETE

All planned LSP enhancements have been implemented.

## Implemented Features (18 tools)

### Navigation
| Tool | Description |
|------|-------------|
| `lsp_goto_definition` | Jump to symbol definition |
| `lsp_find_references` | Find all usages of symbol |
| `lsp_hover` | Get documentation/type info |
| `lsp_get_symbols` | List symbols in file |
| `lsp_find_symbol` | Search symbols with filters, name paths, depth/kinds |
| `lsp_find_referencing_symbols` | Find symbols that reference target |

### Search
| Tool | Description |
|------|-------------|
| `lsp_find_symbol_semantic` | Semantic + LSP fusion for meaning-based search |
| `lsp_get_hot_files` | List recently modified files |
| `lsp_get_dependencies` | Cross-file dependency visualization |

### Editing
| Tool | Description |
|------|-------------|
| `lsp_rename_symbol` | Rename across codebase |
| `lsp_replace_symbol_body` | Replace symbol code |
| `lsp_insert_before_symbol` | Insert code before symbol |
| `lsp_insert_after_symbol` | Insert code after symbol |
| `lsp_organize_imports` | Remove unused + sort imports (Ruff) |

### Refactoring
| Tool | Description |
|------|-------------|
| `lsp_move_symbol` | Move symbol between files with import cleanup |
| `lsp_extract_method` | Extract code into method with LLM naming |
| `lsp_batch` | Execute multiple LSP operations with undo support |
| `lsp_batch_history` | List recent batch operations |
| `lsp_batch_undo` | Revert a batch operation |

## Key Features

### Import Convention Learning
- `lsp_move_symbol` detects project's import style (relative vs absolute)
- Applies detected convention when updating imports in dependent files

### Import Auto-Organization  
- After moving symbols, automatically removes unused imports from source
- Uses Ruff for reliable F401 (unused import) and I (import sorting) fixes

### Cross-File Dependencies
- `lsp_get_dependencies` shows:
  - Exported symbols from a file
  - What the file imports
  - Who depends on this file
  - Per-symbol reference analysis

### Batch Undo Support
- Modifying operations in `lsp_batch` save file state before execution
- `lsp_batch_history` lists recent batches
- `lsp_batch_undo` restores files to pre-batch state
- Keeps last 10 snapshots in `~/.delia/batch_history/`

### Profile-Aware Warnings
- All editing tools warn when modifying security/auth, testing, or API code
- Integrates with Delia profiles for context-aware guidance

## Key Files

- `src/delia/lsp_client.py` - Core LSP client methods
- `src/delia/tools/lsp.py` - MCP tool wrappers (2400+ lines)
- `src/delia/orchestration/graph.py` - Hot file awareness
- `tests/test_lsp.py` - LSP tests
