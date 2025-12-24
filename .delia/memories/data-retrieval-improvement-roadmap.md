# Data Retrieval, Search & Manipulation Improvement Roadmap

**Date:** 2024-12-23
**Status:** P0/P1 Items IMPLEMENTED

## Implementation Summary (2024-12-23)

### Completed Implementations

| Tool | File | Lines | Status |
|------|------|-------|--------|
| `semantic_search` | `resources.py` | 76-116 | ✅ MCP exposed |
| `get_related_files` | `resources.py` | 118-154 | ✅ MCP exposed |
| `explain_dependency` | `resources.py` | 156-190 | ✅ MCP exposed |
| `git_log` | `coding.py` | 667-738 | ✅ Implemented |
| `git_blame` | `coding.py` | 741-802 | ✅ Implemented |
| `git_show` | `coding.py` | 805-861 | ✅ Implemented |
| `read_files` | `files.py` | 486-529 | ✅ Bulk read |
| `edit_files` | `files.py` | 532-605 | ✅ Atomic bulk edit |

### MCP Tool Wrappers (handlers.py)

| Tool | Lines | Purpose |
|------|-------|---------|
| `git_log` | 2007-2032 | Wrapper for coding.git_log |
| `git_blame` | 2034-2055 | Wrapper for coding.git_blame |
| `git_show` | 2057-2078 | Wrapper for coding.git_show |
| `read_files` | 2084-2110 | Wrapper with JSON parsing |
| `edit_files` | 2112-2142 | Wrapper with JSON parsing |

---

## New Tool Descriptions

### semantic_search(query, top_k=10, file_pattern=None)
Search codebase semantically using embeddings. Finds files/code by meaning, not just text matching.

### get_related_files(file_path, depth=2)
Get files related to this file via imports/dependencies. Returns files within N hops in the dependency graph.

### explain_dependency(source, target)
Explain why source file depends on target file using the dependency graph.

### git_log(path=".", file=None, n=10, since=None, author=None, oneline=False)
Show git commit history with filtering options.

### git_blame(file, path=".", start_line=None, end_line=None)
Show line-by-line authorship for a file.

### git_show(commit, file=None, path=".", stat=False)
Show commit details and diff.

### read_files(paths: JSON array)
Read multiple files in one call. More efficient than N read_file calls.

### edit_files(edits: JSON array)
Apply multiple edits across files atomically. All edits validated before any applied.

---

## Remaining Work (P2/P3)

### P2 - Medium Priority
- **Type/signature search** - Find functions by return type or parameter types
- **AST pattern search** - Find code by structural patterns

### P3 - Future Work
- Cross-file data flow analysis
- Change impact analysis

---

## Code Locations Reference

| Component | File | Key Functions |
|-----------|------|---------------|
| Semantic search (exposed) | `tools/resources.py` | `semantic_search()` |
| Graph tools (exposed) | `tools/resources.py` | `get_related_files()`, `explain_dependency()` |
| Git history tools | `tools/coding.py` | `git_log()`, `git_blame()`, `git_show()` |
| Bulk file ops | `tools/files.py` | `read_files()`, `edit_files()` |
| MCP wrappers | `tools/handlers.py` | Lines 2003-2142 |
