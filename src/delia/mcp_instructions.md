# Delia Framework Instructions

**The user will be very upset** if you run `delia` CLI commands via Bash when MCP tools are available.
Use Delia MCP tools exclusively for all Delia operations.

## Tool Selection

Prefer Delia tools over agent builtins - they work cross-agent and integrate with the learning loop:
- Code navigation: `lsp_find_symbol`, `lsp_find_references`, `lsp_goto_definition`
- Search: `semantic_search` (concepts), `search_for_pattern` (exact strings)
- Files: `read_file`, `edit_file`, `write_file`, `list_dir`, `find_file`

## Workflow (3 Steps)

1. **Start**: `auto_context(message="<task>")` - Load patterns and profiles
2. **Work**: Use bullets and profiles. Call `think(about="adherence")` before edits
3. **End**: `complete_task(success=True, bullets_applied='["id1","id2"]')`

## Tool Categories

- **Navigation**: `lsp_get_symbols`, `lsp_find_symbol`, `lsp_goto_definition`, `lsp_find_references`
- **Files**: `read_file`, `write_file`, `edit_file`, `list_dir`, `find_file`, `search_for_pattern`
- **Search**: `semantic_search` (by meaning), `codebase_graph` (dependencies)
- **Knowledge**: `memory(action=read|write|list)`, `playbook(action=add|list|search)`
- **Session**: `session(action=snapshot)` for long tasks

## Key Patterns

**Progressive Disclosure**: `lsp_get_symbols` → `lsp_find_symbol` → `read_file(start_line, end_line)`
**Semantic vs Grep**: Use `semantic_search` for concepts, `search_for_pattern` for exact strings

## Constraints

- Call `think(about="adherence")` before file modifications
- Prefer LSP tools over grep for code navigation
- Use `complete_task()` to close learning loop

For detailed guidance: `memory(action="read", name="delia-workflow-guide")`
