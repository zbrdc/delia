# MCP Tool Inventory (Post-ADR-009 Consolidation)

**Updated**: 2024-12-23
**Total**: 54 tools (down from ~57)

## Summary

After ADR-009 consolidation:
- Removed duplicate standalone tools from admin.py
- Kept consolidated action-based tools in consolidated.py
- Fixed admin.py to export module-level functions for consolidated.py imports
- Fixed json import shadowing bug in admin_tool()

## Tool Categories

### ACE Framework (10)
Core workflow tools for the Adaptive Context Enhancement framework.
- `auto_context` - Auto-detect task type, return relevant bullets + profiles
- `check_ace_status` - Check if ACE is initialized for project
- `complete_task` - Report task completion and record bullet feedback
- `get_playbook` - Manual fallback to get playbook by task type
- `get_profile` - Load specific profile by name
- `get_project_context` - Get project overview and patterns
- `read_initial_instructions` - Get ACE manual (for MCP clients without system prompts)
- `reflect` - Trigger ACE Reflector to analyze task execution
- `report_feedback` - Report individual bullet feedback
- `set_project` - Set active project context

### ACE Reflection Checkpoints (3)
Self-check tools to stay on track during tasks.
- `think_about_collected_info` - Verify enough info gathered before proceeding
- `think_about_completion` - Verification checklist before declaring done
- `think_about_task_adherence` - Verify alignment with patterns before code changes

### Consolidated Tools - ADR-009 (6)
Action-based tools that consolidate multiple operations.
- `admin` - Actions: switch_model, queue_status, mcp_servers, cleanup_*
- `memory` - Actions: list, read, write, delete
- `playbook` - Actions: add, write, delete, prune, list, stats, confirm
- `profiles` - Actions: recommend, check, reevaluate, cleanup
- `project` - Actions: init, scan, analyze, sync, read_instructions
- `session` - Actions: list, stats, compact, delete

### LSP Code Intelligence (9)
Language Server Protocol integration for semantic code navigation.
- `lsp_find_references` - Find all references to symbol
- `lsp_find_symbol` - Search symbols by name
- `lsp_get_symbols` - Get all symbols in file
- `lsp_goto_definition` - Jump to symbol definition
- `lsp_hover` - Get documentation/type info
- `lsp_insert_after_symbol` - Insert code after symbol
- `lsp_insert_before_symbol` - Insert code before symbol
- `lsp_rename_symbol` - Rename symbol across codebase
- `lsp_replace_symbol_body` - Replace function/class body

### File Operations (8)
Standalone file tools for agent/delegation use.
- `create_directory` - Create directory
- `delete_file` - Delete file
- `edit_file` - Search and replace in file
- `find_file` - Find files by glob pattern
- `list_dir` - List directory contents
- `read_file` - Read file with line numbers
- `search_for_pattern` - Grep-like pattern search
- `write_file` - Write/create file

### LLM Delegation (6)
Tools for offloading work to configured backends.
- `agent` - Autonomous agent with tool use
- `batch` - Parallel execution across backends
- `chain` - Sequential task execution with piping
- `delegate` - Single task delegation
- `think` - Extended reasoning with thinking models
- `workflow` - DAG workflow with branching

### Admin Standalone (6)
Tools that remain standalone (not consolidated).
- `dashboard_url` - Get dashboard URL
- `get_model_info_tool` - Get model details
- `health` - Check system health
- `mcp_servers` - Manage external MCP servers
- `models` - List all configured models
- `switch_backend` - Switch active LLM backend

### Learning (3)
Detection pattern learning and management.
- `get_learning_stats` - Stats on learned patterns
- `prune_learned_patterns` - Remove ineffective patterns
- `record_detection_feedback` - Teach correct task type detection

### Project Analysis (3)
Project structure and relationship tools.
- `codebase_graph` - Query dependency graph (GraphRAG)
- `project_memories` - List loaded project memories
- `project_overview` - Hierarchical project structure

## Architecture Notes

### admin.py Structure
Functions are defined at module level for importability:
- `init_project()` - Initialize ACE for project
- `scan_codebase()` - Incremental codebase scanning
- `analyze_and_index()` - Create ACE index
- `cleanup_profiles_impl()` - Remove obsolete profiles

MCP tools registered (non-duplicated only):
- health, dashboard_url, models, switch_backend, get_model_info_tool

### consolidated.py Structure
Action-based tools that import from admin.py:
- `project_tool()` uses `init_project`, `scan_codebase`, `analyze_and_index`
- `profiles_tool()` uses `cleanup_profiles_impl`

### handlers.py Structure
Contains `*_impl` internal helper functions - NOT duplicate MCP tools.
Used internally by ACE framework for processing.
