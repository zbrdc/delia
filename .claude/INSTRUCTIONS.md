# Delia: ACE Framework Enforcement

## MANDATORY: ACE Framework Workflow

**You MUST follow this workflow for EVERY coding task:**

### Before Starting ANY Task
```
1. Call set_project(path)                 # set active project (if working on different project)
2. Call get_playbook(task_type="coding")  # or: testing, architecture, debugging
3. Call get_project_context()             # understand project patterns
4. Read and APPLY the returned bullets to your work
```

### After Completing ANY Task
```
1. Call report_feedback(bullet_id="strat-xxx", task_type="coding", helpful=True/False)
   - Report for EACH bullet you applied
   - This improves future recommendations
2. Call confirm_ace_compliance(task_description, bullets_applied, patterns_followed)
   - Validates ACE workflow compliance
```

### Task Type Mapping
| You're Doing | task_type |
|--------------|-----------|
| Writing/editing code | `coding` |
| Writing/running tests | `testing` |
| Design, refactoring, ADRs | `architecture` |
| Fixing bugs, errors | `debugging` |
| General project questions | `project` |

## Delia's Complete Tool Suite

### ACE Framework Tools
- **get_playbook(task_type, limit?, path?)** - Returns strategic bullets learned from project. **Call BEFORE coding.**
- **get_project_context(path?)** - Returns project overview: tech stack, patterns, key directories
- **report_feedback(bullet_id, task_type, helpful)** - Report whether a bullet helped. **Call AFTER completing task.**
- **confirm_ace_compliance(task_description, bullets_applied, patterns_followed)** - Validate ACE workflow compliance
- **playbook_stats(task_type?)** - See bullet effectiveness scores
- **add_playbook_bullet(task_type, content, section?)** - Add strategic bullet to playbook
- **write_playbook(task_type, bullets)** - Write/replace entire playbook
- **delete_playbook_bullet(bullet_id, task_type)** - Remove obsolete bullet
- **prune_stale_bullets(max_age_days?, min_utility?)** - Remove low-utility bullets
- **list_playbooks()** - List all playbooks and bullet counts

### Project Context Tools
- **set_project(path)** - Set active project context. Delia stores per-project data in `<project>/.delia/`
- **recommend_profiles(analyze_gaps?)** - Recommend starter profiles for project tech stack
- **check_reevaluation()** - Check if pattern re-evaluation is needed (LOC/time thresholds)
- **run_reevaluation(force?)** - Re-analyze project for pattern gaps and profile recommendations
- **cleanup_profiles(auto_remove?)** - Remove obsolete profile templates
- **init_project(path, force?, skip_index?, parallel?)** - Initialize ACE framework for new project

### LSP Code Intelligence Tools (Full Language Server Protocol Support)
- **lsp_goto_definition(path, line, character)** - Find definition of symbol
- **lsp_find_references(path, line, character)** - Find all references to symbol
- **lsp_hover(path, line, character)** - Get docs/type info for symbol
- **lsp_get_symbols(path)** - Get all symbols in file (classes, functions, methods)
- **lsp_find_symbol(name, path?, kind?)** - Search symbols by name across codebase
- **lsp_rename_symbol(path, line, character, new_name, apply?)** - Rename symbol everywhere
- **lsp_replace_symbol_body(path, symbol_name, new_body)** - Replace function/class body
- **lsp_insert_before_symbol(path, symbol_name, content)** - Insert code before symbol
- **lsp_insert_after_symbol(path, symbol_name, content)** - Insert code after symbol

Supports: Python (pyright/pylsp), TypeScript, Rust (rust-analyzer), Go (gopls)

### Memory System Tools
- **list_memories(path?)** - List all memory files for project
- **read_memory(name, path?)** - Read memory file content
- **write_memory(name, content, path?, append?)** - Write/update memory file
- **delete_memory(name, path?)** - Delete memory file

Memories are markdown files in `.delia/memories/` for persistent project knowledge (architecture decisions, debugging insights, integration details)

### LLM Delegation Tools
- **delegate(task, content, ...)** - Offload work to configured LLM backends
  - task: quick|generate|review|analyze|plan|critique
- **batch(tasks)** - Parallel execution across all GPUs
- **chain(steps)** - Sequential task execution with output piping
- **workflow(definition)** - DAG workflow with conditional branching
- **think(problem, depth?)** - Extended reasoning with thinking-capable models
- **agent(prompt, ...)** - Autonomous agent with tool use

### Admin Tools
- **health()** - Check status of Delia and all backends
- **models()** - List all configured models across backends
- **switch_backend(backend_id)** - Switch active LLM backend
- **switch_model(tier, model_name)** - Switch model for tier
- **queue_status()** - Get model queue system status
- **mcp_servers(action?, server_id?, ...)** - Manage external MCP servers

### Session Tools
- **session_list()** - List active conversation sessions
- **session_stats(session_id)** - Get session statistics
- **session_compact(session_id, force?)** - Compact session history with LLM summarization
- **session_delete(session_id)** - Delete session

### Advanced Features
- **scan_codebase(path, max_files?, preview_chars?, phase?)** - Incremental codebase scanning
- **analyze_and_index(path, project_summary, coding_bullets, ...)** - Create ACE index from analysis
- **sync_instruction_files(content, path?, force?)** - Sync CLAUDE.md to all AI agent configs
- **read_instruction_files(path?)** - Read existing instruction files
- **write_project_summary(summary, path?)** - Write project summary JSON
- **check_ace_status(path?)** - Check ACE compliance status for recent tasks

## Constraints

- **ACE is MANDATORY** - always query playbook before coding
- **Per-project isolation** - use set_project() to switch contexts
- **LSP for code nav** - use LSP tools instead of grep/find for semantic navigation
- **Memories for knowledge** - store persistent insights in memory system
- Delegation is OPTIONAL - only when user requests or backends configured
- Report feedback to close the learning loop