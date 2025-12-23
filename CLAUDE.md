# Delia: AI-Assisted Coding Enhancement

## Why Use Delia's Tools?

Delia makes you **faster and better** at coding tasks:

| Tool | Benefit |
|------|---------|
| **auto_context / Playbooks** | Project-specific patterns learned over time. Don't reinvent - apply what works. |
| **Memories** | Persistent knowledge survives context limits. Store architecture decisions, integration details. |
| **LSP/Symbols** | Semantic code navigation. `lsp_find_references()` beats grep. Jump to definitions, rename safely. |
| **CodeRAG** | Semantic code search. Find related code by meaning, not just text matching. |
| **Profiles** | Framework-specific best practices (FastAPI, React, etc.) loaded automatically. |

**Using these tools = better code, fewer mistakes, faster iteration.**

## MANDATORY: ACE Framework Workflow

**You MUST follow this workflow for EVERY task:**

### Before Starting ANY Task (AUTOMATIC)
```
1. Call auto_context(message="<user's request>")  # AUTOMATIC detection - ONE call does everything
   - Detects task type from message (coding/testing/debugging/etc)
   - Returns relevant playbook bullets automatically
   - Returns profile recommendations
   - Returns recommended_tools (LSP, memories, CodeRAG) for your task type
   - NO need to manually pick task_type
2. APPLY the returned bullets AND USE the recommended tools
```

### RE-CALL auto_context Frequently to Stay on Track
**CRITICAL**: Call auto_context more often than you think necessary. Every task phase should load fresh context.

**Call auto_context when ANY of these happen:**
1. **Starting a new file** - Different file = potentially different patterns
2. **Task type shifts** - coding → testing → git → deployment
3. **User gives short response** - "yes", "do it", "sounds good" - use prior_context
4. **After completing a subtask** - About to move on? Refresh context
5. **Before git operations** - ALWAYS get git playbook before commit/push/PR
6. **Switching languages/frameworks** - Python to TypeScript? Refresh

| Trigger | Action |
|---------|--------|
| Starting work on user request | `auto_context(message)` |
| Finished coding, now committing | `auto_context("commit this fix", prior_context="<your last message>")` |
| Finished fixing, now testing | `auto_context("run tests")` |
| User asks about git after coding work | `auto_context(message, prior_context="Would you like me to commit?")` |
| About to push or create PR | `auto_context("push to remote")` |

**Use `prior_context`** when user's response is short/ambiguous:
```python
# User says "yes" after you offered to commit
auto_context(message="yes", prior_context="Would you like me to commit this fix to dev?")
# -> Detects "git" instead of "project"
```

**Rule of thumb**: If you're about to perform an action that wasn't part of your last auto_context call, refresh it first.

### Alternative (Manual) - Only if auto_context unavailable
```
1. Call get_playbook(task_type="coding")  # Manual task type selection
2. Call get_project_context()             # understand project patterns
```

### After Completing ANY Task
```
1. Call complete_task(success=True/False, bullets_applied='["strat-xxx", "strat-yyy"]')
   - Pass ALL bullet IDs you applied (from auto_context's bullet_ids field)
   - Automatically records feedback for all bullets in one call
   - Optionally add new_insight if you learned something valuable
   - This closes the ACE learning loop
```

### Task Types (auto-detected by auto_context)
| Keywords in Message | Detected Type |
|---------------------|---------------|
| implement, add, create, build, write, refactor | `coding` |
| test, pytest, coverage, mock, assert | `testing` |
| bug, error, fix, debug, broken, failing | `debugging` |
| design, architecture, pattern, ADR, plan, think through, approach, trade-off | `architecture` |
| git, commit, branch, merge, PR, push, pull, check in, land, squash, amend, revert | `git` |
| deploy, docker, CI/CD, production, ship | `deployment` |
| security, auth, password, token | `security` |
| how, what, where, explain, project | `project` |

## Delia's Complete Tool Suite

### ACE Framework Tools (Automatic)
- **auto_context(message, path?, prior_context?)** - **PRIMARY TOOL** - Auto-detects task type and returns relevant bullets + profiles + recommended_tools (LSP/memories/CodeRAG). Returns `bullet_ids` list for easy reference. Call at start AND when task shifts.
- **complete_task(success, bullets_applied, task_summary?, new_insight?)** - **CALL WHEN DONE** - Records feedback for all applied bullets in one call. Closes the ACE learning loop.
- **check_ace_status(path?)** - **CALL THIS FIRST** if unsure about project status. Returns whether playbooks exist and what's needed.
- **read_initial_instructions()** - **CRITICAL** for MCP clients that don't show system prompts. Call immediately if you haven't read the ACE manual.
- **get_playbook(task_type, limit?, path?)** - Manual fallback - Returns bullets for specific task type.
- **get_project_context(path?)** - Returns project overview: tech stack, patterns, key directories
- **get_profile(name, path?)** - Load a specific profile by name when auto_context indicates more are available.
- **report_feedback(bullet_id, task_type, helpful)** - Individual bullet feedback (prefer complete_task() instead).

### ACE Workflow Checkpoint Tools (call these to stay on track!)
- **think_about_task_adherence()** - **ALWAYS call BEFORE modifying code**. Prompts you to verify alignment with project patterns.
- **think_about_collected_info()** - **ALWAYS call after searching/reading**. Ensures you have enough information before proceeding.
- **think_about_completion()** - **Call when you think you're done**. Verification checklist before declaring completion.

### Playbook Management Tools
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

- **ACE is MANDATORY** - always call auto_context() before coding
- **Per-project isolation** - use set_project() to switch contexts
- **LSP for code nav** - use LSP tools instead of grep/find for semantic navigation
- **Memories for knowledge** - store persistent insights in memory system
- Delegation is OPTIONAL - only when user requests or backends configured
- **Call complete_task()** to close the ACE learning loop when done
