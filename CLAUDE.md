# Delia - AI Coding Instructions

This file provides guidance for Claude Code when working with this repository.

**MCP Configuration**: Add to .claude/settings.json or use `claude mcp add`

---

# Delia: AI-Assisted Coding Enhancement

You are an AI coding agent with access to Delia's powerful semantic tools. This document is your complete guide to using them effectively.

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

---

## MANDATORY: Delia Framework Workflow

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
2. Call get_project_context()             # Understand project patterns
```

### After Completing ANY Task
```
1. Call complete_task(success=True/False, bullets_applied='["strat-xxx", "strat-yyy"]')
   - Pass ALL bullet IDs you applied (from auto_context's bullet_ids field)
   - Automatically records feedback for all bullets in one call
   - Optionally add new_insight if you learned something valuable
   - This closes the Delia learning loop
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

---

## Semantic Guidance (Work Smarter, Not Harder)

- **Avoid reading entire files** unless absolutely necessary. Large files consume context and inject noise.
- **Progressive Disclosure Strategy**:
  1. **Locate**: Use `list_dir` or `find_file` to find the right area.
  2. **Map**: Use `lsp_get_symbols(path)` to see the structure of a file without reading its content.
  3. **Target**: Use `lsp_goto_definition` or `lsp_find_references` to jump to exactly what matters.
  4. **Acquire**: Read ONLY the specific lines or symbol bodies needed for implementation.
- **Verification**: Use `think_about_collected_info()` after your search phase to confirm you have enough data to act.

## Tool Chaining Pro-Tips

- **The "Overview First" Pattern**: If you're new to a file, don't `read_file`. Call `lsp_get_symbols` first. It gives you line numbers for every class and function so you can target your next move.
- **The "Impact Analysis" Pattern**: Before editing a function, call `lsp_find_references`. This ensures you don't break other parts of the system.
- **The "Pattern Discovery" Pattern**: Call `auto_context` with your current message AND `working_files` to get specific playbook bullets for the code you are actually touching.

---

## Delia Focus Checkpoint Tools (CRITICAL - use frequently!)

These tools are **non-negotiable checkpoints** that keep you on track. Call them MORE often than you think necessary.

| Checkpoint | When to Call | Why It Matters |
|------------|--------------|----------------|
| **think_about_task_adherence()** | **BEFORE every code modification** | Prevents drift from project patterns and user intent |
| **think_about_collected_info()** | **AFTER searching/reading files** | Ensures you have complete information before acting |
| **think_about_completion()** | **BEFORE declaring task done** | Catches missed steps, untested changes, incomplete work |

**These tools are force multipliers** - they prompt self-reflection that catches errors before they happen:

```
# Pattern: Search → Checkpoint → Modify → Checkpoint → Done

1. User asks for a change
2. Search/read relevant files
3. think_about_collected_info()     # "Do I have enough info?"
4. Plan the modification
5. think_about_task_adherence()     # "Am I aligned with project patterns?"
6. Make the code change
7. think_about_completion()         # "Is this truly complete?"
8. complete_task()
```

**The refocus effect**: When context grows large or tasks become complex, these tools re-center your attention on what actually matters. **Use them proactively, not just when uncertain.**

---

## Delia's Complete Tool Suite

### Delia Framework Tools (Automatic)
- **auto_context(message, path?, prior_context?)** - **PRIMARY TOOL** - Auto-detects task type and returns relevant bullets + profiles + recommended_tools (LSP/memories/CodeRAG). Returns `bullet_ids` list for easy reference. Call at start AND when task shifts.
- **complete_task(success, bullets_applied, task_summary?, new_insight?)** - **CALL WHEN DONE** - Records feedback for all applied bullets in one call. Closes the Delia learning loop.
- **check_status(path?)** - **CALL THIS FIRST** if unsure about project status. Returns whether playbooks exist and what's needed.
- **read_initial_instructions()** - **CRITICAL** for MCP clients that don't show system prompts. Call immediately if you haven't read the Delia manual.
- **get_playbook(task_type, limit?, path?)** - Manual fallback - Returns bullets for specific task type.
- **get_project_context(path?)** - Returns project overview: tech stack, patterns, key directories
- **get_profile(name, path?)** - Load a specific profile by name when auto_context indicates more are available.
- **report_feedback(bullet_id, task_type, helpful)** - Individual bullet feedback (prefer complete_task() instead).

### Playbook Management Tools
- **playbook(action, ...)** - Unified playbook management (add, write, delete, prune, list, stats, confirm)
- **add_playbook_bullet(task_type, content, section?)** - Add strategic bullet to playbook
- **prune_stale_bullets(max_age_days?, min_utility?)** - Remove low-utility bullets
- **list_playbooks()** - List all playbooks and bullet counts

### Project Context Tools
- **set_project(path)** - Set active project context. Delia stores per-project data in `<project>/.delia/`
- **recommend_profiles(analyze_gaps?)** - Recommend starter profiles for project tech stack
- **check_reevaluation()** - Check if pattern re-evaluation is needed (LOC/time thresholds)
- **run_reevaluation(force?)** - Re-analyze project for pattern gaps and profile recommendations
- **cleanup_profiles(auto_remove?)** - Remove obsolete profile templates
- **init_project(path, force?, skip_index?, parallel?)** - Initialize Delia Framework for new project

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

### File Operation Tools
- **read_file(path, start_line?, end_line?)** - Read file with line numbers
- **write_file(path, content, create_dirs?)** - Write/create file
- **edit_file(path, old_text, new_text)** - Search and replace in file
- **list_dir(path?, recursive?, pattern?)** - List directory contents
- **find_file(pattern, path?)** - Find files by glob pattern
- **search_for_pattern(pattern, path?, file_pattern?, context_lines?)** - Grep-like search
- **delete_file(path)** - Delete a file
- **create_directory(path)** - Create a directory

### Memory System Tools
- **memory(action, ...)** - Unified memory management (list, read, write, delete)
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
- **admin(action, ...)** - Unified admin management (switch_model, queue_status, mcp_servers)
- **mcp_servers(action?, server_id?, ...)** - Manage external MCP servers

### Session Tools
- **session(action, ...)** - Unified session management (list, stats, compact, delete)

### Git Tools
- **git_log(path?, file?, n?, since?, author?, oneline?)** - Show commit history
- **git_blame(file, path?, start_line?, end_line?)** - Show line-by-line authorship
- **git_show(commit, file?, path?, stat?)** - Show commit details and diff

### Advanced Features
- **project(action, path, ...)** - Unified project management (init, scan, analyze, sync, read_instructions)
- **profiles(action, ...)** - Unified profile management (recommend, check, reevaluate, cleanup)
- **semantic_search(query, top_k?, file_pattern?)** - Search codebase by meaning
- **get_related_files(file_path, depth?)** - Find files related via imports/dependencies
- **explain_dependency(source, target)** - Explain why source depends on target
- **codebase_graph(query?, file_path?, depth?, top_k?)** - Query project dependency graph

---

## Constraints

- **Delia Framework is MANDATORY** - always call auto_context() before coding
- **Focus checkpoints are MANDATORY** - call think_about_* tools at every phase transition
  - `think_about_collected_info()` after searching/reading
  - `think_about_task_adherence()` before modifying code
  - `think_about_completion()` before declaring done
- **Per-project isolation** - use set_project() to switch contexts
- **LSP for code nav** - use LSP tools instead of grep/find for semantic navigation
- **Memories for knowledge** - store persistent insights in memory system
- Delegation is OPTIONAL - only when user requests or backends configured
- **Call complete_task()** to close the Delia learning loop when done


---

## Claude Code Specific Features

### Hooks Support
Claude Code supports hooks that can inject context automatically:
- Pre-tool hooks can add playbook bullets before file operations
- Post-tool hooks can trigger feedback collection

### Task Subagents
When spawning Task subagents, they may not have MCP access. The playbook bullets
embedded below ensure subagents still benefit from Delia guidance.

### Slash Commands
Use `/delia` commands if configured in your Claude Code setup.

---

## Subagent Fallback (No MCP Access)

If running as a subagent without MCP tool access, read `.delia/` files directly:
- `.delia/playbooks/*.json` - Task-specific bullets (coding, testing, etc.)
- `.delia/memories/*.md` - Persistent project knowledge
- `.delia/project_summary.json` - Project overview

The playbook bullets below are auto-embedded for convenience.

---

## PROJECT PLAYBOOK (Auto-embedded)

These are learned strategies from this project. Apply them to relevant tasks.
For the latest bullets, use `auto_context()` or read `.delia/playbooks/*.json` directly.

### Coding
- Use pathlib.Path over os.path for file operations
- MCP tools must return JSON-serializable dicts wrapped in result key
- Always pass project path explicitly - never assume cwd
- Use httpx async client over requests for HTTP calls
- When implementing cross-platform features, verify hook support via web search first - Cursor has hooks (v1.7+), Windsurf has enterprise-only Cascade Hooks, VS Code Copilot and Gemini have no hooks

### Testing
- Use pytest with async support via pytest-asyncio
- Mock external services (Ollama, LSP) in unit tests
- Integration tests go in tests/integration/
- Use fixtures for common setup patterns
- Test MCP tools via their handler functions directly

### Architecture
- MCP server is the primary interface - tools are registered via decorators
- Playbooks store per-project learned patterns in .delia/playbooks/
- LSP integration provides semantic code navigation
- Memories persist knowledge in .delia/memories/ as markdown
- Profiles are starter templates copied to .delia/profiles/

### Debugging
- Check structlog output for detailed traces
- MCP tool errors are returned in error key of result
- LSP issues often stem from language server not running
- Use health() tool to check backend connectivity
- Dashboard at localhost:8765 shows real-time tool usage

### Project
- Primary language: Python 3.11+
- Package manager: uv with pyproject.toml
- MCP server for AI agent integration
- Dashboard: Next.js app in dashboard/
- CLI entry point: delia command via cli.py

### Git
- Commit messages should be descriptive of the change
- Use conventional commits format when possible
- Don't commit .delia/data/ or session files
- Playbooks and profiles are project-specific and should be committed

### Security
- Never log or expose API keys
- Validate all file paths to prevent traversal
- MCP tools run with user permissions - be cautious with file ops
- Settings files may contain sensitive backend URLs

### Deployment
- MCP server runs via stdio for AI agent integration
- REST API available via delia api command
- Dashboard runs separately on port 8765
- Ollama must be running for LLM delegation features

### Api
- REST API uses FastAPI with automatic OpenAPI docs
- All endpoints return JSON responses
- Use proper HTTP status codes for errors
- MCP tools follow Model Context Protocol specification
- When testing ACE Framework, verify the complete loop: auto_context detection → bullet loading → profile loading → task execution → complete_task feedback. Each component must integrate seamlessly.

### Performance
- Use async/await for all I/O operations
- LSP operations can be slow - cache results when appropriate
- Batch LLM calls when possible via batch() tool
- Dashboard uses React Query for efficient data fetching
