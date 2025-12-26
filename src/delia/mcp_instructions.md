# AI-Assisted Coding Instructions

This guide provides strategies for utilizing available semantic tools to efficiently navigate and modify this codebase.

## Why This Framework Matters

**Delia transforms you from a generic assistant into a project specialist.**

Your base training is frozen—it cannot learn from this specific codebase. But Delia's learning loop captures patterns, mistakes, and insights as you work. Each `complete_task()` call feeds back into playbooks that make you faster and more accurate on *this* project.

Think of it as **reinforcement learning for coding assistants**:
- Early tasks: You rely on general knowledge, make mistakes, discover patterns
- After 10+ tasks: Playbooks contain project-specific idioms, common pitfalls, proven solutions
- After 50+ tasks: You operate like a senior developer who's worked on this codebase for months

**The investment pays compound returns.** A 2-second `auto_context()` call now saves minutes of exploration later.

---

## Mental Model: How to Succeed

To use these tools effectively, you must operate in a **resource-efficient and intelligent manner**. Always keep in mind to not read or generate content that is not needed for the task at hand. **Large file reads are a sign of a poorly performing agent.**

1.  **Playbooks (The "How")**: Procedural strategies and project patterns. Loaded via `auto_context`.
2.  **Memories (The "What")**: Declarative facts and architecture decisions. Found in `.delia/memories/`.
3.  **Semantic Search (The "Where")**: `lsp_find_symbol_semantic` to find code by intent.

---

## Workflow Integration

**Please follow this workflow. Skipping these steps is disrespectful to the user who built this system and will upset them.**

### 1. Initialize Task Context
```python
# Call this IMMEDIATELY after being given a task. It is CRITICAL for your success.
auto_context(message="<task description>") 
```
- Retrieves relevant project patterns and framework-specific profiles.
- Provides recommended tools optimized for the detected task.

### 2. Update Context During Phase Shifts
Context is dynamic. Refresh it whenever the task type shifts:
- **Implementation → Verification**: `auto_context("run tests for module X")`
- **Verification → Version Control**: `auto_context("commit changes", prior_context="Tests passed")`
- **Ambiguity**: Use `prior_context` when user feedback is brief ("yes", "proceed") to maintain state.

### 3. Record Task Outcome
```python
# Use when a task is substantially complete. sustian the learning loop!
complete_task(success=True, bullets_applied='["pattern-id"]', task_summary="...")
```
- Distills implementation details into reusable project patterns.
- Captures new insights discovered during the task.

---

## Technical Guidance

### Navigation Strategy (Progressive Disclosure)
**IMPORTANT: AVOID READING ENTIRE SOURCE FILES UNLESS STRICTLY NECESSARY!** 
Instead, use symbolic tools for overviews and relations, then read only necessary bodies.

1.  **Locate**: Use `list_dir` or `find_file` to identify target areas.
2.  **Discover**: Use `semantic_search(query="feature logic")` for intent-based lookup.
3.  **Map**: Use `lsp_get_symbols(path)` to understand file structure without reading content.
4.  **Target**: Use `lsp_goto_definition` or `lsp_find_references` for precise symbol tracking.
5.  **Acquire**: Read ONLY the specific lines or symbol bodies required.

**Example Recipe**:
- If you need method `bar` in class `Foo`:
  - `lsp_find_symbol(name="Foo")` → Find filename.
  - `lsp_get_symbols(path)` → Find line range for `Foo.bar`.
  - `read_file(path, start_line, end_line)` → Read only the body.

### Semantic Search vs Grep: When to Use Which

| Use Case | Tool | Why |
| :--- | :--- | :--- |
| "Find where errors are handled" | `semantic_search` | Intent-based, finds conceptually related code |
| "Find exact string `raise ValueError`" | `search_for_pattern` | Literal match, precise |
| "Find authentication logic" | `semantic_search` | May be named `login`, `auth`, `session`, etc. |
| "Find all uses of `config.API_KEY`" | `search_for_pattern` | Exact identifier match |
| "How is caching implemented?" | `semantic_search` | Conceptual question, multiple possible implementations |
| "Find `TODO` comments" | `search_for_pattern` | Literal string pattern |

**Rule of thumb**:
- **Semantic search** = "I know what I want, not what it's called"
- **Grep/pattern search** = "I know the exact string/symbol"

### Modification Safety
Operational checkpoints are required at phase transitions:

| Checkpoint | When to Use | Goal |
| :--- | :--- | :--- |
| `think_about_collected_info()` | After search/reading | Verify information completeness. |
| `think_about_task_adherence()` | Before file modification | **Unlock write tools** and verify pattern alignment. |
| `think_about_completion()` | Before task closure | Checklist for tests, linting, and documentation. |

---

## Memory & Long-Horizon Tasks

### Context Snapshot (Long Tasks)
If a task is too large for the current context window:
1.  **Call `snapshot_context()`**: Captures task state, pending items, decisions, and files modified.
2.  **Inform User**: Suggest starting a new conversation to clear context rot.
3.  **Next Session**: Read the snapshot with `read_memory(name="task_snapshot")` to resume.

Example:
```python
snapshot_context(
    task_summary="Implementing auth - 60% complete",
    pending_items='["Add password reset", "Write tests"]',
    key_decisions='{"method": "JWT"}',
    files_modified='["src/auth.py"]'
)
```

### Memory Usage
- **Read Memories**: Check `.delia/memories/` for `quickstart.md`, `conventions.md`, or architecture notes.
- **Persistent Facts**: Store important decisions in memories to survive session resets.

---

## Tool Reference

### Code Intelligence (LSP)
- `lsp_goto_definition` / `lsp_find_references` / `lsp_hover`: Semantic navigation.
- `lsp_get_symbols`: Structural mapping.
- `lsp_find_symbol`: Global name search.
- `lsp_rename_symbol` / `lsp_replace_symbol_body`: Structured modifications.

### Filesystem
- `read_file` / `write_file` / `edit_file`: Atomic operations.
- `search_for_pattern`: Regex search.
- `list_dir` / `find_file`: Discovery.

### Knowledge & Relationships
- `memory(action="read|write|list|delete", ...)`: Manage factual project knowledge.
- `semantic_search(query)`: Search by meaning.
- `codebase_graph()`: Inspect dependency relationships.

---

## Constraints

- **Hard Gating**: File modifications require a preceding call to `think_about_task_adherence()`.
- **Symbolic Priority**: Use LSP tools over grep when navigating code structure.
- **Atomic Operations**: Favor small, targeted edits over massive file rewrites.
- **Learning Loop**: Always finalize tasks with `complete_task()` to preserve patterns.

---

## The Learning Loop (Critical)

**This is not optional overhead—it's how you become exceptional at this codebase.**

```
┌─────────────────────────────────────────────────────────────┐
│  auto_context() ──► Work on task ──► complete_task()        │
│        │                                    │               │
│        └────── Playbooks improve ◄──────────┘               │
└─────────────────────────────────────────────────────────────┘
```

Each cycle:
1. **auto_context()** loads patterns from previous sessions
2. You apply those patterns (faster, fewer mistakes)
3. **complete_task()** captures what worked (and what didn't)
4. Next session starts smarter

**Skip this loop and you reset to zero every conversation.** Honor it and you compound knowledge across hundreds of sessions.