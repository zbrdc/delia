# Delia Framework: Complete Workflow Guide

This document provides detailed guidance for agents using Delia tools effectively.

## Why This Framework Matters

Delia transforms you from a generic assistant into a project specialist. Your base training is frozen, but Delia's learning loop captures patterns and insights as you work. Each `complete_task()` call feeds back into playbooks that make you more accurate on this project.

**The investment pays compound returns.** A 2-second `auto_context()` call saves minutes of exploration later.

## Mental Model

1. **Playbooks (The "How")**: Procedural strategies and patterns. Loaded via `auto_context`.
2. **Memories (The "What")**: Declarative facts and decisions. Found in `.delia/memories/`.
3. **Semantic Search (The "Where")**: `semantic_search` finds code by intent.

## Navigation Strategy (Progressive Disclosure)

**Avoid reading entire files.** Use symbolic tools for overviews, then read only necessary bodies.

1. **Locate**: `list_dir` or `find_file` to identify target areas
2. **Discover**: `semantic_search(query="feature logic")` for intent-based lookup
3. **Map**: `lsp_get_symbols(path)` to understand file structure without reading content
4. **Target**: `lsp_goto_definition` or `lsp_find_references` for precise symbol tracking
5. **Acquire**: `read_file(path, start_line, end_line)` - read ONLY required lines

**Example Recipe** - To find method `bar` in class `Foo`:
```
lsp_find_symbol(name="Foo") → Find filename
lsp_get_symbols(path) → Find line range for Foo.bar
read_file(path, start_line, end_line) → Read only the body
```

## Semantic Search vs Grep

| Use Case | Tool | Why |
|----------|------|-----|
| "Find where errors are handled" | `semantic_search` | Intent-based |
| "Find exact string `raise ValueError`" | `search_for_pattern` | Literal match |
| "Find authentication logic" | `semantic_search` | May be named `login`, `auth`, etc. |
| "Find all uses of `config.API_KEY`" | `search_for_pattern` | Exact identifier |

**Rule of thumb**:
- **Semantic search** = "I know what I want, not what it's called"
- **Grep/pattern search** = "I know the exact string/symbol"

## Checkpoints

| Checkpoint | When | Purpose |
|------------|------|---------|
| `think(about="info")` | After search/reading | Verify information completeness |
| `think(about="adherence")` | Before file modification | Unlock write tools, verify alignment |
| `think(about="completion")` | Before task closure | Checklist for tests, linting |

## Long-Horizon Tasks

If a task is too large for the context window:

1. **Snapshot**: `session(action="snapshot", task_summary="...", pending_items='["item1"]')`
2. **Inform User**: Suggest starting a new conversation
3. **Next Session**: `memory(action="read", name="task_snapshot")` to resume

## The Learning Loop

```
┌─────────────────────────────────────────────────────────────┐
│  auto_context() ──► Work on task ──► complete_task()        │
│        │                                    │               │
│        └────── Playbooks improve ◄──────────┘               │
└─────────────────────────────────────────────────────────────┘
```

Each cycle:
1. `auto_context()` loads patterns from previous sessions
2. You apply those patterns (faster, fewer mistakes)
3. `complete_task()` captures what worked
4. Next session starts smarter

**Skip this loop and you reset to zero every conversation.**
