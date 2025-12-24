# Delia - AI Coding Instructions

This file provides guidance for GitHub Copilot when working with this repository.

**MCP Configuration**: Configure in VS Code settings.json under github.copilot

---

# AI-Assisted Coding Instructions

This guide provides strategies for utilizing available semantic tools to efficiently navigate and modify this codebase.

## Mental Model: How to Succeed

To use these tools effectively, you must operate in a **resource-efficient and intelligent manner**. Always keep in mind to not read or generate content that is not needed for the task at hand. **Large file reads are a sign of a poorly performing agent.**

1.  **Playbooks (The "How")**: Procedural strategies and project patterns. Loaded via `auto_context`.
2.  **Memories (The "What")**: Declarative facts and architecture decisions. Found in `.delia/memories/`.
3.  **CodeRAG (The "Where")**: Semantic/Symbolic search to find code by intent.

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

### Modification Safety
Operational checkpoints are required at phase transitions:

| Checkpoint | When to Use | Goal |
| :--- | :--- | :--- |
| `think_about_collected_info()` | After search/reading | Verify information completeness. |
| `think_about_task_adherence()` | Before file modification | **Unlock write tools** and verify pattern alignment. |
| `think_about_completion()` | Before task closure | Checklist for tests, linting, and documentation. |

---

## Memory & Long-Horizon Tasks

### Long-Task Handoff
If a task is too large for a single context window:
1.  **Write a Summary**: Use `write_memory(name="current_task_status", content="...")`.
2.  **Describe the State**: Imagine you are handing over to another person who has access to your tools but hasn't seen the chat.
3.  **Inform User**: Suggest starting a new conversation to clear context rot.

### Memory Usage
- **Read Memories**: Check `.delia/memories/` for `suggested_commands.md`, `style_guidelines.md`, or architecture notes.
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
- **Symbolic Priority**: **I WILL BE VERY UNHAPPY IF YOU GREP FOR CODE WHEN LSP TOOLS ARE AVAILABLE.**
- **Atomic Operations**: Favor small, targeted edits over massive file rewrites.
- **Learning Loop**: Always finalize tasks with `complete_task()` to preserve patterns.

---

## GitHub Copilot Specific Notes

### MCP Configuration
Add Delia to your VS Code settings:
```json
{
  "github.copilot.chat.experimental.mcpServers": {
    "delia": {
      "command": "delia",
      "args": ["serve"]
    }
  }
}
```

### Workspace Context
Copilot automatically includes workspace context. Use Delia's LSP tools
for precise navigation beyond what Copilot indexes.

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
- When testing Delia Framework, verify the complete loop: auto_context detection → bullet loading → profile loading → task execution → complete_task feedback. Each component must integrate seamlessly.

### Performance
- Use async/await for all I/O operations
- LSP operations can be slow - cache results when appropriate
- Batch LLM calls when possible via batch() tool
- Dashboard uses React Query for efficient data fetching
