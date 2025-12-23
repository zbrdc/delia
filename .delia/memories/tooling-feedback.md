# Delia Tooling Feedback

From: Claude (Opus 4.5) after extended usage session
Date: 2024-12-22

## What Works Well

### ACE Workflow
- `auto_context` → work → `complete_task` provides good structure
- Playbook learning has real potential for project-specific patterns
- Memory system elegantly solves context limit problem
- **ACE ceremony is ESSENTIAL** - LLMs cannot remember across sessions. Playbooks ARE external memory. The "ceremony" is the price of consistency.

### Reflection Checkpoints
- `think_about_task_adherence()` - forces pause before modifying code
- `think_about_collected_info()` - ensures sufficient context before acting
- `think_about_completion()` - verification before declaring done
- **These need to be triggered MORE often, not less** - LLMs drift without explicit checkpoints

### LSP Tools
- Semantic code navigation beats grep/find
- `lsp_find_references`, `lsp_goto_definition` are genuinely useful
- Symbol-aware editing (`lsp_replace_symbol_body`) is powerful

### Unified Tools (ADR-009)
- Consolidating to `playbook(action="add")` instead of 7 separate tools is right
- Reduces cognitive load
- Easier to discover what's possible

## Pain Points

### Too Many Tools (~80+)
- Overwhelming to know what exists
- Many tools overlap in purpose
- Discovery is hard

### Redundancy with Native Tools
Why do these exist when Claude already has them?
- `mcp__delia__read_file` vs native `Read`
- `mcp__delia__write_file` vs native `Write`
- `mcp__delia__search_for_pattern` vs native `Grep`
- `mcp__delia__find_file` vs native `Glob`
- `mcp__delia__list_dir` vs native `Bash(ls)`

These add latency and confusion without clear benefit.

### Unclear Boundaries
- What's "Delia" vs "external MCP servers configured alongside Delia"?
- CodeRAG confusion was a symptom of this
- `.mcp.json` in delia repo configures non-delia servers

### Dead Code Creates Confusion
- Modules exist but aren't wired up (personas, sandbox, eval_harness)
- Creates false impression of capabilities
- See: cleanup-backlog.md

## Recommendations

### 1. Fewer, Smarter Tools
Consolidate aggressively. 20 well-designed tools > 80 overlapping ones.

### 2. Don't Reimplement Native Tools
Trust Claude's native Read/Write/Grep/Glob. Only add tools that provide NEW capabilities:
- LSP (semantic navigation) ✓
- Playbooks (learning) ✓
- Memory (persistence) ✓
- File ops (redundant) ✗

### 3. Enforce Reflection Checkpoints More
Current problem: I don't call them enough, not that they exist.
Options:
- Auto-inject reminders in tool responses
- Make them part of the ACE workflow (not optional)
- Trigger based on action type (editing code? must call think_about_task_adherence first)

### 4. Clean Up Dead Code
Remove personas.py, sandbox.py, eval_harness.py, frustration.py.
See cleanup-backlog.md.

### 5. Clarify Project Boundaries
- Delia = the MCP server and its tools
- .mcp.json = user's personal MCP config (shouldn't be in delia repo)
- Document this clearly
