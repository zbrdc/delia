# Framework Tools

Core tools for Delia's learning loop.

## auto_context

Load relevant patterns at task start. **Call this first for every task.**

```python
auto_context(
    message="implement user authentication",  # Required: task description
    path="/path/to/project",                  # Optional: project path
    prior_context="...",                      # Optional: previous context
    working_files=["auth.py"],                # Optional: files in focus
    code_snippet="def auth():"                # Optional: relevant code
)
```

**Returns**:
- Detected task type and confidence
- Relevant playbook bullets (with IDs)
- Framework profiles
- Tool recommendations

**Example response**:
```json
{
  "task_type": "coding",
  "confidence": 0.85,
  "bullets": [
    {"id": "strat-pathlib", "content": "Use pathlib.Path..."},
    {"id": "strat-async", "content": "Use async/await..."}
  ],
  "profiles": ["python.md", "fastapi.md"],
  "tools": ["lsp_find_symbol", "edit_file"]
}
```

## complete_task

Record task outcome. **Call this at the end of every task.**

```python
complete_task(
    success=True,                              # Required: did task succeed?
    bullets_applied='["strat-pathlib"]',       # Required: bullet IDs used
    task_summary="Added rate limiting",        # Optional: what was done
    failure_reason="...",                      # Optional: if success=False
    new_insight="Use sliding window...",       # Optional: lesson learned
    path="/path/to/project"                    # Optional: project path
)
```

**Returns**: Confirmation with learning status

**Critical**: Skipping this prevents Delia from learning.

## think

Reflection checkpoints during work.

```python
# Before making changes
think(about="adherence")
# Response: Checks if you're following loaded patterns

# When gathering information
think(about="info")
# Response: What do you still need to understand?

# Before completing
think(about="completion")
# Response: Is the task actually done?
```

**Use**: Insert checkpoints to maintain quality.

## reflect

Analyze execution and extract insights.

```python
reflect(
    task_description="Added rate limiting",
    task_type="coding",
    success=True,
    outcome="Rate limiting works",
    error_trace=None,
    applied_bullets=["strat-middleware"],
    path="/path/to/project"
)
```

**Returns**: Structured insights for playbook updates

## check_status

Get Delia Framework status.

```python
check_status(path="/path/to/project")
```

**Returns**:
- Current project path
- Active session info
- Playbook statistics
- Embedding service status

## snapshot_context

Save task state for long-running work.

```python
snapshot_context(
    task_summary="Implementing OAuth2 flow",
    pending_items=["Add token refresh", "Write tests"],
    key_decisions=["Using JWT", "Redis for sessions"],
    files_modified=["auth.py", "middleware.py"],
    next_steps=["Implement refresh endpoint"],
    path="/path/to/project"
)
```

**Returns**: Snapshot saved to memories

**Use**: Before context loss in long conversations.

## read_initial_instructions

Get the Delia Framework manual.

```python
read_initial_instructions()
```

**Returns**: Full workflow instructions and playbook summary

## Workflow Example

```python
# 1. START
context = auto_context(message="Add rate limiting to API")
# Note the bullet IDs: ["strat-middleware", "strat-redis"]

# 2. CHECKPOINT
think(about="adherence")
# Verify following patterns before making changes

# 3. WORK
# Apply patterns, make changes...

# 4. CHECKPOINT
think(about="completion")
# Verify task is complete

# 5. END
complete_task(
    success=True,
    bullets_applied='["strat-middleware", "strat-redis"]',
    new_insight="Use sliding window for rate limiting"
)
```

## See Also

- [Workflow](../user-guide/workflow.md) - Understanding the learning loop
- [Best Practices](../user-guide/best-practices.md) - Effective usage
