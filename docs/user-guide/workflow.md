# Workflow: The Learning Loop

Delia's power comes from its learning loop. Every task follows three steps that help Delia learn what works for your project.

## The Three Steps

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  auto_context   │ ──▶ │      WORK       │ ──▶ │  complete_task  │
│  Load patterns  │     │  Apply bullets  │     │ Record feedback │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         └───────────── Learning Loop ◀─────────────────┘
```

### Step 1: auto_context

**When**: At the START of every task

```python
auto_context(message="implement user authentication")
```

**What happens**:
1. Analyzes your task description
2. Detects task type (coding, testing, debugging, etc.)
3. Retrieves relevant playbook bullets via semantic search
4. Loads matching framework profiles
5. Returns context to guide your work

**Returns**:
- Detected task type and confidence
- Relevant playbook bullets (scored by utility)
- Framework profiles (e.g., fastapi.md, react.md)
- Recommended tools for this task

### Step 2: Work

Apply the loaded bullets and profiles while doing the actual work.

**During work**:
- Reference the bullets from auto_context
- Use `think(about="adherence")` before file modifications
- Track which bullet IDs you're applying

### Step 3: complete_task

**When**: At the END of every task

```python
complete_task(
    success=True,
    bullets_applied='["strat-pathlib", "strat-httpx"]'
)
```

**What happens**:
1. Records which bullets were helpful
2. Updates utility scores in playbooks
3. Optionally extracts new insights
4. Closes the learning loop

**Critical**: This step teaches Delia what works. Skip it and the system can't learn.

## Why This Works

**Utility scoring**: Each bullet tracks `helpful_count` and `harmful_count`. Over time, good advice rises and bad advice sinks.

**Semantic retrieval**: Bullets are matched by meaning, not just keywords. "implement auth" finds patterns about security, sessions, and user management.

**Compound learning**: Patterns accumulate over months. Your project's playbooks become increasingly valuable.

## Example Session

```python
# User asks: "Add rate limiting to the API"

# Step 1: Load context
auto_context(message="Add rate limiting to the API")
# Returns:
# - Task type: coding (security)
# - Bullets: ["strat-middleware", "strat-redis-cache", "strat-httpx"]
# - Profile: fastapi.md

# Step 2: Work
# Apply the patterns while implementing rate limiting
# Use middleware pattern from bullets
# Follow FastAPI conventions from profile

# Step 3: Record
complete_task(
    success=True,
    bullets_applied='["strat-middleware", "strat-redis-cache"]',
    new_insight="Use sliding window algorithm for rate limiting"
)
```

## Checkpoints: think()

Use `think()` for reflection during work:

```python
# Before making changes
think(about="adherence")  # Am I following the loaded patterns?

# When gathering info
think(about="info")  # What do I still need to understand?

# Before completing
think(about="completion")  # Is the task actually done?
```

## Long-Running Tasks

For tasks spanning multiple messages, use snapshots:

```python
# Save state before context loss
snapshot_context(
    task_summary="Implementing OAuth2 flow",
    pending_items=["Add token refresh", "Write tests"],
    files_modified=["auth.py", "middleware.py"]
)

# Later, resume with
memory(action="read", name="task-snapshot-...")
```

## Common Mistakes

| Mistake | Impact | Fix |
|---------|--------|-----|
| Skip auto_context | No patterns loaded | Always call at task start |
| Skip complete_task | No learning happens | Always call at task end |
| Wrong bullets_applied | Scores incorrect patterns | Track IDs during work |
| Ignore loaded bullets | Repeats past mistakes | Apply the patterns |

## See Also

- [Playbooks](playbooks.md) - How patterns are stored
- [Framework Tools](../tools/framework.md) - Tool reference
- [Best Practices](best-practices.md) - Effective usage
