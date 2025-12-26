# Consolidated Tools

Action-based tools that consolidate multiple operations.

## playbook

Unified playbook management.

### Actions

```python
# Add a bullet
playbook(
    action="add",
    task_type="coding",
    content="Always validate input before processing",
    section="security"
)

# List bullets
playbook(action="list", task_type="coding")

# Search bullets
playbook(action="search", query="error handling")

# Delete bullet
playbook(action="delete", bullet_id="strat-old-pattern")

# View statistics
playbook(action="stats")

# Index bullets to ChromaDB
playbook(action="index")

# Record feedback on a bullet
playbook(action="feedback", bullet_id="strat-pathlib", helpful=True)

# Get learning statistics
playbook(action="learning_stats")

# Prune low-utility bullets
playbook(action="prune", min_utility=0.3, min_uses=5)
```

## project

Project management operations.

### Actions

```python
# Initialize project
project(
    action="init",
    path="/path/to/project",  # Optional: defaults to current
    force=False               # Optional: reinitialize
)

# Scan codebase structure
project(action="scan")

# Analyze and index code
project(action="analyze")

# Get project overview
project(action="overview")
```

### init Details

When you run `project(action="init")`:

1. Creates `.delia/` directory structure
2. Detects tech stack (languages, frameworks)
3. Seeds playbooks from templates
4. Copies relevant profiles
5. Indexes code to ChromaDB
6. Creates CLAUDE.md instruction file

## session

Session management.

### Actions

```python
# List sessions
session(action="list")

# Get session statistics
session(action="stats")

# Compact session (reduce size)
session(action="compact", session_id="abc123")

# Delete session
session(action="delete", session_id="abc123")

# Create snapshot
session(action="snapshot", summary="Working on auth")
```

## profiles

Profile management.

### Actions

```python
# Get a profile
profiles(action="get", name="fastapi")

# Write/update a profile
profiles(
    action="write",
    name="react",
    content="# React Patterns\n\n..."
)

# List all profiles
profiles(action="list")

# Cleanup unused profiles
profiles(action="cleanup")
```

## admin

Administrative operations.

### Actions

```python
# Health check
admin(action="health")

# Switch model (for delegation)
admin(action="switch_model", model="qwen2.5:32b")

# View queue status
admin(action="queue_status")

# List available tools
admin(action="tools")

# Framework statistics
admin(action="framework_stats")
```

## git

Git operations.

### Actions

```python
# View commit history
git(
    action="log",
    file="src/auth.py",       # Optional: specific file
    n=10,                     # Optional: number of commits
    since="2024-01-01",       # Optional: date filter
    oneline=True              # Optional: compact format
)

# View file blame
git(
    action="blame",
    file="src/auth.py",
    start_line=10,            # Optional: line range
    end_line=50
)

# Show commit details
git(
    action="show",
    commit="abc123",
    file="src/auth.py",       # Optional: specific file
    stat=True                 # Optional: show stats
)
```

## memory

Memory management (from framework tools).

### Actions

```python
# List memories
memory(action="list")

# Read a memory
memory(action="read", name="architecture")

# Write a memory
memory(
    action="write",
    name="api-design",
    content="# API Design\n\n..."
)

# Delete a memory
memory(action="delete", name="old-notes")

# Search memories
memory(action="search", query="authentication")
```

## Why Consolidated?

Instead of 30+ individual tools, actions reduce cognitive load:

| Old Way | New Way |
|---------|---------|
| `add_bullet()` | `playbook(action="add")` |
| `list_bullets()` | `playbook(action="list")` |
| `search_bullets()` | `playbook(action="search")` |
| `delete_bullet()` | `playbook(action="delete")` |

Same functionality, cleaner interface.

## See Also

- [Framework Tools](framework.md) - Learning loop
- [User Guide: Playbooks](../user-guide/playbooks.md) - Playbook concepts
- [User Guide: Memories](../user-guide/memories.md) - Memory concepts
