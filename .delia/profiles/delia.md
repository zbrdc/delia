# Delia Profile: Building MCP Tools & ACE Components

This profile is REQUIRED when working on the Delia codebase.

---

## 1. MCP Tool Implementation

**Every MCP tool MUST follow this exact pattern:**

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

# Tool definition with complete schema
TOOL_DEFINITION = Tool(
    name="my_tool",
    description="Clear description of what this tool does and when to use it.",
    inputSchema={
        "type": "object",
        "properties": {
            "required_param": {
                "type": "string",
                "description": "What this parameter does",
            },
            "optional_param": {
                "type": "integer",
                "default": 10,
                "description": "Optional with sensible default",
            },
        },
        "required": ["required_param"],
    },
)

async def handle_my_tool(
    required_param: str,
    optional_param: int = 10,
    path: str | None = None,  # ALWAYS accept optional path for project context
) -> dict:
    """
    Tool handler implementation.

    Args:
        required_param: What this parameter does
        optional_param: Optional with default
        path: Project path (defaults to cwd if not specified)

    Returns:
        dict with 'result' key containing JSON-serializable data
    """
    # Resolve project path - NEVER assume cwd
    project_path = Path(path) if path else Path.cwd()

    try:
        # Tool logic here
        result = do_something(required_param, optional_param)

        return {
            "result": json.dumps(result, indent=2)
        }

    except SpecificError as e:
        log.warning("my_tool_failed", error=str(e), param=required_param)
        return {
            "error": str(e),
            "suggestion": "Try doing X instead"
        }
```

**CRITICAL Rules:**
1. Tool returns MUST have `result` key with JSON string, or `error` key
2. ALWAYS accept optional `path` parameter for project context
3. NEVER assume current working directory is the project
4. Log all failures with structured context
5. Include helpful `suggestion` in error responses

---

## 2. Playbook Management

**Adding bullets correctly:**

```python
from delia.playbook import get_playbook_manager

async def add_learned_pattern(
    task_type: str,
    content: str,
    section: str = "learned",
    path: str | None = None,
) -> dict:
    """Add a bullet to the playbook with proper structure."""

    # Get manager for specific project
    manager = get_playbook_manager()
    if path:
        manager.set_project(path)

    # CRITICAL: Check for semantic duplicates before adding
    existing = manager.load_playbook(task_type)
    for bullet in existing:
        if is_semantically_similar(bullet.content, content):
            # Update existing bullet instead of adding duplicate
            bullet.helpful_count += 1
            manager.save_playbook(task_type, existing)
            return {"action": "updated_existing", "bullet_id": bullet.id}

    # Add new bullet with full metadata
    bullet = manager.add_bullet(
        task_type=task_type,
        content=content,
        section=section,
        source="learned",
    )

    return {"action": "added", "bullet_id": bullet.id}
```

**Playbook bullet format:**
```json
{
  "id": "strat-abc12345",
  "content": "Use httpx async client for HTTP calls in async contexts",
  "section": "code_standards",
  "helpful_count": 5,
  "harmful_count": 0,
  "created_at": "2025-01-15T10:30:00Z",
  "last_used": "2025-01-20T14:22:00Z",
  "source_task": "coding",
  "source": "learned"
}
```

**Sections for organizing bullets:**
- `code_standards` - Mandatory patterns
- `best_practices` - Recommended approaches
- `anti_patterns` - AVOID: prefixed warnings
- `delia_specific` - Delia codebase patterns
- `learned` - Agent-learned patterns

---

## 3. Auto-Context Detection

**How task type detection works:**

```python
TASK_TYPE_PATTERNS = {
    "coding": ["implement", "add", "create", "build", "write", "refactor"],
    "testing": ["test", "pytest", "coverage", "mock", "assert"],
    "debugging": ["bug", "error", "fix", "debug", "broken", "failing"],
    "git": ["commit", "branch", "merge", "PR", "push", "pull"],
    "architecture": ["design", "architecture", "pattern", "ADR"],
}

def detect_task_type(message: str) -> str:
    """Detect task type from user message."""
    message_lower = message.lower()

    for task_type, keywords in TASK_TYPE_PATTERNS.items():
        if any(kw in message_lower for kw in keywords):
            return task_type

    return "project"  # Default fallback
```

**Using auto_context correctly:**
```python
# At start of task
context = await auto_context(message=user_request)
bullets = context["bullets"]  # Apply these to your work

# When task type shifts (coding → testing)
context = await auto_context(message="run tests")

# When user response is ambiguous ("yes", "do it")
context = await auto_context(
    message="yes",
    prior_context="Would you like me to commit this fix?"
)  # Detects "git" from prior context
```

---

## 4. Memory System

**When to use memories vs playbooks:**

| Use Case | System | Example |
|----------|--------|---------|
| Reusable patterns | Playbook | "Use httpx for HTTP" |
| One-time decisions | Memory | "Chose PostgreSQL over SQLite" |
| Architecture context | Memory | "Auth uses JWT with refresh tokens" |
| Learned best practices | Playbook | "Always validate path params" |

**Writing memories:**
```python
await memory(
    action="write",
    name="architecture-decisions",
    content="""# Architecture Decisions

## Database Choice: PostgreSQL
- Chosen for: JSONB support, full-text search
- Trade-off: More complex than SQLite
- Date: 2025-01-15

## Authentication: JWT with Refresh Tokens
- Access token: 15 min expiry
- Refresh token: 7 day expiry
- Storage: httpOnly cookies
""",
    path="/path/to/project"
)
```

---

## 5. LSP Integration

**Use LSP tools instead of grep for semantic operations:**

```python
# BAD - Text search for symbol
grep("def process_user", path="src/")

# GOOD - Semantic symbol search
await lsp_find_symbol(name="process_user", kind="function")

# BAD - Manual search for references
grep("process_user", path="src/")

# GOOD - Find all references semantically
await lsp_find_references(
    path="src/users/service.py",
    line=42,
    character=4
)

# GOOD - Safe rename across codebase
await lsp_rename_symbol(
    path="src/users/service.py",
    line=42,
    character=4,
    new_name="handle_user",
    apply=True
)
```

---

## 6. Project Path Handling

**CRITICAL: Never assume cwd is the project:**

```python
# BAD - Assumes cwd
def load_config():
    with open(".delia/config.json") as f:
        return json.load(f)

# BAD - Hardcoded home path
def load_config():
    path = Path.home() / ".delia" / "config.json"
    return json.load(path.open())

# GOOD - Explicit project path
def load_config(project_path: Path | None = None) -> dict:
    if project_path is None:
        project_path = Path.cwd()

    config_path = project_path / ".delia" / "config.json"
    if not config_path.exists():
        return {}

    return json.loads(config_path.read_text())
```

**Tool pattern for path handling:**
```python
async def handle_tool(
    path: str | None = None,
    **kwargs
) -> dict:
    # Resolve to Path object
    project_path = Path(path) if path else Path.cwd()

    # Validate it's a directory
    if not project_path.is_dir():
        return {"error": f"Not a directory: {project_path}"}

    # Use project_path for all file operations
    playbook_dir = project_path / ".delia" / "playbooks"
    ...
```

---

## 7. Feedback Loop (ACE Framework)

**Every task MUST close the feedback loop:**

```python
# After applying playbook bullets
applied_bullets = ["strat-abc123", "strat-def456"]

# When task succeeds
for bullet_id in applied_bullets:
    await report_feedback(
        bullet_id=bullet_id,
        task_type="coding",
        helpful=True
    )

# When task fails or bullet was irrelevant
await report_feedback(
    bullet_id="strat-xyz789",
    task_type="coding",
    helpful=False
)
```

---

## Anti-Patterns for Delia Development

```python
# NEVER do these:

# 1. Return non-JSON-serializable data
return {"result": custom_object}  # WRONG
return {"result": json.dumps(custom_object.to_dict())}  # CORRECT

# 2. Ignore project path
manager.set_project(None)  # Uses global, loses isolation

# 3. Skip error context
except Exception:
    return {"error": "Failed"}  # No context!

# 4. Add duplicate bullets
manager.add_bullet(content)  # Check similarity first!

# 5. Forget to log
result = process()  # Add log.info/debug/error

# 6. Use sync file I/O in async context
with open(path) as f:  # Blocks event loop!
    data = json.load(f)
```
