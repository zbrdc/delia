# Coding Profile

Load this profile for: code generation, reviews, refactoring, implementation tasks.

## Code Style

```python
# Function signatures: Clear types, optional parameters with defaults
async def delegate(
    task: str,
    content: str,
    files: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Docstring with Args and Returns."""
```

## Error Handling

```python
from .errors import BackendError, CircuitBreakerError

try:
    response = await call_llm(backend, prompt, model)
except CircuitBreakerError as e:
    log.warning("circuit_breaker_open", backend=backend.id, error=str(e))
    fallback = get_fallback_backend(backend)
    if fallback:
        response = await call_llm(fallback, prompt, model)
    else:
        raise BackendError(f"All backends unavailable: {e}") from e
```

## Pydantic Models

```python
class BackendConfig(BaseModel):
    id: str
    name: str
    provider: str  # "ollama" | "llamacpp" | "gemini"
    enabled: bool = True
    priority: int = 1
    model_config = ConfigDict(extra="forbid")
```

## Anti-Patterns (NEVER DO)

```python
# BAD: Placeholder delegation
async def new_function(...):
    from ..old_module import old_function
    return await old_function(...)  # NOT extraction!

# BAD: Duplicate state
# old_module.py has: LIVE_LOGS = []
# new_module.py has: class LoggingService  # Same functionality = bug
```

## Critical Files

| File | Purpose |
|------|---------|
| `mcp_server.py` | MCP interface |
| `orchestration/service.py` | Unified pipeline |
| `routing.py` | Model selection |
| `llm.py` | Centralized LLM calling |
| `delegation.py` | Task delegation |
