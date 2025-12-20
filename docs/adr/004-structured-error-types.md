# ADR-004: Structured Error Types

## Status
Accepted

## Context
Errors were raised using generic Python exceptions:
- `RuntimeError` for initialization failures
- `ValueError` for validation errors
- `dict` returns with `{"success": False, "error": "message"}`

This made error handling inconsistent and debugging difficult.

## Decision
Create a hierarchy of typed exceptions in `errors.py`.

### Exception Hierarchy
```
DeliaError (base)
├── InitError      # Module not initialized
├── ConfigError    # Invalid configuration
├── BackendError   # Provider/backend failures
├── ValidationError# Input validation failures
└── QueueError     # Model queue errors
```

### Rich Exceptions
Some exceptions carry context:
```python
class BackendError(DeliaError):
    def __init__(self, message, backend_id=None, provider=None):
        super().__init__(message)
        self.backend_id = backend_id
        self.provider = provider

class QueueError(DeliaError):
    def __init__(self, message, model=None, wait_seconds=None):
        ...
```

## Rationale
- **Catch all Delia errors**: `except DeliaError`
- **Specific handling**: `except BackendError`
- **Debugging**: Exception attributes provide context
- **Documentation**: Exception types document failure modes

## Migration
Existing code updated incrementally:
```python
# Before
raise RuntimeError("LLM module not initialized")

# After
from .errors import InitError
raise InitError("LLM module not initialized")
```

## Consequences
- All custom exceptions in one file (`errors.py`)
- Exported from `delia` package for external consumers
- Dict-based errors (`{"error": ...}`) remain for MCP responses
- Gradual migration of existing code
