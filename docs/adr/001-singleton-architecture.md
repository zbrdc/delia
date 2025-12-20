# ADR-001: Singleton Architecture for Core Services

## Status
Accepted

## Context
Delia is a single-process MCP server. Core services (`backend_manager`, `stats_service`, `model_queue`) need to be shared across all MCP tool handlers and provider implementations.

We considered several patterns:
1. **Dependency injection** - Pass services to each function/class
2. **Service locator** - Central registry that hands out services
3. **Module-level singletons** - Import the same instance everywhere

## Decision
Use module-level singletons for core services.

```python
# backend_manager.py
backend_manager = BackendManager()

# Usage anywhere
from .backend_manager import backend_manager
```

## Rationale
- **Simplicity**: MCP servers are inherently single-process, single-tenant
- **No threading**: asyncio concurrency doesn't require DI complexity
- **Fast imports**: No setup code, services ready at import time
- **Test isolation**: Tests can reset singletons (`routing._router = None`)

## Trade-offs
- **Testing**: Must reset singletons between tests
- **Configuration**: Services initialized with defaults, reconfigured later
- **Multi-tenant**: Not supported (but not a requirement)

## Consequences
- Keep singleton pattern for: `backend_manager`, `stats_service`, `model_queue`, `config`
- Document reset patterns for tests
- If multi-tenant needed later, refactor to context-based injection
