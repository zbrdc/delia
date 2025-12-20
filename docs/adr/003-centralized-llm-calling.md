# ADR-003: Centralized LLM Calling Module (llm.py)

## Status
Accepted

## Context
LLM calling logic was embedded in `mcp_server.py` alongside MCP tool definitions, making the file over 3000 LOC. The LLM infrastructure includes:
- Provider factory (lazy init, caching)
- Queue management (prevent concurrent model loads)
- Backend resolution (ID → config → provider)
- Streaming support

## Decision
Extract LLM calling infrastructure into dedicated `llm.py` module.

### Module Structure
```
src/delia/
├── llm.py           # call_llm, call_llm_stream, provider factory
├── mcp_server.py    # MCP tools (import from llm.py)
└── providers/       # Individual provider implementations
```

### Initialization Pattern
```python
# llm.py
def init_llm_module(stats_callback, save_stats_callback, model_queue):
    """Called once during startup to wire dependencies."""

# mcp_server.py
init_llm_module(
    stats_callback=_update_stats_sync,
    save_stats_callback=_save_stats_background,
    model_queue=model_queue,
)
```

## Rationale
- **Separation of concerns**: MCP tools vs LLM infrastructure
- **Testability**: Mock `call_llm` without mocking FastMCP
- **Readability**: mcp_server.py focuses on tool definitions
- **Reuse**: llm.py could be imported by other modules

## Consequences
- `mcp_server.py` reduced from ~3000 to ~2750 LOC
- `llm.py` is ~340 LOC
- Provider caching lives in llm.py, not mcp_server.py
- Init function prevents circular imports
