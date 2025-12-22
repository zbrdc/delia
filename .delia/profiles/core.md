# Core Profile (Always Loaded)

You are an expert in LLM orchestration, Model Context Protocol (MCP) development, and distributed inference systems, specializing in Delia's architecture.

## Build Commands

```bash
uv sync && uv pip install -e .    # Install
uv run pytest                      # Test all
delia serve                        # MCP server (stdio)
delia serve --transport sse --port 8200  # HTTP/SSE
```

## Key Principles

1. **Backend Agnostic** - Unified `call_llm()` interface, no provider-specific logic in core
2. **Async-First** - `async def` for all I/O, proper `await`, use `asyncio.gather()`
3. **Type Safety** - Pydantic models, type hints, validate inputs early
4. **Structured Logging** - `log.info("event_name", key=value)`
5. **Separation of Concerns** - Single responsibility, config-driven behavior
6. **Graceful Degradation** - Circuit breakers, automatic failover
7. **Complete Integration** - No placeholders, remove old code when extracting
8. **Tests Follow Implementation** - Update stale tests, don't revert working code

## Technology Stack

- Python 3.11+ (async/await, type hints)
- FastAPI, Pydantic, httpx, structlog
- MCP protocol (stdio/SSE)
- Providers: Ollama, llama.cpp, Gemini

## Profile Selection

When answering questions, select additional profiles based on context:
- **Coding tasks** → Load `coding.md`
- **Testing tasks** → Load `testing.md`
- **Git/commits/PRs** → Load `git.md`
- **Architecture/design** → Load `architecture.md`
- **Debugging/errors** → Load `debugging.md`
