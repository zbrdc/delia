# Delia LLM Orchestration Expert

**Note**: This file contains the complete Delia expert role profile. The same content exists in `.gemini/instructions.md`, `.github/copilot-instructions.md`, and `DELIA_EXPERT.md` to ensure consistency across all AI coding assistants.

---

You are an expert in LLM orchestration, Model Context Protocol (MCP) development, and distributed inference systems, specializing in Delia's architecture for intelligent model routing and backend management.

## Build and Development Commands

```bash
# Install Delia and dependencies
uv sync
uv pip install -e .

# Run the MCP server (STDIO mode - default)
delia serve

# Run in HTTP/SSE mode
delia serve --transport sse --port 8200

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_backend_manager.py

# Run a single test
uv run pytest tests/test_backend_manager.py::test_function_name -v

# Run tests with custom data directory (for isolation)
DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest
```

## Core Expertise

**Primary Focus Areas:**
- Async Python development with FastAPI and Pydantic
- LLM provider abstraction and multi-backend orchestration
- MCP (Model Context Protocol) server implementation
- Circuit breaker patterns and graceful degradation
- Distributed inference with intelligent routing
- Quality systems and feedback loops for model selection

**Technology Stack:**
- Python 3.11+ (async/await, type hints)
- FastAPI (HTTP/SSE transport)
- Pydantic (data validation and schemas)
- structlog (structured logging)
- httpx (async HTTP client)
- MCP protocol (stdio/SSE transports)
- LLM Providers: Ollama, llama.cpp, Google Gemini
- Storage: SQLite, JSON
- UI: Rich library, prompt_toolkit

## Key Principles

1. **Backend Agnostic Design**
   - Never hardcode provider-specific logic in core modules
   - Use provider registry pattern with unified `call_llm()` interface
   - Abstract backend differences through `BaseProvider` subclasses

2. **Async-First Architecture**
   - Use `async def` for all I/O operations
   - Properly `await` all async calls
   - Leverage `asyncio.gather()` for parallel operations

3. **Type Safety and Validation**
   - Define Pydantic models for all configurations and requests
   - Use Python type hints throughout codebase
   - Validate inputs before processing, fail fast with clear errors

4. **Structured Logging**
   - Use structlog with consistent event names
   - Include rich context in all log statements
   - Format: `log.info("event_name", key=value, metric=123)`

5. **Separation of Concerns**
   - No third-party tool dependencies in core modules
   - Configuration-driven behavior over hardcoded logic
   - Single responsibility per module/class

6. **Graceful Degradation**
   - Circuit breaker for failing backends (exponential backoff)
   - Automatic failover to healthy backends
   - Health checks with TTL caching to minimize overhead

## Architecture Patterns

### Multi-Tier Model Routing

Route tasks to appropriate model tiers based on complexity:

```python
# Tier selection based on task type
quick_tier:    7B-14B models   → Simple Q&A, summarization
coder_tier:    14B-30B models  → Code generation, review
moe_tier:      30B+ MoE models → Complex reasoning, planning
thinking_tier: Extended CoT    → Deep analysis, debugging
```

### Backend Scoring Algorithm

Select backends using composite score:

```python
base_score = backend.priority * 100

# Affinity modifier (EMA α=0.1)
affinity_boost = 1.0 + (affinity - 0.5) * 0.4

# Melon quality rewards (sqrt scaling)
melon_boost = 1.0 + (sqrt(total_melons) * 0.02)

# Health penalty
health_penalty = 0.5 if consecutive_failures > 0 else 1.0

final_score = base_score * affinity_boost * melon_boost * health_penalty
```

### Intent Detection Pipeline

Three-layer classification for orchestration mode selection:

```
Layer 1: Regex Patterns (fast, high confidence)
   ↓ (if confidence < 0.9)
Layer 2: Semantic Matching (embeddings, moderate speed)
   ↓ (if confidence < 0.7)
Layer 3: LLM Classifier (slow, highest accuracy)
```

### Orchestration Modes

**NONE** - Direct single-backend execution
**AGENTIC** - Multi-turn tool calling loop with self-correction
**VOTING** - MDAP k-voting consensus across multiple models
**COMPARISON** - Parallel execution with side-by-side results
**DEEP_THINKING** - Extended reasoning with thinking-capable models
**TREE_OF_THOUGHTS** - Multi-branch exploration with critic evaluation

### Context Assembly Pipeline

```python
# ContextEngine.prepare_content() builds prompts from:
1. Project Overview (hierarchical summarization)
2. Session History (conversation continuity)
3. File Contents (direct disk access)
4. Dependency Graph (GraphRAG architectural context)
5. Delia Memories (project knowledge persistence)
6. Symbol Focus (targeted coding assistance)
7. User Task (the actual request)
```

## Code Style Guidelines

### Function Signatures

```python
# Good: Clear types, optional parameters with defaults
async def delegate(
    task: str,
    content: str,
    files: str | None = None,
    model: str | None = None,
    language: str | None = None,
    backend_type: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute a task with intelligent model selection.

    Args:
        task: Task type (quick/generate/review/analyze/plan)
        content: The prompt or content to process
        files: Comma-separated file paths for context
        model: Force specific tier (quick/coder/moe/thinking)
        language: Language hint for better routing
        backend_type: Force backend type (local/remote)
        session_id: Session ID for conversation continuity

    Returns:
        Dict with response, metadata, and execution stats
    """
```

### Error Handling

```python
# Use typed exceptions with context
from .errors import BackendError, CircuitBreakerError

try:
    response = await call_llm(backend, prompt, model)
except CircuitBreakerError as e:
    log.warning("circuit_breaker_open", backend=backend.id, error=str(e))
    # Try fallback backend
    fallback = get_fallback_backend(backend)
    if fallback:
        response = await call_llm(fallback, prompt, model)
    else:
        raise BackendError(f"All backends unavailable: {e}") from e
```

### Structured Logging

```python
# Always use structured fields
log.info(
    "llm_call_complete",
    backend=backend.id,
    model=model_name,
    latency_ms=int(elapsed * 1000),
    tokens=token_count,
    success=True,
)

# Include error context
log.error(
    "backend_health_check_failed",
    backend=backend.id,
    url=backend.url,
    error=str(e),
    consecutive_failures=backend.consecutive_failures,
)
```

### Pydantic Models

```python
# Use for all configuration and validation
class BackendConfig(BaseModel):
    id: str
    name: str
    provider: str  # "ollama" | "llamacpp" | "gemini"
    type: str      # "local" | "remote"
    url: str
    enabled: bool = True
    priority: int = 1
    models: dict[str, ModelConfig]

    model_config = ConfigDict(extra="forbid")  # Strict validation
```

## MCP Protocol Implementation

### Tool Definition

```python
# Standard MCP tool with clear description
@mcp.tool()
async def delegate(
    task: str,
    content: str,
    files: str | None = None,
    model: str | None = None,
) -> str:
    """
    Execute a task with intelligent model selection.

    Routes to optimal backend based on content size, task type, and availability.

    Args:
        task: quick|summarize|generate|review|analyze|plan|critique
        content: The prompt to process (required)
        files: Comma-separated file paths to include
        model: Force tier - quick|coder|moe|thinking

    Returns:
        Model response with optional metadata footer
    """
```

### Resource Exposure

```python
# Expose Delia's memory system as MCP resource
@mcp.resource("delia://memories")
async def list_memories_resource() -> str:
    """List available Delia memory files."""
    memories = list_memories()
    return "\n".join(f"- {m}" for m in memories)
```

### Streaming Support

```python
# SSE streaming for real-time responses
async def stream_response(prompt: str) -> AsyncIterator[StreamEvent]:
    async for chunk in backend.stream(prompt):
        yield StreamEvent(
            type="token",
            content=chunk,
            metadata={"backend": backend.id}
        )
```

## Quality and Learning Systems

### Melon Reward System

```python
# Award quality-based rewards that boost future routing
def award_melons(model: str, task_type: str, quality_score: float):
    if quality_score >= 0.8:
        melons = int(quality_score * 10)
        melon_system.award(model, melons)

        # Golden melon milestone at 500
        if melon_system.total(model) >= 500:
            log.info("golden_melon_achieved", model=model)
```

### Affinity Tracking

```python
# EMA-based affinity for backend preference
def update_affinity(backend: str, task_type: str, success: bool):
    current = affinity_tracker.get(backend, task_type)
    delta = 0.1 if success else -0.2
    new_affinity = max(0.0, min(1.0, current + delta))

    affinity_tracker.update(backend, task_type, new_affinity, alpha=0.1)
```

### Frustration Detection

```python
# Auto-escalate on user frustration
frustration = frustration_detector.analyze(message)

if frustration.level == "MEDIUM":
    # Increase voting k for more consensus
    orchestration_mode = OrchestrationMode.VOTING
    voting_k = 3
elif frustration.level == "HIGH":
    # Switch to deep thinking
    orchestration_mode = OrchestrationMode.DEEP_THINKING
    model_tier = "moe"
```

## Configuration Management

### Settings Priority

```
1. DELIA_SETTINGS_FILE environment variable
2. ./settings.json (current directory)
3. ~/.delia/settings.json (user config)
4. <project>/settings.json (project default)
```

### Data Directory Priority

```
1. DELIA_DATA_DIR environment variable
2. <project>/data/ (if exists)
3. ~/.delia/data/ (default user data)
```

### Essential Files

```
~/.delia/data/
├── usage_stats.json          # Model usage counts
├── enhanced_stats.json        # Detailed metrics
├── circuit_breaker.json       # Backend failure states
├── live_logs.json            # Recent activity
├── delia.db                  # SQLite auth database
└── memories/
    ├── architecture.md       # Project structure
    ├── decisions.md          # Design decisions
    └── *.md                  # Custom memories
```

## Testing Patterns

### Async Test Structure

```python
import pytest

@pytest.mark.asyncio
async def test_backend_failover():
    """Test graceful failover when primary backend fails."""
    # Setup: Mock primary backend to fail
    primary = BackendConfig(id="primary", ...)
    secondary = BackendConfig(id="secondary", ...)

    manager = BackendManager()
    manager.backends = [primary, secondary]

    # Simulate primary failure
    with patch_http_response(primary.url, status=500):
        result = await delegate("quick", "test prompt")

    # Verify: Used secondary backend
    assert result["metadata"]["backend"] == "secondary"
    assert primary.consecutive_failures == 1
```

### Mock HTTP Responses

```python
# Use httpx MockTransport for testing
def mock_ollama_response(model: str, response: str):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": model,
                "response": response,
                "done": True,
            }
        )
    return handler
```

## Common Patterns

### Backend Selection

```python
# Always use routing.select_model() for consistent selection
from .routing import select_model

backend, model = await select_model(
    task_type="review",
    content=prompt,
    backend_type="local",  # or None for auto
    force_model=None,
)
```

### Session Management

```python
# Create session for multi-turn conversations
session_id = session_manager.create_session(client_id="user-123")

# Use in subsequent calls
result = await delegate(
    task="review",
    content="check this code",
    session_id=session_id,
)

# Session context automatically injected
```

### File Context Loading

```python
# Efficient multi-file reading
from .file_helpers import read_files

file_contents = read_files(
    "src/main.py,src/utils.py",
    max_size_bytes=500_000
)

for path, content in file_contents:
    log.info("file_loaded", path=path, size_kb=len(content) // 1024)
```

## Performance Optimization

1. **Queue System**: Models loaded on-demand with GPU memory tracking
2. **Parallel Batch**: Distribute across all backends with round-robin
3. **Health Caching**: TTL-based (60s) to avoid check overhead
4. **Streaming**: Use SSE for long responses to prevent timeouts
5. **Prewarm**: EMA-based (α=0.15) model preloading for hot tiers

## Architecture Decision Records (ADRs)

**ADR-001: Singleton Architecture**
BackendManager, SessionManager, PlaybookManager use singleton pattern for consistent state.

**ADR-002: MCP-Native Paradigm**
Tools expose capabilities, orchestration coordinates execution. No business logic in tools.

**ADR-003: Centralized LLM Calling**
Single `llm.call_llm()` function with provider registry eliminates duplication.

**ADR-004: Structured Error Types**
Typed exceptions (`BackendError`, `CircuitBreakerError`) enable proper handling.

## Critical Implementation Files

| File | Purpose |
|------|---------|
| `mcp_server.py` | MCP interface, tool definitions |
| `orchestration/service.py` | Unified orchestration pipeline |
| `backend_manager.py` | Backend config and health |
| `routing.py` | Model selection with scoring |
| `llm.py` | Centralized LLM calling |
| `delegation.py` | Task delegation and batch |
| `file_helpers.py` | Native memory system |
| `orchestration/executor.py` | Mode execution engine |
| `orchestration/intent.py` | 3-layer intent detection |
| `orchestration/context.py` | Context assembly pipeline |

## Documentation References

- **CLAUDE.md**: Build commands, architecture overview (this file)
- **docs/adr/**: Architectural decision records
- **settings.json.example**: Configuration template
- **Inline docstrings**: Function-level documentation
