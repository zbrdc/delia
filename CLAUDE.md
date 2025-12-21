# Delia LLM Orchestration Expert

**Note**: This file contains the complete Delia expert role profile. The same content exists in `.gemini/instructions.md`, `.github/copilot-instructions.md`, and `DELIA_EXPERT.md` to ensure consistency across all AI coding assistants.

---

## ACE Framework: Delia-Controlled Playbooks

Delia manages project-specific playbooks that provide learned strategies, patterns, and guidance.
Instead of reading static profile files, query Delia for dynamic, feedback-refined guidance.

### Getting Playbook Guidance

**Before starting a task**, query Delia for relevant playbook bullets:

```python
# Get task-specific guidance
get_playbook(task_type="coding")  # or: testing, architecture, debugging, project

# Get project context (tech stack, patterns, conventions)
get_project_context()
```

The playbooks contain strategic bullets with:
- **Learned lessons** from past tasks
- **Project-specific patterns** (detected from codebase analysis via `delia index --summarize`)
- **Utility scores** - bullets that helped more rank higher

### Applying Guidance

Use the bullet content to guide your work. Each bullet has an ID for feedback:

```json
{
  "id": "strat-a1b2c3d4",
  "content": "This project uses async/await patterns. Prefer async def for I/O operations.",
  "utility_score": 0.85
}
```

### Closing the Learning Loop

**After completing a task**, report whether the guidance helped:

```python
# If the bullet helped complete the task successfully
report_feedback(bullet_id="strat-a1b2c3d4", task_type="coding", helpful=True)

# If the bullet was misleading or irrelevant
report_feedback(bullet_id="strat-a1b2c3d4", task_type="coding", helpful=False)
```

This feedback updates bullet utility scores, improving future recommendations.

### Task Type Mapping

| Task Type | Keywords | Playbook |
|-----------|----------|----------|
| coding | implement, add, create, write | `coding` |
| testing | test, pytest, coverage, assert | `testing` |
| architecture | design, ADR, refactor, pattern | `architecture` |
| debugging | error, bug, fix, stack trace | `debugging` |
| project | general project context | `project` |

---

## INLINED CRITICAL RULES (From All Profiles)

These rules are mandatory regardless of which profile is loaded:

### From core.md - Universal Rules
```
ALWAYS:
- Backend Agnostic: Use unified call_llm() interface
- Async-First: async def for I/O, proper await, asyncio.gather()
- Type Safety: Pydantic models, type hints, validate early
- Structured Logging: log.info("event_name", key=value)
- Graceful Degradation: Circuit breakers, automatic failover
- Complete Integration: No placeholders, remove old code
```

### From coding.md - Before Writing Code
```
PRE-IMPLEMENTATION CHECKLIST (Cannot skip):
[] Search codebase for similar patterns (DRY)
[] Check existing utilities before creating new ones
[] Verify function signatures match project style
[] Ensure Pydantic models for all configs
[] Run tests before committing
```

### From git.md - Git Operations
```
BRANCH DECISION:
- Multi-file changes? -> Create branch: feature/, fix/, refactor/
- Single-file trivial fix? -> Commit directly to main
- Experimental? -> Always branch

COMMIT FORMAT:
type(scope): description
Examples: feat(orchestration):, fix(llm):, refactor(routing):

NEVER:
- Force-push to main
- Skip tests before push
- Leave branches hanging
```

### From debugging.md - Fixing Issues
```
DEBUG ORDER:
1. Stack trace -> exact file/line
2. Recent commits -> what changed?
3. Test output -> which test failing?
4. Logs -> structured log events

FIX STRATEGY:
- Small bug (<10 lines): Direct to main
- Complex bug (>10 lines): Create fix/ branch
- ALWAYS add regression test
```

### From architecture.md - Design Decisions
```
ARCHITECTURE RULES:
- ADRs for significant decisions (docs/adr/)
- Singleton pattern for managers (BackendManager, SessionManager)
- MCP-Native: Tools expose capabilities, orchestration coordinates
- Centralized LLM calling via llm.call_llm()
```

---

## Anti-Patterns (NEVER DO)

```
Code:
- Placeholder delegations -> Actually extract and integrate
- Duplicate state (old + new module) -> Remove old, use new
- Direct imports when using DI -> Use container
- TypeScript-style 'any' -> Proper types or Pydantic
- Hardcode provider logic in core -> Use provider registry

Process:
- Commit without running tests -> NEVER
- Force-push to main -> NEVER
- Create abstractions for one-time use -> Keep it simple
- Leave dead code "just in case" -> DELETE it
```

---

## Validation Checklist (Before Marking Complete)

```
[] Playbook bullets queried and applied (get_playbook)
[] Feedback reported for useful bullets (report_feedback)
[] Code passes: uv run pytest
[] No placeholder delegations
[] Old code removed if extracting
[] Type hints on all functions
[] Structured logging for key events
[] Tests updated to match new implementation
```

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

7. **Complete Integration Over Scaffolding**
   - NEVER create placeholder code that delegates back to the original implementation
   - When extracting code to a new module, REMOVE it from the source module
   - Verify old code paths are deleted, not just bypassed
   - A feature is NOT complete until old globals/patterns are eliminated
   - Test that the new integration is actually being used (grep for old patterns)

8. **Tests Follow Implementation, Not Vice Versa**
   - When a feature is refactored, validated, and working, UPDATE THE TESTS to match
   - Old tests that fail due to changed APIs are NOT regressions - they are stale tests
   - NEVER revert working code to satisfy outdated test expectations
   - Fix the test imports, mocks, and assertions to reflect the new architecture
   - Legacy test failures should trigger test updates, not feature rollbacks

## Architecture Patterns

### Multi-Tier Model Routing

Route tasks to appropriate model tiers based on complexity:

```python
# Tier selection based on task type (8 tiers total)
quick_tier:      7B-14B models   → Simple Q&A, summarization
coder_tier:      14B-30B models  → Code generation, review
moe_tier:        30B+ MoE models → Complex reasoning, planning
thinking_tier:   Extended CoT    → Deep analysis, debugging
dispatcher_tier: 270M models     → Fast task routing classification
agentic_tier:    7B-14B models   → Tool-calling loops, self-correction
swe_tier:        32B+ models     → Repo-scale refactoring, migrations
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

### Orchestration Modes (ADR-008 Aligned)

**Core Modes:**
- **NONE** - Direct single-backend execution (default for 90%+ of queries)
- **AGENTIC** - Multi-turn tool calling loop with self-correction
- **DEEP_THINKING** - Extended reasoning with thinking-capable models (primary advanced mode)
- **VOTING** - Confidence-weighted adaptive k-voting (default k=1, escalates on disagreement; also handles comparison via `return_all=True`)
- **TREE_OF_THOUGHTS** - Multi-branch exploration with critic evaluation (opt-in only via `tot=True` flag)

**Pipeline Modes:**
- **CHAIN** - Sequential execution with variable piping between steps
- **WORKFLOW** - DAG-based execution with branching and parallel paths
- **BATCH** - Parallel task execution across backends

**Utility Modes:**
- **STATUS_QUERY** - Fast introspection for health/status questions

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

## Git Workflow

### When to Create Branches

- **Feature branches**: For new features, refactors, or multi-file changes (`feature/add-voting-confidence`, `refactor/extract-tools`)
- **Fix branches**: For bug fixes (`fix/circuit-breaker-timeout`)
- **Stay on main**: For single-file documentation updates, typo fixes, or trivial changes
- **Ask if uncertain**: If scope is unclear, ask before creating a branch

### When to Commit

- **Atomic commits**: Each commit should be a logical unit that compiles and passes tests
- **Commit after validation**: Only commit after verifying the change works (ran tests, checked behavior)
- **Don't batch unrelated changes**: Separate commits for separate concerns
- **Commit message format**: Imperative mood, concise subject, optional body for context

```bash
# Good commit messages
feat: Add confidence-weighted voting per ADR-008
fix: Resolve circuit breaker false positives on timeout
refactor: Extract tool handlers to dedicated module
docs: Update orchestration modes for ADR-008 alignment
test: Update stale tests for new delegation API
```

### When to Push

- **Push after stable checkpoint**: When a feature/fix is complete and tested
- **Push before context switch**: If stepping away or switching tasks
- **Don't push broken code**: Ensure CI would pass before pushing
- **Force push only on feature branches**: Never force push main/master

### Pull Request Guidelines

- **PR from feature branch to main**: Standard workflow for reviewed changes
- **Draft PR for WIP**: Use draft PRs to share progress without requesting review
- **PR description**: Include summary, test plan, and link to relevant ADRs/issues
- **Small PRs preferred**: Break large changes into reviewable chunks

### What NOT to Commit

- **Secrets**: `.env`, API keys, credentials (check `.gitignore`)
- **Generated files**: `__pycache__`, `.pyc`, build artifacts
- **Local config**: `.delia.local.md`, user-specific settings
- **Incomplete refactors**: Don't commit half-extracted code

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

## Integration Verification Checklist

**Before claiming any refactoring or extraction is "complete", verify ALL of these:**

### Code Extraction Checklist

```bash
# 1. Verify old code is REMOVED, not just bypassed
grep -r "OLD_PATTERN\|OLD_GLOBAL" src/delia/  # Should return nothing

# 2. Verify new module is actually USED
grep -r "from.*new_module import\|new_module\." src/delia/  # Should show usage

# 3. Verify no placeholder delegations exist
grep -r "from.*original_module import.*as original" src/delia/  # Red flag!

# 4. Run tests to confirm behavior unchanged
uv run pytest
```

### Integration Anti-Patterns (NEVER DO THESE)

```python
# BAD: Placeholder that delegates back
async def new_function(...):
    from ..old_module import old_function
    return await old_function(...)  # This is NOT extraction!

# BAD: New module exists but old globals still active
# old_module.py still has: LIVE_LOGS = [], _lock = Lock()
# new_module.py has: class LoggingService with same functionality
# Result: Two systems, duplicate state, bugs

# BAD: Container created but services still imported directly
from .config import config  # Direct import
# vs
container = get_container()
config = container.config  # Proper DI
```

### Completion Criteria

A refactoring phase is complete when:
1. **Old code deleted**: The original implementation is removed from the source file
2. **New code integrated**: All call sites use the new module/pattern
3. **No duplicates**: Only ONE way to access the functionality exists
4. **Tests pass**: All existing tests work with the new implementation
5. **Line count reduced**: If goal was to shrink a file, verify with `wc -l`

### Singleton Consistency Rules

When using DI container with singletons:
- If a service is a singleton, it MUST be created in ONE place only
- Either use module-level singleton OR container, never both
- Container should import existing singletons, not create new instances

```python
# GOOD: Container uses existing singleton
from .config import config  # Module singleton
self.config = config  # Reference, not new instance

# BAD: Container creates duplicate
self.config = Config()  # New instance = duplicate state!
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

**ADR-007: Conversation Compaction**
LLM-based session summarization to prevent context overflow in long conversations.

**ADR-008: ACE-Aligned Orchestration Simplification**
Based on 2025 research (ICLR Inference Scaling Laws, ACE Framework, CISC, DAAO):
- NONE is default for 90%+ of queries
- DEEP_THINKING is primary advanced mode
- VOTING uses confidence-weighted adaptive k (not fixed k=3)
- TREE_OF_THOUGHTS is opt-in only (removed from auto-detection)
- COMPARISON mode removed (folded into VOTING with `return_all=True`)

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
