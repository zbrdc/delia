# Delia - Technical Documentation

## Overview

Delia is a Python-based MCP (Model Context Protocol) server that provides local LLM delegation with an adaptive learning framework. It routes tasks to appropriate language models, learns project-specific patterns through playbooks, and integrates semantic code intelligence via LSP.

**Version:** 1.0.0
**Python:** 3.11+
**Package Manager:** uv with pyproject.toml
**License:** GPLv3

---

## Architecture

```
src/delia/
├── cli.py                     # CLI entry point (Typer)
├── mcp_server.py             # MCP server - main AI agent interface
├── api.py                     # REST API (FastAPI)
├── config.py                  # Configuration and tracking classes
├── backend_manager.py         # Multi-backend orchestration
├── lsp_client.py              # Language Server Protocol client
│
├── learning/                  # Adaptive Learning Framework
│   ├── reflector.py          # Post-task insight extraction
│   ├── curator.py            # Playbook curation and dedup
│   ├── retrieval.py          # Hybrid scoring for bullets
│   └── deduplication.py      # Semantic similarity detection
│
├── orchestration/             # Task Routing
│   ├── service.py            # Main processing pipeline
│   ├── dispatcher.py         # Task→model routing
│   ├── context.py            # Context preparation
│   ├── intrinsics.py         # Pre-execution checks
│   └── stakes.py             # High-stakes detection
│
├── providers/                 # LLM Backends
│   ├── base.py               # LLMProvider abstract class
│   ├── ollama.py             # Ollama backend
│   ├── llamacpp.py           # llama.cpp direct GGUF
│   └── gemini.py             # Google Gemini
│
├── tools/                     # MCP Tool Implementations
│   ├── handlers.py           # ACE framework tools
│   ├── consolidated.py       # ADR-009 action tools
│   ├── lsp.py                # LSP code intelligence
│   ├── files.py              # File operations
│   └── admin.py              # System administration
│
├── semantic/                  # Semantic Processing
│   ├── cache.py              # Embedding cache
│   ├── coherence.py          # Response coherence
│   └── compression.py        # Context compression
│
├── playbook.py               # Playbook CRUD and validation
├── context_detector.py       # Task type detection
├── project_memory.py         # .delia/memories/ management
├── profiles.py               # Profile recommendation
│
├── templates/                 # Built-in Templates
│   ├── playbooks/            # Default playbook bullets
│   └── profiles/             # 33 profile templates
│
└── dashboard/                 # Next.js monitoring dashboard
```

---

## Core Components

### 1. MCP Server (`mcp_server.py`)

The primary interface for AI coding agents (Claude Code, Cursor, etc.).

**Key Features:**
- Dynamic instruction building from playbooks
- Tool registration via fastmcp decorators
- Multi-transport support (stdio, SSE)

**Tool Count:** ~54 tools across categories:
- ACE Framework: 10 tools
- Consolidated (ADR-009): 6 action-based tools
- LSP: 15+ functions
- File operations: 8 tools
- LLM delegation: 6 tools

### 2. Learning Framework (`learning/`)

The adaptive learning system that captures project-specific patterns.

| Component | Class | Purpose |
|-----------|-------|---------|
| `reflector.py` | `Reflector` | Post-task insight extraction via LLM |
| `curator.py` | `Curator` | Playbook curation, dedup, feedback recording |
| `retrieval.py` | `HybridRetriever` | Weighted scoring for bullet retrieval |
| `deduplication.py` | `SemanticDeduplicator` | Cosine similarity for duplicate detection |

**Reflector Insight Types:**
```python
class InsightType(str, Enum):
    STRATEGY = "strategy"
    ANTI_PATTERN = "anti_pattern"
    FAILURE_MODE = "failure_mode"
    TOOL_USAGE = "tool_usage"
    CONTEXT_HINT = "context_hint"
```

**Retrieval Scoring Formula:**
```python
# Formula: score = relevance^α × utility^β × recency^γ
ALPHA = 1.0   # Relevance weight
BETA = 0.5    # Utility (helpful/harmful) weight
GAMMA = 0.3   # Recency weight
RECENCY_HALF_LIFE = 30.0  # Days until 50% decay
```

**Deduplication Thresholds:**
```python
DUPLICATE_THRESHOLD = 0.90  # Reject as duplicate
MERGE_THRESHOLD = 0.85      # Consider merging
SIMILAR_THRESHOLD = 0.75    # Flag as related
```

### 3. Playbook System (`playbook.py`)

Stores and manages learned patterns per task type.

**PlaybookBullet Model:**
```python
class PlaybookBullet(BaseModel):
    content: str                    # The guidance (10-500 chars)
    id: str                         # strat-{hash}
    section: str                    # general_strategies, code_standards, etc.
    helpful_count: int              # Positive feedback
    harmful_count: int              # Negative feedback
    created_at: str                 # ISO timestamp
    last_used: str | None           # Last retrieval
    source_task: str | None         # Task that generated this
    source: Literal["seed", "learned", "manual", "reflector", "curator"]

    @computed_field
    def utility_score(self) -> float:
        total = helpful_count + harmful_count
        return helpful_count / total if total > 0 else 0.5
```

**PlaybookManager Methods:**
- `load_playbook(task_type)` - Load bullets for task type
- `save_playbook(task_type, bullets)` - Persist bullets
- `add_bullet(task_type, content, section)` - Add with validation
- `record_feedback(bullet_id, task_type, helpful)` - Update counts
- `get_top_bullets(task_type, limit)` - Sorted by utility
- `prune_low_utility(task_type, threshold)` - Cleanup stale bullets

**BulletQualityValidator Rules:**
- Length: 10-500 characters
- Must contain action patterns (use, implement, always, never, etc.)
- No placeholders or vague content
- Not excessively repetitive

### 4. Context Detection (`context_detector.py`)

Detects task type from user messages.

**Task Types:**
```python
TaskType = Literal[
    "coding", "testing", "debugging", "architecture",
    "git", "deployment", "security", "project",
    "api", "performance"
]
```

**Pattern Learning:**
```python
class PatternLearner:
    def add_pattern(pattern, task_type, weight)
    def record_feedback(message, detected_task, correct_task, was_correct)
    def get_boosted_patterns(task_type) -> list[tuple[Pattern, float]]
    def prune_ineffective(min_effectiveness, min_uses) -> int
```

Supports negative patterns to penalize incorrect detections.

### 5. LLM Providers (`providers/`)

Abstract provider interface for multiple backends.

**Base Classes:**
```python
class LLMResponse(BaseModel):
    success: bool
    response: str
    tokens: int
    elapsed_ms: float
    error: str | None
    circuit_breaker: bool

class LLMProvider(ABC):
    async def call(model, prompt, system, ...) -> LLMResponse
    async def call_stream(model, prompt, ...) -> AsyncIterator[StreamChunk]
    async def load_model(model, backend_obj) -> ModelLoadResult
    async def unload_model(model, backend_obj) -> ModelLoadResult
```

**Available Providers:**

| Provider | File | Features |
|----------|------|----------|
| Ollama | `ollama.py` | Chat/generate, streaming, thinking extraction |
| llama.cpp | `llamacpp.py` | Direct GGUF, grammar support |
| Gemini | `gemini.py` | Google Generative AI |

### 6. Backend Manager (`backend_manager.py`)

Orchestrates multiple LLM backends.

**BackendConfig Model:**
```python
class BackendConfig(BaseModel):
    id: str                    # "ollama-local"
    name: str                  # "Ollama Local"
    provider: str              # "ollama"
    type: str                  # "local" | "remote"
    url: str                   # http://localhost:11434
    enabled: bool
    priority: int
    models: dict[str, str]     # tier → model mapping
    health_endpoint: str
    context_limit: int
    timeout_seconds: int
    supports_native_tool_calling: bool
```

**BackendManager Methods:**
- `_detect_available_backends()` - Auto-discovery
- `check_all_health(use_cache)` - Health monitoring
- `get_enabled_backends()` - Active backends
- `add_backend()`, `update_backend()`, `remove_backend()`
- `get_mcp_servers()` - External MCP server management

### 7. Orchestration Service (`orchestration/service.py`)

Main processing pipeline for task execution.

**OrchestrationService Flow:**
1. Detect intent from message
2. Check for repeats (frustration detection)
3. Auto-compact session if needed
4. Prepare context with files/symbols
5. Run intrinsics checks (answerability)
6. Route to appropriate model
7. Execute and collect result
8. Update affinity/prewarm trackers
9. Award melons (gamification)

**ProcessingResult:**
```python
class ProcessingResult(BaseModel):
    result: str
    intent: str
    quality_score: float
    melons_awarded: int
    affinity_updated: bool
    prewarm_updated: bool
    frustration_penalty: float
    elapsed_ms: float
```

### 8. Config System (`config.py`)

Configuration and tracking classes.

**Model Tiers:**
```python
model_quick: str      # Fast, small tasks
model_coder: str      # Code generation
model_moe: str        # Mixture of experts
model_critic: str     # Review/critique
model_thinking: str   # Extended reasoning
model_agentic: str    # Tool-using agent
model_swe: str        # Software engineering
```

**Tracking Classes:**

| Class | Purpose |
|-------|---------|
| `BackendHealth` | Circuit breaker with failure tracking |
| `BackendMetrics` | Latency/success rate per backend |
| `AffinityTracker` | Backend×task affinity scores |
| `ModelAffinityTracker` | Model selection optimization |
| `PrewarmTracker` | Hour-based tier prediction |

### 9. LSP Client (`lsp_client.py`)

Multi-language Language Server Protocol integration.

**DeliaLSPClient Methods:**
```python
class DeliaLSPClient:
    def get_client(language_id) -> pylsp.Client
    async def goto_definition(file_path, line, character)
    async def find_references(file_path, line, character)
    async def hover(file_path, line, character)
    async def document_symbols(file_path)
    async def workspace_symbol(query, language_id)
    async def rename_symbol(file_path, line, character, new_name)
```

**Supported Languages:**
- Python (pyright)
- TypeScript
- Go
- Rust

---

## MCP Tools

### ACE Framework Tools (`tools/handlers.py`)

| Tool | Description |
|------|-------------|
| `auto_context` | Auto-detect task type, return relevant bullets + profiles |
| `complete_task` | Report completion and record bullet feedback |
| `reflect` | Trigger Reflector for insight extraction |
| `get_profile` | Load specific profile by name |
| `check_status` | Check ACE initialization status |
| `read_initial_instructions` | Get ACE manual for MCP clients |
| `snapshot_context` | Save task state for continuation |
| `record_detection_feedback` | Teach correct task type detection |
| `get_learning_stats` | Stats on learned patterns |
| `prune_learned_patterns` | Remove ineffective patterns |

### Reflection Checkpoints

| Tool | When to Use |
|------|-------------|
| `think_about_collected_info` | After search/reading - verify completeness |
| `think_about_task_adherence` | Before code modification - verify alignment |
| `think_about_completion` | Before task closure - checklist verification |

### Consolidated Tools (`tools/consolidated.py`)

Action-based tools per ADR-009:

| Tool | Actions |
|------|---------|
| `playbook` | add, write, delete, prune, list, stats, confirm |
| `memory` | list, read, write, delete |
| `session` | list, stats, compact, delete |
| `profiles` | recommend, check, reevaluate, cleanup |
| `project` | init, scan, analyze, sync, read_instructions |
| `admin` | switch_model, queue_status, mcp_servers |

### LSP Tools (`tools/lsp.py`)

| Tool | Description |
|------|-------------|
| `lsp_goto_definition` | Jump to symbol definition |
| `lsp_find_references` | Find all references to symbol |
| `lsp_hover` | Get documentation/type info |
| `lsp_get_symbols` | Get all symbols in file |
| `lsp_find_symbol` | Search symbols by name |
| `lsp_find_symbol_semantic` | Semantic + LSP fusion search |
| `lsp_find_referencing_symbols` | Find containing symbols for references |
| `lsp_rename_symbol` | Rename across codebase |
| `lsp_replace_symbol_body` | Replace function/class body |
| `lsp_insert_before_symbol` | Insert code before symbol |
| `lsp_insert_after_symbol` | Insert code after symbol |
| `lsp_move_symbol` | Move symbol to different file |
| `lsp_extract_method` | Extract code block to method |
| `lsp_batch` | Execute multiple LSP operations |
| `lsp_get_hot_files` | Get recently modified files |

### File Tools (`tools/files.py`)

| Tool | Description |
|------|-------------|
| `read_file` | Read file with line numbers |
| `write_file` | Write/create file |
| `edit_file` | Search and replace in file |
| `list_dir` | List directory contents |
| `find_file` | Find files by glob pattern |
| `search_for_pattern` | Grep-like pattern search |
| `delete_file` | Delete file |
| `create_directory` | Create directory |
| `read_files` | Batch read multiple files |
| `edit_files` | Batch edit with atomicity |

### LLM Delegation Tools (`tools/handlers.py`)

| Tool | Description |
|------|-------------|
| `delegate` | Single task delegation to backend |
| `think` | Extended reasoning with thinking models |
| `batch` | Parallel execution across backends |
| `chain` | Sequential task execution with piping |
| `workflow` | DAG workflow with branching |
| `agent` | Autonomous agent with tool use |

---

## CLI Commands

```bash
# Server
delia serve [--transport stdio|sse] [--port 8765]
delia api [--port 8766]

# Initialization
delia init [--force]              # Initialize globally
delia init-project [--force]      # Initialize .delia/ for project

# Editor Integration
delia install [--client claude|cursor|vscode|...]
delia uninstall [--client ...] [--full]
delia doctor                      # Health check

# Project Management
delia index [--force] [--summarize]
delia validate [--fix]            # Schema validation
delia config [--show] [--edit]

# Session Management
delia compact [--session-id] [--force]
delia memory [--reload] [--show-content]
delia audit [--limit 20] [--tool ...]
delia undo [--list-stack] [--session]

# Interactive
delia chat [--model ...] [--backend ...]
delia agent TASK [--model ...] [--workspace ...]

# Tracking
delia melons [--task ...] [--json]
delia prewarm
```

---

## Data Storage

### Global (`~/.delia/`)

```
~/.delia/
├── settings.json          # Backend configuration
├── data/
│   ├── sessions/          # Conversation history
│   ├── economics/         # Token tracking
│   ├── metrics/           # Backend metrics
│   └── affinity/          # Affinity scores
└── cache/
    └── embeddings/        # Embedding cache
```

### Per-Project (`.delia/`)

```
.delia/
├── playbooks/             # Learned patterns (JSON)
│   ├── coding.json
│   ├── testing.json
│   └── ...
├── memories/              # Persistent knowledge (Markdown)
│   └── *.md
├── profiles/              # Active guidance (Markdown)
│   └── *.md
├── project_summary.json   # Project metadata
└── learned_patterns.json  # Detection learning
```

### Schema Version

```python
SCHEMA_VERSION = 2

# v1: Flat array of bullets
# v2: Wrapped format with metadata
{
    "schema_version": 2,
    "task_type": "coding",
    "updated_at": "2024-12-23T12:00:00.000000",
    "bullets": [...]
}
```

---

## Profile Templates

33 built-in profile templates in `templates/profiles/`:

| Category | Profiles |
|----------|----------|
| **Languages** | python, typescript, golang, rust, c, cpp |
| **Frameworks** | react, vue, svelte, angular, nextjs, django, fastapi, laravel, nestjs, flutter |
| **Mobile** | android, ios |
| **Specialized** | ml, deeplearning, llm, solidity |
| **Task Types** | coding, testing, debugging, architecture, git, deployment, security, performance, api |

Core profile (`core.md`) is always loaded.

---

## Dependencies

Key packages:

| Package | Purpose |
|---------|---------|
| mcp>=1.23.0 | MCP protocol |
| fastmcp>=2.13.3 | MCP server framework |
| httpx>=0.28.1 | Async HTTP client |
| pydantic>=2.12.5 | Data validation |
| sentence-transformers>=3.0.0 | Semantic routing |
| structlog>=24.4.0 | Structured logging |
| typer>=0.15.0 | CLI framework |
| pygls>=1.3.1 | LSP client |
| google-generativeai>=0.8.5 | Gemini provider |
| tiktoken>=0.8.0 | Token counting |

---

## Dashboard

Next.js monitoring dashboard in `dashboard/`:

- System health and backend status
- Tool usage metrics and analytics
- Session browser
- Playbook/memory editor
- Dependency graph visualization

Access via `delia dashboard` or `http://localhost:8765` after starting server.

---

## Security

### File Safety

Path validation in `tools/files.py`:
- Blocked patterns: `.git`, `node_modules`, secrets files
- Workspace-relative resolution
- Symlink attack prevention

### Authentication (Optional)

FastAPI-Users integration:
- OAuth support
- SQLite session storage
- JWT tokens
- Setup via `delia-setup-auth`

### Sandboxing (Optional)

Docker-based sandboxing with `llm-sandbox[docker]`:
- Isolated code execution
- Resource limits
- Network isolation

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DELIA_BACKEND` | Default backend (ollama, llamacpp, gemini) |
| `OLLAMA_BASE_URL` | Ollama server URL |
| `GEMINI_API_KEY` | Google Gemini API key |
| `DELIA_AUTH_ENABLED` | Enable authentication |

### settings.json Structure

```json
{
  "backends": [...],
  "routing": {
    "prefer_loaded_model": true,
    "swap_penalty_ms": 500
  },
  "models": {
    "quick": "qwen2.5:7b",
    "coder": "qwen2.5-coder:14b",
    "thinking": "deepseek-r1:32b"
  },
  "mcp_servers": [...],
  "topology": {
    "single_gpu": true
  }
}
```

---

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src/delia

# Type checking
uv run mypy src/delia

# Linting
uv run ruff check src/
```

Dev dependencies:
- pytest, pytest-asyncio, pytest-cov
- hypothesis (property-based testing)
- mypy, ruff, bandit

---

## Performance Considerations

1. **Embedding Cache** - Cached in `~/.delia/cache/embeddings/`
2. **Health TTL** - 30-second cache for backend health
3. **Batch Operations** - `read_files`, `edit_files` for atomic multi-file ops
4. **Prewarm** - Hour-based model preloading predictions
5. **Affinity Tracking** - Learn optimal backend×task pairings
6. **Circuit Breaker** - Exponential backoff on failures

---

*Last Updated: December 2024*
