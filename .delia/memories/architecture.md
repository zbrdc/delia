# Delia Architecture

## Entry Points

| File | Purpose |
|------|---------|
| `cli.py` | CLI commands (`delia serve`, `init`, `agent`, etc.) |
| `mcp_server.py` | MCP server - main interface for AI agents (Claude, Cursor, etc.) |
| `api.py` | REST API (FastAPI) for programmatic access |

## ACE Framework (Playbook Learning System)

The core differentiator - learns project-specific patterns over time.

```
ace/
├── reflector.py      # Post-task reflection & insight extraction
├── curator.py        # Playbook curation & bullet management
├── deduplication.py  # Bullet dedup/merging (semantic similarity)
└── retrieval.py      # Bullet retrieval for auto_context
```

| File | Purpose |
|------|---------|
| `playbook.py` | Playbook CRUD operations, bullet scoring |
| `context_detector.py` | Task type detection from user messages |
| `profiles.py` | Starter profile templates (FastAPI, React, etc.) |
| `project_memory.py` | Persistent project knowledge in `.delia/memories/` |

## Orchestration (Task Routing & Execution)

Routes tasks to appropriate models and execution modes.

```
orchestration/
├── service.py         # Main orchestration service
├── executor.py        # Task execution (47K lines - largest file)
├── dispatcher.py      # Route to appropriate model/tier
├── dispatcher_embed.py # Embedding-based dispatch
├── intent.py          # Intent classification
├── intrinsics.py      # Pre-execution checks (answerability, frustration)
├── stakes.py          # High-stakes detection
├── meta_learning.py   # Learning from execution outcomes
├── exemplars.py       # Intent examples for classification
├── critic.py          # Response critique
├── summarizer.py      # Context summarization
├── graph.py           # Dependency graph
└── tuning.py          # Model tuning/calibration
```

## Tools (MCP Tool Implementations)

Tools exposed via MCP server for AI agents.

```
tools/
├── handlers.py      # ACE tools (auto_context, complete_task, get_playbook)
├── admin.py         # Admin tools (init_project, health, scan_codebase)
├── consolidated.py  # Unified tools per ADR-009 (playbook, memory, session, etc.)
├── lsp.py           # LSP code intelligence (goto_definition, find_references)
├── files.py         # File operations (read, write, edit, find)
├── coding.py        # Code-specific tools
├── builtins.py      # Built-in utilities
├── agent.py         # Agent execution
├── mcp_client.py    # External MCP server management
└── sandbox_tools.py # Sandboxed execution (DEAD - uses sandbox.py)
```

## LLM Backends

Connects to local/remote LLM providers.

```
providers/
├── base.py      # Base provider interface
├── ollama.py    # Ollama backend
├── llamacpp.py  # llama.cpp backend (direct GGUF)
├── gemini.py    # Google Gemini backend
└── registry.py  # Provider registration
```

| File | Purpose |
|------|---------|
| `backend_manager.py` | Backend coordination, health checks |
| `routing.py` | Model routing logic (tier selection) |
| `llm.py` | LLM call abstraction |
| `model_detection.py` | Detect model capabilities |

## Infrastructure

| File | Purpose |
|------|---------|
| `config.py` | Settings management (`~/.delia/settings.json`) |
| `session_manager.py` | Conversation session handling |
| `session_backends.py` | Session storage (SQLite, file) |
| `lsp_client.py` | Language Server Protocol client |
| `queue.py` | Task queue with priorities |
| `voting.py` | Multi-model voting for reliability |
| `voting_stats.py` | Voting statistics |
| `embeddings.py` | Embedding generation |
| `hardware.py` | GPU/hardware detection |

## Semantic Processing

```
semantic/
├── cache.py           # Embedding cache
├── coherence.py       # Response coherence checking
├── compression.py     # Context compression
└── playbook_search.py # Semantic playbook search
```

## Supporting Modules

| File | Purpose |
|------|---------|
| `agent_sync.py` | Sync CLAUDE.md to other AI agents (Cursor, Copilot, etc.) |
| `paths.py` | Path resolution for `.delia/` directories |
| `file_helpers.py` | File utilities |
| `text_utils.py` | Text processing |
| `messages.py` | Status messages |
| `errors.py` | Error types |
| `types.py` | Type definitions |
| `validation.py` | Input validation |
| `prompts.py` | Prompt templates |
| `tracing.py` | Distributed tracing |
| `logging_service.py` | Structured logging |
| `middleware.py` | API middleware |
| `container.py` | Dependency injection |

## Auth (Optional)

| File | Purpose |
|------|---------|
| `auth.py` | Authentication logic |
| `auth_routes.py` | Auth API routes |
| `setup_auth.py` | Auth setup wizard |
| `security.py` | Security utilities |

## Dead/Deprecated (See cleanup-backlog.md)

| File | Status |
|------|--------|
| `eval_harness.py` | DEAD - broken stub |
| `personas.py` | DEAD - never wired up |
| `sandbox.py` | DEAD - never imported |
| `frustration.py` | DEPRECATED - moved to intrinsics.py |
| `melons.py` | QUESTIONABLE - gamification, tests only |
| `multi_user_tracking.py` | QUESTIONABLE - tests only |

## Data Storage

```
~/.delia/                    # Global Delia data
├── settings.json            # Global settings
├── data/                    # Runtime data (sessions, economics)
└── cache/                   # Embedding cache

<project>/.delia/            # Per-project data
├── playbooks/               # Learned patterns (JSON)
├── memories/                # Persistent knowledge (Markdown)
├── profiles/                # Active profile templates
└── project_summary.json     # Project analysis
```
