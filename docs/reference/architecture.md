# Architecture

Overview of Delia's system design.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AI Assistants                          │
│           (Claude Code, Cursor, Copilot, etc.)              │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Delia MCP Server                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Tools     │  │  Framework  │  │      Learning       │  │
│  │  (40+)      │  │   Loop      │  │       Loop          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌───────────┐ ┌───────────────┐
│   ChromaDB      │ │    LSP    │ │   Ollama/     │
│  (Embeddings)   │ │  Servers  │ │   LLM APIs    │
└─────────────────┘ └───────────┘ └───────────────┘
```

## Core Components

### MCP Server

Entry point for all AI assistant interactions.

- **Location**: `src/delia/mcp_server.py`
- **Transport**: HTTP or stdio
- **Protocol**: Model Context Protocol (MCP)

### Tool System

40+ tools organized by profile:

| Profile | Tools | Use Case |
|---------|-------|----------|
| `light` | ~23 | Minimal deployments |
| `standard` | ~31 | Default |
| `full` | ~40+ | Full capabilities |

**Locations**:
- `src/delia/tools/files.py` - File operations
- `src/delia/tools/lsp.py` - Code navigation
- `src/delia/tools/framework.py` - Learning loop
- `src/delia/tools/semantic.py` - Embeddings search
- `src/delia/tools/consolidated.py` - Management tools

### Learning Loop

The core feedback mechanism:

```
auto_context() → Work → complete_task()
       │                      │
       │    ┌─────────────┐   │
       └───▶│  Retriever  │◀──┘
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │   Curator   │
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  Playbooks  │
            └─────────────┘
```

**Components**:
- **Retriever** (`src/delia/learning/retrieval.py`) - Semantic bullet search
- **Curator** (`src/delia/learning/curator.py`) - Delta operations on playbooks
- **Reflector** (`src/delia/learning/reflector.py`) - Insight extraction

### Embeddings

Vector embeddings for semantic search:

```
┌──────────────┐
│   Query      │
└──────┬───────┘
       │ embed
       ▼
┌──────────────┐     ┌──────────────┐
│  Embedding   │────▶│   ChromaDB   │
│   Service    │     │   Search     │
└──────────────┘     └──────────────┘
```

**Providers** (in priority order):
1. Voyage AI (API)
2. Ollama (local)
3. Sentence Transformers (fallback)

**Location**: `src/delia/embeddings.py`

### LSP Integration

Language Server Protocol for semantic code navigation:

```
┌───────────────┐     ┌───────────────┐
│  Delia LSP    │────▶│  Language     │
│   Tools       │     │  Server       │
└───────────────┘     └───────────────┘
                            │
                      ┌─────▼─────┐
                      │  Source   │
                      │   Code    │
                      └───────────┘
```

**Location**: `src/delia/tools/lsp.py`

## Data Storage

### Project Data (`.delia/`)

```
.delia/
├── chroma/           # Vector database
├── playbooks/        # Learned patterns (JSON)
│   ├── coding.json
│   ├── testing.json
│   └── ...
├── memories/         # Project knowledge (Markdown)
├── profiles/         # Framework guidance (Markdown)
├── symbol_graph.json # Code structure
└── project_summary.json
```

### User Data (`~/.delia/`)

```
~/.delia/
├── .env              # API keys (Voyage, etc.)
├── settings.json     # LLM backend config
└── DELIA.md          # Global instructions
```

## Data Flow

### Context Loading

```
1. auto_context("implement auth")
2. → Detect task type
3. → Embed query
4. → Search ChromaDB for similar bullets
5. → Score: relevance × utility × recency
6. → Load matching profiles
7. → Return context to AI
```

### Learning Flow

```
1. complete_task(success=True, bullets_applied=["id1"])
2. → Curator receives feedback
3. → Increment helpful_count on used bullets
4. → Recalculate utility scores
5. → Save to playbooks/*.json
6. → Re-index to ChromaDB
```

## Design Principles

### Atomic Bullets

Patterns are stored as small, actionable bullets (15-300 chars) rather than monolithic documents. This enables:
- Precise retrieval
- Individual scoring
- Incremental updates

### Delta Operations

Learning happens via atomic deltas (ADD, BOOST, DEMOTE, REMOVE, MERGE) rather than regenerating entire playbooks. This prevents:
- Context collapse
- Cascading errors
- Loss of accumulated knowledge

### Cross-Agent

Delia works with any MCP-compatible client:
- Claude Code
- Cursor
- VS Code Copilot
- Windsurf
- Custom agents

### Progressive Disclosure

Tools reveal information gradually:
1. `lsp_get_symbols` → file overview
2. `lsp_find_symbol` → specific symbol
3. `read_file(start_line, end_line)` → exact code

This saves context and improves focus.

## See Also

- [Tools Overview](../tools/README.md) - Available tools
- [Workflow](../user-guide/workflow.md) - Learning loop details
- [Configuration](../getting-started/configuration.md) - Settings
