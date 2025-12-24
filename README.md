# Delia: Framework for AI Coding Assistants

Delia enhances AI coding assistants (Claude Code, Cursor, Windsurf) with **persistent learning** and **semantic code intelligence**. It remembers what works across sessions and applies proven patterns to every task.

## Quick Start (MCP Server)

### 1. Install
```bash
# Clone and install
git clone https://github.com/zbrdc/delia.git
cd delia
uv sync  # or: pip install -e .
```

### 2. Configure MCP Client

Add to your MCP configuration:

**Claude Code** (`~/.claude/claude_code_config.json`):
```json
{
  "mcpServers": {
    "delia": {
      "command": "uv",
      "args": ["--directory", "/path/to/delia", "run", "delia", "serve"]
    }
  }
}
```

**Cursor** (`.cursor/mcp.json` in project root):
```json
{
  "mcpServers": {
    "delia": {
      "command": "uv",
      "args": ["--directory", "/path/to/delia", "run", "delia", "serve"]
    }
  }
}
```

### 3. Initialize Your Project
```bash
cd your-project
delia init .
```

This creates `.delia/` with playbooks tailored to your tech stack.

### 4. Use It

The AI assistant now has access to Delia's tools. Key workflow:

```
auto_context("implement user auth")  → Get relevant patterns
[do the work, applying patterns]
complete_task(success=True, bullets_applied=["strat-xxx"])  → Learn from it
```

## What Delia Provides

| Feature | What It Does |
|---------|--------------|
| **Playbooks** | Per-project patterns learned over time. Coding, testing, debugging, git, security, etc. |
| **Memories** | Persistent knowledge in `.delia/memories/`. Architecture decisions, integration details. |
| **LSP Tools** | Semantic code navigation. Find references, go to definition, rename symbols. |
| **Profiles** | Framework-specific guidance (FastAPI, React, etc.) loaded automatically. |
| **Learning Loop** | Reflector → Curator pipeline that extracts insights from completed tasks. |

## CLI Usage

Beyond MCP, Delia has a standalone CLI:

```bash
# Interactive chat with local LLMs
delia chat

# Single-shot agent task
delia agent "Scan for security vulnerabilities"

# Initialize config (detect Ollama, etc.)
delia init

# Health check
delia doctor
```

## Backend Support

Delia works with local LLM backends:

- **Ollama** (recommended) - `ollama serve`
- **llama.cpp** - OpenAI-compatible server
- **LM Studio** / **vLLM** - Any OpenAI-compatible endpoint

### Recommended Models

| Purpose | Models |
|---------|--------|
| Quick tasks | `qwen3:4b`, `ministral-3b` |
| Coding | `deepcoder:14b`, `qwen-coder:14b` |
| Thinking | `qwen3:14b`, `openthinker:7b` |

## Semantic Search (Embeddings)

Delia uses embeddings for semantic playbook retrieval. Supported providers (in priority order):

### 1. Voyage AI (Recommended)
Best quality embeddings using `voyage-code-3` model (1024 dimensions).

```bash
# Add to ~/.delia/.env
echo "DELIA_VOYAGE_API_KEY=your-key-here" >> ~/.delia/.env
```

Get an API key at [voyageai.com](https://www.voyageai.com/).

### 2. Ollama (Local)
Uses `mxbai-embed-large` or `nomic-embed-text` from your local Ollama:

```bash
ollama pull mxbai-embed-large
```

### 3. Sentence Transformers (Fallback)
CPU-based local fallback using `sentence-transformers` library. Slower but works offline.

### Initialize Semantic Search
After setting up embeddings, index your playbooks:

```bash
delia migrate  # Index playbooks to ChromaDB
```

This creates `.delia/chroma/` with vector indices for semantic search.

## Project Structure

```
your-project/
├── .delia/
│   ├── playbooks/      # Learned patterns (coding.json, testing.json, ...)
│   ├── memories/       # Persistent knowledge (architecture.md, ...)
│   ├── profiles/       # Framework guides (fastapi.md, react.md, ...)
│   └── chroma/         # Vector database for semantic search
└── CLAUDE.md           # Auto-generated instructions for AI assistants
```

### What's Stored in ChromaDB

| Collection | Contents |
|------------|----------|
| `delia_playbook` | Playbook bullets with embeddings for semantic retrieval |
| `delia_memories` | Memory files indexed for search |
| `delia_code` | Code file summaries (optional, for codebase search) |
| `delia_profiles` | Profile templates |

Search playbooks semantically:
```python
playbook(action="search", query="async HTTP patterns")
```

## Architecture

Delia implements an ACE (Autonomous Cognitive Entity) framework:

| Component | Purpose |
|-----------|---------|
| **Playbooks** | Per-project learned patterns (`.delia/playbooks/`) |
| **OrchestrationExecutor** | Task routing, voting, model selection |
| **ContextDetector** | Intent detection from messages |
| **Reflector → Curator** | Extracts insights from completed tasks |
| **ToolRegistry** | File I/O, LSP, shell execution |

The learning loop: `auto_context()` → apply patterns → `complete_task()` → Reflector extracts insights → Curator updates playbooks.

## Troubleshooting

**MCP server not connecting:**
```bash
# Test the server directly
uv run delia serve

# Check health
uv run delia doctor
```

**Playbooks empty:**
```bash
# Re-initialize project
delia init --force .
```

**Ollama not detected:**
```bash
# Ensure Ollama is running
ollama serve

# Check available models
ollama list
```

## License

GPL-3.0
