# Delia

MCP server that adds persistent learning and semantic code intelligence to AI coding assistants.

## What It Does

- **Playbooks** - Per-project patterns learned over time, indexed in ChromaDB for semantic retrieval
- **Memories** - Persistent knowledge (markdown), searchable via embeddings
- **Profiles** - Framework-specific guidance, semantically matched to your task
- **Code Index** - Codebase summaries and symbols indexed for intelligent navigation
- **LSP Tools** - Semantic code navigation: find references, go to definition, rename symbols
- **Learning Loop** - Extracts insights from completed tasks and updates playbooks

All knowledge is stored in `.delia/chroma/` for fast semantic search.

**[Full Documentation](docs/README.md)** | [Quick Start](docs/getting-started/quick-start.md) | [Tool Reference](docs/tools/README.md)

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/zbrdc/delia.git
cd delia
uv sync

# 2. Start HTTP server (recommended for multi-project)
uv run delia run -t http --port 8765

# 3. Initialize your project (from project directory)
cd ~/your-project
uv run --directory ~/git/delia delia init-project
```

## Complete Setup Guide

### Step 1: Install Delia

```bash
git clone https://github.com/zbrdc/delia.git
cd delia
uv sync
```

### Step 2: Configure MCP Clients

Auto-detect and configure all supported AI clients:
```bash
uv run delia install
```

Or install to a specific client:
```bash
uv run delia install claude    # Claude Code
uv run delia install cursor    # Cursor
uv run delia install vscode    # VS Code
```

List available clients:
```bash
uv run delia install --list
```

### Step 3: Start the Server

**Option A: HTTP Transport (Recommended)**

Best for multi-project setups. One server handles all projects.

```bash
uv run delia run -t http --port 8765
```

Add to each project's `.mcp.json`:
```json
{
  "mcpServers": {
    "delia": {
      "type": "http",
      "url": "http://localhost:8765/mcp"
    }
  }
}
```

**Note**: HTTP servers won't appear in Claude Code's `/mcp` list, but tools work normally.

**Option B: stdio Transport**

Per-project server, managed by the AI client. Shows in `/mcp` list.

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

### Step 4: Initialize Your Project

**Option A: Via MCP (Recommended)**

Let the AI agent initialize the project - it handles summarization:
```
# In Claude Code or Cursor, just ask:
"Initialize this project with Delia"
# Or use the MCP tool directly:
project(action="init", path="/path/to/your-project")
```

**Option B: Via CLI (requires Ollama)**

If you have Ollama running locally with a model:
```bash
cd ~/your-project
uv run --directory /path/to/delia delia init-project
```

This creates `.delia/` with playbooks tailored to your tech stack.

### Step 5: Verify Setup

```bash
uv run delia doctor
```

## Usage

The AI assistant calls these tools:

```
auto_context("implement user auth")  # Load relevant patterns
[work on the task]
complete_task(success=True, bullets_applied=["id1"])  # Record feedback
```

## Project Structure

```
your-project/
├── .delia/
│   ├── chroma/         # Vector database (primary storage)
│   ├── playbooks/      # Learned patterns (JSON, indexed to ChromaDB)
│   ├── memories/       # Persistent knowledge (Markdown, indexed to ChromaDB)
│   └── profiles/       # Framework guides (Markdown, indexed to ChromaDB)
└── CLAUDE.md           # Instructions for AI assistants
```

## CLI Commands

```bash
delia run -t http        # Start HTTP server (MCP)
delia serve              # Start stdio server (MCP)
delia doctor             # Health check
delia init-project       # Initialize project (requires Ollama)
delia chat               # Interactive chat (requires Ollama)
delia agent "task"       # Single-shot task (requires Ollama)
```

## Configuration

### Embeddings (for semantic search)

Create `~/.delia/.env`:
```bash
DELIA_VOYAGE_API_KEY=your-key-here
```

Fallback options (no API key needed):
- **Ollama** - Run `ollama pull mxbai-embed-large`
- **Sentence Transformers** - CPU fallback, works offline

### LLM Backends (for CLI features)

For `init-project`, `chat`, `agent` commands, configure backends in `~/.delia/settings.json`:
```json
{
  "backends": [{
    "name": "ollama-local",
    "url": "http://localhost:11434",
    "model": "llama3.2"
  }]
}
```

## Requirements

- Python 3.11+
- uv (package manager)
- Ollama (optional, for CLI LLM features - not needed if using MCP only)

## License

GPL-3.0
