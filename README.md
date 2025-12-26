# Delia

MCP server that adds persistent learning and semantic code intelligence to AI coding assistants.

## What It Does

- **Playbooks** - Per-project patterns learned over time (`.delia/playbooks/`)
- **Memories** - Persistent knowledge stored as markdown (`.delia/memories/`)
- **Profiles** - Framework-specific guidance loaded automatically (`.delia/profiles/`)
- **LSP Tools** - Semantic code navigation: find references, go to definition, rename symbols
- **Learning Loop** - Extracts insights from completed tasks and updates playbooks

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

From your project directory:
```bash
# Using uv (no venv activation needed)
uv run --directory /path/to/delia delia init-project

# Or if delia is in PATH
delia init-project
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
│   ├── playbooks/      # Learned patterns (JSON)
│   ├── memories/       # Persistent knowledge (Markdown)
│   ├── profiles/       # Framework guides (Markdown)
│   └── chroma/         # Vector database (optional)
└── CLAUDE.md           # Instructions for AI assistants
```

## CLI Commands

```bash
delia run -t http        # Start HTTP server
delia serve              # Start stdio server
delia init               # Initialize globally
delia init .             # Initialize project
delia doctor             # Health check
delia chat               # Interactive chat (requires Ollama)
delia agent "task"       # Single-shot task (requires Ollama)
```

## Embedding Providers

For semantic search in playbooks (optional):

1. **Voyage AI** - Set `DELIA_VOYAGE_API_KEY` in `~/.delia/.env`
2. **Ollama** - Run `ollama pull mxbai-embed-large`
3. **Sentence Transformers** - CPU fallback, works offline

## Requirements

- Python 3.11+
- uv (package manager)
- Ollama (optional, for local LLM features)

## License

GPL-3.0
