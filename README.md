# Delia

MCP server that adds persistent learning and semantic code intelligence to AI coding assistants.

## What It Does

- **Playbooks** - Per-project patterns learned over time (`.delia/playbooks/`)
- **Memories** - Persistent knowledge stored as markdown (`.delia/memories/`)
- **Profiles** - Framework-specific guidance loaded automatically (`.delia/profiles/`)
- **LSP Tools** - Semantic code navigation: find references, go to definition, rename symbols
- **Learning Loop** - Extracts insights from completed tasks and updates playbooks

## Installation

```bash
git clone https://github.com/zbrdc/delia.git
cd delia
uv sync
```

## Setup

### Option 1: HTTP Transport (Recommended)

Best for multi-project setups. One server handles all projects.

Start the server (runs in background):
```bash
delia run -t http --port 8765
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

**Note**: HTTP servers won't appear in Claude Code's `/mcp` list, but tools work normally. Verify with `mcp__delia__health` or by using any Delia tool.

### Option 2: stdio Transport

Per-project server, managed by Claude Code. Shows in `/mcp` list.

Add to Claude Code settings (`~/.claude.json` for global, or project `.mcp.json`):
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

For Cursor, use `.cursor/mcp.json` in your project root.

## Initialize a Project

```bash
cd your-project
delia init .
```

Creates `.delia/` with playbooks tailored to your tech stack.

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
