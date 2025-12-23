# Delia: ACE Framework for AI Coding Assistants

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
| **ACE Loop** | Reflector → Curator pipeline that extracts insights from completed tasks. |

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
| Embeddings | `mxbai-embed-large` (required for semantic search) |
| Quick tasks | `qwen3:4b`, `ministral-3b` |
| Coding | `deepcoder:14b`, `qwen-coder:14b` |
| Thinking | `qwen3:14b`, `openthinker:7b` |

## Project Structure

```
your-project/
├── .delia/
│   ├── playbooks/      # Learned patterns (coding.json, testing.json, ...)
│   ├── memories/       # Persistent knowledge (architecture.md, ...)
│   └── profiles/       # Framework guides (fastapi.md, react.md, ...)
└── CLAUDE.md           # Auto-generated instructions for AI assistants
```

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
