# Delia

A Model Context Protocol (MCP) server that cultivates your local LLM garden. Plant a seed, let Delia pick the right vine, and harvest a fresh melon.

*Delia - from Greek Δηλία, "from Delos" (the sacred island). Also, she grows watermelons.*

## Features

- **Smart Model Selection**: Automatically routes prompts to optimal model tier (quick/coder/moe/thinking)
- **Multi-Backend Support**: Ollama, llama.cpp, Gemini, vLLM, OpenAI-compatible APIs with automatic failover
- **Context-Aware Routing**: Routes large prompts to models with sufficient context windows
- **Circuit Breaker**: Protects against cascading failures with automatic recovery
- **Parallel Processing**: Batch multiple requests across backends simultaneously
- **Authentication**: Optional user auth with per-user quotas (HTTP mode)
- **Usage Tracking**: Token counts, cost estimates, and performance metrics
- **Dashboard**: Real-time status monitoring with activity feed

## Requirements

### Hardware
| Component | Minimum | Recommended | Large Models |
|-----------|---------|-------------|--------------|
| GPU | 4GB VRAM | 12GB VRAM | 24GB+ VRAM |
| RAM | 8GB | 16GB | 32GB+ |
| Storage | 10GB | 30GB | 50GB+ |

### Software
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- One or more backends:
  - [Ollama](https://ollama.ai) (recommended)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp)
  - Google Gemini API (optional cloud fallback)

## Quick Start

### Prerequisites

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
2. Install [Ollama](https://ollama.ai/download) and ensure it's running (`ollama serve`)

### Installation

```bash
# Clone and install dependencies
git clone https://github.com/zbrdc/delia.git
cd delia
uv sync

# Pull at least one model (choose based on your VRAM)
ollama pull qwen3:14b           # 8GB+ VRAM - general purpose
ollama pull qwen2.5-coder:14b   # 8GB+ VRAM - code specialized
ollama pull qwen3:30b-a3b       # 16GB+ VRAM - complex reasoning

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Run the setup wizard
uv run delia init
```

The setup wizard will:
- Detect available backends (Ollama, llama.cpp, vLLM)
- Auto-assign models to tiers based on capabilities
- Optionally configure detected MCP clients (Claude Code, VS Code, etc.)

See [Configuration](#configuration) to customize further.

## Integration

Delia works with AI coding assistants via MCP. Choose your tool:

### VS Code / GitHub Copilot

Add to `~/.config/Code/User/mcp.json`:
```json
{
  "servers": {
    "delia": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/delia", "python", "mcp_server.py"],
      "type": "stdio"
    }
  }
}
```
Reload VS Code to activate.

### Claude Code

Create `~/.claude/mcp.json`:
```json
{
  "mcpServers": {
    "delia": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/delia", "python", "mcp_server.py"]
    }
  }
}
```
Then run `claude` and use `@delia` to delegate tasks.

### Gemini CLI

**Option 1: HTTP Mode (Recommended)**
```bash
# Start server
uv run python mcp_server.py --transport sse --port 8200
```

Add to `~/.gemini/settings.json`:
```json
{
  "mcpServers": {
    "delia": {
      "url": "http://localhost:8200/sse",
      "transport": "sse"
    }
  }
}
```

**Option 2: STDIO Mode**

Add to `~/.gemini/settings.json`:
```json
{
  "mcpServers": {
    "delia": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/delia", "python", "mcp_server.py"]
    }
  }
}
```

### GitHub Copilot CLI

Create `~/.copilot-cli/mcp.json`:
```json
{
  "servers": {
    "delia": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/delia", "python", "mcp_server.py"]
    }
  }
}
```

## Configuration

### Backend Configuration

Delia stores configuration in `settings.json` (created on first run). Copy the example to customize:

```bash
cp settings.json.example settings.json
```

#### Configuration Fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier for the backend |
| `provider` | Backend type: `ollama`, `llamacpp`, `gemini`, `vllm`, `openai` |
| `type` | `local` (GPU on this machine) or `remote` (cloud/network) |
| `url` | API endpoint URL |
| `priority` | Lower = preferred (used for failover ordering) |
| `models` | Map of tier → model name for this backend |

Example `settings.json`:

```json
{
  "backends": [
    {
      "id": "ollama-local",
      "name": "Ollama Local",
      "provider": "ollama",
      "type": "local",
      "url": "http://localhost:11434",
      "enabled": true,
      "priority": 1,
      "models": {
        "quick": "qwen3:14b",
        "coder": "qwen2.5-coder:14b",
        "moe": "qwen3:30b-a3b",
        "thinking": "deepseek-r1:14b"
      }
    }
  ],
  "routing": {
    "prefer_local": true,
    "fallback_enabled": true
  }
}
```

### Gemini Cloud Backend (Optional)

Add Gemini as a cloud fallback:

```bash
# Install dependency
uv add google-generativeai

# Set API key
export GEMINI_API_KEY="your-key-from-aistudio.google.com"
```

Add to `settings.json`:
```json
{
  "id": "gemini-cloud",
  "name": "Gemini Cloud",
  "provider": "gemini",
  "type": "remote",
  "url": "https://generativelanguage.googleapis.com",
  "enabled": true,
  "priority": 10,
  "models": {
    "quick": "gemini-2.0-flash",
    "coder": "gemini-2.0-flash",
    "moe": "gemini-2.0-flash"
  }
}
```

### Authentication (Optional)

For HTTP mode with multiple users:

```bash
# Quick setup
python setup_auth.py

# Or manually
export DELIA_AUTH_ENABLED=true
export DELIA_JWT_SECRET="your-secure-secret"
```

Supports username/password and Microsoft 365 OAuth.

## Transport Modes

```bash
# STDIO (default) - for VS Code, Claude Code, Copilot CLI
uv run python mcp_server.py

# HTTP/SSE - for Gemini CLI, web clients, remote access
uv run python mcp_server.py --transport sse --port 8200

# View all options
uv run python mcp_server.py --help
```

## Tools

Delia provides these MCP tools:

### Core Tools

| Tool | Description | Example |
|------|-------------|---------|
| `delegate` | Route a task to the optimal model tier | `delegate(task="review", content="<code>")` |
| `think` | Deep reasoning with extended thinking | `think(problem="Design auth system", depth="deep")` |
| `batch` | Process multiple tasks in parallel | `batch(tasks='[{"task":"review","content":"..."}]')` |

### Management Tools

| Tool | Description |
|------|-------------|
| `health` | Check backend availability, circuit breaker status, and usage statistics |
| `models` | List configured models per tier and currently loaded models |
| `switch_backend` | Change active backend at runtime (e.g., switch from Ollama to llama.cpp) |
| `switch_model` | Swap the model for a specific tier without restart |
| `get_model_info` | Get VRAM requirements and context window for any model |

## Model Tiers (Vine Selection)

Delia automatically routes requests to the optimal model tier based on task complexity. In garden terminology: prompts are "seeds," model tiers are "vines," and responses are "melons."

| Tier | Model Size | Task Types | Triggers |
|------|------------|------------|----------|
| **quick** | 7B-14B | Summaries, simple Q&A | `task="quick"`, `task="summarize"` |
| **coder** | 14B-30B | Code generation, review, analysis | `task="generate"`, `task="review"`, `task="analyze"` |
| **moe** | 30B+ | Architecture, planning, critique | `task="plan"`, `task="critique"` |
| **thinking** | Specialized | Extended reasoning, research | `model="thinking"` or complex prompts |

Override automatic selection with: `model="quick"`, `model="coder"`, `model="moe"`, or natural hints like "use the large model".

## Troubleshooting

### Server won't start
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Test server import
uv run python -c "import mcp_server; print('OK')"
```

### MCP not connecting
- Verify path in config points to correct directory
- Reload VS Code / restart Claude Code
- Check logs: `~/.cache/delia/live_logs.json`

### "Unknown" responses
- Backend not running or unreachable
- Check `settings.json` configuration
- Run `curl http://localhost:11434/health`

### Slow responses
- Try smaller models
- Check system resources (`nvidia-smi`, `htop`)
- Reduce context size in `settings.json`

## Performance

Typical response times on modern hardware (RTX 3090/4090):

| Tier | Response Time | Use Case |
|------|---------------|----------|
| Quick | 2-5 seconds | Simple queries, summaries |
| Coder | 5-15 seconds | Code review, generation |
| MoE | 15-45 seconds | Complex analysis, planning |
| Thinking | 30-90 seconds | Deep reasoning, research |

Times vary based on prompt length, model size, and hardware.

## License

BSD 3-Clause

## Acknowledgments

- [Ollama](https://ollama.ai) - Local LLM runtime
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Protocol implementation
- [Qwen](https://qwenlm.github.io/) - Base models
