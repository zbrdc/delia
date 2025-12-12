# Delia

A Model Context Protocol (MCP) server that cultivates your local LLM garden. Plant a seed, let Delia pick the right vine, and harvest a fresh melon.

*Delia - from Greek Δηλία, "from Delos" (the sacred island). Also, she grows watermelons.*

## Features

- **Smart Vine Selection**: Routes seeds to the right vine - quick (7B), coder (14B+), moe (30B+), or thinking
- **Multi-Garden Support**: Ollama, llama.cpp, and Gemini gardens with automatic failover
- **Context-Aware Routing**: Handles large seeds with appropriate context windows
- **Circuit Breaker**: Drought protection with graceful recovery
- **Parallel Processing**: Tends multiple seeds simultaneously
- **Authentication**: Optional greenhouse access control
- **Usage Tracking**: Per-gardener quotas and harvest monitoring
- **Dashboard**: Real-time garden status with watermelon-themed activity feed

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

```bash
# Clone and install
git clone https://github.com/zbrdc/delia.git
cd delia
uv sync

# Pull models (examples - choose based on your hardware)
ollama pull qwen3:14b           # General purpose
ollama pull qwen2.5-coder:14b   # Code specialized
ollama pull qwen3:30b-a3b       # Complex reasoning

# Run server
uv run python mcp_server.py
```

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

Copy the example configuration and customize for your setup:

```bash
cp settings.json.example settings.json
```

Then edit `settings.json`:

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

| Tool | Description |
|------|-------------|
| `delegate` | Execute tasks with automatic model selection |
| `think` | Extended reasoning for complex problems |
| `batch` | Process multiple tasks in parallel |
| `health` | Check backend status and statistics |
| `models` | List available models and tiers |
| `switch_backend` | Switch between backends at runtime |
| `switch_model` | Change model for a tier |
| `get_model_info` | Get model specifications |

## Vine Selection

Delia picks the right vine for every seed:

| Vine | Size | Best For |
|------|------|----------|
| Quick | 7B-14B | Summaries, simple questions |
| Coder | 14B-30B | Generation, review, debugging |
| MoE | 30B+ | Architecture, critique, analysis |
| Thinking | Specialized | Extended reasoning, research |

Override with hints in your prompt: "use the large model" or "quick answer".

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

Typical harvest times (modern hardware):
- Quick vine: 2-5 seconds
- Coder vine: 5-15 seconds
- MoE/Thinking vines: 30-60 seconds

## License

BSD 3-Clause

## Acknowledgments

- [Ollama](https://ollama.ai) - Local LLM runtime
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Protocol implementation
- [Qwen](https://qwenlm.github.io/) - Base models
