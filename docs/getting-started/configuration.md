# Configuration

## Embeddings

Embeddings power semantic search. Configure in `~/.delia/.env`:

```bash
# Option 1: Voyage AI (recommended for quality)
DELIA_VOYAGE_API_KEY=your-key-here

# Option 2: Ollama (local, no API key)
# Just run: ollama pull mxbai-embed-large

# Option 3: Sentence Transformers (CPU fallback)
# Works automatically if others unavailable
```

### Embedding Priority

Delia tries providers in order:

1. **Voyage AI** - Best quality, requires API key
2. **Ollama** - Local, requires `mxbai-embed-large` model
3. **Sentence Transformers** - CPU fallback, works offline

## LLM Backends

For CLI features (`init-project`, `chat`, `agent`), configure backends in `~/.delia/settings.json`:

```json
{
  "backends": [
    {
      "name": "ollama-local",
      "url": "http://localhost:11434",
      "model": "llama3.2"
    }
  ]
}
```

### Multiple Backends

```json
{
  "backends": [
    {
      "name": "fast",
      "url": "http://localhost:11434",
      "model": "llama3.2:3b"
    },
    {
      "name": "smart",
      "url": "http://localhost:11434",
      "model": "qwen2.5:32b"
    }
  ]
}
```

## Tool Profiles

Control which tools are registered via environment variable:

```bash
# Minimal tools (~23)
DELIA_TOOLS=light uv run delia serve

# Standard tools (~31) - default
DELIA_TOOLS=standard uv run delia serve

# All tools (~40+)
DELIA_TOOLS=full uv run delia serve
```

### Profile Contents

| Profile | Tools |
|---------|-------|
| `light` | Files, LSP, Framework, Semantic |
| `standard` | Light + Consolidated + Git |
| `full` | Standard + Resources + MCP Management |

## Delegation

Enable local model delegation for offloading tasks:

```bash
DELIA_DELEGATION=true uv run delia serve
```

This adds: `delegate`, `think`, `batch`, `chain`, `workflow`, `agent`

## Project Structure

After initialization, your project has:

```
your-project/
├── .delia/
│   ├── chroma/         # Vector database
│   ├── playbooks/      # Learned patterns (JSON)
│   ├── memories/       # Persistent knowledge (Markdown)
│   └── profiles/       # Framework guides (Markdown)
└── CLAUDE.md           # Instructions for AI assistants
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DELIA_TOOLS` | `standard` | Tool profile: light, standard, full |
| `DELIA_DELEGATION` | `false` | Enable delegation tools |
| `DELIA_VOYAGE_API_KEY` | - | Voyage AI API key for embeddings |
| `DELIA_LOG_LEVEL` | `INFO` | Logging verbosity |

## See Also

- [Installation](installation.md) - Setup instructions
- [MCP Configuration](../reference/mcp-config.md) - Client setup
- [Troubleshooting](troubleshooting.md) - Common issues
