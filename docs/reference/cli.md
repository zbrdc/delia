# CLI Reference

Command-line interface for Delia.

## Commands

### delia run

Start the MCP server.

```bash
# HTTP transport (recommended)
uv run delia run -t http --port 8765

# stdio transport
uv run delia run -t stdio
```

**Options**:
| Flag | Description |
|------|-------------|
| `-t, --transport` | Transport type: `http` or `stdio` |
| `--port` | Port for HTTP server (default: 8765) |
| `--host` | Host to bind (default: 0.0.0.0) |

### delia serve

Alias for `delia run -t stdio`. Used in MCP client configs.

```bash
uv run delia serve
```

### delia install

Configure MCP clients.

```bash
# Auto-detect and configure all clients
uv run delia install

# Specific client
uv run delia install claude
uv run delia install cursor
uv run delia install vscode

# List available clients
uv run delia install --list
```

### delia init-project

Initialize a project for Delia (requires Ollama).

```bash
cd /path/to/project
uv run --directory /path/to/delia delia init-project
```

**Options**:
| Flag | Description |
|------|-------------|
| `--force` | Reinitialize existing project |
| `--skip-index` | Skip code indexing |

**Note**: Prefer `project(action="init")` via MCP when possible.

### delia doctor

Health check.

```bash
uv run delia doctor
```

**Checks**:
- Python version
- Embedding service availability
- ChromaDB status
- Playbook integrity
- LSP server status

### delia chat

Interactive chat mode (requires Ollama).

```bash
uv run delia chat
```

**Options**:
| Flag | Description |
|------|-------------|
| `--model` | Model to use |
| `--backend` | Backend name from settings |

### delia agent

Single-shot task execution (requires Ollama).

```bash
uv run delia agent "Write a function to parse JSON"
```

**Options**:
| Flag | Description |
|------|-------------|
| `--model` | Model to use |
| `--backend` | Backend name |
| `--max-iterations` | Max tool calls |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DELIA_TOOLS` | `standard` | Tool profile: light, standard, full |
| `DELIA_DELEGATION` | `false` | Enable delegation tools |
| `DELIA_LOG_LEVEL` | `INFO` | Log verbosity |

## Examples

### Start Server for Development

```bash
# Terminal 1: Start server
uv run delia run -t http --port 8765

# Terminal 2: Check health
curl http://localhost:8765/health
```

### Initialize Multiple Projects

```bash
# From Delia directory
for project in ~/projects/*/; do
  uv run delia init-project --path "$project"
done
```

### Debug Mode

```bash
DELIA_LOG_LEVEL=DEBUG uv run delia run -t http
```

## See Also

- [Installation](../getting-started/installation.md) - Setup guide
- [MCP Configuration](mcp-config.md) - Client setup
