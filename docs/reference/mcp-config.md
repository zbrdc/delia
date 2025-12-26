# MCP Configuration

How to configure AI clients to use Delia.

## Quick Setup

```bash
# Auto-configure all detected clients
uv run delia install
```

## HTTP Transport (Recommended)

One server handles multiple projects. Best for multi-project setups.

### Start Server

```bash
uv run delia run -t http --port 8765
```

### Client Configuration

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

### Notes

- Server must be running before starting AI client
- Won't appear in Claude Code's `/mcp` list (but tools work)
- Single server for all projects

## stdio Transport

Per-project server, managed by AI client. Shows in `/mcp` list.

### Client Configuration

Add to `.mcp.json`:

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

### Notes

- Client starts/stops server automatically
- One server per project
- Shows in `/mcp` list

## Client-Specific Setup

### Claude Code

Location: Project root `.mcp.json` or `~/.claude/settings.json`

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

### Cursor

Location: `.cursor/mcp.json` in project root

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

### VS Code (Copilot)

Location: `.vscode/mcp.json` or workspace settings

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

### Windsurf

Location: `.windsurf/mcp.json`

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

## Comparison

| Aspect | HTTP | stdio |
|--------|------|-------|
| Server management | Manual | Automatic |
| Projects | Multiple | One per server |
| Shows in /mcp | No | Yes |
| Resource usage | Lower | Higher |
| Setup complexity | Simple | Simple |

## Troubleshooting

### Server Not Connecting

```bash
# Check if server is running
curl http://localhost:8765/health

# Check logs
DELIA_LOG_LEVEL=DEBUG uv run delia run -t http
```

### Tools Not Available

1. Restart AI client after config changes
2. Verify `.mcp.json` syntax
3. Check server is running (HTTP mode)

### Permission Errors

```bash
# Ensure uv is in PATH
which uv

# Try absolute path
"command": "/home/user/.local/bin/uv"
```

## Environment Variables in Config

```json
{
  "mcpServers": {
    "delia": {
      "command": "uv",
      "args": ["--directory", "/path/to/delia", "run", "delia", "serve"],
      "env": {
        "DELIA_TOOLS": "full",
        "DELIA_DELEGATION": "true"
      }
    }
  }
}
```

## See Also

- [Installation](../getting-started/installation.md) - Full setup
- [CLI Reference](cli.md) - Command details
- [Configuration](../getting-started/configuration.md) - Delia settings
