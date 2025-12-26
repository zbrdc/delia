# Installation

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Ollama (optional, for CLI LLM features)

## Install Delia

```bash
# Clone the repository
git clone https://github.com/zbrdc/delia.git
cd delia

# Install dependencies
uv sync
```

## Configure MCP Clients

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

## Start the Server

### Option A: HTTP Transport (Recommended)

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

> **Note**: HTTP servers won't appear in Claude Code's `/mcp` list, but tools work normally.

### Option B: stdio Transport

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

## Initialize Your Project

### Via MCP (Recommended)

Let the AI agent initialize the project:

```
# In Claude Code or Cursor, ask:
"Initialize this project with Delia"

# Or use the MCP tool directly:
project(action="init", path="/path/to/your-project")
```

### Via CLI (requires Ollama)

If you have Ollama running locally:

```bash
cd ~/your-project
uv run --directory /path/to/delia delia init-project
```

This creates `.delia/` with playbooks tailored to your tech stack.

## Verify Setup

```bash
uv run delia doctor
```

## Next Steps

- [Quick Start](quick-start.md) - 5-minute guide
- [Configuration](configuration.md) - Customize settings
- [Workflow](../user-guide/workflow.md) - Learn the learning loop
