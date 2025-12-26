# Quick Start

Get Delia working in 5 minutes.

## 1. Install

```bash
git clone https://github.com/zbrdc/delia.git
cd delia && uv sync
```

## 2. Start Server

```bash
uv run delia run -t http --port 8765
```

## 3. Configure Your Project

Create `.mcp.json` in your project root:

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

## 4. Initialize

In your AI assistant (Claude Code, Cursor, etc.):

```
Initialize this project with Delia
```

Or use the tool directly:

```
project(action="init")
```

## 5. Use the Workflow

Every task follows three steps:

```
# 1. START - Load context
auto_context(message="implement user authentication")

# 2. WORK - Apply the loaded bullets and profiles
# ... do the actual work ...

# 3. END - Record feedback
complete_task(success=True, bullets_applied='["bullet-id-1", "bullet-id-2"]')
```

## What Just Happened?

1. **auto_context** loaded relevant patterns from your playbooks
2. You applied those patterns while working
3. **complete_task** recorded which patterns helped

Next time, the helpful patterns rank higher. Delia learns what works for your project.

## Next Steps

- [Workflow](../user-guide/workflow.md) - Understand the learning loop
- [Tools Overview](../tools/README.md) - See all available tools
- [Configuration](configuration.md) - Customize embeddings and backends
