# Troubleshooting

Common issues and solutions.

## Installation

### "uv not found"

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Dependencies fail to install

```bash
# Clear cache and retry
uv cache clean
uv sync --refresh
```

### Python version mismatch

Delia requires Python 3.11+:

```bash
# Check version
python --version

# Use pyenv to install
pyenv install 3.11
pyenv local 3.11
```

## Server Issues

### "Connection refused" on HTTP

Server isn't running:

```bash
# Start server
uv run delia run -t http --port 8765

# Verify
curl http://localhost:8765/health
```

### Port already in use

```bash
# Find process using port
lsof -i :8765

# Kill it
kill -9 <PID>

# Or use different port
uv run delia run -t http --port 8766
```

### Server crashes on start

Check dependencies:

```bash
uv run delia doctor
```

Enable debug logging:

```bash
DELIA_LOG_LEVEL=DEBUG uv run delia run -t http
```

## MCP Issues

### Tools not appearing

1. Restart your AI client (Claude Code, Cursor, etc.)
2. Verify `.mcp.json` syntax is valid JSON
3. Check server is running (HTTP mode)

For HTTP mode, tools won't appear in `/mcp` list but will work.

### "Server not found"

Check path in config:

```json
{
  "mcpServers": {
    "delia": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/delia", "run", "delia", "serve"]
    }
  }
}
```

Use absolute paths, not `~` or `$HOME`.

### Config not loading

File must be valid JSON. Check for:
- Trailing commas
- Missing quotes
- Syntax errors

```bash
# Validate JSON
python -m json.tool .mcp.json
```

## Embedding Issues

### "No embedding service available"

Install fallback:

```bash
# Option 1: Ollama (recommended)
ollama pull mxbai-embed-large

# Option 2: Sentence Transformers (automatic fallback)
# Already included in dependencies
```

Or configure Voyage AI:

```bash
# ~/.delia/.env
DELIA_VOYAGE_API_KEY=your-key
```

### Slow embeddings

Sentence Transformers on CPU is slow. Use:

1. **Ollama** (GPU accelerated)
2. **Voyage AI** (API, fast)

### ChromaDB errors

Reset the database:

```bash
rm -rf .delia/chroma/
# Reinitialize
project(action="init", force=True)
```

## Learning Loop Issues

### Playbooks not updating

Ensure you're calling `complete_task()`:

```python
complete_task(
    success=True,
    bullets_applied='["bullet-id-1"]'
)
```

### auto_context returns no bullets

1. Project may not be initialized:
   ```python
   project(action="init")
   ```

2. Playbooks may be empty:
   ```python
   playbook(action="stats")
   ```

3. Embeddings may not be indexed:
   ```python
   playbook(action="index")
   ```

### Wrong bullets returned

Utility scores may be off. Check stats:

```python
playbook(action="learning_stats")
```

Prune low-quality bullets:

```python
playbook(action="prune", min_utility=0.3)
```

## LSP Issues

### "Language server not running"

Start the appropriate language server:

```bash
# Python
pip install python-lsp-server
pylsp

# TypeScript
npm install -g typescript-language-server
typescript-language-server --stdio
```

### Symbols not found

Re-index the codebase:

```python
project(action="analyze")
```

### Go to definition fails

LSP may not have analyzed the file:

```python
# Trigger analysis
lsp_get_symbols(path="the_file.py")
```

## Performance Issues

### Slow startup

Reduce tool profile:

```bash
DELIA_TOOLS=light uv run delia serve
```

### High memory usage

- Reduce ChromaDB collection size
- Use HTTP mode (single server) instead of stdio (per-project)

### Slow searches

Ensure embeddings are indexed:

```python
playbook(action="index")
project(action="analyze")
```

## Getting Help

### Check health

```bash
uv run delia doctor
```

### Enable debug logging

```bash
DELIA_LOG_LEVEL=DEBUG uv run delia run -t http
```

### Report issues

https://github.com/zbrdc/delia/issues

Include:
- Delia version
- Python version
- Error messages
- Steps to reproduce

## See Also

- [Installation](installation.md) - Setup guide
- [Configuration](configuration.md) - Settings reference
- [MCP Configuration](../reference/mcp-config.md) - Client setup
