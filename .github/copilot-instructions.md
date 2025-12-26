# Delia Development Instructions

## Project Overview

- **Language**: Python 3.11+
- **Package Manager**: uv with pyproject.toml
- **Frameworks**: FastAPI, Pydantic, structlog
- **Entry Point**: `delia` CLI via cli.py

## Commands

```bash
uv sync              # Install dependencies
uv run pytest        # Run tests
uv run ruff check    # Lint code
delia doctor         # Health check
```

## Patterns

- Search codebase before creating new code (DRY)
- Use type hints and Pydantic models
- Use pathlib.Path over os.path
- Use httpx async client over requests
- MCP tools must return JSON-serializable dicts

## Validation

- [ ] `ruff check` passes
- [ ] `pytest` passes
- [ ] No placeholder implementations
- [ ] Public interface changes documented

---

## Copilot MCP Configuration

Add to VS Code settings:
```json
{
  "github.copilot.chat.experimental.mcpServers": {
    "delia": {
      "command": "delia",
      "args": ["serve"]
    }
  }
}
```

---

## Playbook (Auto-embedded)

These patterns are learned from this project. For latest, read `.delia/playbooks/`.

### Coding
- Use pathlib.Path over os.path for file operations
- MCP tools must return JSON-serializable dicts wrapped in result key
- Always pass project path explicitly - never assume cwd
- Use httpx async client over requests for HTTP calls

### Testing
- Use pytest with async support via pytest-asyncio
- Mock external services (Ollama, LSP) in unit tests
- Integration tests go in tests/integration/
- Test MCP tools via their handler functions directly

### Architecture
- MCP server is the primary interface - tools are registered via decorators
- Playbooks store per-project learned patterns in .delia/playbooks/
- LSP integration provides semantic code navigation
- Memories persist knowledge in .delia/memories/ as markdown

### Debugging
- Check structlog output for detailed traces
- MCP tool errors are returned in error key of result
- When MCP tool changes aren't taking effect, restart the server
- LSP issues often stem from language server not running

### Git
- Use conventional commits: `type(scope): description`
- Don't commit .delia/data/ or session files
- Playbooks and profiles should be committed
