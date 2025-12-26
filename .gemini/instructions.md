# Delia - AI Coding Instructions

This file provides guidance for Google Gemini when working with this repository.

**MCP Configuration**: Configure in Gemini Code Assist settings

---

# delia Development Instructions

This file provides guidance for AI coding assistants working in this repository.

## Project Overview
- **Language:** Python
- **Frameworks:** FastAPI, Pydantic, structlog
- **Primary Tooling:** npm

## Build & Development Commands
```bash
# Install dependencies
npm install

# Run tests
pytest

# Lint code
ruff check
```

## Repository Patterns & Rules
- **Search First**: Before creating new code, search the codebase for similar patterns (DRY).
- **Type Safety**: Use type hints and Pydantic models where applicable.
- **Checkpoints**: Use available semantic tools to validate understanding before making modifications.
- **Atomic Commits**: Use descriptive commit messages: `type(scope): description`.

## Validation Checklist
- [ ] Code passes `ruff check`
- [ ] Tests pass `pytest`
- [ ] No placeholder implementations or temporary code left behind
- [ ] Documentation updated if public interfaces changed


---

## Gemini Specific Notes

### Context Window
Gemini has a large context window. However, still prefer LSP tools over reading
entire files to maintain efficiency and reduce noise.

### MCP Integration
Ensure the Delia MCP server is configured in your Gemini Code Assist settings.

---

## Subagent Fallback (No MCP Access)

If running as a subagent without MCP tool access, read `.delia/` files directly:
- `.delia/playbooks/*.json` - Task-specific bullets (coding, testing, etc.)
- `.delia/memories/*.md` - Persistent project knowledge
- `.delia/project_summary.json` - Project overview

The playbook bullets below are auto-embedded for convenience.

---

## PROJECT PLAYBOOK (Auto-embedded)

These are learned strategies from this project. Apply them to relevant tasks.
For the latest bullets, use `auto_context()` or read `.delia/playbooks/*.json` directly.

### Coding
- Use pathlib.Path over os.path for file operations
- MCP tools must return JSON-serializable dicts wrapped in result key
- Always pass project path explicitly - never assume cwd
- Add quality filters for indexed content (skip boilerplate, tiny fragments, admonitions) to reduce noise in search results
- Use httpx async client over requests for HTTP calls

### Testing
- Use pytest with async support via pytest-asyncio
- Mock external services (Ollama, LSP) in unit tests
- Integration tests go in tests/integration/
- Use fixtures for common setup patterns
- Test MCP tools via their handler functions directly

### Architecture
- MCP server is the primary interface - tools are registered via decorators
- Playbooks store per-project learned patterns in .delia/playbooks/
- Memories persist knowledge in .delia/memories/ as markdown
- LSP integration provides semantic code navigation
- Profiles are starter templates copied to .delia/profiles/

### Debugging
- Check structlog output for detailed traces
- MCP tool errors are returned in error key of result
- When verifying renames/refactors, search for multiple pattern types: imports, string literals, variable names, comments - not just one pattern
- When MCP tool changes aren't taking effect, check if server process is running cached old code - restart required
- LSP issues often stem from language server not running

### Project
- Primary language: Python 3.11+
- Package manager: uv with pyproject.toml
- MCP server for AI agent integration
- Dashboard: Next.js app in dashboard/
- CLI entry point: delia command via cli.py

### Git
- Commit messages should be descriptive of the change
- Use conventional commits format when possible
- Don't commit .delia/data/ or session files
- Playbooks and profiles are project-specific and should be committed

### Security
- Never log or expose API keys
- Validate all file paths to prevent traversal
- MCP tools run with user permissions - be cautious with file ops
- Settings files may contain sensitive backend URLs

### Deployment
- MCP server runs via stdio for AI agent integration
- REST API available via delia api command
- Dashboard runs separately on port 8765
- Ollama must be running for LLM delegation features

### Api
- MCP tools follow Model Context Protocol specification
- REST API uses FastAPI with automatic OpenAPI docs
- All endpoints return JSON responses
- Use proper HTTP status codes for errors
- When testing Delia Framework, verify the complete loop: auto_context detection → bullet loading → profile loading → task execution → complete_task feedback. Each component must integrate seamlessly.

### Performance
- Use async/await for all I/O operations
- LSP operations can be slow - cache results when appropriate
- Batch LLM calls when possible via batch() tool
- Dashboard uses React Query for efficient data fetching
