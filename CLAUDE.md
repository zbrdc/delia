# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Delia is an MCP (Model Context Protocol) server that routes prompts to local LLMs. It provides intelligent model selection across multiple backends (Ollama, llama.cpp, Gemini) with automatic failover, circuit breaker protection, and per-user tracking.

## Build and Development Commands

```bash
# Install Delia and dependencies
uv sync
uv pip install -e .

# Run the MCP server (STDIO mode - default)
delia serve

# Run in HTTP/SSE mode
delia serve --transport sse --port 8200

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_backend_manager.py

# Run a single test
uv run pytest tests/test_backend_manager.py::test_function_name -v

# Run tests with custom data directory (for isolation)
DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest
```

## Architecture

### Core Components

- **`src/delia/mcp_server.py`**: Main entry point. Defines MCP tools (`delegate`, `think`, `batch`, `health`, `models`, `switch_backend`, `switch_model`, `get_model_info`). Contains LLM calling logic for Ollama, llama.cpp, and Gemini backends. Handles model selection based on task type and prompt analysis.

- **`src/delia/backend_manager.py`**: Manages backend configurations and health. `BackendConfig` represents a single backend with its models. `BackendManager` handles loading/saving settings, health checks with TTL caching, and backend switching.

- **`src/delia/config.py`**: Configuration models (`ModelConfig`, `Config`) and backend health tracking with circuit breaker logic (`BackendHealth`). Also contains model name parsing utilities to detect model tiers from names.

- **`src/delia/paths.py`**: Centralized path management. Uses `DELIA_DATA_DIR` env var if set, otherwise `~/.cache/delia`. All data files (stats, logs, database) go through this module.

- **`src/delia/schemas/`**: Typed Pydantic schemas for structured JSON tool interfaces. Contains `enums.py` (TaskType, ModelTier, Language, etc.), `requests.py` (CodeReviewRequest, etc.), and `responses.py` (StructuredResponse, etc.).

- **`src/delia/structured_tools.py`**: Structured MCP tools for LLM-to-LLM communication. Provides JSON input/output alternatives to the NLP-based tools.

### Tool Interfaces

Delia provides three interfaces:

**NLP Interface** (original): `delegate`, `think`, `batch`
- Text input with natural language task descriptions
- Text output with metadata footer

**Structured Interface**: `code_review`, `code_generate`, `code_analyze`, `structured_think`, `structured_delegate`, `batch_structured`
- JSON input with typed schemas (explicit content_type, language, model_tier)
- JSON output with usage metrics and execution info

**Garden-Themed Aliases** (fun alternatives):
| Standard | Garden | Description |
|----------|--------|-------------|
| `delegate` | `plant` | Plant a seed in the garden |
| `think` | `ponder` | Let thoughts grow slowly |
| `batch` | `harvest` | Gather multiple melons |
| `code_review` | `prune` | Examine vines for weeds |
| `code_generate` | `grow` | Cultivate fresh code |
| `code_analyze` | `tend` | Tend to the garden |
| `think(deep)` | `ruminate` | Deep contemplation |

Structured garden aliases: `prune_json`, `grow_json`, `tend_json`, `ponder_json`, `harvest_json`

### Model Tiers

The system routes requests to different model tiers based on task complexity:
- **quick**: Small models (7B-14B) for simple tasks
- **coder**: Code-specialized models (14B-30B)
- **moe**: Large MoE models (30B+) for complex reasoning
- **thinking**: Extended reasoning models (e.g., deepseek-r1)

### Authentication (Optional)

- **`src/delia/auth.py`**: FastAPI-Users integration with SQLite backend
- **`src/delia/multi_user_tracking.py`**: Per-user usage quotas and statistics
- Enable with `DELIA_AUTH_ENABLED=true` and `DELIA_JWT_SECRET`

### Configuration

Backend configuration lives in `settings.json` (created at first run). Structure:
```json
{
  "backends": [{"id", "name", "provider", "type", "url", "enabled", "priority", "models"}],
  "routing": {"prefer_local": true, "fallback_enabled": true}
}
```

### Data Storage

All persistent data stored in `DELIA_DATA_DIR` (default `~/.cache/delia`):
- `usage_stats.json`: Model usage counts
- `enhanced_stats.json`: Detailed statistics
- `circuit_breaker.json`: Backend failure states
- `live_logs.json`: Recent activity for dashboard
- `delia.db`: SQLite database for auth

## Testing Patterns

Tests use `DELIA_DATA_DIR` environment variable for isolation. Many tests mock HTTP responses using `httpx`'s mock transport. Authentication tests use in-memory SQLite.
