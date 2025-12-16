# Delia

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.23.0-green)](https://modelcontextprotocol.io/)
[![Tests](https://img.shields.io/badge/tests-746+-brightgreen)](tests/)

**Delia** is an enterprise-grade Model Context Protocol (MCP) server that orchestrates your local LLM infrastructure. It acts as an intelligent router and delegation layer, automatically selecting the best model tier ("Quick", "Coder", "MoE", or "Thinking") for a given task using semantic embeddings, latency-aware scoring, and learned task affinities.

Delia transforms a collection of local models (via Ollama, llama.cpp, vLLM) and optional cloud fallbacks (Gemini, OpenAI) into a cohesive, fault-tolerant AI system available to any MCP-compliant client (Claude Desktop, VS Code, Cursor, Windsurf, etc.).

**Project Stats:** ~17K LOC | 746+ tests | 3 providers | 4 model tiers

## Features

### Intelligent Routing

Delia uses a sophisticated multi-tier routing system:

| Tier | Models | Use Cases |
|------|--------|-----------|
| **Quick** | 7B-14B | Summaries, simple queries, triage |
| **Coder** | 14B-30B | Code generation, review, refactoring |
| **MoE** | 30B+ | Complex planning, architecture, critique |
| **Thinking** | Extended reasoning | Deep analysis (e.g., DeepSeek-R1) |

**Advanced Routing Features:**
- **Semantic Model Selection**: Uses embeddings (nomic-embed) for intelligent task→model matching
- **Latency-Aware Scoring**: Tracks P50 latency, throughput, and success rates per backend
- **Cost-Aware Routing**: Prefers local/free backends, with configurable cloud cost sensitivity
- **Task-Backend Affinity**: Learns which backends excel at which task types via EMA
- **Weighted Load Balancing**: Distributes load across backends based on performance scores
- **Hedged Requests**: Staggered parallel execution for latency-critical requests
- **Predictive Pre-warming**: Loads models before they're needed based on hourly usage patterns

### Multi-Backend Support

| Provider | Type | Notes |
|----------|------|-------|
| **Ollama** | Local | Recommended for ease of use |
| **llama.cpp** | Local | Maximum performance |
| **vLLM** | Local/Remote | High-throughput production |
| **Gemini** | Cloud | Optional fallback |
| **OpenAI-compatible** | Any | Works with any compatible API |

### MCP Tools

| Tool | Purpose |
|------|---------|
| `delegate` | Route tasks to optimal model tier |
| `think` | Deep multi-step reasoning with extended thinking |
| `batch` | Parallel execution across GPUs |
| `chain` | Sequential task pipelines with variable substitution |
| `workflow` | DAG execution with conditional branching and retry |
| `agent` | Autonomous agent with native tool calling |
| `session_*` | Stateful multi-turn conversations |

### Resilience & Reliability

- **Circuit Breaker**: Disables failing backends with exponential backoff recovery
- **Model Queue**: Prevents OOM by managing concurrent model loads
- **Automatic Failover**: Retries on alternative backends when primary fails
- **Health Monitoring**: Continuous backend health checks with TTL caching

### Observability

- **Real-time Dashboard**: Next.js UI for monitoring requests, tokens, and backend health
- **Backend Metrics**: Success rate, latency P50, throughput (tok/s) per backend
- **Routing Intelligence**: View affinity scores, prewarm predictions, hedging status
- **Cost Tracking**: Estimated savings vs. cloud API costs

### Enterprise Features

- **JWT Authentication**: Optional auth with FastAPI-Users
- **Per-User Quotas**: Track and limit usage by user
- **Session Management**: Persistent conversation state

## Quick Start

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (Recommended package manager)
- A running local LLM backend (e.g., [Ollama](https://ollama.ai/))

### Installation

1.  **Clone and Install:**
    ```bash
    git clone https://github.com/zbrdc/delia.git
    cd delia
    uv tool install .
    ```

2.  **Initialize & Configure:**
    Run the setup wizard to detect your local models and configure MCP clients.
    ```bash
    delia init
    ```
    This command will:
    - Detect running backends (Ollama, etc.).
    - Auto-assign models to tiers (Quick, Coder, MoE).
    - Generate a `~/.delia/settings.json` configuration.
    - Offer to auto-configure detected clients (Claude, VS Code, etc.).

### Using with MCP Clients

If you didn't use the auto-install feature in `delia init`, you can manually add Delia to your client configuration.

**Command:** `delia`
**Args:** `serve`

**Example `mcp.json` (Claude Desktop):**
```json
{
  "mcpServers": {
    "delia": {
      "command": "delia",
      "args": ["serve"]
    }
  }
}
```

## Usage

### Core Tools

Once connected to an MCP client (like Claude), you can use natural language to leverage Delia's power.

-   **Delegation (Smart Routing):**
    > "Review this code using your coder model."
    > "Plan a microservices architecture using the MoE model."
    
    *Behind the scenes, Delia uses the `delegate` tool:*
    ```python
    delegate(task="review", content="...", model="coder")
    ```

-   **Deep Thinking:**
    > "Think deeply about the potential race conditions in this async logic."
    
    *Delia uses the `think` tool:*
    ```python
    think(problem="Analyze race conditions...", depth="deep")
    ```

-   **Batch Processing:**
    > "Summarize these 5 files in parallel."
    
    *Delia uses the `batch` tool to distribute work:*
    ```python
    batch(tasks='[{"task": "summarize", "content": "..."}, ...]')
    ```

### CLI Commands

-   `delia init`: Run the setup wizard.
-   `delia doctor`: Diagnose configuration and connectivity issues.
-   `delia install [client]`: Install Delia to a specific client (e.g., `delia install vscode`).
-   `delia serve`: Start the MCP server (STDIO mode by default).
-   `delia run --transport sse`: Start the server in HTTP/SSE mode (useful for remote access).
-   `delia config --edit`: Edit the global configuration.

## Configuration

Configuration is stored in `~/.delia/settings.json`.

**Example Structure:**
```json
{
  "backends": [
    {
      "id": "ollama-local",
      "provider": "ollama",
      "type": "local",
      "url": "http://localhost:11434",
      "models": {
        "quick": "qwen2.5:14b",
        "coder": "qwen2.5-coder:14b",
        "moe": "qwen2.5:32b",
        "thinking": "deepseek-r1:14b"
      }
    }
  ],
  "routing": {
    "prefer_local": true,
    "fallback_enabled": true,
    "scoring": {
      "enabled": true,
      "weights": {
        "latency": 0.35,
        "throughput": 0.15,
        "reliability": 0.35,
        "availability": 0.15,
        "cost": 0.0
      }
    },
    "hedging": {
      "enabled": false,
      "delay_ms": 50,
      "max_backends": 2
    },
    "prewarm": {
      "enabled": false,
      "threshold": 0.3,
      "check_interval_minutes": 5
    },
    "affinity_learning": {
      "enabled": true,
      "alpha": 0.1
    }
  }
}
```

### Model Tiers

| Tier | Triggers | Best For |
|------|----------|----------|
| **Quick** | "summarize", "explain", simple queries | Low latency responses |
| **Coder** | "write code", "refactor", "review", "test" | Code-specialized tasks |
| **MoE** | "plan", "critique", "architecture" | Complex reasoning |
| **Thinking** | "think", "reason", "analyze deeply" | Extended chain-of-thought |

### Routing Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `prefer_local` | `true` | Prefer local backends over cloud |
| `fallback_enabled` | `true` | Allow fallback to other backends |
| `scoring.enabled` | `true` | Use performance-based backend scoring |
| `hedging.enabled` | `false` | Enable hedged (parallel) requests |
| `prewarm.enabled` | `false` | Enable predictive model pre-warming |
| `affinity_learning.enabled` | `true` | Learn task→backend affinities |

## Architecture

Delia's architecture has evolved through multiple phases to achieve enterprise-grade reliability:

| Component | Sophistication | Implementation |
|-----------|---------------|----------------|
| Model Selection | Advanced | Semantic embeddings + regex + task-based |
| Provider Abstraction | Advanced | Protocol-based with 3 providers |
| Resilience | Advanced | Circuit breaker + exponential backoff |
| Orchestration | Advanced | Chains, DAGs, conditional branching |
| Backend Selection | Advanced | Latency/cost/affinity-aware scoring |
| Load Distribution | Advanced | Weighted random with hedging support |

### Key Components

- **BackendScorer**: Scores backends using configurable weights (latency, throughput, reliability, cost)
- **AffinityTracker**: EMA-based learning of task→backend performance
- **PrewarmTracker**: Hourly usage pattern learning for predictive model loading
- **ModelQueue**: Prevents OOM by serializing model loads
- **CircuitBreaker**: Protects against cascading failures with auto-recovery

## Dashboard

Delia includes a real-time Next.js dashboard for monitoring.

**To start:**

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

### Dashboard Features

| Section | Data Shown |
|---------|------------|
| **Usage Stats** | Calls/tokens per tier, cost savings |
| **Backend Health** | Availability, response time, loaded models, circuit status |
| **Performance Metrics** | Success rate, latency P50, throughput per backend |
| **Routing Config** | Scoring weights, hedging toggle, prewarm toggle |
| **Intelligence** | Affinity pairs learned, prewarm predictions |
| **Recent Calls** | Task type, tokens, elapsed time |

The dashboard reads from `~/.cache/delia/` (live logs, affinity.json, prewarm.json).

## Development

### Setup

```bash
git clone https://github.com/zbrdc/delia.git
cd delia
uv sync
```

### Running

```bash
# Development mode (STDIO)
uv run delia serve

# HTTP/SSE mode
uv run delia serve --transport sse --port 8200

# Check configuration
uv run delia doctor
```

### Testing

```bash
# Run all tests (recommended: use isolated data directory)
DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest

# Run with coverage
uv run pytest --cov=delia

# Run a specific test file
uv run pytest tests/test_backend_manager.py

# Run async tests only
uv run pytest -m asyncio
```

**Test Stats:** 746+ tests across 33 test files

### Property-Based Testing (Hypothesis)

```bash
# Quick iteration
HYPOTHESIS_PROFILE=quick uv run pytest

# Extended fuzzing (overnight)
HYPOTHESIS_PROFILE=overnight uv run pytest
```

| Profile | Examples | Use Case |
|---------|----------|----------|
| quick | 10 | Fast iteration |
| default | 100 | Normal development |
| ci | 500 | CI pipeline |
| overnight | 2000 | Extended fuzzing |

### Project Structure

```
src/delia/                    # ~17K LOC
├── mcp_server.py             # Main entry, MCP tools
├── cli.py                    # CLI commands (init, serve, doctor)
├── backend_manager.py        # Backend config and health
├── session_manager.py        # Session state for multi-turn
├── delegation.py             # Core delegation logic
├── config.py                 # Configuration, circuit breaker
├── routing.py                # Model/backend routing
├── llm.py                    # LLM call dispatcher
├── providers/                # Ollama, llama.cpp, Gemini
└── tools/                    # Agent tools (read_file, search_code, etc.)

tests/                        # ~12K LOC, 33 test files
dashboard/                    # Next.js monitoring UI
```

## License

Delia is licensed under the **GNU General Public License v3 (GPLv3)**.

You are free to use, modify, and distribute this software, but all modifications and derived works must also be open-source under the same license. See [LICENSE](LICENSE) for details.