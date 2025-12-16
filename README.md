# Delia

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.23.0-green)](https://modelcontextprotocol.io/)
[![Tests](https://img.shields.io/badge/tests-910+-brightgreen)](tests/)

**Delia** is an enterprise-grade Model Context Protocol (MCP) server that orchestrates your local LLM infrastructure. It acts as an intelligent router and delegation layer, automatically selecting the best model tier ("Quick", "Coder", "MoE", or "Thinking") for a given task using semantic embeddings, latency-aware scoring, and learned task affinities.

Delia transforms a collection of local models (via Ollama, llama.cpp, vLLM) and optional cloud fallbacks (Gemini, OpenAI) into a cohesive, fault-tolerant AI system available to any MCP-compliant client (Claude Desktop, VS Code, Cursor, Windsurf, etc.).

**Project Stats:** ~17K LOC | 910+ tests | 3 providers | 4 model tiers

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
- **K-Voting Consensus**: MDAP-based reliability guarantees for agentic tasks

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
- **K-Voting Consensus**: Mathematical reliability guarantees (see [Routing Intelligence](#routing-intelligence))

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

## CLI Commands

Delia provides a rich command-line interface for setup, diagnostics, and direct interaction.

### Server Commands

| Command | Description |
|---------|-------------|
| `delia serve` | Start the MCP server (STDIO mode, default) |
| `delia serve -t sse` | Start in HTTP/SSE mode for remote access |
| `delia api` | Start HTTP API server for CLI frontend |

### Setup & Configuration

| Command | Description |
|---------|-------------|
| `delia init` | Interactive setup wizard |
| `delia doctor` | Diagnose configuration and connectivity |
| `delia install [client]` | Install to MCP client (claude, vscode, cursor, etc.) |
| `delia install --list` | List available clients |
| `delia config --show` | Show current configuration |
| `delia config --edit` | Edit configuration in editor |
| `delia uninstall [client]` | Remove from client configuration |

### Agent Commands

```bash
# Run autonomous agent for a task
delia agent "What files are in the src directory?"
delia agent "Find all TODO comments in the codebase"
delia agent "Summarize main.py" --workspace ./myproject

# With options
delia agent "Analyze error handling patterns" \
    --model moe \              # Force model tier
    --workspace ./project \    # Confine to directory
    --max-iterations 15 \      # Limit iterations
    --backend local \          # Force local backend
    --voting                   # Enable k-voting consensus
    --voting-k 3               # Require 3 matching votes
```

### Interactive Chat

Delia includes a TypeScript CLI for rich terminal interaction:

```bash
# Start interactive chat session
cd packages/cli
npm install
npm run build
node dist/index.js chat

# Chat options
node dist/index.js chat --model coder    # Use coder tier
node dist/index.js chat --session abc123 # Resume session
node dist/index.js chat --api-url http://localhost:8201
```

The chat CLI connects to the Delia API server (`delia api`) for SSE streaming.

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

## Routing Intelligence

Delia implements several advanced routing algorithms for optimal backend selection and reliability.

### Backend Scoring

The `BackendScorer` evaluates backends using configurable weights:

| Factor | Default Weight | Description |
|--------|---------------|-------------|
| **Latency** | 0.35 | Lower P50 latency = higher score |
| **Reliability** | 0.35 | Success rate (successes / total requests) |
| **Throughput** | 0.15 | Tokens per second |
| **Availability** | 0.15 | Circuit breaker state |
| **Cost** | 0.0 | Provider cost (disabled by default) |

Scoring formula uses inverse relationships for latency:
- `latency_score = 1 / (1 + latency_ms / 1000)`
- 0ms = 1.0, 500ms = 0.67, 1000ms = 0.5, 2000ms = 0.33

### Task-Backend Affinity Learning

`AffinityTracker` learns which backends excel at specific task types using Exponential Moving Average (EMA):

```
new_score = old_score × (1 - α) + quality × α
```

Where:
- `α = 0.1` (decay factor, configurable)
- `quality` = 1.0 for success, 0.0 for failure, or continuous 0.0-1.0 score

Affinity boosts backend scores for tasks they historically perform well on.

### Predictive Pre-warming

`PrewarmTracker` learns hourly usage patterns to pre-load models before they're needed:

- Tracks (hour, tier) → EMA score
- Recommends pre-warming when score exceeds threshold (default 0.3)
- Adapts with `α = 0.15` for moderate responsiveness

### K-Voting Consensus (MDAP)

For high-reliability requirements, Delia implements k-voting consensus from the MDAP paper "Smashing Intelligence into a Million Pieces":

```
P(correct) = 1 / (1 + ((1-p)/p)^k)
```

With base model accuracy p=0.99 and k=3 votes:
- **P(correct) = 0.999999** (six nines reliability)

**How it works:**
1. Send same request to multiple backends
2. Normalize and compare responses (85% similarity threshold)
3. First response to get k matching votes wins
4. Red-flag responses that are too long (>700 tokens) or show quality issues

**Quality validation includes:**
- Repetition detection (n-gram loops, sentence repeats)
- Length validation (min/max thresholds)
- Coherence checks (vocabulary diversity, encoding issues)
- Hallucination detection (false capability claims)

Enable with `--voting` flag or in configuration.

### Response Quality Scoring

`ResponseQualityValidator` provides continuous quality scores (0.0-1.0) for affinity learning:

| Component | Weight | Checks |
|-----------|--------|--------|
| **Repetition** | 0.30 | N-gram loops, sentence repeats, token stuttering |
| **Coherence** | 0.30 | Vocabulary diversity, encoding issues, truncation |
| **Length** | 0.20 | Min/max thresholds, whitespace ratio |
| **Hallucination** | 0.20 | False capability claims, refusal patterns |

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
    },
    "voting": {
      "enabled": false,
      "k": 2,
      "similarity_threshold": 0.85
    },
    "quality": {
      "enabled": true,
      "repetition_ngram_size": 5,
      "repetition_threshold": 3,
      "min_response_length": 10,
      "min_vocabulary_diversity": 0.15
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
| `voting.enabled` | `false` | Enable k-voting consensus |
| `voting.k` | `2` | Votes needed for consensus |
| `quality.enabled` | `true` | Enable response quality scoring |

### Key Components

| Component | Purpose |
|-----------|---------|
| **BackendScorer** | Scores backends using configurable weights (latency, throughput, reliability, cost) |
| **AffinityTracker** | EMA-based learning of task→backend performance |
| **PrewarmTracker** | Hourly usage pattern learning for predictive model loading |
| **VotingConsensus** | K-voting for mathematical reliability guarantees |
| **ResponseQualityValidator** | Continuous quality scoring for affinity learning |
| **ModelQueue** | Prevents OOM by serializing model loads |
| **CircuitBreaker** | Protects against cascading failures with auto-recovery |

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

## Project Structure

```
src/delia/                    # ~17K LOC
├── mcp_server.py             # Main entry, MCP tools
├── cli.py                    # CLI commands (init, serve, doctor, agent)
├── api.py                    # HTTP API for CLI frontend
├── backend_manager.py        # Backend config and health
├── session_manager.py        # Session state for multi-turn
├── delegation.py             # Core delegation logic
├── config.py                 # Configuration, circuit breaker, metrics
├── routing.py                # Model/backend routing, BackendScorer
├── voting.py                 # K-voting consensus (MDAP)
├── quality.py                # Response quality validation
├── llm.py                    # LLM call dispatcher
├── providers/                # Ollama, llama.cpp, Gemini
└── tools/                    # Agent tools (read_file, search_code, etc.)

packages/cli/                 # TypeScript CLI frontend
├── src/
│   ├── commands/
│   │   ├── agent.tsx         # Agent command
│   │   └── chat.tsx          # Interactive chat
│   └── components/           # React Ink UI components

tests/                        # ~12K LOC, 33+ test files
dashboard/                    # Next.js monitoring UI
```

## License

Delia is licensed under the **GNU General Public License v3 (GPLv3)**.

You are free to use, modify, and distribute this software, but all modifications and derived works must also be open-source under the same license. See [LICENSE](LICENSE) for details.
