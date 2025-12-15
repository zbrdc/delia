# Delia

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.23.0-green)](https://modelcontextprotocol.io/)

**Delia** is an intelligent Model Context Protocol (MCP) server that orchestrates your local LLM infrastructure. It acts as a smart router and delegation layer, automatically selecting the best model tier ("Quick", "Coder", "MoE", or "Thinking") for a given task, balancing performance, quality, and resource usage.

Delia turns a collection of local models (via Ollama, llama.cpp, vLLM) and optional cloud fallbacks (Gemini, OpenAI) into a cohesive, fault-tolerant AI system available to any MCP-compliant client (Claude Desktop, VS Code, Cursor, Windsurf, etc.).

## Features

- **🧠 Intelligent Routing**: Automatically routes tasks to the optimal model tier:
    - **Quick**: Fast 7B-14B models for summaries and simple queries.
    - **Coder**: Specialized 14B-30B models for code generation and review.
    - **MoE** (Mixture of Experts): Large 30B+ models for complex planning and architectural critique.
    - **Thinking**: Models with extended reasoning capabilities (e.g., DeepSeek-R1) for deep analysis.
- **🔌 Multi-Backend Support**: Seamlessly integrates:
    - **Ollama** (Recommended for ease of use)
    - **llama.cpp / server** (For maximum performance)
    - **vLLM** (For high-throughput production setups)
    - **Google Gemini** (Optional cloud fallback)
    - **OpenAI-compatible APIs** (Local or remote)
- **🛠️ Powerful MCP Tools**:
    - `delegate`: The core tool for routing tasks to the right model.
    - `think`: Dedicated tool for deep, multi-step reasoning on complex problems.
    - `batch`: Parallel execution of multiple tasks across available GPUs.
    - `chain` & `workflow`: Execute sequential or DAG-based task pipelines.
    - `agent`: Autonomous agent capable of multi-step tool use.
- **🛡️ Resilience**:
    - **Circuit Breaker**: Automatically disables failing backends to prevent cascading errors.
    - **Queue System**: Manages concurrency to prevent OOM errors on local hardware.
    - **Failover**: Automatically retries on alternative backends if primary fails.
- **📊 Observability**:
    - **Real-time Dashboard**: Next.js-based dashboard for monitoring requests, token usage, and backend health.
    - **Usage Tracking**: Detailed stats on tokens, costs (saved vs. cloud), and model performance.
- **🔐 Enterprise Ready**:
    - **Authentication**: Optional JWT-based auth with per-user quotas.
    - **Session Management**: Stateful conversations with `session_create/get/list`.

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
    "fallback_enabled": true
  }
}
```

### Model Tiers

-   **Quick**: General purpose, low latency. *Triggers: "summarize", "explain", simple queries.*
-   **Coder**: Code specialization. *Triggers: "write code", "refactor", "review", "test".*
-   **MoE**: Complex reasoning, large context. *Triggers: "plan", "critique", "architecture".*
-   **Thinking**: Extended chain-of-thought. *Triggers: "think", "reason", "analyze deeply".*

## Dashboard

Delia includes a real-time Next.js dashboard for monitoring request activity, token usage, and system health.

**To start the dashboard:**

1.  Navigate to the dashboard directory:
    ```bash
    cd dashboard
    ```

2.  Install dependencies:
    ```bash
    npm install
    # or
    yarn install
    ```

3.  Run the development server:
    ```bash
    npm run dev
    ```

4.  Open [http://localhost:3000](http://localhost:3000) in your browser.

The dashboard connects to Delia's log files (typically in `~/.cache/delia/`) to display live activity.

## Development

To develop on Delia:

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/zbrdc/delia.git
    cd delia
    ```

2.  **Install dependencies:**
    ```bash
    uv sync
    ```

3.  **Run in dev mode:**
    ```bash
    uv run delia serve
    ```

4.  **Run tests:**
    ```bash
    uv run pytest
    ```

## License

Delia is licensed under the **GNU General Public License v3 (GPLv3)**.

You are free to use, modify, and distribute this software, but all modifications and derived works must also be open-source under the same license. See [LICENSE](LICENSE) for details.