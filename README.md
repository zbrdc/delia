# Delia: Autonomous Cognitive Entity (ACE) & Local LLM Orchestrator

Delia is a high-performance executive agent and orchestration system for local Large Language Models (LLMs). Designed as a "Tactical ACE" (Autonomous Cognitive Entity), it bridges the gap between simple chatbots and fully autonomous agents by providing a robust operational layer for tool use, strategic planning, and self-correction.

It functions as a unified CLI workspace that integrates local inference (Ollama, llama.cpp) with an advanced routing and execution engine.

## Core Capabilities

-   **Unified Orchestration:** Seamlessly routes tasks between specialized model tiers ("Quick", "Coder", "MoE", "Thinking") based on complexity and intent.
-   **ACE Architecture:** Implements a persistent self-improvement loop:
    -   **Global Strategy:** Learns from execution failures via a `Reflector` -> `Curator` -> `Playbook` cycle.
    -   **Cognitive Control:** Uses tiered NLP (Regex -> Semantic -> LLM) for precise intent detection.
    -   **Task Prosecution:** Robust tool suite for file I/O, shell execution, and web search.
-   **Native Agentic Loop:** A continuous, stateful Read-Eval-Print Loop (REPL) that maintains conversation history while executing multi-step autonomous plans.
-   **Mathematical Reliability:** Implements K-voting consensus algorithms (verified via Wolfram Alpha) to guarantee response accuracy for critical tasks.
-   **Melon Economy:** A performance-based routing weight system that dynamically prioritizes models with a proven track record of quality.

## Installation

Delia is a Python package managed via `uv` or `pip`.

```bash
# Install via uv (recommended for speed)
uv pip install -e .

# Or standard pip
pip install -e .
```

## Usage

### Interactive Chat (Default)

Launch the native TUI (Text User Interface) for an interactive session. This mode supports full agent capabilities, including planning, tool use, and file system access.

```bash
delia
# or
delia chat
```

**Key Features in Chat:**
-   **Standard Chat:** "How do I reverse a list in Python?"
-   **Agentic Tasks:** "Refactor `src/main.py` to use async/await." (Triggers planning & tools)
-   **Interruption:** Press `Ctrl+\` to pause the agent mid-execution and inject new instructions.
-   **Termination:** Press `Ctrl+C` to exit.

### Single-Shot Agent

Execute a specific task non-interactively and exit. Useful for scripting or CI/CD integration.

```bash
delia agent "Scan the current directory for security vulnerabilities"
```

### Configuration

Initialize the configuration wizard to detect local backends (Ollama, etc.) and set up model tiers.

```bash
delia init
```

Configuration is stored in `~/.delia/settings.json` and supports hot-reloading.

## Architecture

Delia operates on a 6-layer ACE framework:

1.  **Aspirational:** Defined via `data/constitution.md` (Mission & Values).
2.  **Global Strategy:** Strategic playbooks (`data/playbooks/*.json`) accumulated from past interactions.
3.  **Agent Model:** Dynamic awareness of available backends and tool capabilities.
4.  **Executive Function:** The `OrchestrationExecutor` manages resource allocation and task routing.
5.  **Cognitive Control:** The `IntentDetector` filters and directs input.
6.  **Task Prosecution:** The `AgentLoop` executes actions via the `ToolRegistry`.

## License

GPL-3.0