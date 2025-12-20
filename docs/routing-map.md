# Delia System Architecture & Routing Logic

This document details the internal architecture, decision logic, and data flow of the Delia system.

## 1. High-Level Architecture

Delia functions as an intelligent orchestration layer between user inputs and backend LLM providers.

```mermaid
flowchart TB
    subgraph entry [Entry Points]
        CLI[CLI (Native TUI)]
        API[HTTP API]
        MCP[MCP Protocol]
    end
    
    subgraph orchestration [Orchestration Layer]
        ID[IntentDetector]
        EX[OrchestrationExecutor]
        PM[PromptGenerator]
    end
    
    subgraph routing [Routing Layer]
        MR[ModelRouter]
        BS[BackendScorer]
        AT[AffinityTracker]
    end
    
    subgraph memory [State & Memory]
        PB[PlaybookManager]
        SM[Serena Memory]
        MT[MelonTracker]
    end
    
    subgraph backends [Inference]
        OL[Ollama]
        LC[Llama.cpp]
        EXT[External APIs]
    end
    
    CLI --> ID
    API --> ID
    MCP --> ID
    
    ID --> EX
    EX --> PB
    EX --> SM
    EX --> PM
    
    EX --> MR
    MR --> BS
    BS --> AT
    BS --> MT
    
    MR --> backends
```

---

## 2. Intent Detection Pipeline

Input is processed through a 3-tier NLP pipeline to determine the optimal execution strategy.

| Layer | Mechanism | Latency | Purpose |
|-------|-----------|---------|---------|
| **1. Regex** | Pre-compiled patterns | ~0ms | Detects explicit commands ("run", "scan") and known triggers. |
| **2. Semantic** | `sentence-transformers` | ~50ms | Detects intent via embedding similarity (paraphrasing). |
| **3. LLM** | Dispatcher Model (270M) | ~500ms | Classifies complex or ambiguous requests. |

### Detected Intents

-   **Quick:** Simple Q&A, summarization.
-   **Coder:** Code generation, debugging, refactoring.
-   **MoE (Mixture of Experts):** Complex reasoning, planning, architectural design.
-   **Agentic:** Tasks requiring tool execution (File I/O, Shell).
-   **Voting:** Tasks requiring high reliability validation.

---

## 3. Orchestration Modes

The `OrchestrationExecutor` routes the request based on the detected intent.

| Mode | Behavior | Use Case |
|------|----------|----------|
| **NONE** | Single pass generation. | General chat, simple questions. |
| **AGENTIC** | Recursive loop: Think → Tool → Observe. | "Refactor src/", "Search web for X". |
| **VOTING** | Generates K samples, selects consensus. | "Verify this logic", "Check for bugs". |
| **COMPARISON** | Queries multiple models in parallel. | "Compare views on X", "Second opinion". |
| **DEEP_THINKING** | Forces Chain-of-Thought (CoT). | "Plan architecture", "Solve puzzle". |

---

## 4. Routing & Model Selection

Delia dynamically selects the backend and model tier based on task requirements and performance metrics.

### Model Tiers
-   **Quick:** Low latency, low VRAM (e.g., `qwen2.5:1.5b`).
-   **Coder:** Specialized for code tasks (e.g., `deepseek-coder:6.7b`).
-   **Thinking:** Reasoning-heavy models (e.g., `deepseek-r1-distill`).
-   **MoE:** Large context/parameter models for complex synthesis.

### Scoring Algorithm
Backends are scored (0.0 - 1.0) based on weighted factors:
1.  **Latency (35%):** Inverse of P50 response time.
2.  **Reliability (35%):** Success rate of recent requests.
3.  **Throughput (15%):** Tokens per second.
4.  **Affinity:** Boost for backends that historically succeed at the specific task type.
5.  **Melon Score:** Dynamic boost based on model performance history.

---

## 5. Self-Correction (The ACE Loop)

Delia implements an autonomous feedback loop for continuous improvement.

1.  **Execution:** Task is performed.
2.  **Validation:** Output is checked against requirements/tests.
3.  **Reflection:** If failed/low-quality, the `Reflector` analyzes the root cause.
4.  **Curation:** The `Curator` extracts a generalized lesson.
5.  **Persistence:** Lesson is saved to `data/playbooks/`.
6.  **Injection:** Lesson is injected into the prompt context for future runs.

---

## 6. Data Structures

### Playbooks (`data/playbooks/*.json`)
JSON-based storage for learned strategies.
```json
{
  "id": "strat-001",
  "content": "When using grep, always check if directory exists first.",
  "section": "terminal_usage",
  "helpful_count": 5
}
```

### Constitution (`data/constitution.md`)
Markdown definition of high-level agent values and operational constraints.

### Melon Tracker
Performance tracking database.
-   **Melon:** Unit of positive reinforcement (successful task).
-   **Golden Melon:** 500 Melons. Signifies a highly trusted model.