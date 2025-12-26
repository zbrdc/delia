# ADR-006: Agent-Aware Model Routing

## Status
Proposed

## Context

Current HuggingFace trends show that **agent-trained models** significantly outperform instruction-tuned models for tool-use tasks:

| Model | Training | Terminal-Bench | SWE-Bench |
|-------|----------|----------------|-----------|
| Qwen3-8B (base) | Instruction | 0.0 | 0.7% |
| OpenThinker-Agent-v1 | Execution traces | **4.9** | **15.7%** |
| DeepSWE-32B | RL on SWE tasks | - | **42.2%** |
| DeepCoder-14B | RL on code traces | - | 60.6% LiveCodeBench |

**Key insight**: Models trained on execution traces (terminal commands, tool outputs, diffs) have dramatically better tool compliance and task completion rates than generic instruction-tuned models.

Delia's current architecture routes agentic tasks to the `coder` tier, which uses generic code models. This is suboptimal.

## Decision

Add two new model tiers optimized for agent workflows:

### 1. `agentic` Tier
- **Purpose**: Tool-calling loops, terminal operations, file manipulation
- **Ideal Models**: OpenThinker-Agent-v1, agent-fine-tuned variants
- **Trigger**: `OrchestrationMode.AGENTIC` detection
- **Key Property**: High tool compliance, deterministic outputs

### 2. `swe` Tier
- **Purpose**: Multi-file refactoring, repo-scale reasoning, architecture changes
- **Ideal Models**: DeepSWE-Preview, SWE-bench optimized models
- **Trigger**: Complex codebase operations detected
- **Key Property**: Long-context reasoning, diff generation

## Architecture Changes

### 1. Config Layer (`config.py`)

```python
# New tier definitions (after line 181)
model_agentic: ModelConfig = field(
    default_factory=lambda: ModelConfig(
        name="agentic",
        default_model="auto",
        vram_gb=float(os.getenv("DELIA_MODEL_AGENTIC_VRAM", "-1")),
        context_tokens=-1,
        num_ctx=-1,
        max_input_kb=int(os.getenv("DELIA_MODEL_AGENTIC_INPUT_KB", "64")),
    )
)

model_swe: ModelConfig = field(
    default_factory=lambda: ModelConfig(
        name="swe",
        default_model="auto",
        vram_gb=float(os.getenv("DELIA_MODEL_SWE_VRAM", "-1")),
        context_tokens=-1,
        num_ctx=-1,
        max_input_kb=int(os.getenv("DELIA_MODEL_SWE_INPUT_KB", "200")),  # Large for repo context
    )
)

# New task sets (after line 205)
agentic_tasks: frozenset[str] = field(
    default_factory=lambda: frozenset({"agent", "tool", "execute", "terminal"})
)

swe_tasks: frozenset[str] = field(
    default_factory=lambda: frozenset({"refactor", "migrate", "architect", "redesign"})
)
```

### 2. Model Detection (`model_detection.py`)

```python
# Updated TIER_KEYWORDS (line 48)
TIER_KEYWORDS = {
    "dispatcher": ["functiongemma"],
    "thinking": ["think", "reason", "r1", "o1", "deepseek-r"],
    "agentic": ["agent", "openthinker", "tool-use", "function-call"],  # NEW
    "swe": ["swe", "deepswe", "software-engineer"],  # NEW
    "coder": ["code", "coder", "codestral", "starcoder", "qwen2.5-coder", "deepcoder"],
    "moe": ["30b", "32b", "70b", "72b", "moe", "mixtral", "qwen3:30"],
    "quick": ["7b", "8b", "3b", "4b", "1b", "small", "mini", "tiny", "14b"],
}

# Updated assign_models_to_tiers (line 127)
tiers: dict[str, list[str]] = {
    "quick": [],
    "coder": [],
    "moe": [],
    "thinking": [],
    "dispatcher": [],
    "agentic": [],  # NEW
    "swe": [],      # NEW
}
```

### 3. Intent Detection (`intent.py`)

```python
# New SWE patterns (after line 87)
SWE_PATTERNS: ClassVar[list[IntentPattern]] = [
    IntentPattern(
        re.compile(r"\b(refactor|redesign|migrate|overhaul|rewrite)\s+(the\s+)?(entire|whole|full|complete)?\s*(codebase|project|system|repo)\b", re.I),
        orchestration_mode=OrchestrationMode.AGENTIC,
        task_type="swe",
        confidence_boost=0.6,
        reasoning="repo-scale operation detected",
    ),
    IntentPattern(
        re.compile(r"\b(multi.?file|across files|all files|every file|codebase.?wide)\b", re.I),
        task_type="swe",
        confidence_boost=0.5,
        reasoning="multi-file operation",
    ),
    IntentPattern(
        re.compile(r"\b(architecture|system design|component diagram|module structure)\b", re.I),
        task_type="swe",
        confidence_boost=0.45,
        reasoning="architectural task",
    ),
]

# Enhanced AGENTIC_PATTERNS - add agent-specific triggers
IntentPattern(
    re.compile(r"\b(use tools?|call tools?|with tools?|tool.?use|function.?call)\b", re.I),
    orchestration_mode=OrchestrationMode.AGENTIC,
    task_type="agentic",
    confidence_boost=0.55,
    reasoning="explicit tool use requested",
),
```

### 4. Routing Logic (`routing.py`)

```python
# New task mappings (after line 322)
_AGENTIC_TASKS = frozenset({"agent", "tool", "execute", "terminal", "agentic"})
_SWE_TASKS = frozenset({"refactor", "migrate", "architect", "redesign", "swe"})

def _task_to_tier(task_type: str) -> str:
    """Map task type to model tier for economic lookups."""
    if task_type in _MOE_TASKS:
        return "moe"
    if task_type in _AGENTIC_TASKS:
        return "agentic"  # NEW
    if task_type in _SWE_TASKS:
        return "swe"      # NEW
    if task_type in _CODER_TASKS:
        return "coder"
    return "quick"

# In ModelRouter.select_model() - add after line 803:
# Priority 2.6: Agentic tasks use agent-trained models
if task_type in self.config.agentic_tasks or task_type == "agentic":
    model_agentic = get_model("agentic")
    if model_agentic != "current":
        log.info("model_selected", source="agentic_task", task=task_type, tier="agentic")
        return model_agentic
    # Fall back to coder if no agentic model configured
    log.info("model_selected", source="agentic_fallback", task=task_type, tier="coder")
    return model_coder

# Priority 2.7: SWE tasks use SWE-optimized models
if task_type in self.config.swe_tasks or task_type == "swe":
    model_swe = get_model("swe")
    if model_swe != "current":
        log.info("model_selected", source="swe_task", task=task_type, tier="swe")
        return model_swe
    # Fall back to moe for complex reasoning
    log.info("model_selected", source="swe_fallback", task=task_type, tier="moe")
    return model_moe
```

### 5. Orchestration Executor (`executor.py`)

```python
# In _execute_agentic(), modify line 174:
# OLD: selected_model = model_override or await select_model(task_type="review", ...)
# NEW:
task_tier = "agentic" if intent.task_type in ("agentic", "agent", "tool") else intent.task_type
if intent.task_type == "swe":
    task_tier = "swe"
selected_model = model_override or await select_model(
    task_type=task_tier,
    content_size=len(message),
    content=message
)
```

### 6. Settings Schema (`settings.json.example`)

```json
{
  "backends": [
    {
      "id": "ollama-local",
      "models": {
        "quick": "qwen3:8b",
        "coder": ["deepcoder:14b", "qwen2.5-coder:14b"],
        "moe": ["qwen3:32b", "openthinker:32b"],
        "thinking": "openthinker:7b",
        "agentic": "openthinker:7b",
        "swe": "deepswe-preview:32b",
        "dispatcher": "functiongemma:270m"
      }
    }
  ]
}
```

## Routing Flow Diagram

```
User Request
     │
     ▼
┌─────────────────────────────────────────┐
│          Intent Detection               │
│  (regex → semantic → LLM classifier)   │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│     Orchestration Mode Selection        │
│                                         │
│  AGENTIC detected?                      │
│    ├─ SWE patterns? → task_type="swe"  │
│    └─ Tool patterns? → task_type="agentic" │
│                                         │
│  Other modes: VOTING, DEEP_THINKING...  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│          Model Selection                │
│                                         │
│  task_type == "agentic"                 │
│    → Use agentic tier (OpenThinker)    │
│                                         │
│  task_type == "swe"                     │
│    → Use swe tier (DeepSWE)            │
│                                         │
│  Fallbacks:                             │
│    agentic → coder → quick             │
│    swe → moe → coder                   │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         Agent Loop Execution            │
│                                         │
│  SWE tier gets:                         │
│    - allow_write=True                   │
│    - allow_exec=True (with gate)        │
│    - Larger max_iterations (15)         │
│                                         │
│  Agentic tier gets:                     │
│    - Standard tool access               │
│    - Native tool calling if supported   │
└─────────────────────────────────────────┘
```

## Fallback Strategy

When specialized tiers aren't configured:

| Missing Tier | Fallback Chain |
|--------------|----------------|
| `agentic` | `coder` → `quick` |
| `swe` | `moe` → `coder` |
| `thinking` | `moe` → `coder` |
| `coder` | `quick` |

## Quality Tracking

The existing melon/affinity system automatically learns which models excel at agentic tasks:

```python
# In BackendScorer.score() - affinity boost applies per task_type
affinity = tracker.get_affinity(backend.id, "agentic")  # Tracks agentic performance
boost_multiplier *= (1.0 + (affinity - 0.5) * 0.4)      # ±20% boost
```

Over time, models that succeed at tool-use tasks accumulate higher affinity scores for agentic routing.

## Consequences

### Positive
- Agent loops use purpose-trained models (10x+ improvement on tool compliance)
- SWE tasks get repo-scale reasoning capability
- Existing fallback system prevents failures if tiers not configured
- Quality system auto-learns optimal routing

### Negative
- Two additional models to manage/pull
- Slight increase in config complexity
- Requires users to understand tier purpose

### Neutral
- Backward compatible - existing configs work unchanged
- No breaking changes to MCP tool interface

## Implementation Order

1. `model_detection.py` - Add tier keywords
2. `config.py` - Add tier definitions and task sets
3. `routing.py` - Add tier routing logic
4. `intent.py` - Add SWE patterns
5. `executor.py` - Wire up tier selection in agentic mode
6. `settings.json.example` - Update example config
7. Pull models: `ollama pull deepcoder:14b openthinker:7b`

## References

- [DeepCoder-14B](https://huggingface.co/agentica-org/DeepCoder-14B-Preview) - 60.6% LiveCodeBench
- [OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1) - SOTA 8B agent
- [DeepSWE-Preview](https://huggingface.co/agentica-org/DeepSWE-Preview) - 42.2% SWE-Bench
