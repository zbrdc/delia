# Architecture Profile

Load this profile for: design decisions, ADRs, system architecture, orchestration.

## Model Tiers (8 Total)

```python
quick_tier:      7B-14B    → Simple Q&A, summarization
coder_tier:      14B-30B   → Code generation, review
moe_tier:        30B+ MoE  → Complex reasoning, planning
thinking_tier:   Extended  → Deep analysis, debugging
dispatcher_tier: 270M      → Fast task routing
agentic_tier:    7B-14B    → Tool-calling loops
swe_tier:        32B+      → Repo-scale refactoring
```

## Orchestration Modes (ADR-008)

**Core:**
- NONE - Default (90%+ of queries)
- AGENTIC - Tool-calling loops
- DEEP_THINKING - Primary advanced mode
- VOTING - Confidence-weighted adaptive k
- TREE_OF_THOUGHTS - Opt-in only (`tot=True`)

**Pipeline:** CHAIN, WORKFLOW, BATCH
**Utility:** STATUS_QUERY

## Backend Scoring

```python
base_score = backend.priority * 100
affinity_boost = 1.0 + (affinity - 0.5) * 0.4
melon_boost = 1.0 + (sqrt(total_melons) * 0.02)
health_penalty = 0.5 if failures > 0 else 1.0
final_score = base_score * affinity_boost * melon_boost * health_penalty
```

## Intent Detection (3-Layer)

```
Layer 1: Regex (fast, high confidence)
   ↓ (if confidence < 0.9)
Layer 2: Semantic (embeddings)
   ↓ (if confidence < 0.7)
Layer 3: LLM Classifier (accurate)
```

## ADRs

| ADR | Title | Status |
|-----|-------|--------|
| 001 | Singleton Architecture | Accepted |
| 002 | MCP-Native Paradigm | Accepted |
| 003 | Centralized LLM Calling | Accepted |
| 004 | Structured Error Types | Accepted |
| 007 | Conversation Compaction | Accepted |
| 008 | ACE-Aligned Simplification | Implemented |
