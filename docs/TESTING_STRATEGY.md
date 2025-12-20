# Delia Testing Strategy

This document outlines the multi-stage testing and hardening strategy for Delia, ensuring robustness in local LLM orchestration.

## 1. Unit Testing (Standard)
- **Location**: `tests/`
- **Focus**: Individual function correctness.
- **Framework**: `pytest`, `pytest-asyncio`.

## 2. Property-Based Fuzzing (Hardening)
Automated edge-case discovery using `hypothesis`.

### Stage 1: Unit & Logic Fuzzing
- **Tests**: `tests/test_fuzz_parser.py`, `tests/test_fuzz_scorer.py`, `tests/test_fuzz_routing.py`, `tests/test_fuzz_stream.py`.
- **Target**: Parsers, math models, and protocol handlers.
- **Outcome**: Fixed brittle regex and sanitized metrics math.

### Stage 2: Protocol & Interface Fuzzing
- **Tests**: `tests/test_fuzz_mcp.py`.
- **Target**: Top-level MCP tools (`delegate`, `batch`, `think`).
- **Outcome**: Verified that public APIs are crash-proof against arbitrary input types.

### Stage 3: Adversarial/Security Fuzzing
- **Tests**: `tests/test_fuzz_security.py`.
- **Target**: Path validation, shell execution, and workspace boundaries.
- **Outcome**: Verified blocking of traversal and injection payloads.

### Stage 4: Stateful Fuzzing
- **Tests**: `tests/test_fuzz_stateful.py`.
- **Target**: `SessionManager`, `MelonTracker`.
- **Outcome**: Verified consistency across long sequences of operations and LRU eviction logic.

## 3. Resiliency & Chaos Testing (System-Level)
Testing the system under stress and failure conditions.

### Stage 5: Concurrency & Race Condition Stress
- **Goal**: Simulate high concurrent load (20+ users) hitting the `ModelQueue`.
- **Focus**: Race conditions in stats tracking, file locking in sessions, and GPU resource contention logic.

### Stage 6: Chaos & Failover Integration
- **Goal**: Mock flaky backends with high latency, partial responses, and random 5xx errors.
- **Focus**: Verifying `BackendScorer` affinity penalties and automatic failover to healthy nodes.

## 4. Model Intelligence Evals (Future)
- **Intent Accuracy**: Measuring precision/recall of the `IntentDetector`.
- **Tool Use Robustness**: Verifying agentic loop completion using "Perfect" vs "Flaky" mocked LLMs.
