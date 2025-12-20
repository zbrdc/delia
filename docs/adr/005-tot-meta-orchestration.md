# ADR-005: Tree of Thoughts as Meta-Orchestration

**Status:** Proposed
**Date:** 2025-12-20
**Deciders:** Delia Core Team

## Context

We have multiple orchestration modes (VOTING, AGENTIC, DEEP_THINKING, COMPARISON) but no way to:
1. Know which mode is best for a given task type
2. Learn from comparative outcomes
3. Handle extremely high-stakes decisions that need maximum confidence

Additionally, TREE_OF_THOUGHTS is defined but unimplemented (currently aliases to DEEP_THINKING).

## Decision

Redefine **Tree of Thoughts (ToT)** as **Meta-Orchestration** that:
- Executes multiple orchestration modes in parallel as "branches"
- Uses Critic to evaluate and select the best outcome
- Feeds results to ACE Framework for meta-learning
- Reserves ToT for high-stakes scenarios (not default)

### Key Principles

1. **ToT is the Explorer**: Tries multiple orchestration approaches on the same problem
2. **ACE is the Learner**: Analyzes why one approach won, updates playbook
3. **Intent Detector is the Exploiter**: Uses learned patterns for future routing
4. **Escalation-based**: ToT triggers on high stakes, not routine tasks

## Architecture

### ToT Execution Flow

```python
async def _execute_tree_of_thoughts(
    self,
    intent: DetectedIntent,
    message: str,
    backend_type: str | None,
    model_override: str | None,
    messages: list[dict[str, Any]] | None = None,
) -> OrchestrationResult:
    """
    Meta-orchestration: Try multiple orchestration modes,
    critic picks best, ACE learns from outcome.
    """
    # 1. Define branches (different orchestration modes)
    branches = [
        ("voting", self._execute_voting),
        ("agentic", self._execute_agentic),
        ("deep_thinking", self._execute_deep_thinking),
    ]

    # 2. Execute all branches in parallel
    results = await asyncio.gather(*[
        executor(intent, message, backend_type, model_override, messages)
        for _, executor in branches
    ])

    # 3. Critic evaluates all results with reasoning
    critique = await self._critic_evaluate_branches(
        message=message,
        branches=[(name, res) for (name, _), res in zip(branches, results)]
    )

    # 4. Select winner
    winner_idx = critique["winner_index"]
    winner_mode = branches[winner_idx][0]
    winner_result = results[winner_idx]

    # 5. Feed to ACE for learning
    await self._ace_learn_from_tot(
        task_type=intent.task_type,
        message=message,
        branches=branches,
        results=results,
        winner_mode=winner_mode,
        critic_reasoning=critique["reasoning"],
    )

    # 6. Return winner with ToT metadata
    winner_result.mode = OrchestrationMode.TREE_OF_THOUGHTS
    winner_result.debug_info["tot_branches"] = [b[0] for b in branches]
    winner_result.debug_info["tot_winner"] = winner_mode
    winner_result.debug_info["tot_reasoning"] = critique["reasoning"]

    return winner_result
```

### Critic Evaluation

```python
async def _critic_evaluate_branches(
    self,
    message: str,
    branches: list[tuple[str, OrchestrationResult]],
) -> dict[str, Any]:
    """
    Critic compares all branch outcomes and picks best with reasoning.
    """
    comparison_prompt = f"""
Original Task: {message}

Branch Results:
{self._format_branches_for_critic(branches)}

As the Senior Critic, evaluate all branches and pick the BEST result.

Scoring criteria:
1. Correctness (does it solve the task?)
2. Completeness (are all requirements met?)
3. Quality (code quality, reasoning depth, security)
4. Confidence (how certain are we this is right?)

OUTPUT FORMAT (JSON):
{{
  "winner_index": 0,  // Index of best branch (0-based)
  "winner_mode": "voting",  // Name of winning orchestration mode
  "reasoning": "VOTING produced the most thorough security analysis...",
  "scores": [
    {{"mode": "voting", "correctness": 9, "completeness": 9, "quality": 9, "confidence": 8}},
    {{"mode": "agentic", "correctness": 7, "completeness": 8, "quality": 7, "confidence": 6}},
    {{"mode": "deep_thinking", "correctness": 8, "completeness": 7, "quality": 8, "confidence": 7}}
  ],
  "insights": "VOTING's consensus mechanism caught edge cases that single-model approaches missed."
}}
"""

    from ..llm import call_llm
    from ..routing import select_model

    critic_model = await select_model(task_type="moe", content_size=len(comparison_prompt))
    result = await call_llm(
        model=critic_model,
        prompt=comparison_prompt,
        system=ROLE_PROMPTS[ModelRole.CRITIC],
        enable_thinking=True,
    )

    import json
    from ..text_utils import strip_thinking_tags
    return json.loads(strip_thinking_tags(result["response"]))
```

### ACE Meta-Learning

```python
async def _ace_learn_from_tot(
    self,
    task_type: str,
    message: str,
    branches: list[tuple[str, callable]],
    results: list[OrchestrationResult],
    winner_mode: str,
    critic_reasoning: str,
) -> None:
    """
    ACE Framework learns meta-patterns from ToT outcomes.

    Updates playbook with orchestration selection strategies.
    """
    # Extract task characteristics
    task_features = self._extract_task_features(message)

    # Build learning prompt
    learning_prompt = f"""
Task Type: {task_type}
Task Features: {json.dumps(task_features)}

Orchestration Modes Tried:
{self._format_tot_results(branches, results)}

Winner: {winner_mode}
Critic Reasoning: {critic_reasoning}

As the ACE Reflector, analyze WHY {winner_mode} won for this type of task.

OUTPUT FORMAT (JSON):
{{
  "task_pattern": "security-critical code review with crypto primitives",
  "winning_mode": "{winner_mode}",
  "why_it_won": "Detailed analysis of what properties made this mode optimal",
  "when_to_use": "Generalized rule for when to use {winner_mode}",
  "playbook_update": "Actionable strategy bullet for future tasks",
  "confidence": 0.85
}}
"""

    # ACE Reflector analyzes
    from ..llm import call_llm
    reflector_result = await call_llm(
        model=config.model_moe.default_model,
        prompt=learning_prompt,
        system=ACE_REFLECTOR_PROMPT,
        enable_thinking=True,
    )

    if reflector_result.get("success"):
        lesson = json.loads(strip_thinking_tags(reflector_result["response"]))

        # ACE Curator integrates into playbook
        await self._curate_orchestration_strategy(
            task_type=task_type,
            lesson=lesson,
        )

        log.info(
            "tot_meta_learning_complete",
            task_type=task_type,
            winner_mode=winner_mode,
            confidence=lesson.get("confidence"),
            pattern=lesson.get("task_pattern"),
        )
```

### Intent Detection Enhancement

```python
# In orchestration/intent.py

def _check_tot_triggers(self, message: str, intent: DetectedIntent) -> bool:
    """
    Determine if ToT meta-orchestration should be used.

    Triggers:
    1. Explicit user request ("try multiple approaches", "high stakes")
    2. High-risk keywords (security, crypto, auth, payment, medical)
    3. User frustration level CRITICAL
    4. Previous attempts failed (session history)
    5. Playbook recommends ToT for this task pattern
    """
    msg_lower = message.lower()

    # Explicit ToT request
    if any(kw in msg_lower for kw in [
        "tree of thoughts", "tot", "multiple approaches",
        "try different ways", "meta-orchestrate", "high stakes"
    ]):
        return True

    # High-risk domains (consult playbook)
    high_risk_keywords = [
        "security", "crypto", "auth", "payment", "medical",
        "safety-critical", "production", "mission-critical"
    ]
    if any(kw in msg_lower for kw in high_risk_keywords):
        # Check playbook: does it recommend ToT for this domain?
        playbook_context = playbook_manager.format_for_prompt(intent.task_type)
        if "meta-orchestration" in playbook_context or "tree-of-thoughts" in playbook_context:
            return True

    # User frustration (from frustration detector)
    from ..frustration import get_frustration_detector
    frustration = get_frustration_detector().analyze(message)
    if frustration.level == "CRITICAL":
        log.info("tot_triggered_by_frustration", level=frustration.level)
        return True

    return False
```

## Benefits

### 1. Self-Improving System
- ToT explores → ACE learns → Intent detector exploits
- Continuous improvement without manual tuning

### 2. Domain-Specific Optimization
```
Learned patterns:
- "Security audits → VOTING (k=5) for consensus"
- "Refactoring → AGENTIC for tool access"
- "Architecture design → DEEP_THINKING for reasoning"
```

### 3. Confidence Guarantee
- For high-stakes tasks, ToT provides maximum confidence
- User gets best result from multiple approaches
- Transparent reasoning about why one approach won

### 4. Cost-Effective Escalation
- Normal tasks use direct modes (fast, cheap)
- Only escalate to ToT when needed
- ToT cost justified by learning value

## Trade-offs

### Costs
- **Latency**: ToT runs multiple modes in parallel (3-5x slower)
- **Compute**: Multiple LLM calls per request
- **Complexity**: More code to maintain

### Mitigations
- Reserve ToT for high-stakes only (not default)
- Parallel execution keeps latency reasonable
- Learning amortizes cost (fewer ToT calls over time as intent detector improves)

## Implementation Phases

### Phase 1: Core ToT Execution
- [x] Design architecture
- [ ] Implement `_execute_tree_of_thoughts()`
- [ ] Implement `_critic_evaluate_branches()`
- [ ] Add ToT result metadata

### Phase 2: ACE Meta-Learning Integration
- [ ] Implement `_ace_learn_from_tot()`
- [ ] Implement `_curate_orchestration_strategy()`
- [ ] Add orchestration playbook support
- [ ] Test learning loop

### Phase 3: Intent Detection Enhancement
- [ ] Add `_check_tot_triggers()` logic
- [ ] Integrate playbook recommendations
- [ ] Add frustration-based escalation
- [ ] Add session history analysis

### Phase 4: Optimization
- [ ] Add branch caching (skip redundant executions)
- [ ] Implement adaptive branch selection
- [ ] Add cost controls (max branches, timeouts)
- [ ] Performance benchmarking

## Success Metrics

1. **Learning Rate**: Playbook accumulates useful orchestration patterns
2. **Accuracy**: Intent detector selects correct mode more often over time
3. **ToT Frequency**: Decreases as system learns (good sign)
4. **User Satisfaction**: High-stakes tasks get better results

## Alternatives Considered

### Alternative 1: Full ToT Implementation
Traditional tree search with branching/pruning - too complex, marginal benefit

### Alternative 2: Remove ToT
Simple but loses opportunity for meta-learning

### Alternative 3: Manual Mode Selection
Users pick orchestration mode - puts burden on user

## References

- ToolOrchestra: Training LLMs to Think Like Tools
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- MDAP: Massively Decomposed Agentic Processes
- ACE Framework (Autonomous Cognitive Entity)

## Decision

**APPROVED** - Implement ToT as Meta-Orchestration with ACE integration

This creates a complete learning system where:
- ToT explores orchestration space
- ACE learns meta-patterns
- Intent detector exploits learned knowledge
- System continuously improves
