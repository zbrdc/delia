# ADR-008: Intrinsics Consolidation

## Status
Accepted

## Date
2024-12-20

## Context

Delia had multiple overlapping systems for pre-execution checks and orchestration decisions:

| System | Purpose | Location |
|--------|---------|----------|
| **Intent Detection** | Classify task type and orchestration mode | `orchestration/intent.py` |
| **Meta-learning** | Decide when to use ToT | `orchestration/meta_learning.py` |
| **Frustration Tracker** | Detect user frustration, escalate | `frustration.py` |
| **Intrinsics Engine** | Answerability, groundedness checks | `orchestration/intrinsics.py` |
| **Profiles** | Specialist model routing | `profiles.py` |

This created:
- Redundant code paths (frustration + intrinsics both trigger escalation)
- Multiple singletons to initialize
- Unclear execution order
- Maintenance burden

## Decision

Consolidate into a simpler pipeline:

```
Before: Intent → Meta-learning → Frustration → Routing → Profiles → Intrinsics → Execute

After:  Intent (includes ToT) → Intrinsics (includes user state) → Execute
                                        ↓
                                    Profiles (specialist routing)
```

### Changes Made

1. **Frustration → Intrinsics**
   - Added `FrustrationLevel` enum to `intrinsics.py`
   - Added `UserStateResult` dataclass
   - Added `check_user_state()` method (no LLM needed, pure regex)
   - Added `ESCALATE_VOTING` and `ESCALATE_DEEP` actions

2. **Meta-learning already in Intent**
   - `should_use_tot()` was already called from `intent.py:245`
   - No changes needed

3. **Service.py Simplified**
   - Removed `FrustrationTracker` dependency
   - Added inline `_check_repeat()` for session tracking
   - STEP 2-3 now use single `intrinsics.check_user_state()` call

4. **frustration.py Deprecated**
   - Added deprecation notice
   - No active imports remain

### New IntrinsicsEngine API

```python
from delia.orchestration.intrinsics import (
    get_intrinsics_engine,
    IntrinsicAction,
    FrustrationLevel,
)

intrinsics = get_intrinsics_engine()

# Pre-execution checks (all in one place)
answerability = await intrinsics.check_answerability(task, context)
user_state = intrinsics.check_user_state(message, is_repeat=True, repeat_count=2)

# Post-execution checks
groundedness = await intrinsics.check_groundedness(response, sources)

# Action-based routing
if user_state.action == IntrinsicAction.ESCALATE_DEEP:
    intent.orchestration_mode = OrchestrationMode.DEEP_THINKING
elif user_state.action == IntrinsicAction.ESCALATE_VOTING:
    intent.orchestration_mode = OrchestrationMode.VOTING
```

### Frustration Detection Logic

```python
# Scoring (from check_user_state):
score = 0.0
if repeat_count > 0: score += repeat_count * 1.5
if has_angry_keywords: score += 3.5
if has_negative_feedback: score += 2.5

# Thresholds:
# score >= 5.0 → HIGH → ESCALATE_DEEP
# score >= 3.0 → MEDIUM → ESCALATE_VOTING
# score >= 1.5 → LOW → ESCALATE_VOTING
# else → NONE → PROCEED
```

## Consequences

### Positive
- Single location for all pre-execution checks
- Clearer mental model: "intrinsics" = sanity checks
- Fewer singletons (removed FrustrationTracker)
- Easier to add new checks (just add method to IntrinsicsEngine)

### Negative
- Session-level repeat tracking now in OrchestrationService (simple dict)
- `frustration.py` still exists (deprecated, for backwards compat)

### Neutral
- Meta-learning remains separate (already integrated into intent)
- Profiles remain separate (orthogonal concern: which model, not whether to escalate)

## Files Changed

| File | Change |
|------|--------|
| `orchestration/intrinsics.py` | Added FrustrationLevel, UserStateResult, check_user_state() |
| `orchestration/service.py` | Removed FrustrationTracker, use intrinsics |
| `frustration.py` | Marked DEPRECATED |
