# ACE Implementation Principles for Delia

## Core Principles

### 1. Delta Updates Over Rewrites
- NEVER ask LLM to regenerate entire playbooks
- Use atomic operations: add, remove, modify
- Preserve exact wording of learned knowledge

### 2. Feedback Loop is Mandatory
- Every bullet MUST have helpful/harmful tracking
- `report_feedback()` closes the learning loop
- Bullets without feedback should decay over time

### 3. Semantic Deduplication
- Before adding new bullet, check semantic similarity
- Merge similar bullets to prevent bloat
- Use embeddings for similarity, not just text matching

### 4. Hybrid Retrieval
- Don't dump entire playbook into context
- Score bullets by: relevance + utility_score + recency
- Keep context focused on task at hand

### 5. Per-Project Isolation
- Each project has its own playbook
- Patterns learned in Project A shouldn't leak to Project B
- Global patterns (language best practices) can be shared

## Delia's Current Implementation Status

### ✅ Implemented
- Playbook storage with bullet IDs
- `add_playbook_bullet()`, `delete_playbook_bullet()`
- `report_feedback()` for helpful/harmful tracking
- `utility_score` calculation
- Per-project playbook isolation (`.delia/playbooks/`)
- `auto_context()` for task detection

### ⚠️ Needs Improvement
- Semantic deduplication (currently no embedding check)
- Auto-reflection after task completion
- Curator agent for playbook maintenance
- Hybrid retrieval scoring (currently just top-N by utility)

### ❌ Not Implemented
- Automatic Reflector agent analyzing outcomes
- Semantic similarity before bullet insertion
- Decay for bullets never used
- Cross-project pattern sharing (opt-in)

## Recommendations

1. **Add embedding-based dedup** before `add_playbook_bullet()`
2. **Implement auto-reflection** hook after task completion
3. **Add decay mechanism** for stale bullets
4. **Build Curator agent** for periodic playbook maintenance
5. **Hybrid scoring**: `score = relevance * utility * recency_factor`
