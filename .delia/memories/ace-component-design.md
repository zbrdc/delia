# ACE Framework Component Design

## Summary

Complete design for the missing ACE Framework components based on Stanford research and Anthropic's context engineering principles.

## 1. Reflector Component

**Purpose**: Analyzes task execution outcomes to extract structured insights.

**Key Interface**:
```python
class Reflector:
    async def reflect(
        task_description: str,
        task_type: str,
        outcome: str,
        tool_calls: list[dict] | None,
        applied_bullets: list[str] | None,
        error_trace: str | None,
    ) -> ReflectionResult
```

**Output**: `ReflectionResult` with:
- `insights: list[ExtractedInsight]` - Strategies, anti-patterns, failure modes
- `bullets_to_tag_helpful: list[str]` - Bullet IDs that worked
- `bullets_to_tag_harmful: list[str]` - Bullet IDs that hurt
- `root_causes`, `correct_approaches`

## 2. Curator Component

**Purpose**: Maintains playbooks through incremental delta updates (NEVER regenerate).

**Key Interface**:
```python
class Curator:
    async def curate(reflection: ReflectionResult) -> CurationResult
    async def add_bullet(task_type, content, skip_dedup=False) -> tuple[bool, str|None]
    async def merge_similar_bullets(task_type, threshold=0.85) -> list[tuple]
    async def run_maintenance(task_type, max_age_days=90, min_utility=0.3)
```

**Delta Operations**: ADD, REMOVE, MERGE, MODIFY, BOOST, DEMOTE

## 3. Semantic Deduplication

**Purpose**: Embedding-based similarity check before adding bullets.

**Thresholds**:
- `>0.90`: Definite duplicate - skip add
- `0.85-0.90`: Merge candidate - combine insights
- `0.75-0.85`: Related - add but link

**Key Interface**:
```python
class SemanticDeduplicator:
    async def check_similarity(new_content, existing_bullets, threshold=0.85) -> DeduplicationResult
    async def find_clusters(bullets, threshold=0.80) -> list[list[PlaybookBullet]]
```

## 4. Hybrid Retrieval Scoring

**Formula**: `score = relevance^α × utility^β × recency^γ`

**Default Weights**:
- α = 1.0 (relevance - linear)
- β = 0.5 (utility - sqrt emphasis)
- γ = 0.3 (recency - weak emphasis)

**Utility Score**: `(helpful + 1) / (helpful + harmful + 2)` (Laplace smoothing)
**Recency Score**: `exp(-days_since_use / 30)` (30-day half-life)

## Integration Points

1. **OrchestrationService**: After task completion, call `reflector.reflect()` then `curator.curate()`
2. **auto_context tool**: Replace `get_top_bullets()` with `HybridRetriever.retrieve()`
3. **playbook tool**: Add `action="maintain"` for curator maintenance

## New Module Structure

```
src/delia/ace/
    __init__.py
    reflector.py
    curator.py
    deduplication.py
    retrieval.py
```

## Critical Files to Modify

- `playbook.py` - Extend with HybridRetriever integration
- `embeddings.py` - Reuse for deduplication
- `orchestration/service.py` - Add reflection trigger
- `tools/consolidated.py` - Expose new tools
