# Fix: Migrate Code Indexing to ChromaDB

## Problem Identified (2024-12-24)

There was a design inconsistency in Delia's vector storage:

| Data Type | Before | After |
|-----------|--------|-------|
| Playbooks | ChromaDB `delia_playbook` | ✅ Unchanged |
| Memories | ChromaDB `delia_memories` | ✅ Unchanged |
| **Code** | `project_summary.json` | ✅ Now uses ChromaDB `delia_code` |

## Implementation (2024-12-25)

### Changes Made

**File: `src/delia/orchestration/summarizer.py`**

1. **Import**: Added `from .vector_store import get_vector_store`

2. **`sync_project()` (line ~270)**: Added call to `_sync_to_chromadb()` after saving JSON index

3. **New method `_sync_to_chromadb()`**: Syncs all file embeddings to ChromaDB
   - Uses `get_vector_store(self.root)` singleton
   - Calls `store.add_code_file()` for each file with embedding
   - Handles non-string exports by converting to str

4. **`search()` method**: Replaced numpy cosine similarity with ChromaDB query
   - Uses `store.search_code(query_embedding, project, n_results)`
   - Falls back to in-memory numpy if ChromaDB fails

5. **New method `_fallback_search()`**: Fallback numpy search for robustness

### VectorStore API Used

```python
from .vector_store import get_vector_store

store = get_vector_store(self.root)
store.add_code_file(
    file_path=rel_path,
    content=summary.summary or f"File: {rel_path}",
    embedding=summary.embedding,
    summary=summary.summary,
    exports=exports,  # list[str]
    project=project_name,
)

results = store.search_code(
    query_embedding=query_embedding,
    project=project_name,
    n_results=top_k,
)
```

## Test Results

```
Files indexed in memory: 285
ChromaDB delia_code collection: {'count': 284}
Search results for "vector store embeddings":
  - src/delia/orchestration/vector_store.py: 0.6879
  - src/delia/learning/retrieval.py: 0.6364
  - src/delia/embeddings.py: 0.5836
```

## Status

**COMPLETED** - 2024-12-25
