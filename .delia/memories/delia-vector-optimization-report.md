# Delia Vector Optimization Report

Based on nebnet-mcp optimizations analysis, adapted for Delia's architecture.

## System Comparison

| Aspect | nebnet-mcp | Delia |
|--------|-----------|-------|
| **Content Type** | Documentation (markdown) | Code files, bullets, memories, profiles |
| **Chunking** | Yes (1500 chars, 150 overlap) | No - full content indexed |
| **Vector DB** | ChromaDB (single collection) | ChromaDB (4 collections) |
| **Embeddings** | Voyage AI (voyage-3) | Voyage AI + Ollama fallback |
| **Search** | Hybrid semantic + keyword + rerank | Hybrid semantic + keyword + utility×recency |
| **Result Format** | JSON with contextual headers | Raw content with score |

## Applicable Optimizations

### 1. Contextual Headers in Search Results ✅ HIGH IMPACT
**Location**: `orchestration/vector_store.py` search methods

nebnet-mcp adds breadcrumb context:
```
{"context": "React Native Docs > Components > Button", "content": "...", "score": 85.2}
```

Delia could add collection-specific context:
```
{"context": "Playbook > coding > pathlib-pattern", "content": "...", "score": 0.85}
{"context": "Code > src/auth.py > login()", "content": "...", "score": 0.78}
```

### 2. Quality Filtering for Code Files ✅ MEDIUM IMPACT
**Location**: `orchestration/vector_store.py` add_code_file()

Skip indexing:
- Empty files or <50 chars
- Auto-generated files (__pycache__, .pyc, node_modules artifacts)
- Files with only imports/docstrings

### 3. Hybrid Score Normalization ✅ LOW IMPACT
**Current**: Delia returns raw distance-converted scores (1 - distance)
**Improved**: Normalize to 0-100 scale like nebnet for consistency

### 4. Large File Chunking - NEW OPPORTUNITY 🆕
**NOT in nebnet, but valuable for Delia**

Large code files (>5KB) could be chunked by:
- Function/class boundaries (using LSP symbols)
- Logical sections with context headers

This would improve recall for symbol-level searches.

## NOT Applicable (Delia doesn't chunk)

- Min chunk size filtering (Delia uses full files)
- Sentence-based splitting (no paragraph chunking)
- Overlap tuning (no overlapping chunks)

## Implementation Priority

1. **Contextual Headers** - Easiest win, improves LLM comprehension
2. **Quality Filtering** - Prevents noise in vector store
3. **Score Normalization** - Consistency across all collections
4. **LSP-guided Code Chunking** - Complex but high value (future)

## Metrics to Track

- Tokens per search result (aim for 300-500)
- Retrieval precision (relevant results in top 5)
- Index size reduction after quality filtering
