# Nebnet-MCP Context Optimization (2025-12-26)

## Summary
Optimized documentation chunking and retrieval for better LLM context efficiency.
Test reindex of react-native source showed **19.6% chunk reduction** (2871 → 2309).

## Changes Made

### 1. Minimum Chunk Quality Filter (embedder.py)
- Increased minimum chunk size from 50 to 100 chars (configurable via `min_chunk_size`)
- Added `_is_low_value_content()` method that filters:
  - Docusaurus admonitions (:::tip, :::warning) under 300 chars
  - Tiny code blocks (<80 chars)
  - License/contributing boilerplate
  - Markdown table separators only
  - Badge/shield images only

### 2. Contextual Headers in Search Results (server.py)
- Added `context` field with breadcrumb: "Source Docs > Title > Section"
- Removed redundant `scope` field for shared docs (saves tokens)
- Renamed `relevance` to `score` for brevity

### 3. Config Optimizations (config.py)
- `chunk_size`: 1000 → 1500 chars (allows more context per chunk)
- `chunk_overlap`: 200 → 150 chars (less duplication)
- `min_chunk_size`: 100 (new config option)

### 4. Improved Large Content Splitting (embedder.py)
- Added `_split_by_sentences()` method for large paragraphs
- Splits at sentence boundaries (., !, ?) instead of arbitrary character positions
- Falls back to character split only when no sentence boundaries exist

## Results (react-native test)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Chunks | 2871 | 2309 | -19.6% |
| Tiny (<100 chars) | 5.3% | 0.2% | -96% |
| Max size | 5490 | 1650 | -70% |
| Avg size | 530 | 609 | +15% |

## Files Modified
- `src/nebnet/indexer/embedder.py` - chunking logic + quality filters
- `src/nebnet/server.py` - search result formatting
- `src/nebnet/config.py` - default settings

## Next Steps
- Full reindex of all 43 sources to apply new settings
- Monitor search quality after reindex
- Consider adding more boilerplate patterns as discovered
