# ADR-007: Conversation Compaction System

## Status
Accepted

## Context

Long-running Delia sessions accumulate conversation history that can exceed model context windows, leading to:
- Degraded response quality as important context gets truncated
- Increased latency from processing large contexts
- Memory pressure from storing extensive message histories
- Loss of early task context when truncation occurs

Claude Code handles this elegantly with its `/compact` command, which summarizes older messages while preserving recent context and key information.

## Decision

Implement a conversation compaction system for Delia that:

1. **LLM-Based Summarization**: Use the `quick` model tier to generate concise summaries of older messages, preserving:
   - Main task/goal being worked on
   - Key decisions and their reasoning
   - File/code modifications made
   - Problems encountered and solutions
   - Current state and progress

2. **Key Information Extraction**: Automatically detect and preserve:
   - Tool calls (file operations, commands, etc.)
   - File modifications
   - Important decisions

3. **Threshold-Based Triggers**:
   - Warning at 12,000 tokens (configurable)
   - Auto-compaction at 18,000 tokens (1.5x threshold)

4. **Message Preservation**: Always keep the last N message pairs (default: 6 pairs = 12 messages) untouched for immediate context

5. **MCP Tools**:
   - `session_compact`: Manual compaction trigger
   - `session_stats`: View compaction statistics

## Implementation

### Core Components

**`src/delia/compaction.py`** - Main compaction logic:
- `ConversationCompactor` class with configurable thresholds
- `CompactionResult` dataclass for operation results
- `CompactionMetadata` for tracking compaction history
- Key info extraction via regex patterns

**SessionManager Integration**:
- `SessionState.needs_compaction()` - Check if session needs compaction
- `SessionState.get_compaction_stats()` - Get token counts and status
- `SessionManager.compact_session()` - Async compaction method

**Orchestration Integration** (`service.py`):
- `_check_auto_compaction()` - Check and optionally auto-compact
- Warning events for large contexts
- Automatic compaction above critical threshold

### Compaction Algorithm

```
1. Count total tokens in session messages
2. If tokens > threshold:
   a. Identify messages to compact (all except last N pairs)
   b. Extract key information (tool calls, file mods, decisions)
   c. Generate LLM summary of compactable messages
   d. Replace compacted messages with single summary message
   e. Store compaction metadata in session
3. Save updated session to disk
```

### Key Information Patterns

```python
TOOL_CALL_PATTERNS = [
    r"\b(created|wrote|edited)\s+file\s+...",
    r"\b(running|executed)\s+command\s+...",
    r"\b(pip|npm|git|docker)\s+\w+",
]

DECISION_PATTERNS = [
    r"(?:decided|choosing|will use)\s+...",
    r"(?:approach|strategy|plan):\s*...",
]
```

## Consequences

### Positive
- Prevents context overflow in long sessions
- Preserves important information across compaction
- Matches Claude Code's user-friendly `/compact` workflow
- Automatic protection against context overflow
- Configurable thresholds for different use cases

### Negative
- Summarization may lose some nuance from original messages
- Adds latency during compaction (LLM call required)
- Storage overhead for compaction metadata

### Trade-offs
- Quality vs. compression: Higher compression ratios may lose more detail
- Threshold tuning: Too low = frequent compaction, too high = context issues

## Configuration

```python
DEFAULT_COMPACTION_THRESHOLD_TOKENS = 12000  # Warning threshold
DEFAULT_PRESERVE_RECENT_MESSAGES = 6         # Message pairs to keep
DEFAULT_SUMMARY_MAX_TOKENS = 2000           # Max summary length
```

## Usage Examples

### Manual Compaction (MCP)
```python
# Check if compaction is needed
session_stats(session_id="abc-123")
# Returns: {"total_tokens": 15000, "needs_compaction": true, ...}

# Compact the session
session_compact(session_id="abc-123")
# Returns: {"success": true, "messages_compacted": 20, "tokens_saved": 8000, ...}
```

### Automatic Compaction
When processing a message and tokens exceed 1.5x threshold:
```
StreamEvent(type="compaction", message="Session auto-compacted...")
```

## Related ADRs
- ADR-001: Singleton Architecture (SessionManager singleton)
- ADR-002: MCP-Native Paradigm (session tools)
- ADR-003: Centralized LLM Calling (summarization via call_llm)
