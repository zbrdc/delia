# Anthropic: Effective Context Engineering for AI Agents

Source: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

## Core Principle
> Find the **smallest possible set of high-signal tokens** that maximize the likelihood of your desired outcome.

## Context as Finite Resource
- **Context rot**: As tokens increase, model's ability to accurately recall information decreases
- LLMs have an "attention budget" - every new token depletes it
- n² pairwise relationships for n tokens (transformer architecture constraint)
- Thoughtful context engineering is essential for capable agents

## System Prompts: Right Altitude
Find the Goldilocks zone between:
- ❌ **Too brittle**: Hardcoded complex logic, exact behaviors
- ❌ **Too vague**: High-level guidance, assumed shared context
- ✅ **Optimal**: Specific enough to guide, flexible enough for heuristics

## Just-in-Time Context Strategy
Rather than pre-loading all data:
1. Maintain lightweight identifiers (file paths, stored queries, links)
2. Use references to dynamically load data at runtime
3. Progressive disclosure - incrementally discover through exploration
4. Metadata provides efficient signals (file names, folder hierarchies, timestamps)

**Claude Code example**: Writes targeted queries, stores results, uses bash `head`/`tail` to analyze large data without loading full objects into context.

## Long-Horizon Task Techniques

### 1. Compaction
- Summarize conversation nearing context limit
- Reinitiate new context with summary
- Preserve: architectural decisions, unresolved bugs, implementation details
- Discard: redundant tool outputs, old messages
- **Art of compaction**: Balance recall vs precision

### 2. Structured Note-Taking (Agentic Memory)
- Agent writes notes persisted outside context window
- Notes pulled back in at later times
- Examples: to-do lists, NOTES.md files
- Track progress across complex tasks
- **Claude playing Pokémon**: Maintains tallies across thousands of steps, develops maps, tracks objectives

### 3. Sub-Agent Architectures
- Specialized sub-agents handle focused tasks with clean context
- Main agent coordinates with high-level plan
- Each subagent may use tens of thousands of tokens
- Returns only condensed summary (1,000-2,000 tokens)
- Clear separation of concerns

## Tools Design
- Well understood by LLMs
- Minimal overlap in functionality
- Self-contained, robust to error
- Clear intended use
- Descriptive, unambiguous parameters
- **Avoid**: Bloated tool sets, ambiguous decision points

## Examples (Few-Shot)
- Curate diverse, canonical examples
- Don't stuff laundry list of edge cases
- Examples are "pictures worth a thousand words"

## Relevance to Delia ACE Framework

| Anthropic Concept | Delia Implementation |
|-------------------|---------------------|
| Smallest high-signal tokens | Playbook bullets |
| Structured note-taking | Memories system |
| Curated starting context | Profile seeds |
| Learning/refining over time | ACE feedback loop |
| Just-in-time retrieval | auto_context() |
| Compaction | Session compaction |
| Sub-agents | Task tool delegation |

## Key Quote
> "As capabilities scale, treating context as a precious, finite resource will remain central to building reliable, effective agents."
