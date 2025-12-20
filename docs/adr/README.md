# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) documenting significant technical decisions in Delia.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-singleton-architecture.md) | Singleton Architecture for Core Services | Accepted |
| [002](002-mcp-native-paradigm.md) | MCP-Native Paradigm (No REST API) | Accepted |
| [003](003-centralized-llm-calling.md) | Centralized LLM Calling Module (llm.py) | Accepted |
| [004](004-structured-error-types.md) | Structured Error Types | Accepted |

## Template

New ADRs should follow this structure:

```markdown
# ADR-NNN: Title

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
What is the issue we're addressing?

## Decision
What have we decided to do?

## Rationale
Why did we make this decision?

## Consequences
What are the resulting context and trade-offs?
```
