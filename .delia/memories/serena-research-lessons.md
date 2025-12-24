# Serena Research: Context Engineering & Tool Orchestration Lessons

This document captures the comparative research between **Delia** and **Serena** (located in `~/git/serena`), specifically investigating why Serena's tools were invoked more frequently and reliably by LLMs.

## Core Lessons Learned

### 1. Authoritative Instruction Density
Serena uses highly imperative and "emotive" language in tool descriptions and system prompts. 
*   **Lesson**: LLMs respond to authority. Vague guidance leads to drift; explicit constraints (e.g., "I WILL BE VERY UNHAPPY IF...") reduce hallucination and lazy patterns.
*   **Delia Implementation**: Refactored `mcp_instructions.md` to use stronger, task-oriented language and explicit symbolic navigation recipes.

### 2. The "Handoff" Pattern (Long-Horizon Tasks)
Serena explicitly addresses the "Context Rot" problem by providing a `prepare_for_new_conversation` tool that summarizes the current state into a memory file.
*   **Lesson**: When a task exceeds the context window, the agent should proactively "checkpoint" its intent and progress to a physical file so the next session can resume without loss of fidelity.
*   **Delia Implementation**: Added "Long-Task Handoff" strategy to canonical instructions.

### 3. Progressive Disclosure (Tool Chaining)
Serena's instructions strictly enforce a hierarchy: **Locate → Map → Target → Acquire**.
*   **Lesson**: Prevent "File Gulping" (reading huge files) by making the cost of context clear and providing a step-by-step symbolic alternative.
*   **Delia Implementation**: Explicitly documented the `lsp_get_symbols` → `read_file(lines=...)` recipe.

### 4. Semantic over Line-Based Editing
Serena moved away from line-number-based edits (which shift during operations) to symbolic and string-matching edits.
*   **Lesson**: Line numbers are brittle. Symbols (LSP) and unique string anchors are the only reliable way for an LLM to edit code.

---

## Actionable Improvement Roadmap for Delia

### 🔴 High Priority: Prompt Autogeneration & Management
Serena uses a `PromptFactory` (via the `interprompt` subpackage) to autogenerate Python classes from Jinja2 YAML templates.
*   **Improvement**: Delia should consolidate its prompt logic (currently split between `mcp_server.py`, `mcp_instructions.md`, and `agent_sync.py`) into a template-driven factory to allow for easier user customization and consistency.

### 🔴 High Priority: Process Isolation for LSP/Asyncio
Serena solved asyncio deadlocks by putting the agent and language servers into a separate process.
*   **Improvement**: Verify Delia's concurrency model. If we notice "Not connected" or timeout issues under high load, we should adopt Serena's process-isolation hammer to prevent event loop contamination.

### 🟡 Medium Priority: Opinionated Onboarding
Serena's onboarding proactively creates specific memory files like `suggested_commands.md` and `style_guidelines.md`.
*   **Improvement**: Enhance `delia init-project` to not just index symbols, but to generate these high-level human-readable guides in `.delia/memories/`. This provides an immediate "boost" to new agents.

### 🟡 Medium Priority: Tool Categorization Visibility
Serena's `ToolRegistry` allows for printing a formatted tool overview.
*   **Improvement**: While Delia has `list_tools`, it should be more prominent in the `initial_instructions` to help agents navigate the ~80 available tools without cognitive overload.

### 🟢 Low Priority: Emotive Constraint Enforcement
*   **Improvement**: Audit all `src/delia/tools/*.py` docstrings. Replace "Reads a file" with "Reads a file. WARNING: Use lsp_get_symbols first to find target lines to save tokens."

---

## Conclusion
The transition from a prototype to a production-grade agent requires moving from "providing tools" to "enforcing methodology." Serena's success is rooted in its **opinionated orchestration**. Delia must continue to bake these methodologies (Progressive Disclosure, State Checkpointing, Symbolic Priority) into its core protocol layer.
