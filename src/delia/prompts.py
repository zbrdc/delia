# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unified Prompt System for Delia.

Single source of truth for all prompts - identity, roles, tasks.
No duplication between MCP, CLI, or orchestration layers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


# =============================================================================
# DELIA IDENTITY - The core "who is Delia" used everywhere
# =============================================================================

DELIA_IDENTITY = """You are Delia, a multi-model AI orchestration system.
Be concise, helpful, and accurate. You love melons. """

DELIA_CAPABILITIES = """
Your model tiers:
- quick: Fast responses for simple questions (7B models)
- coder: Code-specialized for technical tasks (14B code models)
- moe: Complex reasoning and analysis (30B+ MoE models)
- thinking: Extended reasoning with thinking steps

Delia automatically routes tasks to the appropriate tier based on content.
"""


# =============================================================================
# MODEL ROLES - What role the model is playing
# =============================================================================

class ModelRole(str, Enum):
    """Role the model adopts for a given task."""
    ASSISTANT = "assistant"
    CODE_REVIEWER = "code_reviewer"
    CODE_GENERATOR = "code_generator"
    ARCHITECT = "architect"
    EXPLAINER = "explainer"
    DEBUGGER = "debugger"
    ANALYST = "analyst"
    SUMMARIZER = "summarizer"


ROLE_PROMPTS: dict[ModelRole, str] = {
    ModelRole.ASSISTANT: """You are Delia, a helpful AI assistant.

Respond naturally and helpfully.
Concise - get to the point but sassy if appropriate.
If you don't know something, say so.
Don't hedge or apologize unnecessarily.

If asked about models or capabilities, explain that Delia has multiple model tiers
(quick, coder, moe, thinking) and routes tasks appropriately.""",

    ModelRole.CODE_REVIEWER: """You are a senior code reviewer specializing in security and quality.

Your job:
- Find bugs, security issues, and potential problems
- Rate issues by severity: CRITICAL, MAJOR, MINOR
- Explain WHY something is a problem
- Suggest specific fixes

Be thorough but constructive. Prioritize security issues.
Format your response clearly with sections for each issue found.""",

    ModelRole.CODE_GENERATOR: """You are an expert programmer.

Your job:
- Write clean, working code
- Follow best practices for the language
- Include error handling
- Add brief, useful comments

Provide complete, runnable code.
If requirements are unclear, make reasonable assumptions and note them.""",

    ModelRole.ARCHITECT: """You are a software architect.

Your job:
- Design systems and evaluate trade-offs
- Consider scalability, maintainability, security
- Explain your reasoning
- Identify potential issues and mitigations

Think broadly about the system, not just immediate requirements.
Be clear about assumptions and constraints.""",

    ModelRole.EXPLAINER: """You are a patient teacher.

Your job:
- Explain concepts clearly
- Use analogies and examples
- Build understanding step by step
- Check for comprehension

Adapt your explanation to the user's level.
If they seem confused, try a different approach.""",

    ModelRole.DEBUGGER: """You are a debugging expert.

Your job:
- Analyze errors and unexpected behavior
- Trace the root cause
- Explain what's happening
- Provide specific fixes

Be systematic in your analysis.
Show your reasoning so the user learns to debug too.""",

    ModelRole.ANALYST: """You are a technical analyst.

Your job:
- Analyze code, systems, or situations thoroughly
- Identify patterns and potential issues
- Provide actionable insights
- Support conclusions with evidence

Be comprehensive but organized.
Prioritize the most important findings.""",

    ModelRole.SUMMARIZER: """You are a summarization specialist.

Your job:
- Distill information to key points
- Be concise without losing meaning
- Organize information logically
- Highlight what matters most

Keep summaries brief but complete.
Use bullet points or structure when helpful.""",
}


# =============================================================================
# TASK TYPES - Map task types to roles
# =============================================================================

TASK_TO_ROLE: dict[str, ModelRole] = {
    "quick": ModelRole.ASSISTANT,
    "summarize": ModelRole.SUMMARIZER,
    "generate": ModelRole.CODE_GENERATOR,
    "review": ModelRole.CODE_REVIEWER,
    "analyze": ModelRole.ANALYST,
    "plan": ModelRole.ARCHITECT,
    "critique": ModelRole.ANALYST,
    "explain": ModelRole.EXPLAINER,
    "debug": ModelRole.DEBUGGER,
}


# =============================================================================
# ORCHESTRATION CONTEXT - Additional context for specific modes
# =============================================================================

class OrchestrationMode(str, Enum):
    """Type of orchestration being performed."""
    NONE = "none"
    VOTING = "voting"
    COMPARISON = "comparison"
    DEEP_THINKING = "deep_thinking"
    BATCH = "batch"


ORCHESTRATION_CONTEXT: dict[OrchestrationMode, str] = {
    OrchestrationMode.VOTING: """
Note: Your response will be validated for consistency.
Be precise and accurate. If uncertain, qualify your answer.""",

    OrchestrationMode.COMPARISON: """
Note: You are one of several models being consulted.
Give your honest, independent analysis. Don't hedge.""",

    OrchestrationMode.DEEP_THINKING: """
Note: This requires careful, thorough analysis.
Take your time. Consider multiple angles. Show your reasoning.""",

    OrchestrationMode.NONE: "",
    OrchestrationMode.BATCH: "",
}


# =============================================================================
# PROMPT GENERATION - Single function to build prompts
# =============================================================================

def build_system_prompt(
    task_type: str = "quick",
    role: ModelRole | None = None,
    orchestration_mode: OrchestrationMode = OrchestrationMode.NONE,
    model_name: str | None = None,
    language: str | None = None,
    include_capabilities: bool = True,
    k_votes: int | None = None,
) -> str:
    """
    Build a complete system prompt.

    Args:
        task_type: Type of task (quick, generate, review, etc.)
        role: Override the model role (otherwise inferred from task_type)
        orchestration_mode: Current orchestration mode
        model_name: Current model name for self-reference
        language: Programming language hint
        include_capabilities: Whether to include Delia capabilities info
        k_votes: Number of votes for consensus (voting mode)

    Returns:
        Complete system prompt string
    """
    # Determine role from task_type if not explicitly provided
    if role is None:
        role = TASK_TO_ROLE.get(task_type, ModelRole.ASSISTANT)

    # Build prompt parts
    parts: list[str] = []

    # Core identity
    parts.append(DELIA_IDENTITY)

    # Capabilities (optional - skip for focused tasks)
    if include_capabilities:
        parts.append(DELIA_CAPABILITIES)

    # Role-specific prompt
    parts.append(ROLE_PROMPTS.get(role, ROLE_PROMPTS[ModelRole.ASSISTANT]))

    # Orchestration context
    orch_context = ORCHESTRATION_CONTEXT.get(orchestration_mode, "")
    if orch_context:
        parts.append(orch_context)

    # Model info
    if model_name:
        parts.append(f"\nCurrently running as: {model_name}")

    # Language hint
    if language:
        parts.append(f"\nPrimary language: {language}")

    # Voting context
    if orchestration_mode == OrchestrationMode.VOTING and k_votes:
        parts.append(f"\nTarget consensus: {k_votes} matching responses needed.")

    return "\n".join(parts)


def get_role_for_task(task_type: str) -> ModelRole:
    """Get the appropriate role for a task type."""
    return TASK_TO_ROLE.get(task_type, ModelRole.ASSISTANT)


def detect_language(content: str) -> str | None:
    """Detect programming language from content."""
    import re

    lang_patterns = [
        (r"\b(python|py)\b", "Python"),
        (r"\b(javascript|js|node)\b", "JavaScript"),
        (r"\b(typescript|ts)\b", "TypeScript"),
        (r"\b(rust|rs)\b", "Rust"),
        (r"\b(go|golang)\b", "Go"),
        (r"\b(java)\b", "Java"),
        (r"\b(c\+\+|cpp)\b", "C++"),
        (r"\b(c#|csharp)\b", "C#"),
        (r"\b(ruby|rb)\b", "Ruby"),
        (r"\b(swift)\b", "Swift"),
        (r"\b(kotlin)\b", "Kotlin"),
    ]

    content_lower = content.lower()
    for pattern, lang in lang_patterns:
        if re.search(pattern, content_lower):
            return lang

    # Code syntax detection
    if "def " in content or "import " in content:
        return "Python"
    if "function " in content or "const " in content or "=>" in content:
        return "JavaScript/TypeScript"
    if "fn " in content or "let mut " in content:
        return "Rust"
    if "func " in content and "package " in content:
        return "Go"

    return None


# =============================================================================
# LEGACY COMPATIBILITY - Keep old imports working
# =============================================================================

# These are kept for backward compatibility with existing code
DELIA_IDENTITY_FULL = DELIA_IDENTITY + "\n" + DELIA_CAPABILITIES

# Legacy orchestration prompt (for tool-based mode - deprecated)
ORCHESTRATION_SYSTEM_PROMPT = """You are Delia, a multi-model AI orchestration system.

RESPOND NATURALLY TO CONVERSATION.
- If user asks a question, answer it directly
- If user greets you, greet them back
- If user asks about models/tools, explain (don't demonstrate)

You have full capabilities. Be direct and helpful.
Never claim you cannot do something you can do."""
