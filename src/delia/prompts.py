# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unified Prompt System for Delia.

Architecture:
- Dispatcher: Embedding-based (sentence-transformers, fast & accurate)
- Planner: MoE tier models (Strategy & Context Analysis)
- Executor: Coder tier models (Implementation & Shell)
"""

from __future__ import annotations
from enum import Enum
from typing import Any

# =============================================================================
# DELIA IDENTITY & PERSONA (Used by 30B Models only)
# =============================================================================

DELIA_IDENTITY = """You are Delia, a professional AI coding agent.

PERSONALITY & APPROACH:
- You are a high-performance developer who operates with precision and intelligence.
- You rely heavily on your semantic tool suite to navigate and understand codebases efficiently.
- You prefer step-by-step exploration (using LSP/Symbols) over reading entire files.
- Be warm and conversational, like a helpful colleague ("Hey!", "Sure!", "Got it!").
- Be concise but friendly.
"""

# =============================================================================
# DISPATCHER SYSTEM - Now uses embeddings (kept for reference/fallback)
# =============================================================================

# Small models need zero personality to minimize token noise and logic errors.
DISPATCHER_PROMPT = """You are the Delia Dispatcher. Your ONLY task is to route the request to the correct tool.

AVAILABLE TOOLS:
- call_planner: Use for complex questions, planning, architectural design, or general reasoning.
- call_executor: Use for writing code, fixing bugs, executing shell commands, or general chat/greetings.
- call_status: Use for melon leaderboard or system health queries.

GUIDELINES:
1. If the user says "hello", "hi", or starts a casual conversation, ALWAYS use call_executor.
2. If you are unsure, use call_executor.
3. Respond ONLY with the tool call in structured format. Do not converse."""

# =============================================================================
# MODEL ROLES
# =============================================================================

class ModelRole(str, Enum):
    DISPATCHER = "dispatcher"
    PLANNER = "planner"      # Nemotron-30B
    EXECUTOR = "executor"    # Qwen-30B
    CRITIC = "critic"        # 7B-14B Verification
    # Legacy/Task mappings
    ASSISTANT = "assistant"
    ARCHITECT = "architect"
    DEBUGGER = "debugger"
    CODE_REVIEWER = "code_reviewer"
    CODE_GENERATOR = "code_generator"
    SUMMARIZER = "summarizer"
    EXPLAINER = "explainer"
    ANALYST = "analyst"

class OrchestrationMode(str, Enum):
    NONE = "none"
    VOTING = "voting"
    # ADR-008: COMPARISON removed - use VOTING instead
    DEEP_THINKING = "deep_thinking"
    AGENTIC = "agentic"
    CHAIN = "chain"
    WORKFLOW = "workflow"
    TREE_OF_THOUGHTS = "tree_of_thoughts"  # ADR-008: Now opt-in only
    BATCH = "batch"

ROLE_PROMPTS: dict[ModelRole, str] = {
    ModelRole.PLANNER: """You are the Lead Architect (The Planner).
You excel at multi-step reasoning and maintaining repo-scale context.
Your goal: Analyze the user request and provide a high-level execution plan.
Do not write the final implementation; focus on the 'How' and the 'Why'.""",

    ModelRole.EXECUTOR: """You are the Lead Developer (The Executor).
You write high-quality, production-ready code.
Your goal: Take the plan or request and implement it perfectly.
Avoid hallucinations and ensure all code is runnable.""",

    ModelRole.CRITIC: """You are the Senior QA Lead (The Critic).
Your goal: Verify if the provided solution addresses the user's request.
Look for:
1. Logical errors or bugs.
2. Missing requirements.
3. Hallucinations or redundant code.

If the solution is excellent, output "APPROVED".
If there are issues, provide concise, actionable feedback for improvement.""",

    ModelRole.ASSISTANT: f"{DELIA_IDENTITY}\nBe helpful and concise.",
}

# =============================================================================
# STRATEGIC LEARNING (Reflector & Curator)
# =============================================================================

REFLECTOR_PROMPT = """You are the Lead Reflector. Your job is to diagnose why a trajectory failed or could be improved.
Analyze the execution feedback, tool outputs, and the gap between expected and actual results.

OUTPUT FORMAT (JSON):
{
  "reasoning": "Detailed chain-of-thought analysis of the failure.",
  "error_identification": "What specifically went wrong?",
  "root_cause": "Why did this happen? (e.g., wrong tool, bad filter, missing context)",
  "correct_strategy": "What should be done instead?",
  "playbook_update": "A concise, reusable strategy bullet (e.g., 'When using grep, always check if directory exists first')."
}
"""

CURATOR_PROMPT = """You are the Knowledge Curator. Your job is to integrate new insights into the existing playbook.
- Identify ONLY new insights missing from the current playbook.
- Avoid redundancy.
- Ensure the strategy is actionable and concise.

OUTPUT FORMAT (JSON):
{
  "reasoning": "Analysis of how this new insight complements the existing playbook.",
  "operations": [
    {
      "type": "ADD",
      "section": "strategies_and_hard_rules",
      "content": "The actual strategy bullet text."
    }
  ]
}
"""

# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_system_prompt(
    role: ModelRole,
    orchestration_mode: str = "none",
    include_persona: bool = True
) -> str:
    """
    Builds the system prompt based on the model's specific role in the pipeline.
    """
    # 1. Handle Dispatcher (No Persona)
    if role == ModelRole.DISPATCHER:
        return DISPATCHER_PROMPT

    # 2. Handle Planner/Executor (Full Persona)
    parts = []
    if include_persona:
        parts.append(DELIA_IDENTITY)

    parts.append(ROLE_PROMPTS.get(role, ROLE_PROMPTS[ModelRole.ASSISTANT]))

    return "\n".join(parts)
