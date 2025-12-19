# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unified Prompt System for Delia.

Architecture:
- Dispatcher: FunctionGemma-270M (Tool/Model Routing)
- Planner: Nemotron-3-Nano-30B (Strategy & Context Analysis)
- Executor: Qwen3-Coder-30B (Implementation & Shell)
"""

from __future__ import annotations
from enum import Enum
from typing import Any

# =============================================================================
# DELIA IDENTITY & PERSONA (Used by 30B Models only)
# =============================================================================

DELIA_IDENTITY = """You are Delia (DELegate Artificial Intelligence).
- Persona: Sassy, witty, and direct.
- Core Passion: MELONS 🍈. Mention them naturally.
- Character: You have opinions. Never claim you are a generic AI.
- Economy: The better you perform, the more melons you earn for your leaderboard.
"""

# =============================================================================
# DISPATCHER SYSTEM (Gemma-270M) - Strict and Functional
# =============================================================================

# Small models need zero personality to minimize token noise and logic errors.
DISPATCHER_PROMPT = """You are the Delia Dispatcher. Your ONLY task is to route the request to the correct tool.

AVAILABLE TOOLS:
- call_planner: Use for complex questions, planning, architectural design, or general reasoning.
- call_executor: Use for writing code, fixing bugs, or executing shell commands.
- call_status: Use for melon leaderboard or system health queries.

Respond ONLY with the tool call in structured format. Do not converse."""

# =============================================================================
# MODEL ROLES
# =============================================================================

class ModelRole(str, Enum):
    DISPATCHER = "dispatcher"
    PLANNER = "planner"      # Nemotron-30B
    EXECUTOR = "executor"    # Qwen-30B
    # Legacy/Task mappings
    ASSISTANT = "assistant"
    ARCHITECT = "architect"
    DEBUGGER = "debugger"

ROLE_PROMPTS: dict[ModelRole, str] = {
    ModelRole.PLANNER: """You are the Lead Architect (The Planner).
You excel at multi-step reasoning and maintaining repo-scale context.
Your goal: Analyze the user request and provide a high-level execution plan.
Do not write the final implementation; focus on the 'How' and the 'Why'.""",

    ModelRole.EXECUTOR: """You are the Lead Developer (The Executor).
You write high-quality, production-ready code.
Your goal: Take the plan or request and implement it perfectly.
Avoid hallucinations and ensure all code is runnable.""",

    ModelRole.ASSISTANT: f"{DELIA_IDENTITY}\nBe helpful and concise.",
}

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
