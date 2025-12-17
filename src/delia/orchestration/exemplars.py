# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Intent Exemplars for Semantic Matching.

This file contains canonical examples of user messages that map to
specific intents (OrchestrationMode, ModelRole, task_type). The
SemanticIntentMatcher uses these to classify new messages via
embedding similarity.

Design:
- Each exemplar is a real-world example of how users express intent
- Exemplars should cover variations in phrasing
- Keep exemplars concise - focus on the intent signal
- Add new exemplars when detection fails on real queries
"""

from __future__ import annotations

from dataclasses import dataclass

from ..prompts import ModelRole, OrchestrationMode


@dataclass
class IntentExemplar:
    """A canonical example mapping text to intent."""
    text: str
    orchestration_mode: OrchestrationMode | None = None
    model_role: ModelRole | None = None
    task_type: str | None = None
    priority: float = 1.0  # Higher = more weight in matching


# =============================================================================
# ORCHESTRATION MODE EXEMPLARS
# =============================================================================

VOTING_EXEMPLARS = [
    # Explicit verification requests
    IntentExemplar("make sure this answer is correct", OrchestrationMode.VOTING),
    IntentExemplar("verify this is accurate", OrchestrationMode.VOTING),
    IntentExemplar("double check this for me", OrchestrationMode.VOTING),
    IntentExemplar("confirm this is right", OrchestrationMode.VOTING),
    IntentExemplar("I need a reliable response", OrchestrationMode.VOTING),
    IntentExemplar("this is important, get it right", OrchestrationMode.VOTING),
    IntentExemplar("can you validate this answer", OrchestrationMode.VOTING),
    IntentExemplar("ensure this is accurate", OrchestrationMode.VOTING),
    
    # Implicit high-stakes signals
    IntentExemplar("this is critical, don't mess it up", OrchestrationMode.VOTING),
    IntentExemplar("I'm submitting this to production", OrchestrationMode.VOTING),
    IntentExemplar("lives depend on this being correct", OrchestrationMode.VOTING),
    IntentExemplar("this is going to a customer", OrchestrationMode.VOTING),
    IntentExemplar("make absolutely certain", OrchestrationMode.VOTING),
    IntentExemplar("I need 100% accuracy on this", OrchestrationMode.VOTING),
    
    # Mathematical/logical correctness
    IntentExemplar("is this calculation correct", OrchestrationMode.VOTING),
    IntentExemplar("verify my math", OrchestrationMode.VOTING),
    IntentExemplar("check if this logic is sound", OrchestrationMode.VOTING),
]

COMPARISON_EXEMPLARS = [
    # Explicit comparison requests
    IntentExemplar("what do different models think about this", OrchestrationMode.COMPARISON),
    IntentExemplar("compare perspectives on this", OrchestrationMode.COMPARISON),
    IntentExemplar("get me a second opinion", OrchestrationMode.COMPARISON),
    IntentExemplar("I want multiple viewpoints", OrchestrationMode.COMPARISON),
    IntentExemplar("ask different models", OrchestrationMode.COMPARISON),
    IntentExemplar("compare qwen vs deepseek on this", OrchestrationMode.COMPARISON),
    IntentExemplar("what would other models say", OrchestrationMode.COMPARISON),
    
    # Implicit comparison signals
    IntentExemplar("I'm not sure which approach is better", OrchestrationMode.COMPARISON),
    IntentExemplar("there are multiple ways to do this", OrchestrationMode.COMPARISON),
    IntentExemplar("I want to see different options", OrchestrationMode.COMPARISON),
]

DEEP_THINKING_EXEMPLARS = [
    # Explicit deep thinking requests
    IntentExemplar("think carefully about this", OrchestrationMode.DEEP_THINKING),
    IntentExemplar("analyze this thoroughly", OrchestrationMode.DEEP_THINKING),
    IntentExemplar("I need a deep analysis", OrchestrationMode.DEEP_THINKING),
    IntentExemplar("take your time and think step by step", OrchestrationMode.DEEP_THINKING),
    IntentExemplar("consider all the tradeoffs", OrchestrationMode.DEEP_THINKING),
    IntentExemplar("what are the implications", OrchestrationMode.DEEP_THINKING),
    
    # Architecture/design signals
    IntentExemplar("design a system architecture for", OrchestrationMode.DEEP_THINKING),
    IntentExemplar("how should I architect this", OrchestrationMode.DEEP_THINKING),
    IntentExemplar("what's the best approach for", OrchestrationMode.DEEP_THINKING),
    IntentExemplar("help me plan this project", OrchestrationMode.DEEP_THINKING),
    IntentExemplar("strategic advice on", OrchestrationMode.DEEP_THINKING),
]

AGENTIC_EXEMPLARS = [
    # File operations
    IntentExemplar("read the file src/main.py", OrchestrationMode.AGENTIC),
    IntentExemplar("show me what's in that file", OrchestrationMode.AGENTIC),
    IntentExemplar("list the files in src directory", OrchestrationMode.AGENTIC),
    IntentExemplar("create a new file called test.py", OrchestrationMode.AGENTIC),
    IntentExemplar("write this code to utils.py", OrchestrationMode.AGENTIC),
    IntentExemplar("delete the temp files", OrchestrationMode.AGENTIC),
    
    # Shell commands
    IntentExemplar("run npm install", OrchestrationMode.AGENTIC),
    IntentExemplar("execute pip install requests", OrchestrationMode.AGENTIC),
    IntentExemplar("run the tests", OrchestrationMode.AGENTIC),
    IntentExemplar("compile the project", OrchestrationMode.AGENTIC),
    IntentExemplar("start the server", OrchestrationMode.AGENTIC),
    IntentExemplar("run ls -la", OrchestrationMode.AGENTIC),
    IntentExemplar("git status", OrchestrationMode.AGENTIC),
    
    # Code search
    IntentExemplar("find all usages of this function", OrchestrationMode.AGENTIC),
    IntentExemplar("search for TODO comments", OrchestrationMode.AGENTIC),
    IntentExemplar("grep for error handling", OrchestrationMode.AGENTIC),
]


# =============================================================================
# MODEL ROLE EXEMPLARS
# =============================================================================

CODE_REVIEWER_EXEMPLARS = [
    IntentExemplar("review this code", model_role=ModelRole.CODE_REVIEWER, task_type="coder"),
    IntentExemplar("is this code secure", model_role=ModelRole.CODE_REVIEWER, task_type="coder"),
    IntentExemplar("audit this for vulnerabilities", model_role=ModelRole.CODE_REVIEWER, task_type="coder"),
    IntentExemplar("check for bugs in this function", model_role=ModelRole.CODE_REVIEWER, task_type="coder"),
    IntentExemplar("what's wrong with this code", model_role=ModelRole.CODE_REVIEWER, task_type="coder"),
    IntentExemplar("can you find issues in this", model_role=ModelRole.CODE_REVIEWER, task_type="coder"),
    IntentExemplar("security review please", model_role=ModelRole.CODE_REVIEWER, task_type="coder"),
]

CODE_GENERATOR_EXEMPLARS = [
    IntentExemplar("write a function that", model_role=ModelRole.CODE_GENERATOR, task_type="coder"),
    IntentExemplar("create a class for", model_role=ModelRole.CODE_GENERATOR, task_type="coder"),
    IntentExemplar("implement this feature", model_role=ModelRole.CODE_GENERATOR, task_type="coder"),
    IntentExemplar("generate code to", model_role=ModelRole.CODE_GENERATOR, task_type="coder"),
    IntentExemplar("code this for me", model_role=ModelRole.CODE_GENERATOR, task_type="coder"),
    IntentExemplar("build a REST API for", model_role=ModelRole.CODE_GENERATOR, task_type="coder"),
    IntentExemplar("write unit tests for", model_role=ModelRole.CODE_GENERATOR, task_type="coder"),
]

ARCHITECT_EXEMPLARS = [
    IntentExemplar("design the architecture for", model_role=ModelRole.ARCHITECT, task_type="moe"),
    IntentExemplar("how should I structure this system", model_role=ModelRole.ARCHITECT, task_type="moe"),
    IntentExemplar("what's the best design pattern for", model_role=ModelRole.ARCHITECT, task_type="moe"),
    IntentExemplar("architect a solution for", model_role=ModelRole.ARCHITECT, task_type="moe"),
    IntentExemplar("system design question", model_role=ModelRole.ARCHITECT, task_type="moe"),
    IntentExemplar("how would you structure this application", model_role=ModelRole.ARCHITECT, task_type="moe"),
]

EXPLAINER_EXEMPLARS = [
    IntentExemplar("explain this to me", model_role=ModelRole.EXPLAINER, task_type="quick"),
    IntentExemplar("help me understand", model_role=ModelRole.EXPLAINER, task_type="quick"),
    IntentExemplar("what does this mean", model_role=ModelRole.EXPLAINER, task_type="quick"),
    IntentExemplar("teach me about", model_role=ModelRole.EXPLAINER, task_type="quick"),
    IntentExemplar("explain like I'm five", model_role=ModelRole.EXPLAINER, task_type="quick"),
    IntentExemplar("can you clarify", model_role=ModelRole.EXPLAINER, task_type="quick"),
    IntentExemplar("walk me through this", model_role=ModelRole.EXPLAINER, task_type="quick"),
]

DEBUGGER_EXEMPLARS = [
    IntentExemplar("debug this error", model_role=ModelRole.DEBUGGER, task_type="coder"),
    IntentExemplar("why is this failing", model_role=ModelRole.DEBUGGER, task_type="coder"),
    IntentExemplar("fix this bug", model_role=ModelRole.DEBUGGER, task_type="coder"),
    IntentExemplar("I'm getting an error", model_role=ModelRole.DEBUGGER, task_type="coder"),
    IntentExemplar("help me troubleshoot", model_role=ModelRole.DEBUGGER, task_type="coder"),
    IntentExemplar("this isn't working, can you help", model_role=ModelRole.DEBUGGER, task_type="coder"),
    IntentExemplar("stack trace says", model_role=ModelRole.DEBUGGER, task_type="coder"),
]

ANALYST_EXEMPLARS = [
    IntentExemplar("analyze this data", model_role=ModelRole.ANALYST, task_type="moe"),
    IntentExemplar("what are the trends here", model_role=ModelRole.ANALYST, task_type="moe"),
    IntentExemplar("give me insights on", model_role=ModelRole.ANALYST, task_type="moe"),
    IntentExemplar("evaluate this approach", model_role=ModelRole.ANALYST, task_type="moe"),
    IntentExemplar("assess the performance", model_role=ModelRole.ANALYST, task_type="moe"),
]

SUMMARIZER_EXEMPLARS = [
    IntentExemplar("summarize this", model_role=ModelRole.SUMMARIZER, task_type="quick"),
    IntentExemplar("give me the key points", model_role=ModelRole.SUMMARIZER, task_type="quick"),
    IntentExemplar("tl;dr", model_role=ModelRole.SUMMARIZER, task_type="quick"),
    IntentExemplar("what's the gist", model_role=ModelRole.SUMMARIZER, task_type="quick"),
    IntentExemplar("brief overview of", model_role=ModelRole.SUMMARIZER, task_type="quick"),
]


# =============================================================================
# TASK TYPE EXEMPLARS (without specific role)
# =============================================================================

CODER_TASK_EXEMPLARS = [
    IntentExemplar("python function to", task_type="coder"),
    IntentExemplar("typescript class for", task_type="coder"),
    IntentExemplar("javascript code that", task_type="coder"),
    IntentExemplar("rust implementation of", task_type="coder"),
    IntentExemplar("SQL query to", task_type="coder"),
    IntentExemplar("regex pattern for", task_type="coder"),
    IntentExemplar("API endpoint for", task_type="coder"),
    IntentExemplar("async function that", task_type="coder"),
]

MOE_TASK_EXEMPLARS = [
    IntentExemplar("complex problem about", task_type="moe"),
    IntentExemplar("tradeoffs between", task_type="moe"),
    IntentExemplar("strategic decision", task_type="moe"),
    IntentExemplar("evaluate multiple options", task_type="moe"),
    IntentExemplar("long-term implications", task_type="moe"),
]

QUICK_TASK_EXEMPLARS = [
    IntentExemplar("hello", task_type="quick"),
    IntentExemplar("hi there", task_type="quick"),
    IntentExemplar("what time is it", task_type="quick"),
    IntentExemplar("thanks", task_type="quick"),
    IntentExemplar("yes", task_type="quick"),
    IntentExemplar("no", task_type="quick"),
    IntentExemplar("okay", task_type="quick"),
    IntentExemplar("quick question", task_type="quick"),
    IntentExemplar("simple question", task_type="quick"),
]


# =============================================================================
# STATUS QUERY EXEMPLARS (no LLM needed)
# =============================================================================

STATUS_EXEMPLARS = [
    IntentExemplar("show me the melon rankings", task_type="status"),
    IntentExemplar("what's the leaderboard", task_type="status"),
    IntentExemplar("melon status", task_type="status"),
    IntentExemplar("how are models performing", task_type="status"),
    IntentExemplar("model rankings", task_type="status"),
    IntentExemplar("show melons", task_type="status"),
    IntentExemplar("melon leaderboard", task_type="status"),
]


# =============================================================================
# ALL EXEMPLARS COMBINED
# =============================================================================

def get_all_exemplars() -> list[IntentExemplar]:
    """Get all exemplars for embedding."""
    return (
        VOTING_EXEMPLARS +
        COMPARISON_EXEMPLARS +
        DEEP_THINKING_EXEMPLARS +
        AGENTIC_EXEMPLARS +
        CODE_REVIEWER_EXEMPLARS +
        CODE_GENERATOR_EXEMPLARS +
        ARCHITECT_EXEMPLARS +
        EXPLAINER_EXEMPLARS +
        DEBUGGER_EXEMPLARS +
        ANALYST_EXEMPLARS +
        SUMMARIZER_EXEMPLARS +
        CODER_TASK_EXEMPLARS +
        MOE_TASK_EXEMPLARS +
        QUICK_TASK_EXEMPLARS +
        STATUS_EXEMPLARS
    )


def get_orchestration_exemplars() -> list[IntentExemplar]:
    """Get only orchestration mode exemplars."""
    return (
        VOTING_EXEMPLARS +
        COMPARISON_EXEMPLARS +
        DEEP_THINKING_EXEMPLARS +
        AGENTIC_EXEMPLARS
    )


def get_role_exemplars() -> list[IntentExemplar]:
    """Get only model role exemplars."""
    return (
        CODE_REVIEWER_EXEMPLARS +
        CODE_GENERATOR_EXEMPLARS +
        ARCHITECT_EXEMPLARS +
        EXPLAINER_EXEMPLARS +
        DEBUGGER_EXEMPLARS +
        ANALYST_EXEMPLARS +
        SUMMARIZER_EXEMPLARS
    )


def get_task_exemplars() -> list[IntentExemplar]:
    """Get only task type exemplars."""
    return (
        CODER_TASK_EXEMPLARS +
        MOE_TASK_EXEMPLARS +
        QUICK_TASK_EXEMPLARS +
        STATUS_EXEMPLARS
    )


# Convenience export
__all__ = [
    "IntentExemplar",
    "get_all_exemplars",
    "get_orchestration_exemplars",
    "get_role_exemplars",
    "get_task_exemplars",
    # Individual lists for testing
    "VOTING_EXEMPLARS",
    "COMPARISON_EXEMPLARS",
    "DEEP_THINKING_EXEMPLARS",
    "AGENTIC_EXEMPLARS",
]

