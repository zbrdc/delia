# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Legacy prompt templates module.

This module is DEPRECATED. Use src/delia/prompts.py instead.

All exports are re-exported from the unified prompts module
for backward compatibility.
"""

# Re-export everything from the unified system
from .prompts import (
    DELIA_IDENTITY,
    DELIA_IDENTITY_FULL,
    DELIA_CAPABILITIES,
    ORCHESTRATION_SYSTEM_PROMPT,
    ModelRole,
    OrchestrationMode,
    ROLE_PROMPTS,
    TASK_TO_ROLE,
    ORCHESTRATION_CONTEXT,
    build_system_prompt,
    get_role_for_task,
    detect_language,
)

# Legacy Jinja template support - kept for backward compatibility
from jinja2 import Template
from pydantic import BaseModel, Field


class TaskContext(BaseModel):
    """Structured context for task execution."""

    task_type: str = Field(..., description="Type of task (quick, generate, review, etc.)")
    content: str = Field(..., description="Main content to process")
    language: str | None = Field(None, description="Programming language if applicable")
    symbols: list[str] | None = Field(default_factory=list, description="Code symbols to focus on")
    file_path: str | None = Field(None, description="File path if applicable")
    context_files: list[str] | None = Field(default_factory=list, description="Related files for context")
    user_instructions: str | None = Field(None, description="Additional user instructions")


class PromptTemplateManager:
    """
    DEPRECATED: Use build_system_prompt() from prompts.py instead.
    
    Manages structured prompt templates for different task types.
    Kept for backward compatibility.
    """

    def __init__(self) -> None:
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, Template]:
        """Load Jinja2 templates for different task types."""
        return {
            "quick": Template(
                """
{{ delia_identity }}

Answer the following question or request concisely and accurately.

{% if language %}Language: {{ language }}{% endif %}

Request: {{ content }}

Provide a direct, helpful response.
""".strip()
            ),
            "generate": Template(
                """
You are an expert {{ language }} developer.

Requirements: {{ content }}

Generate complete, working code with proper error handling.
""".strip()
            ),
            "review": Template(
                """
You are a senior code reviewer.

Code to review:
```
{{ content }}
```

Review for: correctness, performance, security, style.
Provide specific, actionable feedback with severity levels.
""".strip()
            ),
            "analyze": Template(
                """
Analyze the following:

{{ content }}

Provide comprehensive analysis with specific findings.
""".strip()
            ),
            "plan": Template(
                """
Create a detailed implementation plan for:

{{ content }}

Include: architecture, steps, dependencies, timeline.
""".strip()
            ),
        }

    def render_prompt(self, task_context: TaskContext) -> str:
        """Render a structured prompt using the appropriate template."""
        template = self.templates.get(task_context.task_type, self.templates["quick"])
        context_dict = task_context.model_dump()
        context_dict["delia_identity"] = DELIA_IDENTITY
        return template.render(**context_dict)

    def get_available_templates(self) -> list[str]:
        """Get list of available template types."""
        return list(self.templates.keys())


prompt_manager = PromptTemplateManager()


def create_structured_prompt(
    task_type: str,
    content: str,
    language: str | None = None,
    symbols: list[str] | None = None,
    file_path: str | None = None,
    context_files: list[str] | None = None,
    user_instructions: str | None = None,
) -> str:
    """
    DEPRECATED: Use build_system_prompt() from prompts.py instead.
    
    Create a structured prompt using the template system.
    """
    context = TaskContext(
        task_type=task_type,
        content=content,
        language=language,
        symbols=symbols or [],
        file_path=file_path,
        context_files=context_files or [],
        user_instructions=user_instructions,
    )
    return prompt_manager.render_prompt(context)


__all__ = [
    # From unified prompts
    "DELIA_IDENTITY",
    "DELIA_IDENTITY_FULL", 
    "DELIA_CAPABILITIES",
    "ORCHESTRATION_SYSTEM_PROMPT",
    "ModelRole",
    "OrchestrationMode",
    "ROLE_PROMPTS",
    "build_system_prompt",
    "get_role_for_task",
    "detect_language",
    # Legacy (deprecated)
    "TaskContext",
    "PromptTemplateManager",
    "prompt_manager",
    "create_structured_prompt",
]
