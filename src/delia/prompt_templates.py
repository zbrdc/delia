# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Enhanced prompt templating system for Delia MCP server.

Provides structured prompt templates with JSON schema integration and task-specific optimizations.
"""

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
    """Manages structured prompt templates for different task types."""

    def __init__(self) -> None:
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, Template]:
        """Load Jinja2 templates for different task types."""
        return {
            "quick": Template(
                """
You are a helpful AI assistant. Answer the following question or request concisely and accurately.

Context:
{% if context_files %}
Related files: {{ context_files|join(', ') }}
{% endif %}
{% if language %}
Language: {{ language }}
{% endif %}

Request: {{ content }}

{% if symbols %}
Focus areas: {{ symbols|join(', ') }}
{% endif %}

Provide a direct, helpful response.
""".strip()
            ),
            "generate": Template(
                """
You are an expert {{ language }} developer. Generate high-quality code based on the following requirements.

Context:
{% if file_path %}
File: {{ file_path }}
{% endif %}
{% if context_files %}
Related files: {{ context_files|join(', ') }}
{% endif %}
{% if symbols %}
Symbols to consider: {{ symbols|join(', ') }}
{% endif %}

Requirements: {{ content }}

{% if user_instructions %}
Additional instructions: {{ user_instructions }}
{% endif %}

Generate complete, working {{ language }} code with proper error handling and documentation.
""".strip()
            ),
            "review": Template(
                """
You are a senior {{ language }} code reviewer. Analyze the following code for quality, correctness, and best practices.

Code to review:
```
{{ content }}
```

{% if file_path %}
File: {{ file_path }}
{% endif %}
{% if context_files %}
Related context files: {{ context_files|join(', ') }}
{% endif %}

Review criteria:
- Code correctness and logic
- Performance considerations
- Security vulnerabilities
- Code style and readability
- Best practices compliance
- Potential improvements

{% if symbols %}
Key symbols to focus on: {{ symbols|join(', ') }}
{% endif %}

Provide specific, actionable feedback with severity levels (Critical/Major/Minor).
""".strip()
            ),
            "analyze": Template(
                """
You are a technical analyst specializing in {{ language }} code analysis.

Code to analyze:
```
{{ content }}
```

{% if file_path %}
File location: {{ file_path }}
{% endif %}
{% if context_files %}
Related files for context: {{ context_files|join(', ') }}
{% endif %}

Analysis objectives:
- Understand the code's purpose and functionality
- Identify key algorithms and data structures
- Assess code complexity and maintainability
- Find potential bugs or issues
- Suggest improvements and optimizations

{% if symbols %}
Focus on these symbols: {{ symbols|join(', ') }}
{% endif %}

{% if user_instructions %}
Specific analysis requirements: {{ user_instructions }}
{% endif %}

Provide a comprehensive analysis with specific findings and recommendations.
""".strip()
            ),
            "plan": Template(
                """
You are an expert software architect. Create a detailed implementation plan for the following requirements.

Project requirements: {{ content }}

{% if language %}
Primary language: {{ language }}
{% endif %}
{% if file_path %}
Target file: {{ file_path }}
{% endif %}
{% if context_files %}
Existing codebase: {{ context_files|join(', ') }}
{% endif %}

Planning considerations:
1. Architecture and design decisions
2. Implementation steps and phases
3. Dependencies and prerequisites
4. Testing strategy
5. Potential challenges and mitigations
6. Timeline and milestones

{% if symbols %}
Key components to consider: {{ symbols|join(', ') }}
{% endif %}

{% if user_instructions %}
Additional constraints: {{ user_instructions }}
{% endif %}

Provide a structured plan with clear, actionable steps.
""".strip()
            ),
            "critique": Template(
                """
You are a technical lead conducting a thorough code review and critique.

Code under review:
```
{{ content }}
```

{% if file_path %}
Location: {{ file_path }}
{% endif %}
{% if context_files %}
Codebase context: {{ context_files|join(', ') }}
{% endif %}

Critique dimensions:
- Architectural soundness
- Code quality and maintainability
- Performance implications
- Security considerations
- Scalability potential
- Testing adequacy
- Documentation quality

{% if symbols %}
Critical symbols to evaluate: {{ symbols|join(', ') }}
{% endif %}

{% if user_instructions %}
Review focus areas: {{ user_instructions }}
{% endif %}

Provide a balanced critique with strengths, weaknesses, and prioritized improvement recommendations.
""".strip()
            ),
        }

    def render_prompt(self, task_context: TaskContext) -> str:
        """Render a structured prompt using the appropriate template."""
        template = self.templates.get(task_context.task_type, self.templates["quick"])

        # Convert context to dict for template rendering
        context_dict = task_context.model_dump()

        return template.render(**context_dict)

    def get_available_templates(self) -> list[str]:
        """Get list of available template types."""
        return list(self.templates.keys())


# Global instance for easy access
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
    Create a structured prompt using the template system.

    Args:
        task_type: Type of task (quick, generate, review, analyze, plan, critique)
        content: Main content to process
        language: Programming language if applicable
        symbols: Code symbols to focus on
        file_path: File path if applicable
        context_files: Related files for context
        user_instructions: Additional user instructions

    Returns:
        Structured prompt string optimized for the task type
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
