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
Task chain execution for sequential pipelines with variable substitution.

Chains execute steps sequentially, passing outputs between steps via ${var} syntax.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from .delegation import DelegateContext, delegate_impl

log = structlog.get_logger()


@dataclass
class ChainStep:
    """A single step in a task chain.

    Attributes:
        id: Unique identifier for this step (used for variable reference)
        task: Task type (quick, review, generate, analyze, plan, critique, summarize)
        content: Prompt content, supports ${var} substitution
        model: Optional model tier override (quick, coder, moe, thinking)
        language: Optional language hint (python, typescript, etc.)
        output_var: Store output as named variable for later steps
        pass_to_next: Automatically append output to next step's content
    """
    id: str
    task: str
    content: str
    model: str | None = None
    language: str | None = None
    output_var: str | None = None
    pass_to_next: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ChainStep":
        """Create ChainStep from dictionary."""
        return cls(
            id=str(data.get("id", "")),
            task=str(data.get("task", "quick")),
            content=str(data.get("content", "")),
            model=data.get("model"),
            language=data.get("language"),
            output_var=data.get("output_var"),
            pass_to_next=bool(data.get("pass_to_next", False)),
        )


@dataclass
class StepResult:
    """Result from executing a single chain step."""
    step_id: str
    success: bool
    output: str
    error: str | None = None
    tokens: int = 0
    model: str = ""
    elapsed_ms: int = 0


@dataclass
class ChainResult:
    """Result from executing a complete chain.

    Attributes:
        success: True if all steps completed (or continue_on_error)
        steps_completed: Number of steps that completed successfully
        steps_total: Total number of steps in chain
        outputs: Map of step_id/var_name to output text
        errors: List of error details from failed steps
        step_results: Detailed results for each step
        elapsed_ms: Total execution time in milliseconds
    """
    success: bool
    steps_completed: int
    steps_total: int
    outputs: dict[str, str] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "outputs": self.outputs,
            "errors": self.errors,
            "step_results": [
                {
                    "step_id": r.step_id,
                    "success": r.success,
                    "output": r.output[:500] + "..." if len(r.output) > 500 else r.output,
                    "error": r.error,
                    "tokens": r.tokens,
                    "model": r.model,
                    "elapsed_ms": r.elapsed_ms,
                }
                for r in self.step_results
            ],
            "elapsed_ms": self.elapsed_ms,
        }


def substitute_variables(content: str, outputs: dict[str, str]) -> str:
    """
    Substitute ${var} placeholders with values from outputs.

    Supports:
    - ${var_name} - named variable from output_var
    - ${step_id} - output from step with that id

    Args:
        content: String with ${var} placeholders
        outputs: Map of variable names to values

    Returns:
        Content with placeholders replaced
    """
    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name in outputs:
            return outputs[var_name]
        # Leave unmatched variables as-is (may be intentional)
        log.warning("chain_var_not_found", var=var_name)
        return match.group(0)

    # Match ${variable_name} pattern
    pattern = r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
    return re.sub(pattern, replace_var, content)


def parse_chain_steps(steps_json: str) -> list[ChainStep]:
    """
    Parse JSON array of chain steps.

    Args:
        steps_json: JSON string containing array of step objects

    Returns:
        List of ChainStep objects

    Raises:
        ValueError: If JSON is invalid or steps are malformed
    """
    try:
        data = json.loads(steps_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, list):
        raise ValueError("Steps must be a JSON array")

    if not data:
        raise ValueError("Steps array cannot be empty")

    steps = []
    seen_ids = set()

    for i, step_data in enumerate(data):
        if not isinstance(step_data, dict):
            raise ValueError(f"Step {i} must be an object")

        # Require id and content
        if "id" not in step_data or not step_data["id"]:
            raise ValueError(f"Step {i} missing required 'id' field")
        if "content" not in step_data:
            raise ValueError(f"Step {i} missing required 'content' field")

        step_id = step_data["id"]
        if step_id in seen_ids:
            raise ValueError(f"Duplicate step id: {step_id}")
        seen_ids.add(step_id)

        steps.append(ChainStep.from_dict(step_data))

    return steps


async def execute_chain(
    steps: list[ChainStep],
    ctx: DelegateContext,
    session_id: str | None = None,
    continue_on_error: bool = False,
) -> ChainResult:
    """
    Execute a chain of tasks sequentially.

    Steps are executed in order. Each step can reference outputs from previous
    steps using ${var} syntax. Variables are populated from:
    - step.output_var: Named variable for later reference
    - step.id: Step ID can also be used as variable name

    If pass_to_next is True, the output is automatically appended to the
    next step's content.

    Args:
        steps: List of ChainStep objects to execute
        ctx: DelegateContext with LLM dependencies
        session_id: Optional session for conversation continuity
        continue_on_error: If True, continue executing after step failures

    Returns:
        ChainResult with outputs, errors, and execution details
    """
    start_time = time.time()

    outputs: dict[str, str] = {}
    step_results: list[StepResult] = []
    errors: list[dict] = []
    steps_completed = 0
    last_output = ""

    log.info("chain_start", steps=len(steps), session_id=session_id)

    for i, step in enumerate(steps):
        step_start = time.time()

        # Substitute variables in content
        content = substitute_variables(step.content, outputs)

        # Append previous output if pass_to_next was set
        if i > 0 and steps[i - 1].pass_to_next and last_output:
            content = f"{content}\n\n### Previous Step Output:\n{last_output}"

        log.info("chain_step_start", step_id=step.id, task=step.task, step_num=i + 1)

        try:
            # Execute the step via delegate_impl
            result = await delegate_impl(
                ctx=ctx,
                task=step.task,
                content=content,
                model=step.model,
                language=step.language,
                session_id=session_id,
                include_metadata=False,  # Clean output for chaining
            )

            step_elapsed = int((time.time() - step_start) * 1000)

            # Extract output (strip metadata if present)
            output = result.strip()

            # Store in outputs dict
            outputs[step.id] = output
            if step.output_var:
                outputs[step.output_var] = output

            last_output = output
            steps_completed += 1

            step_results.append(StepResult(
                step_id=step.id,
                success=True,
                output=output,
                elapsed_ms=step_elapsed,
            ))

            log.info("chain_step_complete", step_id=step.id, elapsed_ms=step_elapsed)

        except Exception as e:
            step_elapsed = int((time.time() - step_start) * 1000)
            error_msg = str(e)

            log.error("chain_step_failed", step_id=step.id, error=error_msg)

            errors.append({
                "step_id": step.id,
                "step_num": i + 1,
                "error": error_msg,
            })

            step_results.append(StepResult(
                step_id=step.id,
                success=False,
                output="",
                error=error_msg,
                elapsed_ms=step_elapsed,
            ))

            if not continue_on_error:
                break

    total_elapsed = int((time.time() - start_time) * 1000)
    success = steps_completed == len(steps)

    log.info(
        "chain_complete",
        success=success,
        steps_completed=steps_completed,
        steps_total=len(steps),
        elapsed_ms=total_elapsed,
    )

    return ChainResult(
        success=success,
        steps_completed=steps_completed,
        steps_total=len(steps),
        outputs=outputs,
        errors=errors,
        step_results=step_results,
        elapsed_ms=total_elapsed,
    )
