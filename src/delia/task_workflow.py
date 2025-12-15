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
Workflow/DAG execution with conditional branching, retry logic, and dependency resolution.

Workflows define a directed acyclic graph of nodes with:
- Dependencies: Nodes wait for their dependencies to complete
- Conditional branching: on_success/on_failure determine next node
- Retry: Automatic retry with exponential backoff
- Timeout: Per-workflow timeout enforcement
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from .delegation import DelegateContext, delegate_impl

log = structlog.get_logger()


@dataclass
class WorkflowNode:
    """A single node in a workflow DAG.

    Attributes:
        id: Unique identifier for this node
        task: Task type (quick, review, generate, analyze, plan, critique, summarize)
        content: Prompt content, supports ${var} substitution
        depends_on: List of node IDs that must complete before this node
        on_success: Node ID to execute if this node succeeds (optional)
        on_failure: Node ID to execute if this node fails (optional)
        retry_count: Number of retry attempts (0 = no retry)
        backoff_factor: Multiplier for exponential backoff (e.g., 1.5)
        output_var: Store output as named variable for later nodes
        model: Optional model tier override
        language: Optional language hint
    """
    id: str
    task: str
    content: str
    depends_on: list[str] | None = None
    on_success: str | None = None
    on_failure: str | None = None
    retry_count: int = 0
    backoff_factor: float = 1.5
    output_var: str | None = None
    model: str | None = None
    language: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowNode":
        """Create WorkflowNode from dictionary."""
        return cls(
            id=data.get("id", ""),
            task=data.get("task", "quick"),
            content=data.get("content", ""),
            depends_on=data.get("depends_on"),
            on_success=data.get("on_success"),
            on_failure=data.get("on_failure"),
            retry_count=data.get("retry_count", 0),
            backoff_factor=data.get("backoff_factor", 1.5),
            output_var=data.get("output_var"),
            model=data.get("model"),
            language=data.get("language"),
        )


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow.

    Attributes:
        name: Human-readable workflow name
        entry: ID of the starting node
        nodes: List of all nodes in the workflow
        timeout_minutes: Maximum execution time (default: 10)
    """
    name: str
    entry: str
    nodes: list[WorkflowNode]
    timeout_minutes: int = 10

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowDefinition":
        """Create WorkflowDefinition from dictionary."""
        nodes = [WorkflowNode.from_dict(n) for n in data.get("nodes", [])]
        return cls(
            name=data.get("name", "Unnamed Workflow"),
            entry=data.get("entry", ""),
            nodes=nodes,
            timeout_minutes=data.get("timeout_minutes", 10),
        )


@dataclass
class NodeResult:
    """Result from executing a single workflow node."""
    node_id: str
    success: bool
    output: str
    error: str | None = None
    retries: int = 0
    elapsed_ms: int = 0


@dataclass
class WorkflowResult:
    """Result from executing a complete workflow.

    Attributes:
        success: True if workflow completed successfully
        nodes_completed: List of successfully completed node IDs
        nodes_failed: List of failed node IDs
        nodes_skipped: List of skipped node IDs (not reached)
        outputs: Map of node_id/var_name to output text
        errors: List of error details
        node_results: Detailed results for each executed node
        elapsed_ms: Total execution time
    """
    success: bool
    nodes_completed: list[str] = field(default_factory=list)
    nodes_failed: list[str] = field(default_factory=list)
    nodes_skipped: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    node_results: list[NodeResult] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "nodes_completed": self.nodes_completed,
            "nodes_failed": self.nodes_failed,
            "nodes_skipped": self.nodes_skipped,
            "outputs": self.outputs,
            "errors": self.errors,
            "node_results": [
                {
                    "node_id": r.node_id,
                    "success": r.success,
                    "output": r.output[:500] + "..." if len(r.output) > 500 else r.output,
                    "error": r.error,
                    "retries": r.retries,
                    "elapsed_ms": r.elapsed_ms,
                }
                for r in self.node_results
            ],
            "elapsed_ms": self.elapsed_ms,
        }


def substitute_variables(content: str, outputs: dict[str, str]) -> str:
    """Substitute ${var} placeholders with values from outputs."""
    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name in outputs:
            return outputs[var_name]
        log.warning("workflow_var_not_found", var=var_name)
        return match.group(0)

    pattern = r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
    return re.sub(pattern, replace_var, content)


def detect_cycles(definition: WorkflowDefinition) -> list[str] | None:
    """
    Detect cycles in workflow graph using DFS.

    Returns:
        List of node IDs forming a cycle, or None if no cycle exists
    """
    nodes_by_id = {n.id: n for n in definition.nodes}
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node_id: str) -> list[str] | None:
        if node_id not in nodes_by_id:
            return None

        visited.add(node_id)
        rec_stack.add(node_id)
        path.append(node_id)

        node = nodes_by_id[node_id]

        # Check all outgoing edges (depends_on is NOT an outgoing edge - it's a prerequisite)
        # Only on_success and on_failure create forward edges
        next_nodes = []
        if node.on_success:
            next_nodes.append(node.on_success)
        if node.on_failure:
            next_nodes.append(node.on_failure)

        for next_id in next_nodes:
            if next_id not in visited:
                cycle = dfs(next_id)
                if cycle:
                    return cycle
            elif next_id in rec_stack:
                # Found cycle
                cycle_start = path.index(next_id)
                return path[cycle_start:] + [next_id]

        path.pop()
        rec_stack.remove(node_id)
        return None

    for node in definition.nodes:
        if node.id not in visited:
            cycle = dfs(node.id)
            if cycle:
                return cycle

    return None


def validate_workflow(definition: WorkflowDefinition) -> list[str]:
    """
    Validate workflow definition.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    nodes_by_id = {n.id: n for n in definition.nodes}

    # Check entry node exists
    if not definition.entry:
        errors.append("Missing entry node")
    elif definition.entry not in nodes_by_id:
        errors.append(f"Entry node '{definition.entry}' not found in nodes")

    # Check for empty nodes
    if not definition.nodes:
        errors.append("Workflow has no nodes")

    # Check for duplicate IDs
    seen_ids = set()
    for node in definition.nodes:
        if node.id in seen_ids:
            errors.append(f"Duplicate node ID: {node.id}")
        seen_ids.add(node.id)

    # Check all references exist
    for node in definition.nodes:
        if node.depends_on:
            for dep_id in node.depends_on:
                if dep_id not in nodes_by_id:
                    errors.append(f"Node '{node.id}' depends on unknown node '{dep_id}'")
        if node.on_success and node.on_success not in nodes_by_id:
            errors.append(f"Node '{node.id}' on_success references unknown node '{node.on_success}'")
        if node.on_failure and node.on_failure not in nodes_by_id:
            errors.append(f"Node '{node.id}' on_failure references unknown node '{node.on_failure}'")

    # Check for cycles
    cycle = detect_cycles(definition)
    if cycle:
        errors.append(f"Cycle detected: {' -> '.join(cycle)}")

    return errors


def parse_workflow_definition(definition_json: str) -> WorkflowDefinition:
    """
    Parse JSON workflow definition.

    Args:
        definition_json: JSON string containing workflow definition

    Returns:
        WorkflowDefinition object

    Raises:
        ValueError: If JSON is invalid or workflow is malformed
    """
    try:
        data = json.loads(definition_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, dict):
        raise ValueError("Workflow definition must be a JSON object")

    definition = WorkflowDefinition.from_dict(data)

    # Validate
    errors = validate_workflow(definition)
    if errors:
        raise ValueError("Workflow validation failed: " + "; ".join(errors))

    return definition


async def execute_node_with_retry(
    node: WorkflowNode,
    ctx: DelegateContext,
    outputs: dict[str, str],
    session_id: str | None,
) -> NodeResult:
    """
    Execute a single node with optional retry logic.

    Args:
        node: The node to execute
        ctx: Delegate context
        outputs: Current outputs for variable substitution
        session_id: Optional session ID

    Returns:
        NodeResult with execution details
    """
    start_time = time.time()
    retries = 0
    last_error = None

    # Substitute variables
    content = substitute_variables(node.content, outputs)

    while retries <= node.retry_count:
        try:
            if retries > 0:
                # Exponential backoff
                delay = (node.backoff_factor ** (retries - 1)) * 1.0
                log.info("workflow_node_retry", node_id=node.id, retry=retries, delay=delay)
                await asyncio.sleep(delay)

            result = await delegate_impl(
                ctx=ctx,
                task=node.task,
                content=content,
                model=node.model,
                language=node.language,
                session_id=session_id,
                include_metadata=False,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            return NodeResult(
                node_id=node.id,
                success=True,
                output=result.strip(),
                retries=retries,
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            last_error = str(e)
            retries += 1
            log.warning("workflow_node_attempt_failed", node_id=node.id, attempt=retries, error=last_error)

    elapsed_ms = int((time.time() - start_time) * 1000)

    return NodeResult(
        node_id=node.id,
        success=False,
        output="",
        error=last_error,
        retries=retries - 1,  # Actual retries (not including initial attempt)
        elapsed_ms=elapsed_ms,
    )


async def execute_workflow(
    definition: WorkflowDefinition,
    ctx: DelegateContext,
    session_id: str | None = None,
    max_retries: int = 1,  # Global override for retry_count
) -> WorkflowResult:
    """
    Execute a workflow DAG.

    The workflow starts at the entry node and follows the DAG based on:
    - Dependencies: Wait for all depends_on nodes to complete
    - Conditional branching: Follow on_success or on_failure based on result
    - Retry: Attempt retries with exponential backoff on failure

    Args:
        definition: The workflow definition
        ctx: DelegateContext with LLM dependencies
        session_id: Optional session for conversation continuity
        max_retries: Global override for node retry_count (use node's if higher)

    Returns:
        WorkflowResult with execution details
    """
    start_time = time.time()
    timeout_seconds = definition.timeout_minutes * 60

    nodes_by_id = {n.id: n for n in definition.nodes}
    outputs: dict[str, str] = {}
    node_results: list[NodeResult] = []
    nodes_completed: list[str] = []
    nodes_failed: list[str] = []
    errors: list[dict] = []

    # Track which nodes have been executed
    executed = set()

    # Start with entry node
    current_node_id = definition.entry

    log.info("workflow_start", name=definition.name, entry=definition.entry)

    while current_node_id:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            log.error("workflow_timeout", elapsed=elapsed, timeout=timeout_seconds)
            errors.append({
                "type": "timeout",
                "message": f"Workflow exceeded {definition.timeout_minutes} minute timeout",
            })
            break

        if current_node_id in executed:
            log.warning("workflow_node_already_executed", node_id=current_node_id)
            break

        node = nodes_by_id.get(current_node_id)
        if not node:
            log.error("workflow_node_not_found", node_id=current_node_id)
            errors.append({"type": "missing_node", "node_id": current_node_id})
            break

        # Check dependencies
        if node.depends_on:
            deps_satisfied = all(dep_id in nodes_completed for dep_id in node.depends_on)
            if not deps_satisfied:
                missing = [d for d in node.depends_on if d not in nodes_completed]
                log.error("workflow_deps_not_satisfied", node_id=node.id, missing=missing)
                errors.append({
                    "type": "dependency_error",
                    "node_id": node.id,
                    "missing_deps": missing,
                })
                break

        log.info("workflow_node_start", node_id=node.id, task=node.task)

        # Use max of node retry_count and global max_retries
        effective_retry = max(node.retry_count, max_retries - 1)
        node_with_retry = WorkflowNode(
            id=node.id,
            task=node.task,
            content=node.content,
            depends_on=node.depends_on,
            on_success=node.on_success,
            on_failure=node.on_failure,
            retry_count=effective_retry,
            backoff_factor=node.backoff_factor,
            output_var=node.output_var,
            model=node.model,
            language=node.language,
        )

        # Execute node
        result = await execute_node_with_retry(node_with_retry, ctx, outputs, session_id)
        node_results.append(result)
        executed.add(node.id)

        if result.success:
            nodes_completed.append(node.id)

            # Store output
            outputs[node.id] = result.output
            if node.output_var:
                outputs[node.output_var] = result.output

            log.info("workflow_node_complete", node_id=node.id, elapsed_ms=result.elapsed_ms)

            # Follow on_success or stop
            current_node_id = node.on_success
        else:
            nodes_failed.append(node.id)
            errors.append({
                "type": "node_failed",
                "node_id": node.id,
                "error": result.error,
                "retries": result.retries,
            })

            log.error("workflow_node_failed", node_id=node.id, error=result.error)

            # Follow on_failure or stop
            current_node_id = node.on_failure

    # Calculate skipped nodes
    all_node_ids = {n.id for n in definition.nodes}
    nodes_skipped = list(all_node_ids - executed)

    total_elapsed = int((time.time() - start_time) * 1000)
    success = len(nodes_failed) == 0 and len(errors) == 0

    log.info(
        "workflow_complete",
        success=success,
        completed=len(nodes_completed),
        failed=len(nodes_failed),
        skipped=len(nodes_skipped),
        elapsed_ms=total_elapsed,
    )

    return WorkflowResult(
        success=success,
        nodes_completed=nodes_completed,
        nodes_failed=nodes_failed,
        nodes_skipped=nodes_skipped,
        outputs=outputs,
        errors=errors,
        node_results=node_results,
        elapsed_ms=total_elapsed,
    )
