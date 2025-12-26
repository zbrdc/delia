# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Local model delegation tools - OPTIONAL, enabled via DELIA_DELEGATION=true.

These tools delegate work to locally-running LLM backends (Ollama, vLLM, etc.).
They are separate from the core Framework tools and should only be registered
when the user explicitly wants to use local model delegation.

Tools:
- delegate: Execute a task with intelligent model selection
- think: Deep reasoning for complex problems
- batch: Execute multiple tasks in parallel
- chain: Sequential task execution with output piping
- workflow: DAG workflow with conditional branching
- agent: Autonomous agent with tool access
"""

from __future__ import annotations

import json

import structlog
from fastmcp import FastMCP

from .handlers_orchestration import (
    think_impl,
    batch_impl,
    delegate_tool_impl,
    chain_impl,
    workflow_impl,
    agent_impl,
)

log = structlog.get_logger()


def register_delegation_tools(mcp: FastMCP):
    """Register local model delegation tools (optional, controlled by DELIA_DELEGATION)."""

    @mcp.tool()
    async def delegate(
        task: str,
        content: str,
        file: str | None = None,
        model: str | None = None,
        language: str | None = None,
        context: str | None = None,
        symbols: str | None = None,
        include_references: bool = False,
        backend_type: str | None = None,
        files: str | None = None,
        include_metadata: bool = True,
        max_tokens: int | None = None,
        dry_run: bool = False,
        session_id: str | None = None,
        stream: bool = False,
        reliable: bool = False,
        voting_k: int = 3,
        tot: bool = False,
        auto_context: bool = False,
    ) -> str:
        """Execute a task with intelligent 3-tier model selection.

        Offloads work to configured LLM backends. Only use when user explicitly
        requests delegation ("delegate", "offload", "use local model").
        """
        return await delegate_tool_impl(
            task, content, file, model, language, context, symbols, include_references,
            backend_type, files, include_metadata, max_tokens, dry_run, session_id,
            stream, reliable, voting_k, tot, auto_context
        )

    @mcp.tool()
    async def think(
        problem: str,
        context: str = "",
        depth: str = "normal",
        session_id: str | None = None,
    ) -> str:
        """Deep reasoning for complex problems with extended thinking."""
        return await think_impl(problem, context, depth, session_id)

    @mcp.tool()
    async def batch(
        tasks: str,
        include_metadata: bool = True,
        max_tokens: int | None = None,
        session_id: str | None = None,
    ) -> str:
        """Execute multiple tasks in PARALLEL across all available backends."""
        return await batch_impl(tasks, include_metadata, max_tokens, session_id)

    @mcp.tool()
    async def chain(
        steps: str,
        session_id: str | None = None,
        continue_on_error: bool = False,
    ) -> str:
        """Execute a chain of tasks sequentially with output piping."""
        return await chain_impl(steps, session_id, continue_on_error)

    @mcp.tool()
    async def workflow(
        definition: str,
        session_id: str | None = None,
        max_retries: int = 1,
    ) -> str:
        """Execute a DAG workflow with conditional branching and retry logic."""
        return await workflow_impl(definition, session_id, max_retries)

    @mcp.tool()
    async def agent(
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_iterations: int = 10,
        tools: str | None = None,
        backend_type: str | None = None,
        workspace: str | None = None,
    ) -> str:
        """Run an autonomous agent that can use tools to complete tasks."""
        return await agent_impl(prompt, system_prompt, model, max_iterations, tools, backend_type, workspace)
