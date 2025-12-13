# Copyright (C) 2023 the project owner
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
Structured MCP tools for LLM-to-LLM communication.

These tools accept and return JSON with typed schemas, optimized for
AI assistants to communicate with Delia programmatically.
"""

import asyncio
import time
import uuid
from datetime import UTC, datetime

from .backend_manager import BackendConfig

# Import from mcp_server - these are the core functions we'll reuse
from .mcp_server import (
    backend_manager,
    call_llm,
    config,
    create_structured_prompt,
    current_client_id,
    detect_language,
    get_system_prompt,
    mcp,
    prepare_delegate_content,
    select_delegate_model,
    tracker,
    validate_delegate_request,
)
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BackendPreference,
    BatchResponse,
    BatchResponseItem,
    CodeGenerateRequest,
    CodeGenerateResponse,
    CodeReviewRequest,
    CodeReviewResponse,
    ExecutionInfo,
    ModelTier,
    ReasoningDepth,
    # Requests
    StructuredRequest,
    StructuredResponse,
    TaskType,
    ThinkRequest,
    ThinkResponse,
    # Responses
    UsageMetrics,
)


async def _execute_structured(
    request: StructuredRequest,
    task_type_str: str,
) -> tuple[StructuredResponse, str, str]:
    """
    Execute a structured request through the existing delegate infrastructure.

    Returns:
        Tuple of (base_response, raw_content, tier) where base_response has
        usage and execution info filled in, and raw_content is the LLM output.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    # Map structured fields to delegate parameters
    model_hint = request.model_tier.value if request.model_tier else None
    language = request.language.value if request.language else None

    # Map backend preference
    backend_type = None
    if request.backend == BackendPreference.LOCAL:
        backend_type = "local"
    elif request.backend == BackendPreference.REMOTE:
        backend_type = "remote"

    # Validate request
    valid, error = await validate_delegate_request(
        task_type_str,
        request.content,
        request.file_path,
        model_hint,
    )
    if not valid:
        return (
            StructuredResponse(
                success=False,
                content="",
                error=error,
                request_id=request_id,
            ),
            "",
            "quick",
        )

    # Prepare content (context expansion if needed)
    prepared_content = await prepare_delegate_content(
        request.content,
        context=None,
        symbols=None,
        include_references=False,
    )

    # Create structured prompt
    symbols_list = getattr(request, "symbols", None)
    prepared_content = create_structured_prompt(
        task_type=task_type_str,
        content=prepared_content,
        file_path=request.file_path,
        language=language,
        symbols=symbols_list,
        context_files=None,
    )

    # Detect language and get system prompt
    detected_language = language or detect_language(prepared_content, request.file_path or "")
    system = get_system_prompt(detected_language, task_type_str)

    # Select model and backend
    selected_model, tier, target_backend = await select_delegate_model(
        task_type_str,
        prepared_content,
        model_hint,
        backend_type,
        None,  # backend_obj
    )

    # Execute the LLM call
    enable_thinking = task_type_str in config.thinking_tasks
    content_preview = request.content[:200].replace("\n", " ").strip()

    result = await call_llm(
        selected_model,
        prepared_content,
        system,
        enable_thinking,
        task_type=task_type_str,
        original_task=task_type_str,
        language=detected_language,
        content_preview=content_preview,
        backend=target_backend,
        backend_obj=None,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    if not result.get("success"):
        return (
            StructuredResponse(
                success=False,
                content="",
                error=result.get("error", "Unknown error"),
                usage=UsageMetrics(latency_ms=elapsed_ms),
                execution=ExecutionInfo(
                    model=selected_model,
                    model_tier=ModelTier(tier),
                    backend_id=target_backend,
                ),
                request_id=request_id,
            ),
            "",
            tier,
        )

    response_text = result.get("response", "")
    tokens = result.get("tokens", 0)

    # Strip thinking tags if present
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1].strip()

    # Update tracker
    client_id = current_client_id.get()
    if client_id:
        tracker.update_last_request(client_id, tokens=tokens, model_tier=tier)

    # Get backend info
    active_backend: BackendConfig | None = None
    if isinstance(target_backend, BackendConfig):
        active_backend = target_backend
    elif target_backend:
        active_backend = backend_manager.get_backend(target_backend)
    else:
        active_backend = backend_manager.get_active_backend()
    provider = active_backend.provider if active_backend else "unknown"
    backend_type_str = active_backend.type if active_backend else "local"

    # Build response
    response = StructuredResponse(
        success=True,
        content=response_text,
        usage=UsageMetrics(
            input_tokens=0,  # Not tracked separately currently
            output_tokens=tokens,
            total_tokens=tokens,
            latency_ms=elapsed_ms,
        ),
        execution=ExecutionInfo(
            model=selected_model,
            model_tier=ModelTier(tier),
            backend_id=target_backend,
            backend_type=backend_type_str,
            provider=provider,
            timestamp=datetime.now(UTC),
        ),
        request_id=request_id,
    )

    return response, response_text, tier


def _create_error_response(error: str, request_id: str) -> str:
    """Create a JSON error response."""
    return StructuredResponse(
        success=False,
        content="",
        error=error,
        request_id=request_id,
    ).model_dump_json()


# =============================================================================
# STRUCTURED MCP TOOLS
# =============================================================================


@mcp.tool()
async def code_review(request_json: str) -> str:
    """
    Structured code review with typed JSON input/output.

    INPUT: JSON matching CodeReviewRequest schema
    {
        "content": "def foo(): pass",
        "content_type": "code",
        "language": "python",
        "model_tier": "coder",
        "focus_areas": ["security", "performance"],
        "symbols": ["foo"],
        "severity_threshold": "warning"
    }

    OUTPUT: JSON matching CodeReviewResponse schema
    {
        "success": true,
        "content": "...",
        "findings": [...],
        "summary": "...",
        "usage": {...},
        "execution": {...}
    }
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        request = CodeReviewRequest.model_validate_json(request_json)
    except Exception as e:
        return _create_error_response(f"Invalid request JSON: {e}", request_id)

    base_response, raw_content, _tier = await _execute_structured(
        request,
        TaskType.REVIEW.value,
    )

    if not base_response.success:
        return base_response.model_dump_json()

    # Build CodeReviewResponse (findings would need parsing in a future enhancement)
    response = CodeReviewResponse(
        success=True,
        content=raw_content,
        findings=[],  # Post-hoc parsing would go here
        summary=raw_content[:500] if raw_content else "",
        usage=base_response.usage,
        execution=base_response.execution,
        request_id=base_response.request_id,
    )

    return response.model_dump_json()


@mcp.tool()
async def code_generate(request_json: str) -> str:
    """
    Structured code generation with typed JSON input/output.

    INPUT: JSON matching CodeGenerateRequest schema
    {
        "content": "Create a function that calculates fibonacci",
        "content_type": "text",
        "language": "python",
        "model_tier": "coder",
        "include_tests": true,
        "include_docstrings": true
    }

    OUTPUT: JSON matching CodeGenerateResponse schema
    {
        "success": true,
        "content": "...",
        "generated": [...],
        "usage": {...},
        "execution": {...}
    }
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        request = CodeGenerateRequest.model_validate_json(request_json)
    except Exception as e:
        return _create_error_response(f"Invalid request JSON: {e}", request_id)

    base_response, raw_content, _tier = await _execute_structured(
        request,
        TaskType.GENERATE.value,
    )

    if not base_response.success:
        return base_response.model_dump_json()

    response = CodeGenerateResponse(
        success=True,
        content=raw_content,
        generated=[],  # Post-hoc parsing would extract code blocks
        usage=base_response.usage,
        execution=base_response.execution,
        request_id=base_response.request_id,
    )

    return response.model_dump_json()


@mcp.tool()
async def code_analyze(request_json: str) -> str:
    """
    Structured code analysis with typed JSON input/output.

    INPUT: JSON matching AnalyzeRequest schema
    {
        "content": "class MyClass: ...",
        "content_type": "code",
        "language": "python",
        "analysis_type": "complexity",
        "depth": "deep",
        "include_metrics": true
    }

    OUTPUT: JSON matching AnalyzeResponse schema
    {
        "success": true,
        "content": "...",
        "sections": [...],
        "metrics": {...},
        "recommendations": [...],
        "usage": {...},
        "execution": {...}
    }
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        request = AnalyzeRequest.model_validate_json(request_json)
    except Exception as e:
        return _create_error_response(f"Invalid request JSON: {e}", request_id)

    base_response, raw_content, _tier = await _execute_structured(
        request,
        TaskType.ANALYZE.value,
    )

    if not base_response.success:
        return base_response.model_dump_json()

    response = AnalyzeResponse(
        success=True,
        content=raw_content,
        sections=[],  # Post-hoc parsing would extract sections
        usage=base_response.usage,
        execution=base_response.execution,
        request_id=base_response.request_id,
    )

    return response.model_dump_json()


@mcp.tool()
async def structured_think(request_json: str) -> str:
    """
    Structured extended reasoning with typed JSON input/output.

    INPUT: JSON matching ThinkRequest schema
    {
        "problem": "How should we architect the caching layer?",
        "context": "We have a distributed system with...",
        "constraints": ["Must handle 10k QPS", "Budget is limited"],
        "depth": "deep"
    }

    OUTPUT: JSON matching ThinkResponse schema
    {
        "success": true,
        "content": "...",
        "reasoning_steps": [...],
        "final_answer": "...",
        "alternatives_considered": [...],
        "usage": {...},
        "execution": {...}
    }
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        request = ThinkRequest.model_validate_json(request_json)
    except Exception as e:
        return _create_error_response(f"Invalid request JSON: {e}", request_id)

    # Build content from problem and context
    content_parts = [request.problem]
    if request.context:
        content_parts.append(f"\nContext: {request.context}")
    if request.constraints:
        content_parts.append(f"\nConstraints: {', '.join(request.constraints)}")

    # Map depth to task type
    depth_to_task = {
        ReasoningDepth.QUICK: "quick",
        ReasoningDepth.NORMAL: "analyze",
        ReasoningDepth.DEEP: "plan",
    }
    task_type = depth_to_task.get(request.depth, "analyze")

    # Create a StructuredRequest wrapper
    wrapper_request = StructuredRequest(
        content="\n".join(content_parts),
        model_tier=request.model_tier,
        backend=request.backend,
    )

    base_response, raw_content, _tier = await _execute_structured(
        wrapper_request,
        task_type,
    )

    if not base_response.success:
        return base_response.model_dump_json()

    response = ThinkResponse(
        success=True,
        content=raw_content,
        reasoning_steps=[],  # Post-hoc parsing would extract steps
        final_answer=raw_content,  # Full response as final answer
        usage=base_response.usage,
        execution=base_response.execution,
        request_id=base_response.request_id,
    )

    return response.model_dump_json()


@mcp.tool()
async def structured_delegate(request_json: str) -> str:
    """
    Generic structured delegation with typed JSON input/output.

    This is the structured equivalent of the original 'delegate' tool.

    INPUT: JSON matching StructuredRequest schema + task_type field
    {
        "content": "Explain this code...",
        "content_type": "code",
        "language": "python",
        "model_tier": "coder",
        "task_type": "analyze"
    }

    OUTPUT: JSON matching StructuredResponse schema
    {
        "success": true,
        "content": "...",
        "usage": {...},
        "execution": {...}
    }
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        # Parse as dict first to extract task_type
        import json

        data = json.loads(request_json)
        task_type_str = data.pop("task_type", "analyze")
        request = StructuredRequest.model_validate(data)
    except Exception as e:
        return _create_error_response(f"Invalid request JSON: {e}", request_id)

    # Validate task type
    valid_tasks = {"review", "analyze", "generate", "summarize", "critique", "quick", "plan", "think"}
    if task_type_str not in valid_tasks:
        return _create_error_response(
            f"Invalid task_type: '{task_type_str}'. Valid: {valid_tasks}",
            request_id,
        )

    base_response, _raw_content, _tier = await _execute_structured(
        request,
        task_type_str,
    )

    return base_response.model_dump_json()


@mcp.tool()
async def batch_structured(requests_json: str) -> str:
    """
    Batch processing with structured JSON input/output.

    INPUT: JSON matching BatchRequest schema
    {
        "requests": [
            {"content": "...", "task_type": "review", ...},
            {"content": "...", "task_type": "generate", ...}
        ],
        "fail_fast": false,
        "max_parallel": 3
    }

    OUTPUT: JSON matching BatchResponse schema
    {
        "success": true,
        "results": [...],
        "total_usage": {...},
        "failed_count": 0
    }
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    try:
        import json

        data = json.loads(requests_json)
        requests_data = data.get("requests", [])
        fail_fast = data.get("fail_fast", False)
        max_parallel = data.get("max_parallel")
    except Exception as e:
        return _create_error_response(f"Invalid request JSON: {e}", request_id)

    if not requests_data:
        return _create_error_response("No requests provided", request_id)

    results: list[BatchResponseItem] = []
    total_tokens = 0
    failed_count = 0

    # Process requests
    async def process_one(idx: int, req_data: dict) -> BatchResponseItem:
        nonlocal total_tokens, failed_count

        task_type_str = req_data.pop("task_type", "analyze")

        try:
            request = StructuredRequest.model_validate(req_data)
        except Exception as e:
            failed_count += 1
            return BatchResponseItem(
                index=idx,
                response=StructuredResponse(
                    success=False,
                    error=f"Invalid request at index {idx}: {e}",
                    request_id=f"{request_id}-{idx}",
                ),
            )

        base_response, _raw_content, _tier = await _execute_structured(
            request,
            task_type_str,
        )

        if not base_response.success:
            failed_count += 1
        else:
            total_tokens += base_response.usage.total_tokens

        return BatchResponseItem(
            index=idx,
            response=base_response,
        )

    # Execute with optional parallelism limit
    if max_parallel and max_parallel > 0:
        # Use semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_parallel)

        async def limited_process(idx: int, req: dict) -> BatchResponseItem:
            async with semaphore:
                return await process_one(idx, req)

        tasks = [limited_process(i, req) for i, req in enumerate(requests_data)]
    else:
        tasks = [process_one(i, req) for i, req in enumerate(requests_data)]

    if fail_fast:
        # Process sequentially, stop on first failure
        for i, req in enumerate(requests_data):
            result = await process_one(i, req)
            results.append(result)
            if not result.response.success:
                break
    else:
        # Process all in parallel
        results = await asyncio.gather(*tasks)

    elapsed_ms = int((time.time() - start_time) * 1000)

    response = BatchResponse(
        success=failed_count == 0,
        results=sorted(results, key=lambda r: r.index),
        total_usage=UsageMetrics(
            total_tokens=total_tokens,
            latency_ms=elapsed_ms,
        ),
        failed_count=failed_count,
        request_id=request_id,
    )

    return response.model_dump_json()


# =============================================================================
# GARDEN-THEMED STRUCTURED ALIASES
# =============================================================================


@mcp.tool()
async def prune_json(request_json: str) -> str:
    """
    Examine code vines for weeds with structured JSON input/output.
    Garden-themed alias for 'code_review'.

    The gardener inspects your code and returns a structured report
    of weeds (bugs), tangles (complexity), and overgrowth (issues).

    INPUT: JSON matching CodeReviewRequest schema
    OUTPUT: JSON matching CodeReviewResponse schema
    """
    return await code_review(request_json)  # type: ignore[operator]


@mcp.tool()
async def grow_json(request_json: str) -> str:
    """
    Cultivate fresh code with structured JSON input/output.
    Garden-themed alias for 'code_generate'.

    Plant your requirements as JSON and harvest structured code!

    INPUT: JSON matching CodeGenerateRequest schema
    OUTPUT: JSON matching CodeGenerateResponse schema
    """
    return await code_generate(request_json)  # type: ignore[operator]


@mcp.tool()
async def tend_json(request_json: str) -> str:
    """
    Tend to the code garden with structured JSON input/output.
    Garden-themed alias for 'code_analyze'.

    Examine roots, soil, and growth patterns with structured results.

    INPUT: JSON matching AnalyzeRequest schema
    OUTPUT: JSON matching AnalyzeResponse schema
    """
    return await code_analyze(request_json)  # type: ignore[operator]


@mcp.tool()
async def ponder_json(request_json: str) -> str:
    """
    Let thoughts grow slowly with structured JSON input/output.
    Garden-themed alias for 'structured_think'.

    Deep contemplation with typed reasoning steps returned.

    INPUT: JSON matching ThinkRequest schema
    OUTPUT: JSON matching ThinkResponse schema
    """
    return await structured_think(request_json)  # type: ignore[operator]


@mcp.tool()
async def harvest_json(requests_json: str) -> str:
    """
    Gather multiple melons with structured JSON input/output.
    Garden-themed alias for 'batch_structured'.

    Parallel harvest across all garden plots with structured results.

    INPUT: JSON matching BatchRequest schema
    OUTPUT: JSON matching BatchResponse schema
    """
    return await batch_structured(requests_json)  # type: ignore[operator]
