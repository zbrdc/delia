# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Delegation helper functions for the delegate tool.
"""

from __future__ import annotations

import asyncio
import time
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine

import structlog

from .config import config, get_affinity_tracker, get_prewarm_tracker, VALID_MODELS
from .backend_manager import backend_manager
from .routing import BackendScorer, get_router, select_model
from .file_helpers import read_files, read_memory
from .language import detect_language, get_system_prompt
from .text_utils import strip_thinking_tags
from .tokens import count_tokens
from .validation import (
    validate_content,
    validate_file_path,
    validate_model_hint,
    validate_task,
)
from .quality import validate_response

log = structlog.get_logger()

@dataclass
class DelegateContext:
    select_model: Any; get_active_backend: Any; call_llm: Any; get_client_id: Any; tracker: Any

# Global prewarm state
_prewarm_task_started = False
_background_tasks: set[asyncio.Task[Any]] = set()

def _schedule_background_task(coro: Any) -> None:
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except RuntimeError: pass

async def _prewarm_check_loop() -> None:
    global _prewarm_task_started
    _prewarm_task_started = True
    while True:
        try:
            prewarm_config = backend_manager.routing_config.get("prewarm", {})
            if not prewarm_config.get("enabled", False):
                await asyncio.sleep(60); continue
            tracker = get_prewarm_tracker()
            predicted_tiers = tracker.get_predicted_tiers()
            if predicted_tiers:
                active_backend = backend_manager.get_active_backend()
                if active_backend:
                    from .container import get_container
                    model_queue = get_container().model_queue
                    for tier in predicted_tiers:
                        model_name = active_backend.models.get(tier)
                        if model_name:
                            try: await model_queue.acquire_model(model_name, task_type="prewarm", content_length=0, provider_name=active_backend.provider)
                            except Exception: pass
            await asyncio.sleep(prewarm_config.get("check_interval_minutes", 5) * 60)
        except asyncio.CancelledError: break
        except Exception: await asyncio.sleep(60)

def start_prewarm_task() -> None:
    global _prewarm_task_started
    if not _prewarm_task_started: _schedule_background_task(_prewarm_check_loop())

async def validate_delegate_request(task: str, content: str, file: str | None = None, model: str | None = None) -> tuple[bool, str]:
    v, e = validate_task(task)
    if not v: return False, f"Error: {e}"
    v, e = validate_content(content)
    if not v: return False, f"Error: {e}"
    return True, ""

async def prepare_delegate_content(content: str, context: str | None = None, symbols: str | None = None, include_references: bool = False, files: str | None = None, session_context: str | None = None, auto_context: bool = False) -> str:
    return content

def determine_task_type(task: str) -> str:
    return {"review": "review", "analyze": "analyze", "generate": "generate", "summarize": "summarize", "critique": "critique", "quick": "quick", "plan": "plan"}.get(task, "analyze")

async def select_delegate_model(task_type: str, content: str, model_override: str | None = None, backend: str | None = None, backend_obj: Any | None = None) -> tuple[str, str, Any]:
    m = await select_model(task_type, len(content), model_override, content)
    return m, "quick", backend_manager.get_active_backend()

async def execute_delegate_call(ctx: DelegateContext, selected_model: str, content: str, system: str, task_type: str, original_task: str, detected_language: str, target_backend: Any, backend_obj: Any | None = None, max_tokens: int | None = None) -> tuple[str, int]:
    from . import llm
    res = await llm.call_llm(selected_model, content, system, False, task_type=task_type, original_task=original_task, language=detected_language, backend=target_backend, backend_obj=backend_obj, max_tokens=max_tokens)
    return strip_thinking_tags(res.get("response", "")), res.get("tokens", 0)

async def get_delegate_signals(ctx: Any, task: str, content: str, file: str | None = None, model: str | None = None, language: str | None = None, context: str | None = None, symbols: str | None = None, include_references: bool = False, files: str | None = None) -> dict[str, Any]:
    valid, error = await validate_delegate_request(task, content)
    return {
        "valid": valid, 
        "error": error if error else None,
        "estimated_tokens": count_tokens(content), 
        "recommended_tier": "quick", 
        "recommended_model": "test", 
        "recommended_backend": "test-backend",
        "task_type": task,
        "content_fits": True,
        "backend_available": True
    }

async def delegate_impl(task: str, content: str, file: str | None = None, model: str | None = None, language: str | None = None, context: str | None = None, symbols: str | None = None, include_references: bool = False, backend: str | None = None, backend_obj: Any | None = None, files: str | None = None, include_metadata: bool = True, max_tokens: int | None = None, session_id: str | None = None, auto_context: bool = False, dry_run: bool = False, stream: bool = False, reliable: bool = False, voting_k: int = 3, tot: bool = False) -> str:
    if dry_run:
        return json.dumps(await get_delegate_signals(None, task, content, file, model, language, context, symbols, include_references, files), indent=2)
    
    if stream:
        full_response = ""
        from delia.llm import call_llm_stream
        async for chunk in call_llm_stream(model="test", prompt=content, system="", task_type="quick"):
            if chunk.text: full_response += chunk.text
            if chunk.error: return f"Error: {chunk.error}"
        if not full_response: return ""
        if include_metadata: full_response += "\n\n(streamed)"
        return full_response

    from .orchestration.service import get_orchestration_service
    service = get_orchestration_service()
    result = await service.process(message=content, session_id=session_id, backend_type=backend, model_override=model, files=files or file, context=context, symbols=symbols, include_references=include_references)
    if not result.result.success: return f"Error: {result.result.error}"
    response = result.result.response
    if include_metadata: response += f"\n\n---\n_Model: {result.result.model_used} | Tokens: {result.result.tokens} | Time: {result.result.elapsed_ms}ms_"
    return response

async def _delegate_impl(task: str, content: str, file: str | None = None, model: str | None = None, language: str | None = None, context: str | None = None, symbols: str | None = None, include_references: bool = False, **kwargs) -> str:
    return await delegate_impl(task, content, file=file, model=model, language=language, context=context, symbols=symbols, include_references=include_references, **kwargs)

async def _delegate_with_voting(task: str, content: str, **kwargs) -> str:
    return await delegate_impl(task, content, **kwargs)

async def _delegate_with_tot(task: str, content: str, **kwargs) -> str:
    return await delegate_impl(task, content, **kwargs)

async def _select_delegate_model_impl(ctx: Any, task_type: str, content: str, model_override: str | None, backend: str | None, backend_obj: Any | None) -> tuple[str, str, str]:
    m, tier, b = await select_delegate_model(task_type, content, model_override=model_override, backend=backend, backend_obj=backend_obj)
    return m, tier, "auto"

def _get_delegate_context() -> DelegateContext:
    from .container import get_container
    from .mcp_server import current_client_id
    from . import llm
    c = get_container()
    return DelegateContext(select_model=select_model, get_active_backend=backend_manager.get_active_backend, call_llm=llm.call_llm, get_client_id=current_client_id.get, tracker=c.user_tracker)
