# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Model-as-a-Tool Dispatcher Implementation.

Uses a lightweight dispatcher model to route requests to the appropriate 
specialized tier (Planner or Executor) based on task complexity.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from .result import DetectedIntent, OrchestrationMode
from ..prompts import ModelRole, build_system_prompt

if TYPE_CHECKING:
    from ..backend_manager import BackendConfig

log = structlog.get_logger()


class ModelDispatcher:
    """
    Orchestrates models by treating them as tools.
    
    This implements the "Models ARE the Tools" paradigm.
    The Dispatcher Model decides whether to call
    the Planner or the Executor tier.
    """

    def __init__(self, call_llm_fn: Any):
        self.call_llm = call_llm_fn

    async def dispatch(
        self,
        message: str,
        intent: DetectedIntent,
        backend_obj: BackendConfig | None = None,
    ) -> str:
        """
        Use the dispatcher model to select the next model in the chain.
        """
        from ..config import config
        
        # Tools available to the Dispatcher
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "call_planner",
                    "description": "Use for complex questions, multi-step planning, or architectural design.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string", "description": "Why the planner is needed"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_executor",
                    "description": "Use for direct implementation, writing code, or simple tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string", "description": "Why the executor is needed"}
                        }
                    }
                }
            }
        ]

        # 1. Select Dispatcher Model
        # We prefer the 'dispatcher' tier if available
        dispatcher_model = config.model_dispatcher.default_model
        
        # 2. Call Dispatcher
        system_prompt = build_system_prompt(ModelRole.DISPATCHER)
        
        log.info("dispatcher_call", model=dispatcher_model, message_len=len(message))
        
        result = await self.call_llm(
            model=dispatcher_model,
            prompt=message,
            system=system_prompt,
            task_type="dispatch",
            backend_obj=backend_obj,
            tools=tools,
            temperature=0.0, # Deterministic routing
        )

        if not result.get("success"):
            log.warning("dispatcher_failed_falling_back", error=result.get("error"))
            return "executor" # Safe fallback

        # 3. Parse Tool Call
        metadata = result.get("metadata", {})
        tool_calls = metadata.get("tool_calls", [])
        
        if tool_calls:
            tool_name = tool_calls[0]["function"]["name"]
            if tool_name == "call_planner":
                return "planner"
            elif tool_name == "call_executor":
                return "executor"
        
        # 4. Fallback to intent-based selection if no tool call
        log.debug("dispatcher_no_tool_call_using_intent")
        if intent.task_type in config.moe_tasks or intent.orchestration_mode != OrchestrationMode.NONE:
            return "planner"
            
        return "executor"
