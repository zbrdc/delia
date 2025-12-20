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
        from ..routing import select_model
        
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
            },
            {
                "type": "function",
                "function": {
                    "name": "call_status",
                    "description": "Use for checking melon leaderboard, system health, or stats.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Status query"}
                        }
                    }
                }
            }
        ]

        # 1. Select Dispatcher Model Dynamically
        # Prefer 7B Instruct (Quick tier) for nuanced dispatching if resources allow,
        # otherwise fallback to FunctionGemma (Dispatcher tier) on CPU.
        dispatcher_model = await select_model(
            task_type="dispatcher",
            content=message,
        )
        
        # Final safety check for model name
        if not dispatcher_model or dispatcher_model == "current":
            dispatcher_model = config.model_dispatcher.default_model

        system_prompt = build_system_prompt(ModelRole.DISPATCHER)
        
        # 2. Call Dispatcher
        result = await self.call_llm(
            model=dispatcher_model,
            prompt=message,
            system=system_prompt,
            task_type="dispatcher",
            original_task=intent.task_type,
            language="unknown",
            content_preview=message[:100],
            backend_obj=backend_obj,
            tools=tools,
            tool_choice="auto",
        )

        if not result.get("success"):
            log.warning("dispatcher_failed_falling_back", error=result.get("error"))
            return "executor" # Safe fallback

        # 3. Parse Response & Check for Refusals
        metadata = result.get("metadata", {})
        tool_calls = metadata.get("tool_calls", [])
        response_text = result.get("response", "").lower()
        
        # If the tiny model gives a canned refusal (e.g. "I am sorry, I cannot..."),
        # we must ignore it and just use the Executor.
        refusal_keywords = [
            "i cannot assist", "i am sorry", "i'm sorry", "i am unable", 
            "capabilities are limited", "i am a model", "cannot assist with this"
        ]
        if any(k in response_text for k in refusal_keywords):
            log.info("dispatcher_refusal_detected", reason="model_claimed_inability")
            return "executor"

        if tool_calls:
            tool_name = tool_calls[0]["function"]["name"]
            if tool_name == "call_planner":
                return "planner"
            elif tool_name == "call_executor":
                return "executor"
            elif tool_name == "call_status":
                return "status"
        
        # 4. Keyword Fallback (if model outputs text instead of tool)
        if "planner" in response_text or "plan" in response_text:
            return "planner"
        if "status" in response_text or "melon" in response_text:
            return "status"
        if "executor" in response_text or "code" in response_text:
            return "executor"

        # 5. Intent-based selection as final safety net
        log.debug("dispatcher_no_tool_call_using_intent")
        if intent.task_type in config.moe_tasks or intent.orchestration_mode != OrchestrationMode.NONE:
            return "planner"
            
        return "executor"