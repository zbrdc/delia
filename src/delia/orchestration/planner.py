# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Strategic Planner Implementation.

Uses a high-reasoning planner model to decompose goals
into executable task graphs or chains.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from .result import DetectedIntent, OrchestrationResult, OrchestrationMode
from ..prompts import ModelRole, build_system_prompt
from ..text_utils import strip_thinking_tags

if TYPE_CHECKING:
    from ..backend_manager import BackendConfig

log = structlog.get_logger()


class PlanStep(BaseModel):
    """A single executable step in a plan."""
    id: str = Field(description="Unique identifier for the step")
    task: str = Field(description="Task type: quick, summarize, generate, review, analyze, plan, critique")
    content: str = Field(description="Specific instructions for this step")
    reasoning: str = Field(description="Why this step is necessary")


class ExecutionPlan(BaseModel):
    """A complete decomposed plan."""
    title: str
    steps: list[PlanStep]
    estimated_difficulty: float = Field(0.5, description="0.0 to 1.0 difficulty score")


class StrategicPlanner:
    """
    Decomposes complex goals into strategic plans.
    
    The Planner model tier doesn't use tools itself;
    it produces a plan that Delia then executes via the Executor tier.
    """

    def __init__(self, call_llm_fn: Any):
        self.call_llm = call_llm_fn

    async def plan(
        self,
        message: str,
        intent: DetectedIntent,
        backend_obj: BackendConfig | None = None,
    ) -> ExecutionPlan | None:
        """
        Decompose a goal into executable steps.
        """
        from ..config import config
        from .outputs import get_json_schema_prompt
        
        # 1. Select Planner Model
        planner_model = config.model_thinking.default_model
        
        # 2. Build System Prompt
        system_prompt = build_system_prompt(ModelRole.PLANNER)
        
        # Request structured JSON output
        json_instr = get_json_schema_prompt(ExecutionPlan)
        prompt = f"{message}\n\n{json_instr}"
        
        log.info("planner_start", model=planner_model, message_len=len(message))
        
        # 3. Call Planner
        result = await self.call_llm(
            model=planner_model,
            prompt=prompt,
            system=system_prompt,
            task_type="plan",
            backend_obj=backend_obj,
            enable_thinking=True, # Enable chain-of-thought
            temperature=0.2, # Strategic stability
        )

        if not result.get("success"):
            log.error("planner_failed", error=result.get("error"))
            return None

        response_text = strip_thinking_tags(result.get("response", ""))
        
        # 4. Parse Plan
        try:
            # Extract JSON from response
            from .outputs import parse_structured_output
            plan = parse_structured_output(response_text, ExecutionPlan)
            log.info("planner_success", steps=len(plan.steps), difficulty=plan.estimated_difficulty)
            return plan
        except Exception as e:
            log.warning("planner_parse_failed", error=str(e), response=response_text[:200])
            return None
