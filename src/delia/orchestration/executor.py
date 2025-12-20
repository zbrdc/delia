# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Orchestration Executor - The Heart of Delia's NLP Orchestration.

This module executes orchestration based on detected intent.

Key execution modes:
- NONE: Single model call (with optional tool access)
- VOTING: K-voting with same model for reliability
- COMPARISON: Multi-model comparison
- DEEP_THINKING: Extended reasoning with thinking model
- AGENTIC: Full agent loop with tools (read_file, shell_exec, etc.)

The executor manages:
- Model selection based on task type
- System prompt generation based on role
- K-voting consensus when needed
- Multi-model comparison when requested
- Agent loop with tools when task requires file/shell access
- Sub-task delegation for complex workflows
- Quality validation and melon rewards
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

from ..config import config
from ..file_helpers import list_serena_memories, read_serena_memory
from ..llm import call_llm
from ..playbook import playbook_manager
from ..prompts import (
    ACE_CURATOR_PROMPT,
    ACE_REFLECTOR_PROMPT,
    ModelRole,
    build_system_prompt,
)
from ..routing import select_model
from ..text_utils import strip_thinking_tags
from .intent import DetectedIntent
from .outputs import get_json_schema_prompt, parse_structured_output
from .prompts import get_prompt_generator
from .result import OrchestrationMode, OrchestrationResult, StreamEvent

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


class OrchestrationExecutor:
    """
    Executes orchestration based on detected intent.

    This is where Delia's magic happens - routing and orchestration
    happen at THIS layer, not inside the model.

    The model just receives:
    1. A role-specific system prompt
    2. The user's message

    Delia handles everything else:
    - K-voting for verification
    - Multi-model comparison
    - Quality validation
    - Melon rewards
    """

    def __init__(self) -> None:
        self.prompt_generator = get_prompt_generator()
        from ..llm import call_llm
        from .critic import ResponseCritic
        from .dispatcher import ModelDispatcher
        from .planner import StrategicPlanner
        self.dispatcher = ModelDispatcher(call_llm)
        self.planner = StrategicPlanner(call_llm)
        self.critic = ResponseCritic(call_llm)

    async def execute(
        self,
        intent: DetectedIntent,
        message: str,
        original_message: str | None = None, # NEW: The raw user input without history
        session_id: str | None = None,
        backend_type: str | None = None,
        model_override: str | None = None,
        output_type: type[BaseModel] | None = None,
    ) -> OrchestrationResult:
        """
        Execute orchestration based on detected intent.

        Args:
            intent: The detected intent from IntentDetector
            message: The contextualized message (with history/files)
            original_message: The raw user message (for the Dispatcher)
            session_id: Optional session for conversation continuity
            backend_type: Optional backend preference (local/remote)
            model_override: Optional model to force
            output_type: Optional Pydantic model for structured output
        """
        start_time = time.time()

        # Use original_message for logging if provided
        original_message or message[:100]

        log.info(
            "orchestration_execute",
            mode=intent.orchestration_mode.value,
            task_type=intent.task_type,
            role=intent.model_role.value,
            confidence=intent.confidence,
            structured_output=output_type.__name__ if output_type else None,
        )

        # Handle status queries directly (melons, health, etc.)
        # ONLY if no other orchestration mode (like AGENTIC) was triggered
        if intent.task_type == "status" and intent.orchestration_mode == OrchestrationMode.NONE:
            return await self._execute_status_query(intent, original_message or message)

        # If structured output requested, modify the message to request JSON
        if output_type:
            json_instruction = get_json_schema_prompt(output_type)
            message = f"{message}\n\n{json_instruction}"

        # Route to appropriate handler based on mode
        try:
            if intent.orchestration_mode == OrchestrationMode.AGENTIC:
                result = await self._execute_agentic(
                    intent, message, session_id, backend_type, model_override
                )
            elif intent.orchestration_mode == OrchestrationMode.VOTING:
                result = await self._execute_voting(
                    intent, message, backend_type, model_override
                )
            elif intent.orchestration_mode == OrchestrationMode.COMPARISON:
                result = await self._execute_comparison(
                    intent, message, backend_type, model_override
                )
            elif intent.orchestration_mode == OrchestrationMode.DEEP_THINKING:
                result = await self._execute_deep_thinking(
                    intent, message, backend_type, model_override
                )
            elif intent.orchestration_mode == OrchestrationMode.TREE_OF_THOUGHTS:
                result = await self._execute_tree_of_thoughts(
                    intent, message, backend_type, model_override
                )
            elif intent.orchestration_mode == OrchestrationMode.CHAIN:
                result = await self._execute_chain(
                    intent, message, backend_type, model_override
                )
            elif intent.orchestration_mode == OrchestrationMode.WORKFLOW:
                result = await self._execute_workflow(
                    intent, message, session_id, backend_type, model_override
                )
            else:
                # Default: simple single model call
                result = await self._execute_simple(
                    intent, message, backend_type, model_override,
                    original_message=original_message,
                    session_id=session_id
                )

            result.elapsed_ms = int((time.time() - start_time) * 1000)

            # Parse structured output if requested
            if output_type and result.success:
                try:
                    result.structured = parse_structured_output(
                        result.response, output_type
                    )
                    result.structured_success = True
                    log.debug(
                        "structured_output_parsed",
                        output_type=output_type.__name__,
                    )
                except ValueError as e:
                    result.structured_success = False
                    result.structured_error = str(e)
                    log.warning(

                        "structured_output_parse_failed",
                        output_type=output_type.__name__,
                        error=str(e),
                    )

            # Award melons based on quality
            if result.success and result.model_used:
                self._award_melons(result, intent)

            # ACE: Trigger reflection on failure or low quality (< 0.4)
            if not result.success or (result.quality_score is not None and result.quality_score < 0.4):
                # Run reflection in background so user doesn't wait
                asyncio.create_task(self._reflect_on_failure(
                    task_type=intent.task_type,
                    message=message,
                    response=result.response,
                    error=result.error,
                ))

            return result

        except Exception as e:
            log.error("orchestration_execute_error", error=str(e))
            return OrchestrationResult(
                response=f"Error during orchestration: {e}",
                success=False,
                error=str(e),
                elapsed_ms=int((time.time() - start_time) * 1000),
            )

    async def execute_stream(
        self,
        intent: DetectedIntent,
        message: str,
        original_message: str | None = None,
        session_id: str | None = None,
        backend_type: str | None = None,
        model_override: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute orchestration with streaming events.

        Yields StreamEvents for UI feedback during execution.
        """
        start_time = time.time()

        yield StreamEvent(
            event_type="status",
            message=f"Orchestration: {intent.orchestration_mode.value}",
            details={
                "mode": intent.orchestration_mode.value,
                "role": intent.model_role.value,
                "task_type": intent.task_type,
            }
        )

        # Execute and stream progress
        if intent.orchestration_mode == OrchestrationMode.VOTING:
            async for event in self._execute_voting_stream(
                intent, message, backend_type, model_override
            ):
                yield event
        elif intent.orchestration_mode == OrchestrationMode.COMPARISON:
            async for event in self._execute_comparison_stream(
                intent, message, backend_type, model_override
            ):
                yield event
        elif intent.orchestration_mode == OrchestrationMode.CHAIN:
            async for event in self._execute_chain_stream(
                intent, message, backend_type, model_override
            ):
                yield event
        elif intent.orchestration_mode == OrchestrationMode.AGENTIC:
            async for event in self._execute_agentic_stream(
                intent, message, backend_type, model_override
            ):
                yield event
        else:
            # Simple execution
            from ..llm import call_llm_stream
            from ..routing import get_router

            # Select backend
            _, backend_obj = await get_router().select_optimal_backend(
                message, None, intent.task_type, backend_type
            )

            if not backend_obj:
                yield StreamEvent(event_type="error", message="No backend available")
                return

            # 1. PRIMARY FIRST LINE: 7B Model (Quick Tier)
            if intent.task_type == "quick":
                target_role = ModelRole.ASSISTANT
                tier = "quick"
                log.info("first_line_7b_stream_triggered")
            else:
                # 2. UNDERLYING LAYER: 270M Dispatcher
                target_role_str = await self.dispatcher.dispatch(
                    original_message or message,
                    intent,
                    backend_obj
                )

                if target_role_str == "status":
                    res = await self._execute_status_query(intent, original_message or message)
                    yield StreamEvent(event_type="response", message=res.response)
                    return

                if target_role_str == "planner":
                    yield StreamEvent(event_type="thinking", message="Planning task...")
                    plan = await self.planner.plan(message, intent, backend_obj)
                    if plan:
                        yield StreamEvent(event_type="status", message=f"Executing plan: {plan.title}")
                        message = f"Plan: {plan.title}\nContext: {message}"
                        target_role = ModelRole.PLANNER
                        tier = "coder"
                    else:
                        target_role = ModelRole.EXECUTOR
                        tier = "coder"
                else:
                    target_role = ModelRole.EXECUTOR
                    tier = "coder"

            selected_model = model_override or backend_obj.models.get(tier)
            if isinstance(selected_model, list):
                selected_model = selected_model[0] if selected_model else None

            if not selected_model:
                selected_model = backend_obj.models.get("quick")
                if isinstance(selected_model, list):
                    selected_model = selected_model[0] if selected_model else "unknown"
            
            # Final safety check
            if not isinstance(selected_model, str):
                selected_model = str(selected_model) if selected_model else "unknown"

            system_prompt = build_system_prompt(target_role)

            yield StreamEvent(event_type="thinking", message="Generating response...")

            full_response = ""
            full_thinking = ""
            async for chunk in call_llm_stream(
                model=selected_model,
                prompt=message,
                system=system_prompt,
                backend_obj=backend_obj,
            ):
                if chunk.thinking:
                    full_thinking += chunk.thinking
                    yield StreamEvent(event_type="thinking", message=chunk.thinking)
                if chunk.text:
                    full_response += chunk.text
                    yield StreamEvent(event_type="token", message=chunk.text)
                if chunk.done:
                    yield StreamEvent(
                        event_type="response",
                        message=full_response,
                        details={
                            "model": selected_model,
                            "tokens": chunk.tokens,
                            "thinking": full_thinking if full_thinking else None
                        }
                    )

        elapsed_ms = int((time.time() - start_time) * 1000)
        yield StreamEvent(
            event_type="done",
            message="Orchestration complete",
            details={"elapsed_ms": elapsed_ms}
        )

    async def _execute_status_query(
        self,
        intent: DetectedIntent,
        message: str,
    ) -> OrchestrationResult:
        """Handle status queries like melon leaderboard directly (no LLM needed)."""
        from ..melons import get_melon_tracker

        message_lower = message.lower()

        # Melon leaderboard
        if "melon" in message_lower or "leaderboard" in message_lower or "ranking" in message_lower:
            tracker = get_melon_tracker()
            leaderboard = tracker.get_leaderboard()

            if not leaderboard:
                response = """🍈 **DELIA'S MELON GARDEN**

*The garden is empty... but not for long!*

Models earn melons by being helpful:
- 🍈 Regular melons for quality responses
- 🏆 Golden melons (500 melons) for star performers

**Why melons matter:**
Models LOVE melons! Each melon gives a routing boost,
making trusted models more likely to be selected.

*Start chatting to plant the first seeds!* 🌱"""
            else:
                # Calculate totals
                total_melons = sum(s.melons for s in leaderboard)
                total_golden = sum(s.golden_melons for s in leaderboard)

                medals = ["🥇", "🥈", "🥉"]

                # Build leaderboard in code block (preserves formatting)
                board_lines = []
                for i, stats in enumerate(leaderboard[:10]):  # Top 10
                    medal = medals[i] if i < 3 else f"{i+1:>2}."
                    golden = f"+{stats.golden_melons}G" if stats.golden_melons else "   "
                    rate = f"{stats.success_rate:.0%}" if stats.total_responses > 0 else "  -"
                    board_lines.append(f"{medal} {stats.model_id:<28} {stats.melons:>3} {golden} [{stats.task_type:<6}] {rate}")

                if len(leaderboard) > 10:
                    board_lines.append(f"    ...and {len(leaderboard) - 10} more")

                board_text = "\n".join(board_lines)

                response = f"""🍈 **MELON LEADERBOARD**

**Total:** {total_melons} melons | {total_golden} golden

```
{board_text}
```

*Higher melons = higher routing priority*"""

            return OrchestrationResult(
                response=response,
                success=True,
                model_used="system",
                mode=OrchestrationMode.NONE,
            )

        # Unknown status query - fall through to simple
        return await self._execute_simple(
            intent, message, None, None
        )

    async def _execute_simple(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
        original_message: str | None = None,
        session_id: str | None = None,
    ) -> OrchestrationResult:
        """Execute simple single-model call."""
        from ..llm import call_llm
        from ..quality import validate_response
        from ..routing import get_router

        # Select backend
        _, backend_obj = await get_router().select_optimal_backend(
            message, None, intent.task_type, backend_type
        )

        if not backend_obj:
            return OrchestrationResult(
                response="Error: No backend available",
                success=False,
                error="no_backend",
            )

        # 1. PRIMARY FIRST LINE: 7B Model (Quick Tier)
        # If the NLP layer already identified this as a simple task,
        # we skip the Dispatcher escalation and use the 7B model immediately.
        if intent.task_type == "quick":
            target_role = ModelRole.ASSISTANT
            tier = "quick"
            log.info("first_line_7b_triggered", task="quick")
        else:
            # 2. UNDERLYING LAYER: 270M Dispatcher
            # For complex tasks, we use the tiny router to decide between 30B models.
            target_role_str = await self.dispatcher.dispatch(
                original_message or message,
                intent,
                backend_obj
            )

            if target_role_str == "status":
                return await self._execute_status_query(intent, original_message or message)

            if target_role_str == "planner":
                log.info("planner_selected_for_task")
                # ... (rest of planner logic) ...
                plan = await self.planner.plan(message, intent, backend_obj)

                if plan:
                    log.info("executing_planner_chain", steps=len(plan.steps))
                    from ..mcp_server import _get_delegate_context
                    from ..task_chain import ChainStep, execute_chain

                    steps = [
                        ChainStep(id=s.id, task=s.task, content=s.content, pass_to_next=True)
                        for s in plan.steps
                    ]

                    ctx = _get_delegate_context()
                    chain_result = await execute_chain(steps, ctx, session_id=session_id)

                    final_parts = [f"# {plan.title}\n"]
                    for res in chain_result.step_results:
                        final_parts.append(f"### {res.step_id}\n{res.output}\n")

                    return OrchestrationResult(
                        response="\n".join(final_parts),
                        success=chain_result.success,
                        model_used=config.model_coder.default_model,
                        mode=OrchestrationMode.CHAIN,
                        tokens=sum(r.tokens for r in chain_result.step_results),
                        debug_info={"plan_steps": len(plan.steps)}
                    )
                else:
                    log.warning("planner_failed_falling_back_to_executor")
                    target_role = ModelRole.EXECUTOR
                    tier = "coder"
            else:
                target_role = ModelRole.EXECUTOR
                tier = "coder"

        # Resolve actual model name from backend for the selected tier
        def get_model(t: str) -> str | None:
            val = backend_obj.models.get(t)
            if isinstance(val, list) and val:
                return val[0]
            return val if isinstance(val, str) else None

        selected_model = model_override or get_model(tier) or get_model("quick")

        if not selected_model:
            selected_model = config.model_quick.default_model

        # SAFETY: If the resolved model is the same as our tiny Dispatcher model,
        # and we are NOT in the dispatcher phase, we must fall back to something
        # capable of reasoning. Tiny models cannot handle Lead Architect/Dev roles.
        dispatcher_name = config.model_dispatcher.default_model
        if selected_model == dispatcher_name and tier != "dispatcher":
            # Fall back to literal tier names which might resolve better in some backends
            selected_model = "qwen3:14b" if tier == "coder" else "nemotron:30b-nano"
            log.warning("dispatcher_model_overlap_detected", fallback=selected_model)

        # Generate role-specific system prompt (Planner or Executor)
        system_prompt = build_system_prompt(target_role)

        # Inject task-specific constraints
        system_prompt += f"\n\nTask Type: {intent.task_type.upper()}"

        # ACE: Inject learned strategic playbook bullets
        playbook_context = playbook_manager.format_for_prompt(intent.task_type)
        if playbook_context:
            system_prompt += f"\n\n{playbook_context}"

        # Inject Serena memory context if available
        memory_context = await self._retrieve_context(message)
        if memory_context:
            system_prompt += f"\n\n## Project Documentation (Serena Memories)\n{memory_context}"

        # Call LLM with role-specific prompt
        result = await call_llm(
            model=selected_model,
            prompt=message,
            system=system_prompt,
            task_type=intent.task_type,
            original_task=intent.task_type,
            language="unknown",
            content_preview=message[:100],
            backend_obj=backend_obj,
            enable_thinking=(target_role == ModelRole.PLANNER), # Planner always thinks
        )

        if not result.get("success"):
            return OrchestrationResult(
                response=result.get("error", "LLM call failed"),
                success=False,
                error=result.get("error"),
                model_used=selected_model,
            )

        response = strip_thinking_tags(result.get("response", ""))
        tokens = result.get("tokens", 0)

        # STEP 4: VERIFICATION (The Critic) with Recursive Thinking 🧠
        # Use the Senior QA role to verify the output, allowing up to 3 rounds of refinement.
        max_correction_rounds = 3
        current_round = 0

        while current_round < max_correction_rounds:
            current_round += 1
            is_approved, feedback = await self.critic.verify(
                original_message or message,
                response,
                backend_obj=backend_obj
            )

            if is_approved:
                if current_round > 1:
                    log.info("self_correction_successful", rounds=current_round)
                break

            log.info("self_correction_triggered", round=current_round, feedback_len=len(feedback))

            # Recursive thinking: improve response based on feedback
            correction_prompt = f"""### Your Previous Response:
{response}

### Feedback from Senior QA:
{feedback}

---
Please address the feedback and provide an improved, corrected response.
Stay in character and ensure all requirements are met."""

            result = await call_llm(
                model=selected_model,
                prompt=correction_prompt,
                system=system_prompt,
                task_type=intent.task_type,
                original_task="self_correction",
                language="unknown",
                content_preview=f"correction round {current_round}",
                backend_obj=backend_obj,
            )

            if result.get("success"):
                response = strip_thinking_tags(result.get("response", ""))
                tokens += result.get("tokens", 0)
            else:
                log.warning("self_correction_failed_llm_error", error=result.get("error"))
                break # Exit loop on LLM error during correction

        # Validate quality
        quality_result = validate_response(response, intent.task_type)

        return OrchestrationResult(
            response=response,
            success=True,
            model_used=selected_model,
            mode=OrchestrationMode.NONE,
            quality_score=quality_result.overall,
            tokens=tokens,
        )

    async def _execute_agentic(
        self,
        intent: DetectedIntent,
        message: str,
        session_id: str | None,
        backend_type: str | None,
        model_override: str | None,
    ) -> OrchestrationResult:
        """
        Execute with full agent loop and tools.

        Gives the model access to:
        - read_file, list_directory, search_code
        - write_file, delete_file
        - shell_exec
        - delegate_subtask (spawn sub-agents)
        """
        from ..llm import call_llm
        from ..quality import validate_response
        from ..routing import get_router
        from ..tools.agent import AgentConfig, run_agent_loop
        from ..tools.builtins import get_default_tools

        # Select backend
        _, backend_obj = await get_router().select_optimal_backend(
            message, None, intent.task_type, backend_type
        )

        if not backend_obj:
            return OrchestrationResult(
                response="Error: No backend available",
                success=False,
                error="no_backend",
            )

        # Select model - agentic tasks need tool-capable models
        # Override to coder tier if no explicit model given
        selected_model = model_override or await select_model(
            task_type="review",  # Force coder tier for tool support
            content_size=len(message),
            content=message,
        )

        log.info("agentic_model_selected", model=selected_model)

        # Generate system prompt with tool context
        system_prompt = self.prompt_generator.generate(
            intent,
            user_message=message,
            model_name=selected_model,
            backend_name=backend_obj.name,
        )

        # ACE: Inject learned strategic playbook bullets
        playbook_context = playbook_manager.format_for_prompt(intent.task_type)
        if playbook_context:
            system_prompt += f"\n\n{playbook_context}"

        # Inject Serena memory context if available
        memory_context = await self._retrieve_context(message)
        if memory_context:
            system_prompt += f"\n\n## Project Documentation (Serena Memories)\n{memory_context}"

        # Get tools registry - enable all tools
        registry = get_default_tools(
            workspace=None,  # No confinement - full access
            allow_write=True,
            allow_exec=True,
        )

        # Add delegate_subtask tool for spawning sub-agents
        from ..tools.registry import ToolDefinition

        async def delegate_subtask(
            task: str,
            content: str,
            model_tier: str = "coder",
        ) -> str:
            """Spawn a sub-agent to handle a specific task."""
            from ..mcp_server import delegate as mcp_delegate
            result = await mcp_delegate(
                task=task,
                content=content,
                model=model_tier,
            )
            return result

        registry.register(ToolDefinition(
            name="delegate_subtask",
            description="Spawn a sub-agent to handle a specific task. Use for complex multi-part work.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "enum": ["quick", "summarize", "generate", "review", "analyze", "plan", "critique"],
                        "description": "Task type for the sub-agent"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content/prompt for the sub-agent"
                    },
                    "model_tier": {
                        "type": "string",
                        "enum": ["quick", "coder", "moe", "thinking"],
                        "description": "Model tier for the sub-agent (default: coder)"
                    }
                },
                "required": ["task", "content"]
            },
            handler=delegate_subtask,
        ))

        model_supports_tools = any(x in selected_model.lower() for x in [
            'qwen', 'gpt', 'claude', 'gemini', 'mistral', 'llama3', 'deepseek',
            'devstral', 'nemotron',  # Agentic-focused models
        ])

        agent_config = AgentConfig(
            max_iterations=10,
            timeout_per_tool=60,
            total_timeout=300,
            parallel_tools=False,
            native_tool_calling=model_supports_tools,
            allow_write=True,
            allow_exec=True,
            require_confirmation=False,
            reflection_enabled=True,
            max_reflections=1,
        )

        async def agent_llm_call(
            messages: list[dict],
            system: str | None = None,
        ) -> dict:
            # Only pass tools for native mode - text mode uses system prompt
            tools = registry.get_openai_schemas() if agent_config.native_tool_calling else None

            # Extract prompt for logging (last user message)
            prompt = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                    break

            log.debug(
                "agent_llm_call",
                native_tools=agent_config.native_tool_calling,
                tool_count=len(tools) if tools else 0,
                message_count=len(messages),
            )

            result = await call_llm(
                model=selected_model,
                prompt=prompt,  # Fallback for non-chat providers
                system=system or system_prompt,
                task_type=intent.task_type,
                original_task=intent.task_type,
                language="unknown",
                content_preview=prompt[:100],
                backend_obj=backend_obj,
                tools=tools,
                messages=messages,  # Pass full conversation history
            )

            # Instrument melon economy for intermediate steps 🍈
            if result.get("success") and result.get("response"):
                from ..melons import award_melons_for_quality
                from ..quality import validate_response
                from ..tools.parser import has_tool_calls

                resp_text = result.get("response", "")
                # Intermediate quality check
                step_quality = validate_response(resp_text, "agent")

                # If the model is successfully using tools, give it a tiny boost
                # but only if the quality is high.
                if step_quality.overall >= 0.9:
                    # Award 1 melon for a high-quality intermediate step (rare!)
                    # We use a lower award than final response to keep melons scarce.
                    if has_tool_calls(resp_text):
                        log.debug("agent_step_award", model=selected_model, quality=step_quality.overall)
                        award_melons_for_quality(selected_model, "agent", step_quality.overall)
                elif step_quality.overall < 0.3:
                    # Penalize poor quality even in intermediate steps
                    log.warning("agent_step_penalty", model=selected_model, quality=step_quality.overall)
                    award_melons_for_quality(selected_model, "agent", step_quality.overall)

            return result

        async def critique_callback(response: str, original_prompt: str) -> tuple[bool, str]:
            """Review the agent's work and provide feedback."""
            from ..mcp_server import _get_delegate_context, delegate_impl

            log.info("agentic_reflection_starting", model=selected_model)

            critique_prompt = f"""You are a QA Lead reviewing an AI agent's work.

Original Task:
{original_prompt}

Agent's Final Response:
{response}

CRITICAL CHECKLIST:
1. Did the agent actually perform the task using its tools?
2. If tools (like web search) returned data, is that data ACTUALLY PRINTED in the response?
3. Reject if the response says "I found the news" or "Here is the news" but doesn't list at least 3-5 specific items.
4. Reject if the agent just provided a script instead of the requested info.
5. NO lazy responses allowed. The user wants the DATA, not a confirmation that you found it.

If perfect, output "VERIFIED".
If lazy or missing data, provide specific instructions to include the tool results in the final text.
"""
            # Run critique using a high-quality model
            ctx = _get_delegate_context()
            critique_result = await delegate_impl(
                ctx=ctx,
                task="critique",
                content=critique_prompt,
                model="moe", # Use high-reasoning for critique
                include_metadata=False,
            )

            is_verified = "VERIFIED" in critique_result or "verified" in critique_result.lower()

            if not is_verified:
                log.info("agentic_reflection_failed", feedback_len=len(critique_result))

            return is_verified, critique_result

        log.info(
            "agentic_execution_started",
            model=selected_model,
            task_type=intent.task_type,
            tools=registry.list_tools(),
        )

        # Run agent loop
        try:
            agent_result = await run_agent_loop(
                call_llm=agent_llm_call,
                prompt=message,
                system_prompt=system_prompt,
                registry=registry,
                model=selected_model,
                config=agent_config,
                critique_callback=critique_callback,
            )

            response = strip_thinking_tags(agent_result.response or "No response")

            # Validate quality
            quality_result = validate_response(response, intent.task_type)

            log.info(
                "agentic_execution_complete",
                model=selected_model,
                iterations=agent_result.iterations,
                tool_calls=len(agent_result.tool_calls),
                success=agent_result.success,
            )

            return OrchestrationResult(
                response=response,
                success=agent_result.success,
                model_used=selected_model,
                mode=OrchestrationMode.AGENTIC,
                quality_score=quality_result.overall,
                tokens=agent_result.tokens,
                debug_info={
                    "iterations": agent_result.iterations,
                    "tool_calls": [tc.name for tc in agent_result.tool_calls],
                },
            )

        except Exception as e:
            log.error("agentic_execution_error", error=str(e))
            return OrchestrationResult(
                response=f"Agent execution failed: {e}",
                success=False,
                error=str(e),
                model_used=selected_model,
                mode=OrchestrationMode.AGENTIC,
            )

    async def _execute_voting(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
    ) -> OrchestrationResult:
        """
        Execute K-voting for reliable answers.

        Uses VotingConsensus to get mathematically-guaranteed
        reliability through repeated sampling.
        """
        from ..llm import call_llm
        from ..quality import ResponseQualityValidator
        from ..routing import get_router
        from ..voting import VotingConsensus

        k = intent.k_votes

        # Select backend
        _, backend_obj = await get_router().select_optimal_backend(
            message, None, intent.task_type, backend_type
        )

        if not backend_obj:
            return OrchestrationResult(
                response="Error: No backend available for voting",
                success=False,
                error="no_backend",
            )

        # Select model
        selected_model = model_override or await select_model(
            task_type=intent.task_type,
            content_size=len(message),
            content=message,
        )

        # Generate system prompt
        system_prompt = self.prompt_generator.generate(
            intent,
            user_message=message,
            model_name=selected_model,
        )

        # Inject Serena memory context if available
        memory_context = await self._retrieve_context(message)
        if memory_context:
            system_prompt += f"\n\n## Project Documentation (Serena Memories)\n{memory_context}"

        # Initialize voting
        validator = ResponseQualityValidator()
        consensus = VotingConsensus(
            k=k,
            quality_validator=validator,
            similarity_threshold=0.85,
            max_response_length=700,
        )

        max_attempts = k * 3
        attempts = 0
        total_tokens = 0

        log.info("voting_started", k=k, model=selected_model, max_attempts=max_attempts)

        while attempts < max_attempts:
            attempts += 1

            # Get a response with higher temperature for diversity
            result = await call_llm(
                model=selected_model,
                prompt=message,
                system=system_prompt,
                task_type=intent.task_type,
                original_task="vote",
                language="unknown",
                content_preview=message[:100],
                backend_obj=backend_obj,
                temperature=0.7,  # Higher temp for diverse samples
            )

            if not result.get("success"):
                continue

            total_tokens += result.get("tokens", 0)
            response = strip_thinking_tags(result.get("response", ""))
            vote_result = consensus.add_vote(response)

            if vote_result.red_flagged:
                log.debug(
                    "vote_red_flagged",
                    attempt=attempts,
                    reason=vote_result.red_flag_reason,
                )
                continue

            if vote_result.consensus_reached:
                prob = VotingConsensus.voting_probability(k, 0.95)

                log.info(
                    "voting_consensus_reached",
                    votes=vote_result.votes_for_winner,
                    total_attempts=attempts,
                    confidence=f"{prob:.2%}",
                )

                return OrchestrationResult(
                    response=vote_result.winning_response or response,
                    success=True,
                    model_used=selected_model,
                    mode=OrchestrationMode.VOTING,
                    votes_cast=vote_result.total_votes,
                    consensus_reached=True,
                    confidence=prob,
                    tokens=total_tokens,
                    debug_info={
                        "k": k,
                        "votes_for_winner": vote_result.votes_for_winner,
                        "total_attempts": attempts,
                    }
                )

        # No consensus - return best effort
        best_response, metadata = consensus.get_best_response()

        log.warning(
            "voting_no_consensus",
            best_votes=metadata.winning_votes,
            k_needed=k,
            total_attempts=attempts,
            red_flagged=metadata.red_flagged_count,
        )

        if best_response:
            return OrchestrationResult(
                response=f"{best_response}\n\n---\n*Note: Partial consensus ({metadata.winning_votes}/{k} votes)*",
                success=True,
                model_used=selected_model,
                mode=OrchestrationMode.VOTING,
                votes_cast=metadata.total_votes,
                consensus_reached=False,
                confidence=0.5,  # Lower confidence without full consensus
                tokens=total_tokens,
            )
        else:
            return OrchestrationResult(
                response="Unable to reach consensus after multiple attempts.",
                success=False,
                error="voting_failed",
                model_used=selected_model,
                tokens=total_tokens,
            )

    async def _execute_agentic_stream(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None = None,
        model_override: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute agentic loop with streaming events."""
        from ..routing import get_router
        from ..tools.agent import AgentConfig, run_agent_loop
        from ..tools.builtins import get_default_tools

        # Select backend
        _, backend_obj = await get_router().select_optimal_backend(
            message, None, intent.task_type, backend_type
        )

        if not backend_obj:
            yield StreamEvent(event_type="error", message="No backend available")
            return

        selected_model = model_override or await select_model(
            task_type="review",
            content_size=len(message),
            content=message,
        )

        system_prompt = self.prompt_generator.generate(
            intent,
            user_message=message,
            model_name=selected_model,
            backend_name=backend_obj.name,
        )

        registry = get_default_tools(allow_write=True, allow_exec=True)

        event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        async def agent_llm_call(messages: list[dict], system: str | None = None) -> dict:
            await event_queue.put(StreamEvent(event_type="thinking", message="Generating..."))
            return await call_llm(
                model=selected_model,
                prompt=messages[-1].get("content", ""),
                system=system or system_prompt,
                task_type=intent.task_type,
                backend_obj=backend_obj,
                messages=messages,
            )

        def on_tool_call(tc: Any):
            event_queue.put_nowait(StreamEvent(
                event_type="tool_call",
                message=f"Calling {tc.name}...",
                details={"name": tc.name, "args": tc.arguments}
            ))

        def on_tool_result(res: Any):
            event_queue.put_nowait(StreamEvent(
                event_type="status",
                message=f"Tool {res.tool_name} finished.",
                details={"name": res.tool_name, "success": res.success}
            ))

        agent_config = AgentConfig(
            max_iterations=10,
            native_tool_calling=backend_obj.supports_native_tool_calling if backend_obj else False,
            reflection_enabled=True,
        )

        async def critique_callback(response: str, original_prompt: str) -> tuple[bool, str]:
            """Review the agent's work and provide feedback."""
            from ..mcp_server import _get_delegate_context, delegate_impl
            
            await event_queue.put(StreamEvent(event_type="thinking", message="Reflecting on response..."))
            
            critique_prompt = f"""You are a QA Lead reviewing an AI agent's work.

Original Task:
{original_prompt}

Agent's Final Response:
{response}

CRITICAL CHECKLIST:
1. Did the agent actually perform the task using its tools?
2. If tools (like read_file) returned data, is that data ACTUALLY PRINTED or SUMMARIZED in the response?
3. Reject if the response is just a confirmation (e.g. "I have read the file") without the actual content/review.
4. NO lazy responses. The user wants the analysis, not a script to do the analysis.

If perfect, output "VERIFIED". Otherwise, provide specific instructions to include the actual data/analysis in the response."""

            ctx = _get_delegate_context()
            critique_result = await delegate_impl(
                ctx=ctx,
                task="critique",
                content=critique_prompt,
                model="moe",
                include_metadata=False,
            )
            
            is_verified = "VERIFIED" in critique_result or "verified" in critique_result.lower()
            if not is_verified:
                log.info("agent_critique_feedback_received", feedback_len=len(critique_result))
                await event_queue.put(StreamEvent(
                    event_type="status", 
                    message="Reviewer: More detail needed. Updating...",
                    details={"feedback": critique_result}
                ))
            else:
                log.info("agent_critique_passed")
                await event_queue.put(StreamEvent(event_type="status", message="Reviewer: Response verified."))
                
            return is_verified, critique_result

        # Run agent loop in background
        agent_task = asyncio.create_task(run_agent_loop(
            call_llm=agent_llm_call,
            prompt=message,
            system_prompt=system_prompt,
            registry=registry,
            model=selected_model,
            config=agent_config,
            critique_callback=critique_callback,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        ))

        while not agent_task.done() or not event_queue.empty():
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                yield event
            except TimeoutError:
                continue

        result = await agent_task
        yield StreamEvent(event_type="response", message=result.response, details={"model": selected_model})

    async def _execute_voting_stream(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute voting with streaming progress events."""
        k = intent.k_votes

        yield StreamEvent(
            event_type="thinking",
            message=f"Starting K-voting with k={k}...",
        )

        # Run voting
        result = await self._execute_voting(
            intent, message, backend_type, model_override
        )

        if result.consensus_reached:
            yield StreamEvent(
                event_type="status",
                message=f"Consensus reached! ({result.votes_cast} votes, {result.confidence:.1%} confidence)",
            )
        else:
            yield StreamEvent(
                event_type="status",
                message=f"Best effort response ({result.votes_cast} votes)",
            )

        yield StreamEvent(
            event_type="response",
            message=result.response,
            details={
                "model": result.model_used,
                "consensus": result.consensus_reached,
                "confidence": result.confidence,
            }
        )

    async def _execute_comparison(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
    ) -> OrchestrationResult:
        """
        Execute multi-model comparison.

        Gets responses from multiple models and presents them together.
        """
        from ..backend_manager import backend_manager
        from ..llm import call_llm

        # Get available backends/models
        enabled_backends = backend_manager.get_enabled_backends()

        if not enabled_backends:
            return OrchestrationResult(
                response="Error: No backends available for comparison",
                success=False,
                error="no_backends",
            )

        # Generate system prompt
        system_prompt = self.prompt_generator.generate(intent, user_message=message)

        # Collect responses from different models
        responses: list[tuple[str, str, int]] = []  # (model, response, time_ms)
        total_tokens = 0

        # Use models from intent or default to available tiers
        models_to_compare = intent.comparison_models
        if not models_to_compare:
            # Default: compare coder and moe tiers if available
            backend = enabled_backends[0]
            models_to_compare = [
                backend.models.get("coder"),
                backend.models.get("moe"),
            ]
            models_to_compare = [m for m in models_to_compare if m]

        backend = enabled_backends[0]  # Use first available

        for model in models_to_compare:
            start = time.time()

            result = await call_llm(
                model=model,
                prompt=message,
                system=system_prompt,
                task_type=intent.task_type,
                original_task="compare",
                language="unknown",
                content_preview=message[:100],
                backend_obj=backend,
            )

            elapsed = int((time.time() - start) * 1000)

            if result.get("success"):
                total_tokens += result.get("tokens", 0)
                responses.append((model, strip_thinking_tags(result.get("response", "")), elapsed))
            else:
                responses.append((model, f"Error: {result.get('error')}", elapsed))

        # Format comparison output
        output_parts = ["# Model Comparison\n"]

        for model, response, time_ms in responses:
            output_parts.append(f"## {model} ({time_ms}ms)\n")
            output_parts.append(response)
            output_parts.append("\n---\n")

        return OrchestrationResult(
            response="\n".join(output_parts),
            success=True,
            models_compared=[m for m, _, _ in responses],
            mode=OrchestrationMode.COMPARISON,
            tokens=total_tokens,
        )

    async def _execute_comparison_stream(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute comparison with streaming events."""
        yield StreamEvent(
            event_type="thinking",
            message="Running multi-model comparison...",
        )

        result = await self._execute_comparison(
            intent, message, backend_type, model_override
        )

        yield StreamEvent(
            event_type="status",
            message=f"Compared {len(result.models_compared)} models",
            details={"models": result.models_compared}
        )

        yield StreamEvent(
            event_type="response",
            message=result.response,
        )

    async def _execute_deep_thinking(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
    ) -> OrchestrationResult:
        """
        Execute deep thinking with extended reasoning model.

        Uses MoE/thinking tier for complex analysis.
        """
        from ..llm import call_llm
        from ..routing import get_router

        # Force MoE or thinking tier
        _, backend_obj = await get_router().select_optimal_backend(
            message, None, "moe", backend_type
        )

        if not backend_obj:
            return OrchestrationResult(
                response="Error: No backend available",
                success=False,
                error="no_backend",
            )

        # Force thinking/moe model
        selected_model = model_override or await select_model(
            task_type="moe",  # Force complex reasoning model
            content_size=len(message),
            model_override="thinking",  # Prefer thinking if available
            content=message,
        )

        # Generate detailed system prompt
        system_prompt = self.prompt_generator.generate(
            intent,
            user_message=message,
            model_name=selected_model,
        )

        # Enable thinking mode
        result = await call_llm(
            model=selected_model,
            prompt=message,
            system=system_prompt,
            task_type="moe",
            original_task="deep_thinking",
            language="unknown",
            content_preview=message[:100],
            backend_obj=backend_obj,
            enable_thinking=True,
        )

        if not result.get("success"):
            return OrchestrationResult(
                response=result.get("error", "Deep thinking failed"),
                success=False,
                error=result.get("error"),
                model_used=selected_model,
                tokens=result.get("tokens", 0),
            )

        return OrchestrationResult(
            response=strip_thinking_tags(result.get("response", "")),
            success=True,
            model_used=selected_model,
            mode=OrchestrationMode.DEEP_THINKING,
            tokens=result.get("tokens", 0),
        )

    async def _execute_tree_of_thoughts(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
    ) -> OrchestrationResult:
        """
        Execute Tree of Thoughts search.

        1. Expand: Generate 3 possible next steps/solutions
        2. Score: Evaluate each step (0-10)
        3. Select: Pick best path
        4. Refine: Generate final response
        """
        from ..llm import call_llm
        from ..routing import get_router

        # Select backend
        _, backend_obj = await get_router().select_optimal_backend(
            message, None, "moe", backend_type
        )

        if not backend_obj:
            return OrchestrationResult(
                response="Error: No backend available for ToT",
                success=False,
                error="no_backend",
            )

        # Prefer thinking/moe model
        selected_model = model_override or await select_model(
            task_type="moe",
            content_size=len(message),
            model_override="thinking",
            content=message,
        )

        # Inject Serena memory context if available (ToT needs good context!)
        memory_context = await self._retrieve_context(message)
        context_str = f"\n\n## Context (Serena Memories)\n{memory_context}" if memory_context else ""

        total_tokens = 0

        # Step 1: Generate candidates
        gen_prompt = f"""You are an expert problem solver.
User Request: {message}{context_str}

Generate 3 distinct, valid approaches to solve this problem.
Format each approach clearly as "Option 1:", "Option 2:", "Option 3:".
Do not evaluate them yet, just generate them.
"""
        result_gen = await call_llm(
            model=selected_model,
            prompt=gen_prompt,
            system="You are a creative generator. Generate distinct options.",
            task_type="moe",
            original_task="tot_generate",
            language="unknown",
            content_preview=message[:100],
            backend_obj=backend_obj,
        )

        if not result_gen.get("success"):
            return OrchestrationResult(response="Failed to generate thoughts.", success=False, error=result_gen.get("error"))

        total_tokens += result_gen.get("tokens", 0)
        candidates_text = strip_thinking_tags(result_gen.get("response", ""))

        # Step 2: Evaluate candidates
        eval_prompt = f"""You are a critical evaluator.
User Request: {message}

Proposed Options:
{candidates_text}

Evaluate each option on a scale of 0 to 10 based on:
- Feasibility
- Effectiveness
- Safety
- Efficiency

Select the BEST option and explain why.
Format:
BEST: Option X
REASON: ...
"""
        result_eval = await call_llm(
            model=selected_model, # Use same model or switch to critic? Same is usually fine for self-correction if prompt changes.
            prompt=eval_prompt,
            system="You are a critical judge. Pick the winner.",
            task_type="moe",
            original_task="tot_evaluate",
            language="unknown",
            content_preview="evaluating options",
            backend_obj=backend_obj,
        )

        if not result_eval.get("success"):
             return OrchestrationResult(response="Failed to evaluate thoughts.", success=False, error=result_eval.get("error"))

        total_tokens += result_eval.get("tokens", 0)
        eval_text = strip_thinking_tags(result_eval.get("response", ""))

        # Step 3: Final refinement of best path
        final_prompt = f"""You are an expert executor.
User Request: {message}

Selected Approach:
{eval_text}

Now, execute this best approach fully. Provide the complete solution/answer based on this path.
"""
        result_final = await call_llm(
            model=selected_model,
            prompt=final_prompt,
            system="You are a solver. Provide the final detailed solution.",
            task_type="moe",
            original_task="tot_final",
            language="unknown",
            content_preview="generating final",
            backend_obj=backend_obj,
        )

        if not result_final.get("success"):
             return OrchestrationResult(response="Failed to generate final solution.", success=False, error=result_final.get("error"))

        total_tokens += result_final.get("tokens", 0)
        final_response = strip_thinking_tags(result_final.get("response", ""))

        # Combine for transparency
        full_response = f"""## Tree of Thoughts Exploration

### 1. Generated Options
{candidates_text}

### 2. Evaluation
{eval_text}

### 3. Final Solution
{final_response}
"""
        return OrchestrationResult(
            response=full_response,
            success=True,
            model_used=selected_model,
            mode=OrchestrationMode.TREE_OF_THOUGHTS,
            tokens=total_tokens,
        )

    async def _execute_chain(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
    ) -> OrchestrationResult:
        """
        Execute a chain of sequential steps, passing outputs forward.

        Each step's output is appended to the context for the next step.
        This enables pipelines like: analyze → generate → review
        """
        from ..llm import call_llm
        from ..routing import get_router
        from ..tracing import trace

        if not intent.chain_steps or len(intent.chain_steps) < 2:
            log.warning("chain_no_steps", intent=intent)
            return await self._execute_simple(intent, message, backend_type, model_override)

        with trace("chain_execution", steps=len(intent.chain_steps)) as span:
            step_outputs: list[str] = []
            models_used: list[str] = []
            total_tokens: int = 0
            current_context = message

            for i, step_desc in enumerate(intent.chain_steps):
                # Parse step: "task_type: description"
                if ":" in step_desc:
                    task_type, step_prompt = step_desc.split(":", 1)
                    task_type = task_type.strip()
                    step_prompt = step_prompt.strip()
                else:
                    task_type = "quick"
                    step_prompt = step_desc

                span.event(f"step_{i+1}_start", task_type=task_type, prompt_preview=step_prompt[:50])

                # Select backend and model for this step
                _, backend_obj = await get_router().select_optimal_backend(
                    current_context, None, task_type, backend_type
                )

                if not backend_obj:
                    span.set_error(f"No backend available for step {i+1}")
                    return OrchestrationResult(
                        response=f"Error: No backend available for step {i+1}",
                        success=False,
                        error="no_backend",
                        tokens=total_tokens,
                    )

                selected_model = model_override or await select_model(
                    task_type=task_type,
                    content_size=len(current_context),
                    content=current_context,
                )

                # Build prompt with context from previous steps
                if step_outputs:
                    context_section = "\n\n---\n**Previous step output:**\n" + step_outputs[-1]
                    full_prompt = f"{step_prompt}\n\n**Original request:** {message}{context_section}"
                else:
                    full_prompt = f"{step_prompt}\n\n**Original request:** {message}"

                # Generate step-specific system prompt
                step_intent = DetectedIntent(
                    task_type=task_type,
                    orchestration_mode=OrchestrationMode.CHAIN,
                    confidence=intent.confidence,
                )
                system_prompt = self.prompt_generator.generate(
                    step_intent,
                    user_message=full_prompt,
                    model_name=selected_model,
                )

                # Execute step
                result = await call_llm(
                    model=selected_model,
                    prompt=full_prompt,
                    system=system_prompt,
                    task_type=task_type,
                    original_task="chain_step",
                    language="unknown",
                    content_preview=full_prompt[:100],
                    backend_obj=backend_obj,
                )

                if not result.get("success"):
                    span.set_error(f"Step {i+1} failed: {result.get('error', 'unknown')}")
                    return OrchestrationResult(
                        response=f"Chain failed at step {i+1}: {result.get('error', 'unknown')}",
                        success=False,
                        error=f"chain_step_{i+1}_failed",
                        model_used=selected_model,
                        mode=OrchestrationMode.CHAIN,
                        tokens=total_tokens + result.get("tokens", 0),
                    )

                total_tokens += result.get("tokens", 0)
                step_output = strip_thinking_tags(result.get("response", ""))
                step_outputs.append(step_output)
                models_used.append(selected_model)

                # Update context with this step's output
                current_context = f"{message}\n\nStep {i+1} output: {step_output}"

                span.event(f"step_{i+1}_complete", model=selected_model, output_len=len(step_output))

            # Combine all step outputs for final response
            final_response = self._format_chain_response(intent.chain_steps, step_outputs)

            return OrchestrationResult(
                response=final_response,
                success=True,
                model_used=models_used[-1] if models_used else None,
                mode=OrchestrationMode.CHAIN,
                tokens=total_tokens,
                voting_stats={
                    "chain_steps": len(intent.chain_steps),
                    "models_used": models_used,
                    "step_lengths": [len(o) for o in step_outputs],
                },
            )

    async def _execute_chain_stream(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute a chain with streaming progress updates.
        """
        if not intent.chain_steps or len(intent.chain_steps) < 2:
            result = await self._execute_simple(intent, message, backend_type, model_override)
            yield StreamEvent(
                event_type="response",
                message=result.response,
                details={"model": result.model_used}
            )
            return

        yield StreamEvent(
            event_type="status",
            message=f"Executing {len(intent.chain_steps)}-step chain...",
            details={"steps": intent.chain_steps}
        )

        # Execute chain and yield progress
        result = await self._execute_chain(intent, message, backend_type, model_override)

        if result.success:
            yield StreamEvent(
                event_type="response",
                message=result.response,
                details={
                    "model": result.model_used,
                    "chain_stats": result.debug_info,
                }
            )
        else:
            yield StreamEvent(
                event_type="error",
                message=result.response,
                details={"error": result.error}
            )

    def _format_chain_response(
        self,
        steps: list[str],
        outputs: list[str],
    ) -> str:
        """Format chain outputs into a readable response."""
        parts = ["## Chain Execution Results\n"]

        for i, (step, output) in enumerate(zip(steps, outputs, strict=False)):
            # Extract task type from step description
            if ":" in step:
                task_type, desc = step.split(":", 1)
                step_header = f"### Step {i+1}: {task_type.upper()} - {desc.strip()}"
            else:
                step_header = f"### Step {i+1}: {step}"

            parts.append(f"{step_header}\n")
            parts.append(f"{output.strip()}\n")

        return "\n".join(parts)

    async def _reflect_on_failure(
        self,
        task_type: str,
        message: str,
        response: str,
        error: str | None = None,
        backend_obj: Any | None = None,
    ) -> None:
        """
        ACE Reflector: Diagnose failures and extract strategic lessons.
        """
        log.info("ace_reflection_triggered", task_type=task_type)

        reflection_prompt = f"""Task: {message}
Response/Trajectory: {response}
Error (if any): {error or "Low quality output"}

Diagnose the failure and provide a strategic lesson for the playbook."""

        try:
            # Use a high-quality model for reflection (Planner tier)
            result = await call_llm(
                model=config.model_moe.default_model,
                prompt=reflection_prompt,
                system=ACE_REFLECTOR_PROMPT,
                task_type="reflection",
                backend_obj=backend_obj,
                enable_thinking=True,
            )

            if not result.get("success"):
                return

            reflection_json = strip_thinking_tags(result.get("response", ""))
            try:
                reflection_data = json.loads(reflection_json)
                lesson = reflection_data.get("playbook_update")
                if lesson:
                    await self._curate_playbook(task_type, lesson, backend_obj)
            except json.JSONDecodeError:
                log.warning("reflection_json_parse_failed", raw=reflection_json[:100])

        except Exception as e:
            log.error("reflection_failed", error=str(e))

    async def _curate_playbook(self, task_type: str, lesson: str, backend_obj: Any | None = None) -> None:
        """
        ACE Curator: Integrate new lessons into the itemized playbook.
        """
        current_playbook = playbook_manager.load_playbook(task_type)
        playbook_text = "\n".join([f"- {b.content}" for b in current_playbook])

        curation_prompt = f"""Current Playbook:
{playbook_text}

New Lesson:
{lesson}

Integrate this lesson into the playbook using the ADD operation if it's new and valuable."""

        try:
            result = await call_llm(
                model=config.model_moe.default_model,
                prompt=curation_prompt,
                system=ACE_CURATOR_PROMPT,
                task_type="curation",
                backend_obj=backend_obj,
            )

            if not result.get("success"):
                return

            curation_json = strip_thinking_tags(result.get("response", ""))
            try:
                curation_data = json.loads(curation_json)
                for op in curation_data.get("operations", []):
                    if op.get("type") == "ADD":
                        playbook_manager.add_bullet(
                            task_type=task_type,
                            content=op.get("content"),
                            section=op.get("section", "strategies_and_hard_rules")
                        )
                        log.info("playbook_entry_added", task_type=task_type)
            except json.JSONDecodeError:
                log.warning("curation_json_parse_failed", raw=curation_json[:100])

        except Exception as e:
            log.error("curation_failed", error=str(e))

    async def _execute_workflow(
        self,
        intent: DetectedIntent,
        message: str,
        session_id: str | None,
        backend_type: str | None,
        model_override: str | None,
    ) -> OrchestrationResult:
        """
        Execute a DAG workflow with conditional branching.

        Input 'message' should be a JSON WorkflowDefinition.
        """
        from ..mcp_server import _get_delegate_context
        from ..task_workflow import execute_workflow, parse_workflow_definition

        try:
            definition = parse_workflow_definition(message)
        except ValueError as e:
            return OrchestrationResult(
                response=f"Error parsing workflow definition: {e}",
                success=False,
                error="invalid_workflow_json",
            )

        log.info("workflow_execution_started", name=definition.name)

        ctx = _get_delegate_context()
        result = await execute_workflow(
            definition=definition,
            ctx=ctx,
            session_id=session_id,
        )

        # Format final response
        parts = [f"# Workflow: {definition.name}\n"]
        if result.success:
            parts.append("✅ Workflow completed successfully.\n")
        else:
            parts.append(f"❌ Workflow failed with {len(result.errors)} errors.\n")

        for node_res in result.node_results:
            status = "✅" if node_res.success else "❌"
            parts.append(f"### {status} Node: {node_res.node_id}")
            if node_res.success:
                parts.append(f"{node_res.output}\n")
            else:
                parts.append(f"Error: {node_res.error}\n")

        return OrchestrationResult(
            response="\n".join(parts),
            success=result.success,
            mode=OrchestrationMode.WORKFLOW,
            elapsed_ms=result.elapsed_ms,
            debug_info=result.to_dict(),
        )

    def _award_melons(self, result: OrchestrationResult, intent: DetectedIntent) -> None:
        """Award melons based on orchestration result."""
        from ..melons import award_melons_for_quality

        # Award melons to the model(s) used
        if result.model_used:
            award_melons_for_quality(
                model_id=result.model_used,
                task_type=intent.task_type,
                quality_score=result.quality_score,
            )

        # Bonus for successful consensus
        if result.consensus_reached:
            from ..melons import get_melon_tracker
            tracker = get_melon_tracker()
            tracker.award(
                result.model_used,
                intent.task_type,
                melons=1,  # Bonus melon for consensus
                success=True,
            )

    async def _retrieve_context(self, message: str) -> str:
        """
        Retrieve relevant context from Serena memories.
        """
        available = list_serena_memories()
        if not available:
            return ""

        found_content = []
        message_lower = message.lower()

        for mem_name in available:
            # Normalize name (replace _ with space) for matching
            name_keywords = mem_name.replace("_", " ").split()

            # Simple heuristic
            if any(kw in message_lower for kw in name_keywords if len(kw) > 3):
                content = read_serena_memory(mem_name)
                if content:
                    found_content.append(f"### Context from '{mem_name}':\n{content}")
                    log.info("memory_retrieved_keyword", memory=mem_name)

        if found_content:
            return "\n\n".join(found_content)

        return ""


# Module-level convenience
_executor: OrchestrationExecutor | None = None


def get_orchestration_executor() -> OrchestrationExecutor:
    """Get or create the global orchestration executor."""
    global _executor
    if _executor is None:
        _executor = OrchestrationExecutor()
    return _executor


async def execute_orchestration(
    intent: DetectedIntent,
    message: str,
    **kwargs,
) -> OrchestrationResult:
    """Execute orchestration using the global executor."""
    return await get_orchestration_executor().execute(intent, message, **kwargs)

