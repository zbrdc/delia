# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Orchestration Executor - The Heart of Delia's NLP Orchestration.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

from ..config import config
from ..file_helpers import list_memories, read_memory
from ..llm import call_llm
from ..playbook import playbook_manager
from ..session_manager import get_session_manager
from .tuning import TuningAdvisor, get_tuning_advisor
from ..prompts import (
    ModelRole,
    build_system_prompt,
)
from ..personas import get_persona_manager
from ..routing import select_model
from ..text_utils import strip_thinking_tags
from .intent import DetectedIntent
from .outputs import get_json_schema_prompt, parse_structured_output
from .prompts import get_prompt_generator
from .result import OrchestrationMode, OrchestrationResult, StreamEvent

if TYPE_CHECKING:
    from .parser import ParsedToolCall
    from .registry import ToolRegistry

log = structlog.get_logger()


class ToTConfig(BaseModel):
    """
    Research-backed Tree of Thoughts configuration (ADR-008).

    Based on 2025 research showing ToT has diminishing returns for most tasks.
    These defaults optimize for compute efficiency while preserving ToT's value
    for genuinely complex planning problems.
    """

    max_branches: int = 3  # Limit parallel branches (research: fewer is often better)
    timeout_per_branch: float = 20.0  # Reduced from 30s
    early_stop_on_high_confidence: bool = True  # Stop if first branch is confident
    confidence_threshold: float = 0.85  # Confidence level to trigger early stop
    prune_before_critic: bool = True  # Only send top N to critic
    max_branches_for_critic: int = 2  # Max branches to evaluate (reduces LLM calls)


class OrchestrationExecutor:
    """Executes orchestration based on detected intent."""

    def __init__(self) -> None:
        self.prompt_generator = get_prompt_generator()
        from .critic import ResponseCritic
        from .dispatcher import ModelDispatcher
        from .planner import StrategicPlanner
        self.dispatcher = ModelDispatcher(call_llm)
        self.planner = StrategicPlanner(call_llm)
        self.critic = ResponseCritic(call_llm)
        self.tuning_advisor = get_tuning_advisor()

    async def execute(
        self,
        intent: DetectedIntent,
        message: str,
        original_message: str | None = None,
        session_id: str | None = None,
        backend_type: str | None = None,
        model_override: str | None = None,
        output_type: type[BaseModel] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> OrchestrationResult:
        start_time = time.time()
        log.info("orchestration_execute", mode=intent.orchestration_mode.value, task=intent.task_type)

        if intent.task_type == "status" and intent.orchestration_mode == OrchestrationMode.NONE:
            return await self._execute_status_query(intent, original_message or message)

        if output_type:
            message = f"{message}\n\n{get_json_schema_prompt(output_type)}"

        try:
            if intent.orchestration_mode == OrchestrationMode.AGENTIC:
                result = await self._execute_agentic(intent, message, session_id, backend_type, model_override, messages=messages)
            elif intent.orchestration_mode == OrchestrationMode.VOTING:
                result = await self._execute_voting(intent, message, backend_type, model_override, messages=messages)
            elif intent.orchestration_mode == OrchestrationMode.DEEP_THINKING:
                result = await self._execute_deep_thinking(intent, message, backend_type, model_override, messages=messages)
            elif intent.orchestration_mode == OrchestrationMode.TREE_OF_THOUGHTS:
                result = await self._execute_tree_of_thoughts(intent, message, backend_type, model_override, messages=messages)
            else:
                result = await self._execute_simple(intent, message, backend_type, model_override, original_message, session_id, messages=messages)

            result.elapsed_ms = int((time.time() - start_time) * 1000)
            if output_type and result.success:
                try:
                    result.structured = parse_structured_output(result.response, output_type)
                    result.structured_success = True
                except Exception:
                    result.structured_success = False

            if result.success and result.model_used:
                self._award_melons(result, intent)
                # ACE Framework: Record playbook feedback to close the learning loop
                asyncio.create_task(self._record_playbook_feedback(intent.task_type, result.quality_score))

            if not result.success or (result.quality_score is not None and result.quality_score < 0.4):
                asyncio.create_task(self._reflect_on_failure(intent.task_type, message, result.response, result.error))

            return result
        except Exception as e:
            log.error("execute_error", error=str(e))
            return OrchestrationResult(response=f"Error: {e}", success=False, error=str(e))

    async def execute_stream(
        self,
        intent: DetectedIntent,
        message: str,
        original_message: str | None = None,
        session_id: str | None = None,
        backend_type: str | None = None,
        model_override: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        on_tool_call: Callable[[Any], None] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(event_type="status", message=f"Mode: {intent.orchestration_mode.value}")

        # Speculative 'Draft Zero' for complex modes (ADR-008 UX improvement)
        # Launches a quick model in parallel to provide instant feedback
        speculative_task = None
        if intent.orchestration_mode in (OrchestrationMode.VOTING, OrchestrationMode.DEEP_THINKING):
            yield StreamEvent(event_type="thinking", message="Generating speculative draft zero...")
            
            from ..llm import call_llm_stream
            from ..routing import get_router
            _, backend_obj = await get_router().select_optimal_backend(message, None, "quick", backend_type)
            quick_model = backend_obj.models.get("quick") if backend_obj else "auto"

            # Background task to stream quick tokens
            async def stream_speculative():
                yield StreamEvent(event_type="status", message="Draft Zero (Speculative)")
                async for chunk in call_llm_stream(
                    model=quick_model, 
                    prompt=message, 
                    system="You are Delia. Provide a FAST, brief initial response while a deeper analysis is being performed. Focus on the core answer.",
                    backend_obj=backend_obj,
                    max_tokens=150 # Keep it short
                ):
                    yield StreamEvent(event_type="speculative_token", message=chunk.text)
                yield StreamEvent(event_type="status", message="Draft Zero complete. Finalizing analysis...")

            # We can't easily 'background' an iterator into the current yield stream
            # So we'll just run it first or wrap it. 
            # Given the requirement for UX speed, let's stream it first.
            async for event in stream_speculative():
                yield event

        # Handle complex orchestration modes by running non-streaming execution
        # then yielding result as events. This ensures full orchestration coverage.
        if intent.orchestration_mode == OrchestrationMode.AGENTIC:
            async for event in self._execute_agentic_stream(intent, message, backend_type, model_override, messages=messages, on_tool_call=on_tool_call):
                yield event
            return

        if intent.orchestration_mode == OrchestrationMode.VOTING:
            yield StreamEvent(event_type="thinking", message="Running k-voting consensus...")
            result = await self._execute_voting(intent, message, backend_type, model_override, messages=messages)
            yield StreamEvent(event_type="orchestration", message="Voting complete", details={"k_votes": intent.k_votes, "consensus": result.success})
            yield StreamEvent(event_type="response", message=result.response, details={"model": result.model_used, "quality": result.quality_score})
            yield StreamEvent(event_type="done", message="Completed", details={"success": result.success})
            return

        # ADR-008: COMPARISON mode removed - comparison patterns now trigger VOTING

        if intent.orchestration_mode == OrchestrationMode.DEEP_THINKING:
            yield StreamEvent(event_type="thinking", message="Deep analysis with extended reasoning...")
            result = await self._execute_deep_thinking(intent, message, backend_type, model_override, messages=messages)
            yield StreamEvent(event_type="response", message=result.response, details={"model": result.model_used, "quality": result.quality_score})
            yield StreamEvent(event_type="done", message="Completed", details={"success": result.success})
            return

        if intent.orchestration_mode == OrchestrationMode.TREE_OF_THOUGHTS:
            yield StreamEvent(event_type="thinking", message="Tree-of-thoughts exploration...")
            result = await self._execute_tree_of_thoughts(intent, message, backend_type, model_override, messages=messages)
            yield StreamEvent(event_type="orchestration", message="ToT complete", details={"branches_explored": 3})
            yield StreamEvent(event_type="response", message=result.response, details={"model": result.model_used})
            yield StreamEvent(event_type="done", message="Completed", details={"success": result.success})
            return

        # Default: Simple streaming for NONE mode
        from ..llm import call_llm_stream
        from ..routing import get_router
        _, backend_obj = await get_router().select_optimal_backend(message, None, intent.task_type, backend_type)
        if not backend_obj:
            yield StreamEvent(event_type="error", message="No backend")
            return

        # Fast-path: Skip dispatcher for simple quick tasks (greetings, simple Q&A)
        # The dispatcher LLM call is expensive and unnecessary for trivial messages
        if intent.task_type == "quick" and intent.orchestration_mode == OrchestrationMode.NONE:
            target_role_str = "assistant"
            target_role = ModelRole.EXECUTOR
            selected_model = model_override or backend_obj.models.get("quick")
        else:
            target_role_str = await self.dispatcher.dispatch(original_message or message, intent, backend_obj)
            target_role = ModelRole.PLANNER if target_role_str == "planner" else ModelRole.EXECUTOR
            selected_model = model_override or backend_obj.models.get("coder" if target_role_str != "assistant" else "quick")

        # Ensure selected_model is a string and not None
        if isinstance(selected_model, list):
            selected_model = selected_model[0] if selected_model else "auto"
        if not selected_model:
            selected_model = "auto"

        # Dynamic persona loading for chat mode
        # Uses PersonaManager to detect context and blend appropriate personas
        persona_manager = get_persona_manager()

        # Special handling for simple greetings - use minimal prompt
        # Smaller models don't follow complex persona instructions well
        message_lower = message.lower().strip()
        is_greeting = message_lower in ("hello", "hi", "hey", "hello!", "hi!", "hey!",
                                        "good morning", "good afternoon", "good evening",
                                        "yo", "sup", "what's up", "howdy")

        if is_greeting:
            # Ultra-simple prompt that even small models follow
            system_prompt = "You are Delia, a friendly assistant. Respond to greetings briefly and warmly."
        else:
            system_prompt = persona_manager.get_system_prompt(message)
            active_personas = persona_manager.get_active_persona_names()
            if len(active_personas) > 1:
                yield StreamEvent(event_type="status", message=f"Personas: {', '.join(active_personas)}")

        # Skip playbook and task focus for simple quick tasks (greetings, simple Q&A)
        # These confuse the model with unnecessary "Am I on task?" checks
        is_simple_task = intent.task_type in ("quick", "status", "chat")

        if not is_simple_task:
            pb_context = playbook_manager.format_for_prompt(intent.task_type)
            if pb_context:
                system_prompt += f"\n\n{pb_context}"

            # Inject task focus for "Am I on task?" grounding (complex tasks only)
            if session_id:
                session_mgr = get_session_manager()
                session = session_mgr.get_session(session_id)
                if session:
                    # Set the task if this is a new request (first message or new primary task)
                    if not session.original_task or len(session.messages) == 0:
                        session.set_task(message)
                    task_focus = session.get_task_focus_prompt()
                    if task_focus:
                        system_prompt += f"\n\n{task_focus}"

        full_res = ""
        # Limit tokens for greetings to keep responses short
        max_tokens = 50 if is_greeting else None
        async for chunk in call_llm_stream(model=selected_model, prompt=message, system=system_prompt, backend_obj=backend_obj, messages=messages, max_tokens=max_tokens):
            full_res += chunk.text
            yield StreamEvent(event_type="token", message=chunk.text)

        # Strip thinking tags from the response for cleaner output
        clean_response = strip_thinking_tags(full_res)
        yield StreamEvent(event_type="response", message=clean_response, details={"model": selected_model})
        yield StreamEvent(event_type="done", message="Completed")

    async def _execute_agentic(
        self,
        intent: DetectedIntent,
        message: str,
        session_id: str | None,
        backend_type: str | None,
        model_override: str | None,
        messages: list[dict[str, Any]] | None = None,
        on_tool_call: Callable[[Any], None] | None = None,
    ) -> OrchestrationResult:
        from ..routing import get_router
        from ..tools.agent import AgentConfig, run_agent_loop
        from ..tools.builtins import get_default_tools
        from ..tools.registry import TrustLevel

        _, backend_obj = await get_router().select_optimal_backend(message, None, intent.task_type, backend_type)
        if not backend_obj: return OrchestrationResult(response="No backend", success=False)

        # Use agentic tier for agent loops, with fallback based on task type
        agent_tier = "agentic"
        if intent.task_type == "swe":
            agent_tier = "swe"
        elif intent.task_type in ("agentic", "agent", "tool"):
            agent_tier = "agentic"
        selected_model = model_override or await select_model(task_type=agent_tier, content_size=len(message), content=message)
        
        # 1. Strategic Planning Phase (Claude Code / Gemini Style)
        # Decompose the complex goal into milestones before starting the loop
        plan = await self.planner.plan(message, intent, backend_obj)
        plan_context = ""
        if plan:
            plan_context = f"\n\n### EXECUTION PLAN\n{json.dumps(plan.dict(), indent=2)}"
            log.info("agent_plan_created", steps=len(plan.steps))

        system_prompt = self.prompt_generator.generate(intent, message, selected_model, backend_obj.name)
        system_prompt += plan_context
        
        pb_context = playbook_manager.format_for_prompt(intent.task_type)
        if pb_context: system_prompt += f"\n\n{pb_context}"

        # 2. Add Progress Tracking to Registry
        registry = get_default_tools(allow_write=False, allow_exec=False)
        
        async def security_gate(tool_call: ParsedToolCall) -> bool:
            tool_def = registry.get(tool_call.name)
            if not tool_def or tool_def.dangerous is False: return True
            from ..tools.interaction import ask_user
            is_untrusted = any("web_" in str(m) for m in (messages or []))
            prompt = f"Security Gate{' [UNTRUSTED]' if is_untrusted else ''}: Run {tool_call.name}? (y/n)"
            return (await ask_user(prompt)).lower() in ('y', 'yes')

        async def agent_llm_call(msgs: list[dict], sys: str | None = None) -> dict:
            res = await call_llm(model=selected_model, prompt=msgs[-1]["content"], system=sys if sys is not None else system_prompt, messages=msgs, backend_obj=backend_obj)
            return res

        try:
            res = await run_agent_loop(
                call_llm=agent_llm_call,
                prompt=message,
                system_prompt=system_prompt,
                registry=registry,
                model=selected_model,
                config=AgentConfig(
                    max_iterations=10,
                    native_tool_calling=backend_obj.supports_native_tool_calling if backend_obj else False,
                ),
                messages=messages,
                on_tool_call=on_tool_call or security_gate
            )
            return OrchestrationResult(
                response=res.response,
                success=res.success,
                model_used=selected_model,
                backend_used=backend_obj.id if backend_obj else None,  # FIX: Track backend for affinity
                mode=OrchestrationMode.AGENTIC,
                tool_calls=res.tool_calls,
                tokens=res.tokens
            )
        except Exception as e:
            return OrchestrationResult(response=f"Agent failed: {e}", success=False)

    async def _execute_agentic_stream(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None = None,
        model_override: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        on_tool_call: Callable[[Any], None] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        from ..routing import get_router
        from ..tools.agent import AgentConfig, run_agent_loop
        from ..tools.builtins import get_default_tools

        _, backend_obj = await get_router().select_optimal_backend(message, None, intent.task_type, backend_type)
        if not backend_obj:
            yield StreamEvent(event_type="error", message="No backend")
            return

        # Use agentic tier for agent loops, with fallback based on task type
        agent_tier = "agentic"
        if intent.task_type == "swe":
            agent_tier = "swe"
        elif intent.task_type in ("agentic", "agent", "tool"):
            agent_tier = "agentic"
        selected_model = model_override or await select_model(task_type=agent_tier, content_size=len(message), content=message)
        
        # 1. Strategic Planning Phase (Claude Code / Gemini Style)
        # Decompose the complex goal into milestones before starting the loop
        plan = await self.planner.plan(message, intent, backend_obj)
        plan_context = ""
        if plan:
            plan_context = f"\n\n### EXECUTION PLAN\n{json.dumps(plan.dict(), indent=2)}"
            log.info("agent_plan_created", steps=len(plan.steps))

        system_prompt = self.prompt_generator.generate(intent, message, selected_model, backend_obj.name)
        system_prompt += plan_context
        
        pb_context = playbook_manager.format_for_prompt(intent.task_type)
        if pb_context: system_prompt += f"\n\n{pb_context}"

        # 2. Add Progress Tracking to Registry
        registry = get_default_tools(allow_write=False, allow_exec=False)
        event_queue = asyncio.Queue()

        async def agent_llm_call(msgs: list[dict], sys: str | None = None) -> dict:
            await event_queue.put(StreamEvent(event_type="thinking", message="Agent thinking..."))
            return await call_llm(model=selected_model, prompt=msgs[-1]["content"], system=sys if sys is not None else system_prompt, messages=msgs, backend_obj=backend_obj)

        def internal_on_tool_call(tc: Any):
            event_queue.put_nowait(StreamEvent(event_type="tool_call", message=f"Tool: {tc.name}", details={"name": tc.name, "args": tc.arguments}))
            if on_tool_call:
                on_tool_call(tc)

        agent_task = asyncio.create_task(run_agent_loop(
            agent_llm_call, message, system_prompt, registry, selected_model,
            AgentConfig(
                max_iterations=10,
                native_tool_calling=backend_obj.supports_native_tool_calling if backend_obj else False,
            ),
            messages=messages, on_tool_call=internal_on_tool_call
        ))

        while not agent_task.done() or not event_queue.empty():
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                yield event
            except (asyncio.TimeoutError, TimeoutError):
                continue
        
        res = await agent_task
        yield StreamEvent(event_type="response", message=res.response, details={"model": selected_model})

    async def _execute_voting(self, intent: DetectedIntent, message: str, backend_type: str | None, model_override: str | None, messages: list[dict[str, Any]] | None = None) -> OrchestrationResult:
        from ..voting import AdaptiveVotingConsensus, AdaptiveVotingConfig
        from ..quality import ResponseQualityValidator
        from ..routing import get_router

        selected_model = model_override or await select_model(task_type=intent.task_type, content_size=len(message), content=message)
        _, backend_obj = await get_router().select_optimal_backend(message, None, intent.task_type, backend_type)

        # Get tuning for voting with model-specific quirks
        tuning = self.tuning_advisor.advise(
            task_type=intent.task_type,
            content_length=len(message),
            orchestration_mode="voting",
            tier="coder",
            model_name=selected_model,
        )
        # Voting uses higher temp for diversity in responses
        voting_temp = min(tuning.temperature + 0.2, 1.0)

        # Configure adaptive voting
        config = AdaptiveVotingConfig(
            max_k=intent.k_votes or 3,
            confidence_skip_threshold=0.85,
        )
        consensus = AdaptiveVotingConsensus(config=config, quality_validator=ResponseQualityValidator())

        total_tokens = 0
        votes_cast = 0

        # Adaptive voting loop - continues until should_continue() returns False
        while consensus.should_continue():
            res = await call_llm(
                model=selected_model,
                prompt=message,
                messages=messages,
                task_type=intent.task_type,
                temperature=voting_temp,
                max_tokens=tuning.max_tokens,
                backend_obj=backend_obj,
            )

            if res.get("success"):
                total_tokens += res.get("tokens", 0)
                votes_cast += 1

                # Extract confidence from response (default 0.7 if not found)
                confidence = self._extract_confidence(res["response"])

                # Add vote with confidence
                vote = consensus.add_vote(strip_thinking_tags(res["response"]), confidence)

                if vote.consensus_reached:
                    return OrchestrationResult(
                        response=vote.winning_response,
                        success=True,
                        model_used=selected_model,
                        backend_used=backend_obj.id if backend_obj else None,
                        mode=OrchestrationMode.VOTING,
                        votes_cast=votes_cast,
                        consensus_reached=True,
                        tokens=total_tokens,
                    )
            else:
                # Failed LLM call - break to avoid infinite loop
                break

        # Consensus not reached within max_k votes, get best available
        winner = consensus.get_weighted_winner()
        return OrchestrationResult(
            response=winner.winning_response if winner.winning_response else "Consensus failed",
            success=winner.winning_response is not None,
            model_used=selected_model,
            backend_used=backend_obj.id if backend_obj else None,
            mode=OrchestrationMode.VOTING,
            votes_cast=votes_cast,
            consensus_reached=False,
            tokens=total_tokens,
        )

    # ADR-008: _execute_comparison removed - use _execute_voting instead

    async def _execute_simple(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
        original_message: str | None = None,
        session_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> OrchestrationResult:
        from ..routing import get_router
        _, backend_obj = await get_router().select_optimal_backend(message, None, intent.task_type, backend_type)
        if not backend_obj:
            raise RuntimeError("No backend configured. Please configure at least one backend in settings.json.")

        target_role_str = await self.dispatcher.dispatch(original_message or message, intent, backend_obj)
        target_role = ModelRole.PLANNER if target_role_str == "planner" else ModelRole.EXECUTOR
        selected_model = model_override or backend_obj.models.get("coder" if target_role_str != "assistant" else "quick")

        # Ensure selected_model is a string and not None
        if isinstance(selected_model, list):
            selected_model = selected_model[0] if selected_model else "auto"
        if not selected_model:
            selected_model = "auto"

        # Determine model tier for tuning
        tier = "coder" if target_role_str != "assistant" else "quick"

        # Assess stakes for tuning adjustments (sync version for speed)
        stakes = None
        try:
            from .stakes import analyze_stakes_sync
            stakes = analyze_stakes_sync(original_message or message)
        except Exception as e:
            log.debug("stakes_analysis_skipped", error=str(e))

        # Get optimal tuning parameters from advisor (with model-specific quirks)
        tuning = self.tuning_advisor.advise(
            task_type=intent.task_type,
            content_length=len(message),
            stakes=stakes,
            orchestration_mode=intent.orchestration_mode.value,
            tier=tier,
            model_name=selected_model,
        )

        log.debug(
            "tuning_applied",
            task=intent.task_type,
            tier=tier,
            stakes=stakes.score if stakes else None,
            temperature=tuning.temperature,
            max_tokens=tuning.max_tokens,
            reasoning_count=len(tuning.reasoning),
        )

        # Dynamic persona loading - blends base persona with context-specific ones
        persona_manager = get_persona_manager()
        system = persona_manager.get_system_prompt(message)
        pb = playbook_manager.format_for_prompt(intent.task_type)
        if pb: system += f"\n\n{pb}"

        # Apply tuning parameters to LLM call
        res = await call_llm(
            model=selected_model,
            prompt=message,
            system=system,
            messages=messages,
            backend_obj=backend_obj,
            temperature=tuning.temperature,
            max_tokens=tuning.max_tokens,
        )
        from ..quality import validate_response
        qual = validate_response(res.get("response", ""), intent.task_type)
        return OrchestrationResult(
            response=strip_thinking_tags(res.get("response", "")),
            success=res.get("success", False),
            model_used=selected_model,
            backend_used=backend_obj.id,  # FIX: Pass actual backend ID for affinity tracking
            quality_score=qual.overall,
            tokens=res.get("tokens", 0),  # FIX: Track tokens for economics
        )

    async def _execute_status_query(self, intent: DetectedIntent, message: str) -> OrchestrationResult:
        from ..melons import get_melon_tracker
        tracker = get_melon_tracker()
        leaderboard = tracker.get_leaderboard()
        board = "\n".join([f"{i+1}. {s.model_id}: {s.melons} 🍈" for i, s in enumerate(leaderboard[:10])])
        return OrchestrationResult(response=f"🍈 **Leaderboard**\n\n{board or 'Empty garden.'}", success=True, model_used="system")

    async def _execute_deep_thinking(self, intent: DetectedIntent, message: str, backend_type: str | None, model_override: str | None, messages: list[dict[str, Any]] | None = None) -> OrchestrationResult:
        selected_model = model_override or await select_model(task_type="moe", content_size=len(message), content=message)

        # Get tuning for deep thinking with model-specific quirks
        tuning = self.tuning_advisor.advise(
            task_type=intent.task_type,
            content_length=len(message),
            orchestration_mode="deep_thinking",
            tier="thinking",
            model_name=selected_model,
        )

        from ..routing import get_router
        _, backend_obj = await get_router().select_optimal_backend(message, None, "moe", backend_type)

        res = await call_llm(
            model=selected_model,
            prompt=message,
            messages=messages,
            enable_thinking=tuning.enable_thinking or True,
            temperature=tuning.temperature,
            max_tokens=tuning.max_tokens,
            backend_obj=backend_obj,
        )
        return OrchestrationResult(
            response=strip_thinking_tags(res.get("response", "")),
            success=res.get("success", False),
            model_used=selected_model,
            backend_used=backend_obj.id if backend_obj else None,  # FIX: Track backend
            mode=OrchestrationMode.DEEP_THINKING,
            tokens=res.get("tokens", 0),  # FIX: Track tokens for economics
        )

    async def _execute_tree_of_thoughts(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
        messages: list[dict[str, Any]] | None = None,
        config: ToTConfig | None = None,
    ) -> OrchestrationResult:
        """
        Meta-orchestration: Execute multiple orchestration modes in parallel,
        critic picks best, ACE learns from outcome.

        ADR-008 Optimizations:
        - Early stopping: If first branch has high confidence, skip remaining
        - Branch limiting: Max 3 branches (research shows diminishing returns)
        - Pruning: Only send top 2 branches to critic (reduces LLM calls)

        This is the core of Delia's self-improving orchestration system.
        Uses:
        - UCB1 for exploration-exploitation in mode selection
        - Parallel branch execution with error isolation (or sequential with early stop)
        - Weighted critic scoring for winner selection
        - Bayesian learning for pattern confidence
        """
        from .meta_learning import get_orchestration_learner
        from .critic import BranchEvaluation

        # Use provided config or defaults
        cfg = config or ToTConfig()
        start_time = time.time()
        learner = get_orchestration_learner()

        # 1. Select which modes to try (UCB1 exploration) - LIMITED by config
        all_modes = learner.select_exploration_modes(message)
        modes_to_try = all_modes[: cfg.max_branches]  # ADR-008: Limit branches
        log.info("tot_starting", modes=[m.value for m in modes_to_try], limited_from=len(all_modes))

        # 2. Map modes to executor functions (ADR-008: COMPARISON removed)
        mode_executors: dict[OrchestrationMode, Callable] = {
            OrchestrationMode.VOTING: self._execute_voting,
            OrchestrationMode.AGENTIC: self._execute_agentic,
            OrchestrationMode.DEEP_THINKING: self._execute_deep_thinking,
            # ADR-008: COMPARISON removed - redundant with voting
        }

        # 3. Execute branches with early stopping (ADR-008)
        async def safe_execute(mode: OrchestrationMode) -> tuple[OrchestrationMode, OrchestrationResult]:
            """Execute a branch with timeout and error handling."""
            try:
                executor = mode_executors.get(mode)
                if not executor:
                    raise ValueError(f"No executor for mode: {mode}")

                # Create a modified intent for the specific mode
                branch_intent = DetectedIntent(
                    task_type=intent.task_type,
                    orchestration_mode=mode,
                    model_role=intent.model_role,
                    confidence=intent.confidence,
                    k_votes=intent.k_votes if mode == OrchestrationMode.VOTING else 3,
                    contains_code=intent.contains_code,
                )

                # ADR-008: Reduced timeout
                result = await asyncio.wait_for(
                    executor(branch_intent, message, backend_type, model_override, messages),
                    timeout=cfg.timeout_per_branch
                )
                log.info("tot_branch_complete", mode=mode.value, success=result.success)
                return (mode, result)

            except asyncio.TimeoutError:
                log.warning("tot_branch_timeout", mode=mode.value, timeout=cfg.timeout_per_branch)
                return (mode, OrchestrationResult(
                    response=f"Branch timed out after {cfg.timeout_per_branch}s",
                    success=False,
                    error="Timeout",
                    mode=mode,
                ))
            except Exception as e:
                log.error("tot_branch_error", mode=mode.value, error=str(e))
                return (mode, OrchestrationResult(
                    response=f"Branch failed: {e}",
                    success=False,
                    error=str(e),
                    mode=mode,
                ))

        # 4. ADR-008: Execute with early stopping if enabled
        branch_results: list[tuple[OrchestrationMode, OrchestrationResult]] = []

        if cfg.early_stop_on_high_confidence:
            # Sequential execution with early stopping
            for mode in modes_to_try:
                result = await safe_execute(mode)
                branch_results.append(result)

                # Early stop if high confidence result
                _, res = result
                if res.success and (res.quality_score or 0) >= cfg.confidence_threshold:
                    log.info(
                        "tot_early_stop",
                        mode=mode.value,
                        quality_score=res.quality_score,
                        branches_skipped=len(modes_to_try) - len(branch_results),
                    )
                    break
        else:
            # Original parallel execution
            branch_tasks = [safe_execute(mode) for mode in modes_to_try]
            branch_results = await asyncio.gather(*branch_tasks)

        # 5. Filter to successful branches for evaluation
        successful_branches = [(mode, res) for mode, res in branch_results if res.success]

        if not successful_branches:
            # All branches failed - return best effort from failures
            log.warning("tot_all_branches_failed", branches=len(branch_results))
            best_failure = max(branch_results, key=lambda x: len(x[1].response))
            return OrchestrationResult(
                response=f"All ToT branches failed. Best effort:\n\n{best_failure[1].response}",
                success=False,
                error="All branches failed",
                mode=OrchestrationMode.TREE_OF_THOUGHTS,
                elapsed_ms=int((time.time() - start_time) * 1000),
                debug_info={
                    "tot_branches": [m.value for m in modes_to_try],
                    "tot_errors": [res.error for _, res in branch_results if res.error],
                },
            )

        # 5.5 ADR-008: Prune to top N branches before critic (reduces LLM calls)
        if cfg.prune_before_critic and len(successful_branches) > cfg.max_branches_for_critic:
            # Sort by quality score (or response length as fallback)
            successful_branches = sorted(
                successful_branches,
                key=lambda x: x[1].quality_score or len(x[1].response) / 1000,
                reverse=True,
            )[: cfg.max_branches_for_critic]
            log.info(
                "tot_pruned_for_critic",
                kept=len(successful_branches),
                max_allowed=cfg.max_branches_for_critic,
            )

        # 6. Critic evaluates successful branches (now pruned)
        log.info("tot_evaluating_branches", count=len(successful_branches))
        evaluation: BranchEvaluation = await self.critic.evaluate_branches(
            original_prompt=message,
            branches=successful_branches,
        )

        # 7. Get winning result
        winner_mode, winner_result = successful_branches[evaluation.winner_index]

        # 8. Feed to meta-learner (ACE learns about orchestration)
        await self._ace_meta_learn(
            message=message,
            branch_results=branch_results,
            winner_mode=winner_mode,
            evaluation=evaluation,
            intent=intent,
        )

        # 9. Build final result with ToT metadata
        elapsed_ms = int((time.time() - start_time) * 1000)
        total_tokens = sum(res.tokens for _, res in branch_results if res.tokens)

        # Compute quality score from evaluation
        quality_score = winner_result.quality_score
        if evaluation.scores and evaluation.winner_index < len(evaluation.scores):
            quality_score = evaluation.scores[evaluation.winner_index].weighted_score

        log.info(
            "tot_complete",
            winner=winner_mode.value,
            branches_tried=len(modes_to_try),
            branches_succeeded=len(successful_branches),
            elapsed_ms=elapsed_ms,
        )

        return OrchestrationResult(
            response=winner_result.response,
            success=True,
            model_used=winner_result.model_used,
            backend_used=winner_result.backend_used,  # FIX: Propagate from winning branch
            mode=OrchestrationMode.TREE_OF_THOUGHTS,
            quality_score=quality_score,
            elapsed_ms=elapsed_ms,
            tokens=total_tokens,
            debug_info={
                "tot_branches": [m.value for m, _ in branch_results],
                "tot_successful": [m.value for m, _ in successful_branches],
                "tot_winner": winner_mode.value,
                "tot_reasoning": evaluation.reasoning,
                "tot_insights": evaluation.insights,
                "tot_scores": [
                    {"mode": s.mode.value, "score": round(s.weighted_score, 3)}
                    for s in evaluation.scores
                ] if evaluation.scores else [],
            },
        )

    async def _ace_meta_learn(
        self,
        message: str,
        branch_results: list[tuple[OrchestrationMode, OrchestrationResult]],
        winner_mode: OrchestrationMode,
        evaluation: Any,  # BranchEvaluation
        intent: DetectedIntent,
    ) -> None:
        """
        ACE Framework meta-learning: Learn ABOUT orchestration from ToT outcome.

        This is distinct from regular ACE reflection which learns about task strategies.
        Here we learn which orchestration MODE works for which task PATTERN.
        """
        from .meta_learning import get_orchestration_learner

        try:
            learner = get_orchestration_learner()

            # Update pattern with Bayesian confidence and UCB stats
            learner.learn_from_tot(
                message=message,
                branch_results=branch_results,
                winner_mode=winner_mode,
                critic_reasoning=evaluation.reasoning,
            )

            # Also generate a playbook-style lesson if we have insights
            if evaluation.insights:
                # Add to orchestration-specific playbook
                lesson = f"For {intent.task_type} tasks: prefer {winner_mode.value} - {evaluation.insights}"
                playbook_manager.add_bullet(
                    task_type="orchestration",
                    content=lesson,
                    section="mode_selection",
                )

                log.info(
                    "ace_meta_learning_complete",
                    task_type=intent.task_type,
                    winner_mode=winner_mode.value,
                    insight_len=len(evaluation.insights),
                )

        except Exception as e:
            # Meta-learning should never break the main flow
            log.error("ace_meta_learning_failed", error=str(e))

    def _award_melons(self, result: OrchestrationResult, intent: DetectedIntent) -> None:
        from ..melons import award_melons_for_quality
        if result.model_used and result.quality_score is not None:
            # Pass tokens so economics can calculate real savings
            # Rough split: 80% input tokens, 20% output tokens for estimation
            total_tokens = result.tokens or 0
            input_tokens = int(total_tokens * 0.8)
            output_tokens = total_tokens - input_tokens
            award_melons_for_quality(
                model_id=result.model_used,
                task_type=intent.task_type,
                quality_score=result.quality_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=result.elapsed_ms or 0,
            )

    async def _record_playbook_feedback(self, task_type: str, quality_score: float | None) -> None:
        """
        Record feedback for playbook bullets used in this task.
        
        Called after task completion to close the ACE learning loop.
        Bullets with quality >= 0.7 are marked helpful, < 0.4 harmful.
        
        This enables the playbook to learn which strategies actually work.
        """
        if quality_score is None:
            return
            
        # Get bullets that were used for this task type
        bullets = playbook_manager.load_playbook(task_type)
        if not bullets:
            return
            
        # Determine if the task was helpful based on quality threshold
        helpful = quality_score >= 0.7
        
        # Only record feedback for high or low quality (skip neutral 0.4-0.7)
        if 0.4 <= quality_score < 0.7:
            return
            
        # Record feedback for bullets that were loaded (assumed used)
        # We record for the top bullets that would have been injected
        top_bullets = playbook_manager.get_top_bullets(task_type, limit=5)
        for bullet in top_bullets:
            playbook_manager.record_feedback(bullet.id, task_type, helpful)
            
        if top_bullets:
            log.debug(
                "playbook_feedback_recorded",
                task_type=task_type,
                quality_score=round(quality_score, 3),
                helpful=helpful,
                bullets_updated=len(top_bullets),
            )

    def _extract_confidence(self, response: str) -> float:
        """
        Extract confidence score from LLM response.

        Looks for common confidence patterns in the response text:
        - "confidence: 0.85" or "confidence = 0.85"
        - "I am X% confident" or "X% sure"
        - Thinking tags with confidence markers

        Args:
            response: The LLM response text

        Returns:
            Confidence score 0.0-1.0 (default 0.7 if not found)
        """
        # Default confidence if no explicit signal
        default_confidence = 0.7

        # Pattern 1: Explicit confidence score (0.0-1.0)
        confidence_pattern = r"confidence[:\s=]+([0-9]*\.?[0-9]+)"
        match = re.search(confidence_pattern, response.lower())
        if match:
            conf_value = float(match.group(1))
            # If > 1.0, assume it's a percentage
            if conf_value > 1.0:
                conf_value = conf_value / 100.0
            return max(0.0, min(1.0, conf_value))

        # Pattern 2: Percentage confidence ("95% confident", "I'm 80% sure")
        percent_pattern = r"(\d+)%\s*(confident|sure|certain)"
        match = re.search(percent_pattern, response.lower())
        if match:
            conf_value = float(match.group(1)) / 100.0
            return max(0.0, min(1.0, conf_value))

        # Pattern 3: Qualitative confidence signals
        high_confidence_phrases = [
            r"\b(definitely|certainly|absolutely|clearly)\b",
            r"\b(without a doubt|no question|unquestionably)\b",
            r"\b(highly confident|very confident|quite confident)\b",
        ]
        low_confidence_phrases = [
            r"\b(maybe|perhaps|possibly|might|could be)\b",
            r"\b(uncertain|unsure|not sure|not certain)\b",
            r"\b(i think|i believe|it seems|it appears)\b",
        ]

        high_confidence_count = sum(
            len(re.findall(pattern, response.lower())) for pattern in high_confidence_phrases
        )
        low_confidence_count = sum(
            len(re.findall(pattern, response.lower())) for pattern in low_confidence_phrases
        )

        # Adjust default based on qualitative signals
        if high_confidence_count > low_confidence_count:
            return 0.85  # High confidence detected
        elif low_confidence_count > high_confidence_count:
            return 0.55  # Low confidence detected

        return default_confidence

    async def _retrieve_context(self, message: str) -> str:
        available = list_memories()
        found = []
        for mem in available:
            if any(kw in message.lower() for kw in mem.replace("_", " ").split() if len(kw) > 3):
                content = read_memory(mem)
                if content: found.append(f"### Context: {mem}\n{content}")
        return "\n\n".join(found)

    async def _reflect_on_failure(self, task_type: str, message: str, response: str, error: str | None = None, backend_obj: Any | None = None) -> None:
        """
        Use the ACE Reflector → Curator pipeline to learn from failures.

        This replaces the legacy approach of raw LLM calls with the proper
        ACE modules that include:
        - Structured reflection prompts
        - Semantic deduplication before adding bullets
        - Utility/recency scoring for retrieval
        """
        try:
            from ..ace.reflector import get_reflector
            from ..ace.curator import get_curator
            from pathlib import Path

            reflector = get_reflector()
            curator = get_curator(str(Path.cwd()))

            # Get bullet IDs that were used (approximation: top bullets for this task type)
            top_bullets = playbook_manager.get_top_bullets(task_type, limit=5)
            applied_bullet_ids = [b.id for b in top_bullets]

            # Run structured reflection
            reflection = await reflector.reflect(
                task_description=message,
                task_type=task_type,
                task_succeeded=False,
                outcome=response,
                tool_calls=None,
                applied_bullets=applied_bullet_ids,
                error_trace=error,
                user_feedback=None,
            )

            # Curate the playbook with delta updates
            curation = await curator.curate(reflection, auto_prune=False)

            log.info(
                "ace_reflection_complete",
                task_type=task_type,
                insights=len(reflection.insights),
                bullets_added=curation.bullets_added,
                dedup_prevented=curation.dedup_prevented,
            )
        except Exception as e:
            log.warning("ace_reflection_failed", task_type=task_type, error=str(e))

_executor: OrchestrationExecutor | None = None

def get_orchestration_executor() -> OrchestrationExecutor:
    global _executor
    if _executor is None: _executor = OrchestrationExecutor()
    return _executor

async def execute_orchestration(intent: DetectedIntent, message: str, **kwargs) -> OrchestrationResult:
    return await get_orchestration_executor().execute(intent, message, **kwargs)
