# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Orchestration Executor - The Heart of Delia's NLP Orchestration.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

from ..config import config
from ..file_helpers import list_memories, read_memory
from ..llm import call_llm
from ..playbook import playbook_manager
from .tuning import TuningAdvisor, get_tuning_advisor
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
    from .parser import ParsedToolCall
    from .registry import ToolRegistry

log = structlog.get_logger()


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
            elif intent.orchestration_mode == OrchestrationMode.COMPARISON:
                result = await self._execute_comparison(intent, message, backend_type, model_override, messages=messages)
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
        
        if intent.orchestration_mode == OrchestrationMode.AGENTIC:
            async for event in self._execute_agentic_stream(intent, message, backend_type, model_override, messages=messages, on_tool_call=on_tool_call):
                yield event
        else:
            from ..llm import call_llm_stream
            from ..routing import get_router
            _, backend_obj = await get_router().select_optimal_backend(message, None, intent.task_type, backend_type)
            if not backend_obj:
                yield StreamEvent(event_type="error", message="No backend")
                return

            target_role_str = await self.dispatcher.dispatch(original_message or message, intent, backend_obj)
            target_role = ModelRole.PLANNER if target_role_str == "planner" else ModelRole.EXECUTOR
            selected_model = model_override or backend_obj.models.get("coder" if target_role_str != "assistant" else "quick")
            
            # Ensure selected_model is a string and not None
            if isinstance(selected_model, list):
                selected_model = selected_model[0] if selected_model else "auto"
            if not selected_model:
                selected_model = "auto"

            system_prompt = build_system_prompt(target_role)
            pb_context = playbook_manager.format_for_prompt(intent.task_type)
            if pb_context: system_prompt += f"\n\n{pb_context}"

            full_res = ""
            async for chunk in call_llm_stream(model=selected_model, prompt=message, system=system_prompt, backend_obj=backend_obj, messages=messages):
                full_res += chunk.text
                yield StreamEvent(event_type="token", message=chunk.text)
            
            yield StreamEvent(event_type="response", message=full_res)
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

        selected_model = model_override or await select_model(task_type="review", content_size=len(message), content=message)
        system_prompt = self.prompt_generator.generate(intent, message, selected_model, backend_obj.name)
        pb_context = playbook_manager.format_for_prompt(intent.task_type)
        if pb_context: system_prompt += f"\n\n{pb_context}"

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
                config=AgentConfig(max_iterations=10),
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

        selected_model = model_override or await select_model(task_type="review", content_size=len(message), content=message)
        system_prompt = self.prompt_generator.generate(intent, message, selected_model, backend_obj.name)
        pb_context = playbook_manager.format_for_prompt(intent.task_type)
        if pb_context: system_prompt += f"\n\n{pb_context}"

        registry = get_default_tools(allow_write=False, allow_exec=False)
        event_queue = asyncio.Queue()

        async def agent_llm_call(msgs: list[dict], sys: str | None = None) -> dict:
            await event_queue.put(StreamEvent(event_type="thinking", message="Agent thinking..."))
            return await call_llm(model=selected_model, prompt=msgs[-1]["content"], system=sys if sys is not None else system_prompt, messages=msgs, backend_obj=backend_obj)

        def internal_on_tool_call(tc: Any):
            event_queue.put_nowait(StreamEvent(event_type="tool_call", message=f"Tool: {tc.name}", details={"name": tc.name, "args": tc.arguments}))
            if on_tool_call:
                on_tool_call(tc)

        agent_task = asyncio.create_task(run_agent_loop(agent_llm_call, message, system_prompt, registry, selected_model, AgentConfig(max_iterations=10), messages=messages, on_tool_call=internal_on_tool_call))

        while not agent_task.done() or not event_queue.empty():
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                yield event
            except (asyncio.TimeoutError, TimeoutError):
                continue
        
        res = await agent_task
        yield StreamEvent(event_type="response", message=res.response, details={"model": selected_model})

    async def _execute_voting(self, intent: DetectedIntent, message: str, backend_type: str | None, model_override: str | None, messages: list[dict[str, Any]] | None = None) -> OrchestrationResult:
        from ..voting import VotingConsensus
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

        consensus = VotingConsensus(k=intent.k_votes, quality_validator=ResponseQualityValidator())
        total_tokens = 0
        for i in range(intent.k_votes * 3):
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
                vote = consensus.add_vote(strip_thinking_tags(res["response"]))
                if vote.consensus_reached:
                    return OrchestrationResult(
                        response=vote.winning_response, success=True, model_used=selected_model,
                        backend_used=backend_obj.id if backend_obj else None,  # FIX: Track backend
                        mode=OrchestrationMode.VOTING, votes_cast=i+1, consensus_reached=True, tokens=total_tokens
                    )

        best, meta = consensus.get_best_response()
        return OrchestrationResult(
            response=f"Partial consensus: {best}" if best else "Consensus failed",
            success=best is not None, model_used=selected_model,
            backend_used=backend_obj.id if backend_obj else None,  # FIX: Track backend
            mode=OrchestrationMode.VOTING,
            votes_cast=meta.total_votes, consensus_reached=False, tokens=total_tokens
        )

    async def _execute_comparison(self, intent: DetectedIntent, message: str, backend_type: str | None, model_override: str | None, messages: list[dict[str, Any]] | None = None) -> OrchestrationResult:
        from ..backend_manager import backend_manager
        enabled = backend_manager.get_enabled_backends()
        backend = enabled[0] if enabled else None
        models = intent.comparison_models or [backend.models.get("coder"), backend.models.get("moe")] if backend else []
        res_parts = ["# Comparison\n"]
        actual_models = []
        for m in [m for m in models if m]:
            r = await call_llm(model=m, prompt=message, messages=messages, backend_obj=backend)
            if r.get("success"):
                actual_models.append(m)
                res_parts.append(f"## {m}\n{strip_thinking_tags(r.get('response', ''))}\n")
        return OrchestrationResult(
            response="\n".join(res_parts),
            success=len(actual_models) > 0,
            mode=OrchestrationMode.COMPARISON,
            models_compared=actual_models,
            backend_used=backend.id if backend else None,  # FIX: Track backend for affinity
        )

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

        system = build_system_prompt(target_role)
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
    ) -> OrchestrationResult:
        """
        Meta-orchestration: Execute multiple orchestration modes in parallel,
        critic picks best, ACE learns from outcome.

        This is the core of Delia's self-improving orchestration system.
        Uses:
        - UCB1 for exploration-exploitation in mode selection
        - Parallel branch execution with error isolation
        - Weighted critic scoring for winner selection
        - Bayesian learning for pattern confidence
        """
        from .meta_learning import get_orchestration_learner
        from .critic import BranchEvaluation

        start_time = time.time()
        learner = get_orchestration_learner()

        # 1. Select which modes to try (UCB1 exploration)
        modes_to_try = learner.select_exploration_modes(message)
        log.info("tot_starting", modes=[m.value for m in modes_to_try])

        # 2. Map modes to executor functions
        mode_executors: dict[OrchestrationMode, Callable] = {
            OrchestrationMode.VOTING: self._execute_voting,
            OrchestrationMode.AGENTIC: self._execute_agentic,
            OrchestrationMode.DEEP_THINKING: self._execute_deep_thinking,
            OrchestrationMode.COMPARISON: self._execute_comparison,
        }

        # 3. Execute all branches in parallel with error isolation
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

                # Timeout per branch (30 seconds)
                result = await asyncio.wait_for(
                    executor(branch_intent, message, backend_type, model_override, messages),
                    timeout=30.0
                )
                log.info("tot_branch_complete", mode=mode.value, success=result.success)
                return (mode, result)

            except asyncio.TimeoutError:
                log.warning("tot_branch_timeout", mode=mode.value)
                return (mode, OrchestrationResult(
                    response="Branch timed out after 30 seconds",
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

        # 4. Execute all branches concurrently
        branch_tasks = [safe_execute(mode) for mode in modes_to_try]
        branch_results: list[tuple[OrchestrationMode, OrchestrationResult]] = await asyncio.gather(*branch_tasks)

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

        # 6. Critic evaluates all successful branches
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

    async def _retrieve_context(self, message: str) -> str:
        available = list_memories()
        found = []
        for mem in available:
            if any(kw in message.lower() for kw in mem.replace("_", " ").split() if len(kw) > 3):
                content = read_memory(mem)
                if content: found.append(f"### Context: {mem}\n{content}")
        return "\n\n".join(found)

    async def _reflect_on_failure(self, task_type: str, message: str, response: str, error: str | None = None, backend_obj: Any | None = None) -> None:
        try:
            res = await call_llm(model=config.model_moe.default_model, prompt=f"Task: {message}\nResponse: {response}\nError: {error}", system=ACE_REFLECTOR_PROMPT, task_type="reflection", backend_obj=backend_obj, enable_thinking=True)
            if res.get("success"):
                data = json.loads(strip_thinking_tags(res["response"]))
                lesson = data.get("playbook_update")
                if lesson:
                    await self._curate_playbook(task_type, lesson, backend_obj)
                    log.info("ace_reflection_complete", task_type=task_type, lesson_len=len(lesson))
        except Exception as e:
            log.warning("ace_reflection_failed", task_type=task_type, error=str(e))

    async def _curate_playbook(self, task_type: str, lesson: str, backend_obj: Any | None = None) -> None:
        try:
            pb = playbook_manager.load_playbook(task_type)
            pb_text = "\n".join([f"- {b.content}" for b in pb])
            res = await call_llm(model=config.model_moe.default_model, prompt=f"Current: {pb_text}\nNew: {lesson}", system=ACE_CURATOR_PROMPT, task_type="curation", backend_obj=backend_obj)
            if res.get("success"):
                data = json.loads(strip_thinking_tags(res["response"]))
                operations_applied = 0
                for op in data.get("operations", []):
                    if op.get("type") == "ADD":
                        playbook_manager.add_bullet(task_type, op["content"], op.get("section", "strategies"))
                        operations_applied += 1
                if operations_applied > 0:
                    log.info("playbook_curated", task_type=task_type, operations=operations_applied)
        except Exception as e:
            log.warning("playbook_curation_failed", task_type=task_type, error=str(e))

_executor: OrchestrationExecutor | None = None

def get_orchestration_executor() -> OrchestrationExecutor:
    global _executor
    if _executor is None: _executor = OrchestrationExecutor()
    return _executor

async def execute_orchestration(intent: DetectedIntent, message: str, **kwargs) -> OrchestrationResult:
    return await get_orchestration_executor().execute(intent, message, **kwargs)
