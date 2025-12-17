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
from typing import TYPE_CHECKING, AsyncIterator

import structlog

from .intent import DetectedIntent
from .prompts import get_prompt_generator
from .result import OrchestrationMode, OrchestrationResult, StreamEvent

if TYPE_CHECKING:
    from ..backend_manager import BackendConfig

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
    
    async def execute(
        self,
        intent: DetectedIntent,
        message: str,
        session_id: str | None = None,
        backend_type: str | None = None,
        model_override: str | None = None,
    ) -> OrchestrationResult:
        """
        Execute orchestration based on detected intent.
        
        Args:
            intent: The detected intent from IntentDetector
            message: The user's original message
            session_id: Optional session for conversation continuity
            backend_type: Optional backend preference (local/remote)
            model_override: Optional model to force
            
        Returns:
            OrchestrationResult with response and metadata
        """
        start_time = time.time()
        
        log.info(
            "orchestration_execute",
            mode=intent.orchestration_mode.value,
            task_type=intent.task_type,
            role=intent.model_role.value,
            confidence=intent.confidence,
        )
        
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
            else:
                # Default: simple single model call
                result = await self._execute_simple(
                    intent, message, backend_type, model_override
                )
            
            result.elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Award melons based on quality
            if result.success and result.model_used:
                self._award_melons(result, intent)
            
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
        else:
            # Simple execution - just yield result
            result = await self._execute_simple(
                intent, message, backend_type, model_override
            )
            yield StreamEvent(
                event_type="response",
                message=result.response,
                details={"model": result.model_used}
            )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield StreamEvent(
            event_type="done",
            message="Orchestration complete",
            details={"elapsed_ms": elapsed_ms}
        )
    
    async def _execute_simple(
        self,
        intent: DetectedIntent,
        message: str,
        backend_type: str | None,
        model_override: str | None,
    ) -> OrchestrationResult:
        """Execute simple single-model call."""
        from ..backend_manager import backend_manager
        from ..llm import call_llm
        from ..routing import select_model
        from ..mcp_server import _select_optimal_backend_v2
        from ..quality import validate_response
        
        # Select backend
        _, backend_obj = await _select_optimal_backend_v2(
            message, None, intent.task_type, backend_type
        )
        
        if not backend_obj:
            return OrchestrationResult(
                response="Error: No backend available",
                success=False,
                error="no_backend",
            )
        
        # Select model
        selected_model = model_override or await select_model(
            task_type=intent.task_type,
            content_size=len(message),
            content=message,
        )
        
        # Generate role-specific system prompt
        system_prompt = self.prompt_generator.generate(
            intent,
            user_message=message,
            model_name=selected_model,
            backend_name=backend_obj.name,
        )
        
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
        )
        
        if not result.get("success"):
            return OrchestrationResult(
                response=result.get("error", "LLM call failed"),
                success=False,
                error=result.get("error"),
                model_used=selected_model,
            )
        
        response = result.get("response", "")
        
        # Validate quality
        quality_result = validate_response(response, intent.task_type)
        
        return OrchestrationResult(
            response=response,
            success=True,
            model_used=selected_model,
            mode=OrchestrationMode.NONE,
            quality_score=quality_result.overall,
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
        from ..tools.agent import AgentConfig, run_agent_loop
        from ..tools.builtins import get_default_tools
        from ..backend_manager import backend_manager
        from ..llm import call_llm
        from ..routing import select_model
        from ..mcp_server import _select_optimal_backend_v2
        from ..quality import validate_response
        from ..types import Workspace
        
        # Select backend
        _, backend_obj = await _select_optimal_backend_v2(
            message, None, intent.task_type, backend_type
        )
        
        if not backend_obj:
            return OrchestrationResult(
                response="Error: No backend available",
                success=False,
                error="no_backend",
            )
        
        # Select model
        selected_model = model_override or await select_model(
            task_type=intent.task_type,
            content_size=len(message),
            content=message,
        )
        
        # Generate system prompt with tool context
        system_prompt = self.prompt_generator.generate(
            intent,
            user_message=message,
            model_name=selected_model,
            backend_name=backend_obj.name,
        )
        
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
        
        # Create LLM call function for agent loop
        async def agent_llm_call(
            prompt: str,
            system: str | None = None,
            tools: list | None = None,
            tool_choice: str | None = None,
        ) -> dict:
            return await call_llm(
                model=selected_model,
                prompt=prompt,
                system=system or system_prompt,
                task_type=intent.task_type,
                original_task=intent.task_type,
                language="unknown",
                content_preview=prompt[:100],
                backend_obj=backend_obj,
                tools=tools,
                tool_choice=tool_choice,
            )
        
        # Configure agent
        agent_config = AgentConfig(
            max_iterations=10,
            timeout_per_tool=60,
            total_timeout=300,
            parallel_tools=False,
            native_tool_calling=True,
            allow_write=True,
            allow_exec=True,
            require_confirmation=False,  # No confirmation in programmatic mode
        )
        
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
            )
            
            response = agent_result.final_response or agent_result.error or "No response"
            
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
        from ..backend_manager import backend_manager
        from ..llm import call_llm
        from ..routing import select_model
        from ..mcp_server import _select_optimal_backend_v2
        from ..voting import VotingConsensus
        from ..quality import ResponseQualityValidator
        
        k = intent.k_votes
        
        # Select backend
        _, backend_obj = await _select_optimal_backend_v2(
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
            
            response = result.get("response", "")
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
            )
        else:
            return OrchestrationResult(
                response="Unable to reach consensus after multiple attempts.",
                success=False,
                error="voting_failed",
                model_used=selected_model,
            )
    
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
                responses.append((model, result.get("response", ""), elapsed))
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
        from ..backend_manager import backend_manager
        from ..llm import call_llm
        from ..routing import select_model
        from ..mcp_server import _select_optimal_backend_v2
        
        # Force MoE or thinking tier
        _, backend_obj = await _select_optimal_backend_v2(
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
            )
        
        return OrchestrationResult(
            response=result.get("response", ""),
            success=True,
            model_used=selected_model,
            mode=OrchestrationMode.DEEP_THINKING,
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

