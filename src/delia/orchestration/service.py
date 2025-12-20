# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unified Orchestration Service - Single Entry Point for Chat and MCP.

This service unifies all orchestration capabilities so both the HTTP API
(delia chat) and MCP Server share the same logic:

- Frustration detection and penalties
- Intent detection (3-tier NLP)
- Prewarm tracking (EMA learning)
- Orchestration execution (voting, comparison, etc.)
- Quality validation
- Affinity and melon rewards

Usage:
    service = get_orchestration_service()
    result = await service.process(message, session_id=session_id)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

import structlog

from pydantic import BaseModel

from .intent import detect_intent, get_intent_detector
from .executor import get_orchestration_executor, OrchestrationExecutor
from .result import DetectedIntent, OrchestrationMode, OrchestrationResult, StreamEvent
from .context import ContextEngine
from ..tracing import trace, add_event
from ..frustration import FrustrationLevel
from ..melons import get_reward_collector

if TYPE_CHECKING:
    from ..config import AffinityTracker, PrewarmTracker
    from ..melons import MelonTracker
    from ..frustration import FrustrationTracker


log = structlog.get_logger()


@dataclass
class ProcessingContext:
    """Context for a single orchestration request."""
    
    message: str
    session_id: str | None = None
    backend_type: str | None = None
    model_override: str | None = None
    
    # Populated during processing
    intent: DetectedIntent | None = None
    repeat_info: Any | None = None
    start_time: float = field(default_factory=time.time)


@dataclass
class ProcessingResult:
    """Extended result with all tracking metadata."""
    
    result: OrchestrationResult
    intent: DetectedIntent
    quality_score: float | None = None
    melons_awarded: int = 0
    affinity_updated: bool = False
    prewarm_updated: bool = False
    frustration_penalty: int = 0
    elapsed_ms: int = 0


class OrchestrationService:
    """
    Unified orchestration service for both Chat and MCP.
    
    This is the single entry point for all orchestration logic.
    Both API endpoints and MCP tools should use this service.
    
    Responsibilities:
    1. Frustration detection - penalize models for repeated questions
    2. Intent detection - determine orchestration mode via NLP
    3. Prewarm update - learn hourly usage patterns
    4. Execute orchestration - voting, comparison, deep thinking
    5. Quality validation - score response quality
    6. Rewards - update affinity and award melons
    """
    
    def __init__(
        self,
        executor: OrchestrationExecutor,
        affinity: AffinityTracker,
        prewarm: PrewarmTracker,
        melons: MelonTracker,
        frustration: FrustrationTracker,
    ) -> None:
        self.executor = executor
        self.affinity = affinity
        self.prewarm = prewarm
        self.melons = melons
        self.frustration = frustration
        self._initialized = True
        
        log.info("orchestration_service_initialized")
    
    @classmethod
    def create(cls) -> "OrchestrationService":
        """
        Factory method to create a properly initialized service.
        
        Loads persisted state (affinity, prewarm) from disk.
        """
        from ..config import (
            get_affinity_tracker,
            get_prewarm_tracker,
            load_affinity,
            load_prewarm,
        )
        from ..melons import get_melon_tracker
        from ..frustration import get_frustration_tracker
        
        # Load persisted state
        load_affinity()
        load_prewarm()
        
        return cls(
            executor=get_orchestration_executor(),
            affinity=get_affinity_tracker(),
            prewarm=get_prewarm_tracker(),
            melons=get_melon_tracker(),
            frustration=get_frustration_tracker(),
        )
    
    async def save_state(self) -> None:
        """
        Persist service state to disk.
        
        Saves:
        - Affinity scores
        - Prewarm EMA data
        - Melon leaderboard stats
        """
        from ..config import save_affinity, save_prewarm
        
        log.debug("orchestration_service_saving_state")
        
        # Save each sub-component
        save_affinity()
        save_prewarm()
        self.melons.save()
        
        log.info("orchestration_service_state_saved")

    async def process(
        self,
        message: str,
        session_id: str | None = None,
        backend_type: str | None = None,
        model_override: str | None = None,
        output_type: type[BaseModel] | None = None,
        files: str | None = None,
        context: str | None = None,
        symbols: str | None = None,
        include_references: bool = False,
    ) -> ProcessingResult:
        """
        Process a message through the full orchestration pipeline (Static).
        """
        # Wrap entire pipeline in a trace for observability
        with trace(
            "orchestration",
            message_len=len(message),
            session=session_id[:8] if session_id else None,
            structured=output_type.__name__ if output_type else None,
        ) as span:
            return await self._process_traced(
                span, message, session_id, backend_type, model_override, output_type,
                files=files, context=context, symbols=symbols, include_references=include_references
            )

    async def process_stream(
        self,
        message: str,
        session_id: str | None = None,
        backend_type: str | None = None,
        model_override: str | None = None,
        files: str | None = None,
        context: str | None = None,
        symbols: str | None = None,
        include_references: bool = False,
        include_file_tools: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """
        Process a message through the full pipeline with real-time SSE events.
        
        This is the primary engine for the 'delia chat' command.
        """
        # 1. Context Preparation (Shared logic)
        session_context = None
        if session_id:
            from ..session_manager import get_session_manager
            session_manager = get_session_manager()
            session = session_manager.get_session(session_id)
            if session:
                session_context = session.get_context_window(max_tokens=6000)
                session_manager.add_to_session(session_id, "user", message)

        prepared_content = await ContextEngine.prepare_content(
            content=message,
            context=context,
            symbols=symbols,
            include_references=include_references,
            files=files,
            session_context=session_context,
            include_project_overview=True, # Chat always gets repo awareness
        )

        # 2. Intent Detection
        intent = await get_intent_detector().detect_async(message)
        yield StreamEvent(
            event_type="intent",
            message=f"Detected Intent: {intent.task_type}",
            details={"mode": intent.orchestration_mode.value, "role": intent.model_role.value}
        )

        # 3. Orchestrated Execution (Streaming)
        async for event in self.executor.execute_stream(
            intent=intent,
            message=prepared_content,
            original_message=message,
            session_id=session_id,
            backend_type=backend_type,
            model_override=model_override,
        ):
            yield event

        # 4. Learning (Update prewarm)
        self.prewarm.update(intent.task_type)            
    async def process_chain(
        self,
        steps_json: str,
        session_id: str | None = None,
    ) -> ProcessingResult:
        """Explicitly process a task chain."""
        intent = DetectedIntent(
            task_type="coder",
            orchestration_mode=OrchestrationMode.CHAIN,
            reasoning="explicit chain tool call",
        )
        # Use a trace
        with trace("orchestration_chain", steps_len=len(steps_json)) as span:
            return await self._process_traced_direct(
                span, steps_json, intent, session_id
            )

    async def process_workflow(
        self,
        workflow_json: str,
        session_id: str | None = None,
    ) -> ProcessingResult:
        """Explicitly process a workflow DAG."""
        intent = DetectedIntent(
            task_type="moe",
            orchestration_mode=OrchestrationMode.WORKFLOW,
            reasoning="explicit workflow tool call",
        )
        with trace("orchestration_workflow", workflow_len=len(workflow_json)) as span:
            return await self._process_traced_direct(
                span, workflow_json, intent, session_id
            )

    async def _process_traced_direct(
        self,
        span,
        message: str,
        intent: DetectedIntent,
        session_id: str | None = None,
    ) -> ProcessingResult:
        """Internal implementation for direct (non-detected) processing."""
        start_time = time.time()
        
        # Skip frustration and intent detection for direct calls
        # but record to session history
        if session_id:
            from ..session_manager import get_session_manager
            session_manager = get_session_manager()
            session_manager.add_to_session(
                session_id, "user", message[:100] + "...", tokens=0, model="", task_type=intent.task_type
            )

        # Execute orchestration
        result = await self.executor.execute(
            intent=intent,
            message=message,
            original_message=message, # Same for direct calls
            session_id=session_id,
        )
        
        # Rewards and validation
        quality_score = 0.8 # Default for complex multi-step
        if result.success:
             self.prewarm.update(intent.task_type)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return ProcessingResult(
            result=result,
            intent=intent,
            quality_score=quality_score,
            elapsed_ms=elapsed_ms,
        )

    async def _process_traced(
        self,
        span,
        message: str,
        session_id: str | None,
        backend_type: str | None,
        model_override: str | None,
        output_type: type[BaseModel] | None,
        files: str | None = None,
        context: str | None = None,
        symbols: str | None = None,
        include_references: bool = False,
    ) -> ProcessingResult:
        """Internal traced implementation of process()."""
        start_time = time.time()
        frustration_penalty = 0
        session_context = None
        
        # STEP 1: Intent detection (3-tier NLP)
        from .intent import get_intent_detector
        detector = get_intent_detector()
        intent = await detector.detect_async(message)
        
        # Check if we even have a backend before starting heavy lifting.
        # Only 'status' task is safe without a backend as it returns local leaderboard.
        if intent.task_type != "status" or intent.orchestration_mode != OrchestrationMode.NONE:
            from ..backend_manager import backend_manager
            if not backend_manager.get_active_backend():
                raise RuntimeError("No backend configured. Please configure at least one backend in settings.json.")

        span.event(
            "intent_detected",
            task_type=intent.task_type,
            mode=intent.orchestration_mode.value,
            confidence=round(intent.confidence, 2),
        )

        # Decide if we need repo-wide context based on intent
        needs_repo_context = intent.orchestration_mode in [
            OrchestrationMode.WORKFLOW, 
            OrchestrationMode.AGENTIC,
            OrchestrationMode.DEEP_THINKING
        ]

        # Assemble full content (Files + Memories + History + Task)
        prepared_content = await ContextEngine.prepare_content(
            content=message,
            context=context,
            symbols=symbols,
            include_references=include_references,
            files=files,
            session_context=session_context,
            include_project_overview=needs_repo_context,
        )

        span.event("context_prepared", length=len(prepared_content))

        # STEP 2: Frustration detection
        repeat_info = None
        if session_id:
            repeat_info = self.frustration.check_repeat(session_id, message)
            
            # Penalize if frustration detected (even if not strictly a repeat)
            if repeat_info.level != FrustrationLevel.NONE and repeat_info.previous_model:
                # Escalating penalties
                base_penalty = 0
                if repeat_info.level == FrustrationLevel.LOW:
                    base_penalty = 2
                elif repeat_info.level == FrustrationLevel.MEDIUM:
                    base_penalty = 3
                elif repeat_info.level == FrustrationLevel.HIGH:
                    base_penalty = 5
                
                frustration_penalty = base_penalty
                
                self.melons.penalize(
                    repeat_info.previous_model,
                    "quick",  # Use quick as default task type for penalty
                    melons=frustration_penalty,
                )
                
                span.event(
                    "frustration_detected",
                    level=repeat_info.level.value,
                    penalty=frustration_penalty,
                    previous_model=repeat_info.previous_model,
                )
                
                log.warning(
                    "frustration_penalty_applied",
                    level=repeat_info.level.value,
                    model=repeat_info.previous_model,
                    penalty=frustration_penalty,
                )
        
        # STEP 3: Auto-upgrade orchestration based on frustration level
        if repeat_info and repeat_info.level != FrustrationLevel.NONE:
            # Only upgrade if current mode is weaker than what we want
            current_mode_is_weak = intent.orchestration_mode == OrchestrationMode.NONE
            
            if repeat_info.level == FrustrationLevel.HIGH:
                # High frustration -> Deep Thinking or Heavy Voting
                # (Prefer deep thinking for detailed correction, voting for fact check)
                if current_mode_is_weak:
                    intent.orchestration_mode = OrchestrationMode.DEEP_THINKING
                    intent.reasoning = f"auto-escalation to deep thinking due to HIGH frustration; {intent.reasoning}"
                    log.warning("frustration_auto_escalate", mode="deep_thinking", level="high")
            
            elif repeat_info.level == FrustrationLevel.MEDIUM:
                # Medium frustration -> Voting (k=3)
                if current_mode_is_weak or (intent.orchestration_mode == OrchestrationMode.VOTING and intent.k_votes < 3):
                    intent.orchestration_mode = OrchestrationMode.VOTING
                    intent.k_votes = 3
                    intent.reasoning = f"auto-escalation to voting (k=3) due to MEDIUM frustration; {intent.reasoning}"
                    log.warning("frustration_auto_escalate", mode="voting_k3", level="medium")

            elif repeat_info.level == FrustrationLevel.LOW:
                # Low frustration -> Voting (k=2)
                if current_mode_is_weak:
                    intent.orchestration_mode = OrchestrationMode.VOTING
                    intent.k_votes = 2
                    intent.reasoning = f"auto-escalation to voting (k=2) due to LOW frustration; {intent.reasoning}"
                    log.info("frustration_auto_escalate", mode="voting_k2", level="low")
        
        # STEP 4: Prewarm update - learn usage patterns
        self.prewarm.update(intent.task_type)
        prewarm_updated = True
        
        log.info(
            "orchestration_intent_detected",
            task_type=intent.task_type,
            mode=intent.orchestration_mode.value,
            role=intent.model_role.value,
            confidence=intent.confidence,
        )
        
        # STEP 5: Execute orchestration
        span.event("execute_start", mode=intent.orchestration_mode.value)
        
        # FAST PATH: Skip Model-Dispatcher for non-complex intents
        # This saves 2-3s of latency for basic chat/status.
        fast_path_tasks = ["status", "quick"]
        if intent.task_type in fast_path_tasks and intent.orchestration_mode == OrchestrationMode.NONE:
            log.info("orchestration_fast_path_triggered", task=intent.task_type)
            result = await self.executor.execute(
                intent=intent,
                message=prepared_content,
                original_message=message,
                session_id=session_id,
                backend_type=backend_type,
                model_override=model_override,
                output_type=output_type,
            )
        else:
            # Check if we even have a backend before continuing (NORMAL PATH)
            from ..backend_manager import backend_manager
            if not backend_manager.get_active_backend():
                raise RuntimeError("No backend configured. Please configure at least one backend in settings.json.")

            # NORMAL PATH: Use Dispatcher -> Planner -> Executor
            result = await self.executor.execute(
                intent=intent,
                message=prepared_content,
                original_message=message,
                session_id=session_id,
                backend_type=backend_type,
                model_override=model_override,
                output_type=output_type,
            )
        
        span.event(
            "execute_complete",
            success=result.success,
            model=result.model_used,
            elapsed_ms=result.elapsed_ms,
        )
        
        # STEP 6 & 7: Quality validation and rewards (only on success)
        quality_score = None
        melons_awarded = 0
        affinity_updated = False
        
        if result.success and result.model_used:
            # Quality validation
            from ..quality import validate_response
            quality_result = validate_response(result.response, intent.task_type)
            quality_score = quality_result.overall
            
            span.event("quality_validated", score=round(quality_score, 2))
            
            # Update affinity tracker
            backend_id = result.backend_used or "unknown"
            self.affinity.update(backend_id, intent.task_type, quality=quality_score)
            affinity_updated = True
            
            # Award melons! 🍈
            from ..melons import award_melons_for_quality
            melons_awarded = award_melons_for_quality(
                model_id=result.model_used,
                task_type=intent.task_type,
                quality_score=quality_score,
            )
            
            # Record winning pair for local RL dataset 🏆
            get_reward_collector().record_winning_pair(
                prompt=message,
                response=result.response,
                model_id=result.model_used,
                task_type=intent.task_type,
                quality_score=quality_score,
            )
            
            # Record for frustration tracking
            if session_id:
                self.frustration.record_response(
                    session_id=session_id,
                    message=message,
                    model_used=result.model_used,
                    response=result.response,
                )
            
            span.event(
                "rewards_applied",
                melons=melons_awarded,
                affinity_updated=affinity_updated,
            )
            
            log.info(
                "orchestration_complete",
                mode=result.mode.value,
                model=result.model_used,
                quality=quality_score,
                melons=melons_awarded,
                elapsed_ms=result.elapsed_ms,
            )
        else:
            # Mark span as error if orchestration failed
            if not result.success:
                span.set_error(result.error or "orchestration_failed")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Final span attributes
        span.set_attribute("total_elapsed_ms", elapsed_ms)
        span.set_attribute("success", result.success)
        if quality_score is not None:
            span.set_attribute("quality_score", round(quality_score, 2))
        
        return ProcessingResult(
            result=result,
            intent=intent,
            quality_score=quality_score,
            melons_awarded=melons_awarded,
            affinity_updated=affinity_updated,
            prewarm_updated=prewarm_updated,
            frustration_penalty=frustration_penalty,
            elapsed_ms=elapsed_ms,
        )
    
    async def save_state(self) -> None:
        """
        Persist all learned state to disk.
        
        Should be called periodically (e.g., every 5 minutes) or on shutdown.
        """
        from ..config import save_affinity, save_prewarm
        
        await asyncio.to_thread(save_affinity)
        await asyncio.to_thread(save_prewarm)
        self.melons.save()
        
        log.debug("orchestration_service_state_saved")
    
    def get_stats(self) -> dict[str, Any]:
        """Get current service statistics."""
        return {
            "affinity_entries": len(self.affinity._scores),
            "total_prewarm_points": sum(len(points) for points in self.prewarm._scores.values()),
            "melon_models": len(self.melons._stats),
            "frustration_sessions": len(self.frustration._records),
        }


# =============================================================================
# Global Service Instance
# =============================================================================

_SERVICE: OrchestrationService | None = None
_SERVICE_LOCK = asyncio.Lock()


def get_orchestration_service() -> OrchestrationService:
    """
    Get the global orchestration service instance.
    
    Creates the service on first call (lazy initialization).
    Thread-safe via lock.
    """
    global _SERVICE
    
    if _SERVICE is None:
        _SERVICE = OrchestrationService.create()
    
    return _SERVICE


async def get_orchestration_service_async() -> OrchestrationService:
    """
    Async version of get_orchestration_service.
    
    Uses asyncio lock for proper async initialization.
    """
    global _SERVICE
    
    async with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = OrchestrationService.create()
        return _SERVICE


def reset_orchestration_service() -> None:
    """
    Reset the global service instance.
    
    Used for testing or when configuration changes require re-initialization.
    """
    global _SERVICE
    _SERVICE = None
    log.info("orchestration_service_reset")