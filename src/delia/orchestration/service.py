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
from .intrinsics import (
    get_intrinsics_engine,
    IntrinsicAction,
    FrustrationLevel,
)
from ..tracing import trace, add_event
from ..melons import get_reward_collector
from ..semantic import get_conversation_compressor as get_compactor
from ..semantic.compression import DEFAULT_COMPACTION_THRESHOLD_TOKENS

if TYPE_CHECKING:
    from ..config import AffinityTracker, PrewarmTracker
    from ..melons import MelonTracker


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
    ) -> None:
        self.executor = executor
        self.affinity = affinity
        self.prewarm = prewarm
        self.melons = melons
        self._initialized = True
        # Simple repeat tracking per session (replaces FrustrationTracker)
        self._repeat_hashes: dict[str, list[str]] = {}

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

        # Load persisted state
        load_affinity()
        load_prewarm()

        return cls(
            executor=get_orchestration_executor(),
            affinity=get_affinity_tracker(),
            prewarm=get_prewarm_tracker(),
            melons=get_melon_tracker(),
        )

    def _hash_message(self, message: str) -> str:
        """Create normalized hash for repeat detection."""
        import hashlib
        normalized = " ".join(message.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _check_repeat(self, session_id: str, message: str) -> tuple[bool, int]:
        """Check if message is a repeat in this session. Returns (is_repeat, count)."""
        if not session_id:
            return False, 0

        msg_hash = self._hash_message(message)
        hashes = self._repeat_hashes.setdefault(session_id, [])

        # Count how many times this hash appears
        count = hashes.count(msg_hash)

        # Record this message
        hashes.append(msg_hash)

        # Keep only last 50 messages per session
        if len(hashes) > 50:
            self._repeat_hashes[session_id] = hashes[-50:]

        return count > 0, count

    async def _check_auto_compaction(
        self,
        session,
        session_id: str,
    ) -> StreamEvent | None:
        """
        Check if a session needs compaction and optionally auto-compact.

        Returns a warning StreamEvent if compaction is recommended,
        or performs auto-compaction if above critical threshold.
        """
        from ..session_manager import get_session_manager

        compactor = get_compactor()
        stats = compactor.get_compaction_stats(session)

        if not stats["needs_compaction"]:
            return None

        total_tokens = stats["total_tokens"]
        threshold = stats["threshold_tokens"]

        # Critical threshold: 1.5x the normal threshold - auto-compact
        critical_threshold = int(threshold * 1.5)

        if total_tokens >= critical_threshold:
            # Auto-compact to prevent context overflow
            log.info(
                "auto_compaction_triggered",
                session_id=session_id,
                tokens=total_tokens,
                critical_threshold=critical_threshold,
            )

            try:
                from ..semantic.compression import compact_session
                result = await compact_session(session, force=True)

                if result.success:
                    # Save the compacted session
                    sm = get_session_manager()
                    with sm._lock:
                        sm._save_session(session_id)

                    return StreamEvent(
                        event_type="compaction",
                        message=f"Session auto-compacted: {result.messages_compacted} messages summarized, "
                                f"{result.tokens_saved} tokens saved ({result.compression_ratio:.0%} reduction)",
                        details={
                            "auto": True,
                            "messages_compacted": result.messages_compacted,
                            "tokens_saved": result.tokens_saved,
                            "compression_ratio": result.compression_ratio,
                        }
                    )
            except Exception as e:
                log.warning("auto_compaction_failed", error=str(e), session_id=session_id)

        # Just a warning - user should manually compact
        return StreamEvent(
            event_type="warning",
            message=f"Session context is large ({total_tokens} tokens). "
                    f"Consider using session_compact to reduce context.",
            details={
                "total_tokens": total_tokens,
                "threshold": threshold,
                "compactable_messages": stats["compactable_messages"],
            }
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

                # Check for auto-compaction need
                compaction_warning = await self._check_auto_compaction(session, session_id)
                if compaction_warning:
                    yield compaction_warning

        # 2. Intent Detection (moved before context prep)
        intent = await get_intent_detector().detect_async(message)

        # Skip project overview for quick tasks (greetings, simple Q&A)
        # This avoids confusing the model with "Project overview not yet generated"
        needs_overview = intent.task_type not in ("quick", "status", "chat")

        # Skip session history injection for quick tasks (greetings, simple Q&A)
        # Simple messages should get simple responses - history confuses the model
        needs_session_context = intent.task_type not in ("quick", "status", "chat")

        prepared_content = await ContextEngine.prepare_content(
            content=message,
            context=context,
            symbols=symbols,
            include_references=include_references,
            files=files,
            session_context=session_context if needs_session_context else None,
            include_project_overview=needs_overview,  # Only for complex tasks
            include_project_instructions=needs_overview,  # Skip for quick tasks
        )
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

        # STEP 1.5: Answerability check (Intrinsics) - gate before expensive LLM call
        # Only run for non-trivial tasks where context matters
        intrinsics_action = None
        from .intrinsics import get_intrinsics_engine, IntrinsicAction
        intrinsics = get_intrinsics_engine()
        if intent.task_type not in ["status", "quick"] and files:
            try:
                answerability = await intrinsics.check_answerability(message, prepared_content)

                span.event(
                    "answerability_check",
                    score=round(answerability.score, 2),
                    action=answerability.action.value,
                    passed=answerability.passed,
                )

                intrinsics_action = answerability.action

                if answerability.action == IntrinsicAction.FETCH_MORE:
                    # Context insufficient - try to auto-fetch more files
                    log.info(
                        "intrinsics_fetch_more",
                        score=answerability.score,
                        missing=answerability.missing_info,
                    )
                    # Use semantic search to find additional relevant files
                    try:
                        from .summarizer import get_summarizer
                        summarizer = get_summarizer()
                        await summarizer.initialize()
                        results = await summarizer.search(message, top_k=3)
                        if results:
                            extra_paths = [r["path"] for r in results if r["score"] >= 0.3]
                            if extra_paths:
                                # Re-prepare content with additional files
                                extra_files = ",".join(extra_paths)
                                combined_files = f"{files},{extra_files}" if files else extra_files
                                prepared_content = await ContextEngine.prepare_content(
                                    content=message,
                                    context=context,
                                    symbols=symbols,
                                    include_references=include_references,
                                    files=combined_files,
                                    session_context=session_context,
                                    include_project_overview=needs_repo_context,
                                )
                                span.event("intrinsics_context_expanded", extra_files=extra_paths)
                    except Exception as e:
                        log.debug("intrinsics_auto_fetch_failed", error=str(e))

                elif answerability.action == IntrinsicAction.ESCALATE:
                    # Low answerability - escalate orchestration mode
                    log.warning(
                        "intrinsics_escalate",
                        score=answerability.score,
                        current_mode=intent.orchestration_mode.value,
                    )
                    if intent.orchestration_mode == OrchestrationMode.NONE:
                        intent.orchestration_mode = OrchestrationMode.VOTING
                        intent.k_votes = 3
                        intent.reasoning = f"intrinsics escalation (answerability={answerability.score:.2f}); {intent.reasoning}"

            except Exception as e:
                log.debug("answerability_check_failed", error=str(e))
                # Don't block on intrinsics failure

        # STEP 2: User state analysis (consolidated frustration detection via intrinsics)
        is_repeat, repeat_count = self._check_repeat(session_id, message) if session_id else (False, 0)
        user_state = intrinsics.check_user_state(message, is_repeat=is_repeat, repeat_count=repeat_count)

        if user_state.level != FrustrationLevel.NONE:
            span.event("frustration_detected", level=user_state.level.value)
            log.warning("frustration_detected", level=user_state.level.value, action=user_state.action.value)

        # STEP 3: Auto-upgrade orchestration based on user state action
        if user_state.action in (IntrinsicAction.ESCALATE_DEEP, IntrinsicAction.ESCALATE_VOTING, IntrinsicAction.ESCALATE):
            current_mode_is_weak = intent.orchestration_mode == OrchestrationMode.NONE

            if user_state.action == IntrinsicAction.ESCALATE_DEEP:
                if current_mode_is_weak:
                    intent.orchestration_mode = OrchestrationMode.DEEP_THINKING
                    intent.reasoning = f"auto-escalation: {user_state.reasoning}; {intent.reasoning}"
                    log.warning("frustration_auto_escalate", mode="deep_thinking")

            elif user_state.action == IntrinsicAction.ESCALATE_VOTING:
                k = 3 if user_state.level == FrustrationLevel.MEDIUM else 2
                if current_mode_is_weak or (intent.orchestration_mode == OrchestrationMode.VOTING and intent.k_votes < k):
                    intent.orchestration_mode = OrchestrationMode.VOTING
                    intent.k_votes = k
                    intent.reasoning = f"auto-escalation: {user_state.reasoning}; {intent.reasoning}"
                    log.warning("frustration_auto_escalate", mode=f"voting_k{k}")
        
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

            # Groundedness check (Intrinsics) - detect potential hallucination
            # Only run if we have source context (files were provided)
            groundedness_score = None
            if files and len(prepared_content) > 500:
                try:
                    from .intrinsics import get_intrinsics_engine, IntrinsicAction
                    intrinsics = get_intrinsics_engine()
                    groundedness = await intrinsics.check_groundedness(
                        result.response,
                        prepared_content,
                    )
                    groundedness_score = groundedness.score

                    span.event(
                        "groundedness_check",
                        score=round(groundedness.score, 2),
                        passed=groundedness.passed,
                    )

                    if not groundedness.passed:
                        # Flag low groundedness - response may contain hallucination
                        log.warning(
                            "intrinsics_low_groundedness",
                            score=groundedness.score,
                            reasoning=groundedness.reasoning[:100],
                        )
                        # Penalize quality score for ungrounded responses
                        quality_score = quality_score * 0.7

                except Exception as e:
                    log.debug("groundedness_check_failed", error=str(e))
            
            # Update affinity trackers
            backend_id = result.backend_used or "unknown"

            # Backend affinity (for multi-GPU setups)
            from ..backend_manager import backend_manager
            if backend_manager.is_feedback_enabled("backend_affinity"):
                self.affinity.update(backend_id, intent.task_type, quality=quality_score)
                affinity_updated = True

            # Model affinity (for single-GPU setups - tracks which model is best per task)
            if backend_manager.is_feedback_enabled("model_affinity"):
                from ..config import get_model_affinity_tracker, save_model_affinity
                model_affinity = get_model_affinity_tracker()

                # Calculate tokens/sec if we have the data
                tokens_per_sec = None
                if result.tokens and result.elapsed_ms and result.elapsed_ms > 0:
                    tokens_per_sec = result.tokens / (result.elapsed_ms / 1000)

                model_affinity.update(
                    model_id=result.model_used,
                    task_type=intent.task_type,
                    success=True,
                    quality_score=quality_score,
                    tokens_per_sec=tokens_per_sec,
                )
                model_affinity.set_current_model(result.model_used)
                save_model_affinity()  # Persist immediately
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
            # Note: Repeat tracking is done in _check_repeat() at STEP 2

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
            "prewarm_tiers": len(self.prewarm._scores),
            "melon_models": len(self.melons._stats),
            "repeat_sessions": len(self._repeat_hashes),
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