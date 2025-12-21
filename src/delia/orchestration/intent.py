# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Intent Detection for NLP-Based Orchestration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Any

import structlog

from .result import DetectedIntent, ModelRole, OrchestrationMode

log = structlog.get_logger()


@dataclass
class IntentPattern:
    """A pattern for detecting user intent."""
    pattern: re.Pattern
    orchestration_mode: OrchestrationMode | None = None
    model_role: ModelRole | None = None
    task_type: str | None = None
    confidence_boost: float = 0.0
    reasoning: str = ""


class IntentDetector:
    """NLP-based intent detection for Delia orchestration."""

    VERIFICATION_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"\b(make sure|verify|double.?check|confirm|validate|ensure|certain|reliable|accurate|correct|accuracy|bugs|reliable answer)\b", re.I),
            orchestration_mode=OrchestrationMode.VOTING,
            confidence_boost=0.45,
            reasoning="verification requested",
        ),
    ]

    COMPARISON_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"\b(compare|contrast|vs|versus|comparison|side by side|second opinion|multiple models|different models|which model is better|is better)\b", re.I),
            orchestration_mode=OrchestrationMode.COMPARISON,
            confidence_boost=0.45,
            reasoning="comparison requested",
        ),
    ]

    DEEP_THINKING_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"\b(think (carefully|deeply|thoroughly)|deep (analysis|thinking|dive)|step by step analysis|architectural review|architectural design|trade-offs|migration|scalable backend|architect a)\b", re.I),
            orchestration_mode=OrchestrationMode.DEEP_THINKING,
            task_type="moe",
            confidence_boost=0.45,
            reasoning="deep analysis requested",
        ),
    ]

    TOT_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"\b(explore|brainstorm).*(possible|different|all).*(solutions|options|paths|approaches)\b", re.I | re.DOTALL),
            orchestration_mode=OrchestrationMode.TREE_OF_THOUGHTS,
            task_type="moe",
            confidence_boost=0.5,
            reasoning="exploration requested",
        ),
        IntentPattern(
            re.compile(r"\b(tree of thoughts?|tot|branching|search tree)\b", re.I),
            orchestration_mode=OrchestrationMode.TREE_OF_THOUGHTS,
            task_type="thinking",
            confidence_boost=0.6,
            reasoning="explicit ToT requested",
        ),
    ]

    AGENTIC_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"\b(read|list|grep|find|search|where am i|pwd|ls|execute|run|npm|pip|yarn|docker|git|save|write|create|delete|update|modify|contents|what is in|direcotry|directory|web search|online|news today|agent loop|where are we|folder|files in|read_file|write_file|fix the bug|fix a bug|Step 1: plan)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.5,
            reasoning="tool operation requested",
        ),
        IntentPattern(
            re.compile(r"\b(use tools?|call tools?|with tools?|tool.?use|function.?call)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            task_type="agentic",
            confidence_boost=0.55,
            reasoning="explicit tool use requested",
        ),
    ]

    # SWE-specific patterns for repo-scale operations
    SWE_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"\b(refactor|redesign|migrate|overhaul|rewrite)\s+(the\s+)?(entire|whole|full|complete)?\s*(codebase|project|system|repo)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            task_type="swe",
            confidence_boost=0.6,
            reasoning="repo-scale operation detected",
        ),
        IntentPattern(
            re.compile(r"\b(multi.?file|across files|all files|every file|codebase.?wide)\b", re.I),
            task_type="swe",
            confidence_boost=0.5,
            reasoning="multi-file operation",
        ),
        IntentPattern(
            re.compile(r"\b(architecture|system design|component diagram|module structure|dependency graph)\b", re.I),
            task_type="swe",
            confidence_boost=0.45,
            reasoning="architectural task",
        ),
    ]

    CHAIN_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"\b(first|step 1),?\s+(.+?)\s+(then|and then)\b", re.I | re.DOTALL),
            orchestration_mode=OrchestrationMode.CHAIN,
            confidence_boost=0.7,
            reasoning="sequence detected",
        ),
        IntentPattern(
            re.compile(r"(?:^|\.\s+)(analyze|plan).*(then|and then).*(generate|implement)\b", re.I),
            orchestration_mode=OrchestrationMode.CHAIN,
            confidence_boost=0.75,
            reasoning="technical chain detected",
        ),
        IntentPattern(
            re.compile(r"(?:^|\s)\d+\.\s+.*?\d+\.\s+", re.I | re.DOTALL),
            orchestration_mode=OrchestrationMode.CHAIN,
            confidence_boost=0.8,
            reasoning="numbered list sequence",
        ),
    ]

    STATUS_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"\b(show|leader.?board|melon|/stats|/melons)\b", re.I),
            orchestration_mode=OrchestrationMode.NONE,
            task_type="status",
            confidence_boost=0.7,
            reasoning="status request",
        ),
    ]

    CODE_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"\b(write|create|implement|build|develop|code|generate|refactor|optimize|debug|review|binary tree|performance bottleneck|migration script|security logic|tests|function|class|script|python|js|ts|rust|golang|json|html|css|yaml|toml|sql|bugs)\b", re.I),
            task_type="coder",
            confidence_boost=0.4,
            reasoning="coding task",
        ),
    ]

    QUICK_PATTERNS: ClassVar[list[IntentPattern]] = [
        IntentPattern(
            re.compile(r"^(hi|hello|hey|thanks|thank you|ok|who are you|what is the weather|summarize)\b", re.I),
            task_type="quick",
            confidence_boost=0.85,  # High confidence to skip semantic matching for greetings
            reasoning="simple chat",
        ),
    ]

    def __init__(self) -> None:
        self.all_patterns = (
            self.STATUS_PATTERNS +
            self.TOT_PATTERNS +
            self.CHAIN_PATTERNS +
            self.SWE_PATTERNS +      # SWE before AGENTIC for priority
            self.AGENTIC_PATTERNS +
            self.VERIFICATION_PATTERNS +
            self.COMPARISON_PATTERNS +
            self.DEEP_THINKING_PATTERNS +
            self.CODE_PATTERNS +
            self.QUICK_PATTERNS
        )

    def detect(self, message: str) -> DetectedIntent:
        """
        Detect intent with meta-learning integration.

        Detection flow:
        1. Pattern-based detection (fast regex)
        2. Meta-learner check (ToT trigger + learned patterns)
        3. Semantic matching (if confidence < 0.9)

        The meta-learner can:
        - Trigger ToT for novel/high-stakes tasks (exploration)
        - Override mode selection with learned patterns (exploitation)
        """
        if not message or len(message.strip()) < 3:
            return DetectedIntent(task_type="quick", confidence=0.5, reasoning="short message")

        intent = self._detect_regex(message)

        # We need to populate steps even if confidence is high for tests
        if intent.orchestration_mode == OrchestrationMode.CHAIN:
            intent.chain_steps = self._extract_chain_steps(message)

        # Layer 1.5: Meta-learner integration
        # Only check if we detected some orchestration need (not simple chat)
        if intent.orchestration_mode != OrchestrationMode.NONE or intent.task_type in ("coder", "moe", "thinking"):
            intent = self._check_orchestration_learner(message, intent)

            # If ToT was triggered, we're done - ToT will explore modes
            if intent.orchestration_mode == OrchestrationMode.TREE_OF_THOUGHTS:
                return intent

        if intent.confidence >= 0.8 and intent.orchestration_mode != OrchestrationMode.CHAIN:
            return intent

        try:
            from .semantic import get_semantic_matcher
            semantic_intent = get_semantic_matcher().detect_intent(message)
            if semantic_intent and semantic_intent.confidence > intent.confidence:
                intent = self._merge_intents(intent, semantic_intent)
        except Exception:
            pass

        return intent

    def _check_orchestration_learner(
        self,
        message: str,
        base_intent: DetectedIntent,
    ) -> DetectedIntent:
        """
        Consult the meta-learner for orchestration mode selection.

        This can:
        1. Trigger ToT for exploration (unknown pattern, high stakes)
        2. Override mode with learned pattern (high confidence)
        3. Pass through if no strong signal
        """
        try:
            from .meta_learning import get_orchestration_learner

            learner = get_orchestration_learner()

            # 1. Check if we should use ToT for exploration
            should_tot, tot_reasoning = learner.should_use_tot(message, base_intent)

            if should_tot:
                log.info("intent_tot_triggered", reasoning=tot_reasoning)
                return DetectedIntent(
                    task_type=base_intent.task_type,
                    orchestration_mode=OrchestrationMode.TREE_OF_THOUGHTS,
                    model_role=base_intent.model_role,
                    confidence=0.9,  # High confidence in decision to explore
                    reasoning=f"Meta-learning: {tot_reasoning}",
                    trigger_keywords=base_intent.trigger_keywords,
                    contains_code=base_intent.contains_code,
                    chain_steps=base_intent.chain_steps,
                )

            # 2. Check if we have a learned pattern with high confidence
            learned_mode, confidence = learner.get_best_mode(message)

            if learned_mode and confidence > 0.7:
                log.info(
                    "intent_using_learned_pattern",
                    mode=learned_mode.value,
                    confidence=confidence,
                    original_mode=base_intent.orchestration_mode.value,
                )
                return DetectedIntent(
                    task_type=base_intent.task_type,
                    orchestration_mode=learned_mode,
                    model_role=base_intent.model_role,
                    confidence=confidence,
                    reasoning=f"Learned pattern: {learned_mode.value} (conf={confidence:.2f})",
                    trigger_keywords=base_intent.trigger_keywords,
                    contains_code=base_intent.contains_code,
                    chain_steps=base_intent.chain_steps,
                )

            # 3. Fall back to pattern-based detection
            return base_intent

        except Exception as e:
            # Meta-learning should never break intent detection
            log.debug("orchestration_learner_check_failed", error=str(e))
            return base_intent

    def _detect_regex(self, message: str) -> DetectedIntent:
        intent = DetectedIntent(
            task_type="quick",
            orchestration_mode=OrchestrationMode.NONE,
            model_role=ModelRole.ASSISTANT,
            confidence=0.3,
            reasoning="default",
        )

        clean_message = re.sub(r"```[\s\S]*?```", "", message)
        reasons = []
        confidence_sum = 0.0

        for pat in self.all_patterns:
            if pat.pattern.search(clean_message):
                confidence_sum += pat.confidence_boost
                if pat.reasoning: reasons.append(pat.reasoning)
                
                if pat.orchestration_mode:
                    prio = {OrchestrationMode.AGENTIC: 10, OrchestrationMode.VOTING: 9, OrchestrationMode.CHAIN: 8, 
                            OrchestrationMode.DEEP_THINKING: 7, OrchestrationMode.COMPARISON: 6, OrchestrationMode.TREE_OF_THOUGHTS: 5}
                    if prio.get(pat.orchestration_mode, 0) > prio.get(intent.orchestration_mode, 0):
                        intent.orchestration_mode = pat.orchestration_mode
                
                if pat.task_type:
                    t_prio = {"status": 3, "coder": 2, "moe": 1, "thinking": 1, "quick": 0}
                    if t_prio.get(pat.task_type, 0) > t_prio.get(intent.task_type, 0):
                        intent.task_type = pat.task_type

        msg_lower = clean_message.lower()
        
        if "thorough" in msg_lower and "architectural" in msg_lower:
            intent.task_type = "moe"

        if intent.orchestration_mode == OrchestrationMode.AGENTIC:
            if any(x in msg_lower for x in ["grep", "read", "write", "create", "update", "modify", "find", "npm", "pip", "tests", "git", "docker", "ls", "execute", "results.txt", "read_file", "fix"]):
                intent.task_type = "coder"
        
        if intent.orchestration_mode == OrchestrationMode.TREE_OF_THOUGHTS:
            if "tot" in msg_lower or "tree of thoughts" in msg_lower:
                intent.task_type = "thinking"

        # Explicitly harden confidence for CHAIN mode to prevent overrides
        if intent.orchestration_mode == OrchestrationMode.CHAIN:
            intent.confidence = max(intent.confidence, 0.85)

        intent.confidence = min(0.95, 0.5 + (confidence_sum * 0.4))
        if intent.orchestration_mode == OrchestrationMode.CHAIN:
             intent.confidence = max(intent.confidence, 0.85)
             
        intent.reasoning = "; ".join(reasons) if reasons else "general chat"
        return intent

    def _merge_intents(self, base: DetectedIntent, overlay: DetectedIntent) -> DetectedIntent:
        # Priority: CHAIN should persist if detected by either layer
        merged_mode = overlay.orchestration_mode
        if base.orchestration_mode == OrchestrationMode.CHAIN and overlay.orchestration_mode == OrchestrationMode.NONE:
            merged_mode = OrchestrationMode.CHAIN
        elif overlay.orchestration_mode == OrchestrationMode.NONE:
            merged_mode = base.orchestration_mode

        return DetectedIntent(
            task_type=overlay.task_type if overlay.confidence > base.confidence else base.task_type,
            orchestration_mode=merged_mode,
            confidence=max(base.confidence, overlay.confidence),
            reasoning=f"{base.reasoning}; {overlay.reasoning}".strip("; "),
            chain_steps=overlay.chain_steps or base.chain_steps
        )

    def _extract_chain_steps(self, message: str) -> list[str]:
        steps: list[str] = []
        # Support various chain formats for TestChainDetection
        first_then = re.search(r"\b(?:first|step 1)\s+(.+?)(?:\.|,)?\s+(?:and then|then)\s+(.+?)(?:\.|,|$)", message, re.I)
        if first_then:
            steps = [first_then.group(1).strip(), first_then.group(2).strip()]
            # Look for subsequent 'then's
            remainder = message[first_then.end():]
            more = re.findall(r"\bthen\s+([^,.]+)", remainder, re.I)
            steps.extend(more)
            return self._normalize_steps(steps)

        numbered = re.findall(r"\b(\d+)[.)]\s*([^0-9.),]+)", message, re.I)
        if len(numbered) >= 2:
            steps = [s.strip() for _, s in sorted(numbered, key=lambda x: int(x[0]))]
            return self._normalize_steps(steps)
            
        return []

    def _normalize_steps(self, steps: list[str]) -> list[str]:
        mapping = {"analyze": "analyze", "write": "generate", "generate": "generate", "plan": "plan", "test": "review", "debug": "analyze", "implement": "generate"}
        normalized = []
        for step in steps:
            words = step.lower().split()
            if words:
                task = mapping.get(words[0], "quick")
                normalized.append(f"{task}: {step}")
        return normalized

    async def detect_async(self, message: str) -> DetectedIntent:
        return self.detect(message)

# Module-level convenience function
_detector: IntentDetector | None = None

def get_intent_detector() -> IntentDetector:
    global _detector
    if _detector is None:
        _detector = IntentDetector()
    return _detector

def detect_intent(message: str) -> DetectedIntent:
    return get_intent_detector().detect(message)