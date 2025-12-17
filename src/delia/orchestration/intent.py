# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Intent Detection for NLP-Based Orchestration.

The IntentDetector is the brain of Delia's routing layer.
It analyzes user messages using NLP to determine:
1. What task type (quick, coder, moe, thinking)
2. What orchestration mode (none, voting, comparison, deep)
3. What role the model should play

This allows models to just respond naturally - they never see tools.
Delia handles orchestration AROUND them, not THROUGH them.

This is the ToolOrchestra paradigm: intent detection drives orchestration,
models are the tools, not tool-users.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from .result import DetectedIntent, OrchestrationMode, ModelRole

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
    """
    NLP-based intent detection for Delia orchestration.
    
    Instead of giving models tools and hoping they use them correctly,
    we detect intent at the Delia layer and orchestrate appropriately.
    
    Key insight from ToolOrchestra paper:
    - Models ARE the tools
    - Intent detection drives orchestration
    - Models receive task-specific prompts, not tool schemas
    
    Usage:
        detector = IntentDetector()
        intent = detector.detect("Make sure this code is secure: def login()...")
        # intent.orchestration_mode = VOTING
        # intent.model_role = CODE_REVIEWER
        # intent.task_type = "coder"
    """
    
    # Verification signals → Voting mode
    VERIFICATION_PATTERNS = [
        IntentPattern(
            re.compile(r"\b(make sure|verify|double.?check|confirm|validate|ensure|certain)\b", re.I),
            orchestration_mode=OrchestrationMode.VOTING,
            confidence_boost=0.3,
            reasoning="verification requested",
        ),
        IntentPattern(
            re.compile(r"\b(reliable|accurate|correct|trustworthy|confident)\s+(answer|response|result)\b", re.I),
            orchestration_mode=OrchestrationMode.VOTING,
            confidence_boost=0.25,
            reasoning="reliability requested",
        ),
        IntentPattern(
            re.compile(r"\b(high.?stakes|important|critical|crucial)\b", re.I),
            orchestration_mode=OrchestrationMode.VOTING,
            confidence_boost=0.2,
            reasoning="high-stakes context",
        ),
    ]
    
    # Comparison signals → Comparison mode
    COMPARISON_PATTERNS = [
        IntentPattern(
            re.compile(r"\b(compare|contrast|vs|versus|different\s+(models?|approaches?|opinions?))\b", re.I),
            orchestration_mode=OrchestrationMode.COMPARISON,
            confidence_boost=0.35,
            reasoning="comparison requested",
        ),
        IntentPattern(
            re.compile(r"\bwhat\s+do\s+(different|multiple|other)\s+(models?|llms?)\s+think\b", re.I),
            orchestration_mode=OrchestrationMode.COMPARISON,
            confidence_boost=0.4,
            reasoning="multi-model perspective requested",
        ),
        IntentPattern(
            re.compile(r"\b(second|third)\s+opinion\b", re.I),
            orchestration_mode=OrchestrationMode.COMPARISON,
            confidence_boost=0.3,
            reasoning="additional opinion requested",
        ),
    ]
    
    # Deep thinking signals → Deep mode
    DEEP_THINKING_PATTERNS = [
        IntentPattern(
            re.compile(r"\b(think\s+(carefully|deeply|thoroughly)|deep\s+(analysis|thinking|dive))\b", re.I),
            orchestration_mode=OrchestrationMode.DEEP_THINKING,
            task_type="thinking",
            confidence_boost=0.35,
            reasoning="deep analysis requested",
        ),
        IntentPattern(
            re.compile(r"\b(step.?by.?step|careful\s+consideration|thorough\s+analysis)\b", re.I),
            orchestration_mode=OrchestrationMode.DEEP_THINKING,
            task_type="moe",
            confidence_boost=0.25,
            reasoning="methodical analysis requested",
        ),
        IntentPattern(
            re.compile(r"\b(architect|design|trade.?offs?|pros?\s+and\s+cons?)\b", re.I),
            orchestration_mode=OrchestrationMode.DEEP_THINKING,
            task_type="moe",
            model_role=ModelRole.ARCHITECT,
            confidence_boost=0.2,
            reasoning="architecture/design task",
        ),
    ]
    
    # Agentic signals → Agent mode with tools
    AGENTIC_PATTERNS = [
        IntentPattern(
            re.compile(r"\b(read|open|show|cat|view)\s+(the\s+)?(file|contents?|code)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.4,
            reasoning="file read requested",
        ),
        IntentPattern(
            re.compile(r"\b(list|ls|show)\s+(the\s+)?(files?|directory|directories|folder)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.4,
            reasoning="directory listing requested",
        ),
        IntentPattern(
            re.compile(r"\b(run|execute|exec)\s+(this\s+)?(command|script|shell)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.45,
            reasoning="shell execution requested",
        ),
        IntentPattern(
            re.compile(r"\b(search|find|grep|look\s+for)\s+.*(in\s+)?(the\s+)?(code|files?|codebase|project|src|directory)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.4,
            reasoning="code search requested",
        ),
        IntentPattern(
            re.compile(r"\b(search|find|grep)\s+(for\s+)?\w+", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.35,
            reasoning="search operation",
        ),
        IntentPattern(
            re.compile(r"\b(write|create|save|update|modify|edit)\s+.*(to\s+)?\w+\.(py|js|ts|rs|go|sh|yaml|json|toml|txt|md)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.45,
            reasoning="file write requested",
        ),
        IntentPattern(
            re.compile(r"\b(write|create|save)\s+(a\s+)?(file|code|test|script)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.4,
            reasoning="file creation requested",
        ),
        IntentPattern(
            re.compile(r"\b(install|npm|pip|yarn|apt|brew|cargo)\s+\w+", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.45,
            reasoning="package installation requested",
        ),
        IntentPattern(
            re.compile(r"\b(git|docker|kubectl|make|cmake|./)\b", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.35,
            reasoning="CLI tool usage detected",
        ),
        IntentPattern(
            re.compile(r"[`'\"][\w\-./]+\.(py|js|ts|rs|go|sh|yaml|json|toml)[`'\"]", re.I),
            orchestration_mode=OrchestrationMode.AGENTIC,
            confidence_boost=0.3,
            reasoning="specific file referenced",
        ),
    ]
    
    # Code-related signals → Coder task type
    CODE_PATTERNS = [
        IntentPattern(
            re.compile(r"\b(review|audit|check)\s+(this\s+)?(code|function|class|script)\b", re.I),
            task_type="coder",
            model_role=ModelRole.CODE_REVIEWER,
            confidence_boost=0.3,
            reasoning="code review requested",
        ),
        IntentPattern(
            re.compile(r"\b(security|vulnerability|vulnerabilities|exploit|injection|xss|sql)\b", re.I),
            task_type="coder",
            model_role=ModelRole.CODE_REVIEWER,
            confidence_boost=0.25,
            reasoning="security review",
        ),
        IntentPattern(
            re.compile(r"\b(write|create|generate|implement|build)\s+(a\s+)?(function|class|code|script|api)\b", re.I),
            task_type="coder",
            model_role=ModelRole.CODE_GENERATOR,
            confidence_boost=0.3,
            reasoning="code generation requested",
        ),
        IntentPattern(
            re.compile(r"\b(debug|fix|error|bug|issue|broken|not\s+working)\b", re.I),
            task_type="coder",
            model_role=ModelRole.DEBUGGER,
            confidence_boost=0.25,
            reasoning="debugging requested",
        ),
        IntentPattern(
            re.compile(r"```[\w]*\n", re.I),  # Code block
            task_type="coder",
            confidence_boost=0.2,
            reasoning="code block present",
        ),
        IntentPattern(
            re.compile(r"\b(def|class|function|const|let|import|from)\s+\w+", re.I),
            task_type="coder",
            confidence_boost=0.15,
            reasoning="code syntax detected",
        ),
    ]
    
    # Quick/simple signals → Quick task type
    QUICK_PATTERNS = [
        IntentPattern(
            re.compile(r"^(hi|hello|hey|thanks?|thank\s+you|ok|okay|yes|no|bye)\b", re.I),
            task_type="quick",
            confidence_boost=0.4,
            reasoning="greeting/acknowledgment",
        ),
        IntentPattern(
            re.compile(r"^(what\s+is|who\s+is|when|where|how\s+many|how\s+much)\b", re.I),
            task_type="quick",
            model_role=ModelRole.ASSISTANT,
            confidence_boost=0.2,
            reasoning="simple question",
        ),
        IntentPattern(
            re.compile(r"\b(summarize|tldr|brief|short|quick)\b", re.I),
            task_type="quick",
            model_role=ModelRole.SUMMARIZER,
            confidence_boost=0.2,
            reasoning="summary requested",
        ),
    ]
    
    # Explanation signals → Explainer role
    EXPLAIN_PATTERNS = [
        IntentPattern(
            re.compile(r"\b(explain|teach|help\s+me\s+understand|walk\s+me\s+through)\b", re.I),
            model_role=ModelRole.EXPLAINER,
            confidence_boost=0.2,
            reasoning="explanation requested",
        ),
        IntentPattern(
            re.compile(r"\b(like\s+i'?m\s+(5|five|a\s+beginner)|eli5|newbie|basics?)\b", re.I),
            model_role=ModelRole.EXPLAINER,
            task_type="quick",  # Simple explanations
            confidence_boost=0.25,
            reasoning="beginner explanation requested",
        ),
    ]
    
    def __init__(self) -> None:
        """Initialize the intent detector with all patterns."""
        # Combine all patterns in priority order
        # AGENTIC first - file/shell ops take precedence
        self.all_patterns = (
            self.AGENTIC_PATTERNS +
            self.VERIFICATION_PATTERNS +
            self.COMPARISON_PATTERNS +
            self.DEEP_THINKING_PATTERNS +
            self.CODE_PATTERNS +
            self.QUICK_PATTERNS +
            self.EXPLAIN_PATTERNS
        )
    
    def detect(self, message: str) -> DetectedIntent:
        """
        Detect intent from a user message.
        
        This is the core NLP routing logic. It determines:
        1. Task type (which model tier to use)
        2. Orchestration mode (how Delia should orchestrate)
        3. Model role (what system prompt to use)
        
        Args:
            message: The user's message
            
        Returns:
            DetectedIntent with all routing information
        """
        if not message or len(message.strip()) < 3:
            return DetectedIntent(
                task_type="quick",
                model_role=ModelRole.ASSISTANT,
                confidence=0.5,
                reasoning="very short message",
            )
        
        # Start with defaults
        intent = DetectedIntent(
            task_type="quick",
            orchestration_mode=OrchestrationMode.NONE,
            model_role=ModelRole.ASSISTANT,
            confidence=0.5,
            reasoning="default",
        )
        
        # Track matched patterns
        matched_keywords: list[str] = []
        confidence_adjustments: list[float] = []
        reasons: list[str] = []
        
        # Check all patterns
        for pat in self.all_patterns:
            match = pat.pattern.search(message)
            if match:
                matched_keywords.append(match.group(0))
                confidence_adjustments.append(pat.confidence_boost)
                if pat.reasoning:
                    reasons.append(pat.reasoning)
                
                # Apply pattern effects
                if pat.orchestration_mode is not None:
                    # Orchestration mode is important - only override if higher priority
                    if intent.orchestration_mode == OrchestrationMode.NONE:
                        intent.orchestration_mode = pat.orchestration_mode
                
                if pat.model_role is not None:
                    intent.model_role = pat.model_role
                
                if pat.task_type is not None:
                    intent.task_type = pat.task_type
        
        # Check for code content (affects task type)
        intent.contains_code = bool(
            re.search(r"```|\bdef\s|\bclass\s|\bfunction\s|\bconst\s|\blet\s", message)
        )
        if intent.contains_code and intent.task_type == "quick":
            intent.task_type = "coder"
            reasons.append("code content detected")
            confidence_adjustments.append(0.15)
        
        # Calculate final confidence
        if confidence_adjustments:
            intent.confidence = min(0.95, 0.5 + sum(confidence_adjustments))
        
        # Build reasoning
        intent.reasoning = "; ".join(reasons) if reasons else "general chat"
        intent.trigger_keywords = matched_keywords
        
        # Calculate k for voting mode
        if intent.orchestration_mode == OrchestrationMode.VOTING:
            intent.k_votes = self._calculate_k(message, intent)
        
        log.info(
            "intent_detected",
            task_type=intent.task_type,
            orchestration=intent.orchestration_mode.value,
            role=intent.model_role.value,
            confidence=round(intent.confidence, 2),
            reasoning=intent.reasoning,
            keywords=matched_keywords[:5],  # Limit for logging
        )
        
        return intent
    
    def _calculate_k(self, message: str, intent: DetectedIntent) -> int:
        """
        Calculate optimal k for voting based on task complexity.
        
        Uses the MDAP formula: kmin = Θ(ln s) where s is step count.
        We estimate steps from message complexity.
        """
        from ..voting import VotingConsensus, estimate_task_complexity
        
        task_steps = estimate_task_complexity(message)
        
        k = VotingConsensus.calculate_kmin(
            total_steps=task_steps,
            target_accuracy=0.9999,
            base_accuracy=0.95,  # Conservative
        )
        
        return max(2, min(k, 5))  # Clamp to reasonable range
    
    def needs_orchestration(self, intent: DetectedIntent) -> bool:
        """Check if intent requires orchestration (vs simple model call)."""
        return intent.orchestration_mode != OrchestrationMode.NONE
    
    def get_debug_info(self, message: str) -> dict:
        """Get detailed debug info about intent detection (for troubleshooting)."""
        intent = self.detect(message)
        
        return {
            "message_preview": message[:200],
            "detected_intent": {
                "task_type": intent.task_type,
                "orchestration_mode": intent.orchestration_mode.value,
                "model_role": intent.model_role.value,
                "confidence": intent.confidence,
                "reasoning": intent.reasoning,
            },
            "trigger_keywords": intent.trigger_keywords,
            "contains_code": intent.contains_code,
            "k_votes": intent.k_votes if intent.orchestration_mode == OrchestrationMode.VOTING else None,
        }


# Module-level convenience function
_detector: IntentDetector | None = None


def get_intent_detector() -> IntentDetector:
    """Get or create the global intent detector."""
    global _detector
    if _detector is None:
        _detector = IntentDetector()
    return _detector


def detect_intent(message: str) -> DetectedIntent:
    """Detect intent from a message using the global detector."""
    return get_intent_detector().detect(message)

