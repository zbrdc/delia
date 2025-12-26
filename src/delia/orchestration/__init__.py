# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Delia Orchestration Package - NLP-Driven Multi-Model Orchestration.

This package implements the "Models ARE the Tools" paradigm from ToolOrchestra.
Instead of giving models tools and hoping they use them correctly, we:

1. DETECT intent at the Delia layer using NLP
2. DECIDE what orchestration is needed (voting, comparison, etc.)
3. GENERATE task-specific system prompts for the model's role
4. EXECUTE the orchestration AROUND the model

The model never sees tools. It just responds naturally in its assigned role.
Delia handles all orchestration decisions.

Key Components:
    - IntentDetector: NLP-based detection of user intent
    - SystemPromptGenerator: Role-specific prompts for models
    - OrchestrationExecutor: Executes voting, comparison, etc.
    
Usage:
    from delia.orchestration import detect_intent, execute_orchestration
    
    # Detect what the user wants
    intent = detect_intent("Make sure this code is secure: def login()...")
    # intent.orchestration_mode = VOTING
    # intent.model_role = CODE_REVIEWER
    
    # Execute orchestration (Delia handles voting, model gets role-specific prompt)
    result = await execute_orchestration(intent, message)
    
This is the ToolOrchestra paradigm:
- Models are tools, not tool-users
- Intent detection drives orchestration
- Outcome rewards (melons) guide routing

References:
- ToolOrchestra: Training LLMs to Think Like Tools
- MDAP: Massively Decomposed Agentic Processes (k-voting)
"""

from .result import (
    OrchestrationMode,
    ModelRole,
    DetectedIntent,
    OrchestrationResult,
    StreamEvent,
)

from .intent import (
    IntentDetector,
    get_intent_detector,
    detect_intent,
)

from .prompts import (
    SystemPromptGenerator,
    get_prompt_generator,
    generate_system_prompt,
)

from .executor import (
    OrchestrationExecutor,
    get_orchestration_executor,
    execute_orchestration,
)

from .service import (
    OrchestrationService,
    ProcessingContext,
    ProcessingResult,
    get_orchestration_service,
    get_orchestration_service_async,
    reset_orchestration_service,
)

from .outputs import (
    # Output types
    CodeReview,
    CodeIssue,
    Analysis,
    Finding,
    Plan,
    PlanStep,
    Comparison,
    ComparisonItem,
    QuestionAnswer,
    StructuredResponse,
    # Enums
    Severity,
    Priority,
    # Utilities
    get_default_output_type,
    get_json_schema_prompt,
    parse_structured_output,
)

from .semantic import (
    SemanticIntentMatcher,
    get_semantic_matcher,
    semantic_detect,
)

from .llm_classifier import (
    LLMIntentClassifier,
    get_llm_classifier,
    llm_classify,
)

from .exemplars import (
    IntentExemplar,
    get_all_exemplars,
)

from .meta_learning import (
    OrchestrationPattern,
    OrchestrationLearner,
    get_orchestration_learner,
    reset_orchestration_learner,
)

from .critic import (
    BranchScore,
    BranchEvaluation,
    ResponseCritic,
)

from .vector_store import (
    VectorStore,
    get_vector_store,
    reset_vector_store,
)


__all__ = [
    # Result types
    "OrchestrationMode",
    "ModelRole", 
    "DetectedIntent",
    "OrchestrationResult",
    "StreamEvent",
    
    # Intent detection (tiered)
    "IntentDetector",
    "get_intent_detector",
    "detect_intent",
    
    # Semantic matching (Layer 2)
    "SemanticIntentMatcher",
    "get_semantic_matcher",
    "semantic_detect",
    
    # LLM classification (Layer 3)
    "LLMIntentClassifier",
    "get_llm_classifier",
    "llm_classify",
    
    # Exemplars
    "IntentExemplar",
    "get_all_exemplars",
    
    # Prompt generation
    "SystemPromptGenerator",
    "get_prompt_generator",
    "generate_system_prompt",
    
    # Execution
    "OrchestrationExecutor",
    "get_orchestration_executor",
    "execute_orchestration",
    
    # Unified Service
    "OrchestrationService",
    "ProcessingContext",
    "ProcessingResult",
    "get_orchestration_service",
    "get_orchestration_service_async",
    "reset_orchestration_service",
    
    # Structured Output Types
    "CodeReview",
    "CodeIssue",
    "Analysis",
    "Finding",
    "Plan",
    "PlanStep",
    "Comparison",
    "ComparisonItem",
    "QuestionAnswer",
    "StructuredResponse",
    "Severity",
    "Priority",
    
    # Output Utilities
    "get_default_output_type",
    "get_json_schema_prompt",
    "parse_structured_output",

    # Meta-Learning (ToT + Delia)
    "OrchestrationPattern",
    "OrchestrationLearner",
    "get_orchestration_learner",
    "reset_orchestration_learner",

    # Branch Evaluation (Critic)
    "BranchScore",
    "BranchEvaluation",
    "ResponseCritic",

    # Vector Store (ChromaDB)
    "VectorStore",
    "get_vector_store",
    "reset_vector_store",
]

