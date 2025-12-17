# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Orchestration result types.

Defines the data structures for orchestration outcomes, ensuring
clean separation between orchestration decisions and execution.

Note: ModelRole and OrchestrationMode are imported from the unified
prompts module to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

# Import from unified prompt system - single source of truth
from ..prompts import ModelRole, OrchestrationMode

# Type variable for structured outputs
T = TypeVar("T", bound=BaseModel)


@dataclass
class DetectedIntent:
    """Result of NLP intent detection.
    
    This is the key data structure that bridges user input
    to Delia's orchestration decisions.
    """
    # What kind of task is this?
    task_type: str  # quick, coder, moe, thinking
    
    # How should Delia orchestrate it?
    orchestration_mode: OrchestrationMode = OrchestrationMode.NONE
    
    # What role should the model play?
    model_role: ModelRole = ModelRole.ASSISTANT
    
    # How confident are we in this detection?
    confidence: float = 0.5
    
    # Why did we choose this?
    reasoning: str = ""
    
    # For voting: how many votes needed?
    k_votes: int = 3
    
    # For comparison: which models?
    comparison_models: list[str] = field(default_factory=list)
    
    # For chain: sequential steps to execute
    chain_steps: list[str] = field(default_factory=list)
    
    # Is there code in the message?
    contains_code: bool = False
    
    # Extracted keywords that influenced the decision
    trigger_keywords: list[str] = field(default_factory=list)


@dataclass  
class OrchestrationResult:
    """Result of orchestration execution.
    
    Contains the final response plus metadata about
    how it was produced (for logging, learning, UI).
    
    When output_type is specified, the `structured` field will
    contain the parsed Pydantic model instance.
    """
    # The response to show the user (raw text)
    response: str
    
    # Was orchestration successful?
    success: bool = True
    
    # Which model(s) produced this?
    model_used: str = ""
    models_compared: list[str] = field(default_factory=list)
    
    # Which backend was used?
    backend_used: str | None = None
    
    # What orchestration mode was used?
    mode: OrchestrationMode = OrchestrationMode.NONE
    
    # For voting: consensus details
    votes_cast: int = 0
    consensus_reached: bool = False
    confidence: float = 0.0
    
    # Timing
    elapsed_ms: int = 0
    
    # Quality score (for melon rewards)
    quality_score: float = 0.5
    
    # Structured output (when output_type is specified)
    # This will be a Pydantic model instance
    structured: BaseModel | None = None
    
    # Was structured parsing successful?
    structured_success: bool = True
    structured_error: str | None = None
    
    # Errors if any
    error: str | None = None
    
    # Debug info for logging
    debug_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """SSE event for streaming orchestration progress.
    
    Allows the UI to show real-time feedback about
    what Delia is doing (without showing raw tools).
    """
    event_type: str  # status, thinking, progress, response, done, error
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_sse(self) -> str:
        """Format as SSE event."""
        import json
        data = {"message": self.message, **self.details}
        return f"event: {self.event_type}\ndata: {json.dumps(data)}\n\n"


# Re-export for convenience
__all__ = [
    "ModelRole",
    "OrchestrationMode", 
    "DetectedIntent",
    "OrchestrationResult",
    "StreamEvent",
]
