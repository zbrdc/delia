# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Model Tuning Advisor.

Determines optimal inference parameters (temperature, context size, max_tokens)
based on task type, stakes assessment, content characteristics, and model family.

This runs AFTER the dispatcher selects a tier, providing fine-grained control
without burdening the small dispatcher model with parameter decisions.

Supports model-specific quirks for:
- DeepSeek (R1, Coder, V2/V3)
- Qwen (Qwen2.5, Qwen3, QwQ)
- Llama (Llama 3.x, Code Llama)
- Mistral/Mixtral
- Gemma/CodeGemma
- Phi (Phi-3, Phi-4)
- StarCoder/DeepSeek-Coder
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .stakes import StakesAssessment

log = structlog.get_logger()


class ModelFamily(Enum):
    """Known model families with specific tuning requirements."""
    DEEPSEEK = "deepseek"
    DEEPSEEK_CODER = "deepseek-coder"
    DEEPSEEK_R1 = "deepseek-r1"
    QWEN = "qwen"
    QWEN3 = "qwen3"
    QWQ = "qwq"
    LLAMA = "llama"
    CODE_LLAMA = "codellama"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    GEMMA = "gemma"
    CODE_GEMMA = "codegemma"
    PHI = "phi"
    STARCODER = "starcoder"
    NEMOTRON = "nemotron"
    COMMAND_R = "command-r"
    YI = "yi"
    INTERNLM = "internlm"
    UNKNOWN = "unknown"


@dataclass
class ModelProfile:
    """Tuning profile for a model family."""
    family: ModelFamily

    # Temperature bounds
    temp_min: float = 0.1
    temp_max: float = 1.5
    temp_default: float = 0.7

    # Sampling parameters
    top_p_default: float | None = None
    top_k_default: int | None = None
    min_p_default: float | None = None

    # Repetition control
    repeat_penalty_default: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    # Context and output
    preferred_ctx: int = 8192
    max_ctx: int = 32768
    thinking_token_budget: int = 4096  # For thinking models

    # Model quirks
    prefers_low_temp_for_code: bool = False
    supports_extended_thinking: bool = False
    needs_explicit_think_tags: bool = False
    strip_thinking_from_output: bool = True

    # Special handling
    notes: str = ""


# Model family profiles with empirically tuned defaults
MODEL_PROFILES: dict[ModelFamily, ModelProfile] = {
    ModelFamily.DEEPSEEK_R1: ModelProfile(
        family=ModelFamily.DEEPSEEK_R1,
        temp_min=0.0,
        temp_max=1.0,
        temp_default=0.6,
        top_p_default=0.95,
        repeat_penalty_default=1.05,
        preferred_ctx=16384,
        max_ctx=65536,
        thinking_token_budget=8192,
        supports_extended_thinking=True,
        needs_explicit_think_tags=False,  # R1 thinks automatically
        notes="DeepSeek-R1 has native CoT. Lower temp for reasoning coherence.",
    ),
    ModelFamily.DEEPSEEK_CODER: ModelProfile(
        family=ModelFamily.DEEPSEEK_CODER,
        temp_min=0.0,
        temp_max=1.2,
        temp_default=0.4,
        top_p_default=0.9,
        top_k_default=50,
        repeat_penalty_default=1.1,
        preferred_ctx=16384,
        max_ctx=65536,
        prefers_low_temp_for_code=True,
        notes="DeepSeek-Coder excels at low temps. Use 0.0-0.3 for precise code.",
    ),
    ModelFamily.DEEPSEEK: ModelProfile(
        family=ModelFamily.DEEPSEEK,
        temp_min=0.1,
        temp_max=1.3,
        temp_default=0.7,
        top_p_default=0.9,
        repeat_penalty_default=1.05,
        preferred_ctx=8192,
        max_ctx=32768,
        notes="General DeepSeek. Solid all-rounder.",
    ),
    ModelFamily.QWEN3: ModelProfile(
        family=ModelFamily.QWEN3,
        temp_min=0.0,
        temp_max=1.5,
        temp_default=0.7,
        top_p_default=0.8,
        top_k_default=20,
        min_p_default=0.05,
        repeat_penalty_default=1.05,
        preferred_ctx=32768,
        max_ctx=131072,
        thinking_token_budget=8192,
        supports_extended_thinking=True,
        needs_explicit_think_tags=True,  # Qwen3 uses /think command
        notes="Qwen3 supports /think for extended reasoning. Use top_k=20, min_p=0.05.",
    ),
    ModelFamily.QWQ: ModelProfile(
        family=ModelFamily.QWQ,
        temp_min=0.0,
        temp_max=1.0,
        temp_default=0.6,
        top_p_default=0.9,
        repeat_penalty_default=1.0,
        preferred_ctx=32768,
        max_ctx=131072,
        thinking_token_budget=16384,
        supports_extended_thinking=True,
        notes="QwQ is a reasoning model. Keep temp low for coherent chains.",
    ),
    ModelFamily.QWEN: ModelProfile(
        family=ModelFamily.QWEN,
        temp_min=0.1,
        temp_max=1.5,
        temp_default=0.7,
        top_p_default=0.8,
        top_k_default=40,
        repeat_penalty_default=1.05,
        preferred_ctx=8192,
        max_ctx=32768,
        notes="Qwen2.5 series. Balanced defaults.",
    ),
    ModelFamily.LLAMA: ModelProfile(
        family=ModelFamily.LLAMA,
        temp_min=0.1,
        temp_max=2.0,
        temp_default=0.8,
        top_p_default=0.9,
        top_k_default=40,
        repeat_penalty_default=1.1,
        preferred_ctx=8192,
        max_ctx=131072,
        notes="Llama 3.x. Handles higher temps well for creative tasks.",
    ),
    ModelFamily.CODE_LLAMA: ModelProfile(
        family=ModelFamily.CODE_LLAMA,
        temp_min=0.0,
        temp_max=1.0,
        temp_default=0.3,
        top_p_default=0.95,
        repeat_penalty_default=1.15,
        preferred_ctx=16384,
        max_ctx=100000,
        prefers_low_temp_for_code=True,
        notes="Code Llama. Very low temp (0.1-0.3) for code generation.",
    ),
    ModelFamily.MISTRAL: ModelProfile(
        family=ModelFamily.MISTRAL,
        temp_min=0.1,
        temp_max=1.5,
        temp_default=0.7,
        top_p_default=0.9,
        repeat_penalty_default=1.1,
        preferred_ctx=8192,
        max_ctx=32768,
        notes="Mistral. Good general-purpose defaults.",
    ),
    ModelFamily.MIXTRAL: ModelProfile(
        family=ModelFamily.MIXTRAL,
        temp_min=0.1,
        temp_max=1.5,
        temp_default=0.7,
        top_p_default=0.9,
        repeat_penalty_default=1.05,
        preferred_ctx=16384,
        max_ctx=32768,
        notes="Mixtral MoE. Slightly lower repeat penalty than Mistral.",
    ),
    ModelFamily.GEMMA: ModelProfile(
        family=ModelFamily.GEMMA,
        temp_min=0.1,
        temp_max=1.5,
        temp_default=0.8,
        top_p_default=0.95,
        top_k_default=64,
        repeat_penalty_default=1.0,
        preferred_ctx=8192,
        max_ctx=8192,
        notes="Gemma. Higher top_k works well.",
    ),
    ModelFamily.CODE_GEMMA: ModelProfile(
        family=ModelFamily.CODE_GEMMA,
        temp_min=0.0,
        temp_max=1.0,
        temp_default=0.3,
        top_p_default=0.9,
        repeat_penalty_default=1.1,
        preferred_ctx=8192,
        max_ctx=8192,
        prefers_low_temp_for_code=True,
        notes="CodeGemma. Low temp for code tasks.",
    ),
    ModelFamily.PHI: ModelProfile(
        family=ModelFamily.PHI,
        temp_min=0.0,
        temp_max=1.2,
        temp_default=0.6,
        top_p_default=0.9,
        repeat_penalty_default=1.1,
        preferred_ctx=4096,
        max_ctx=16384,
        notes="Phi-3/4. Works well at moderate temps.",
    ),
    ModelFamily.STARCODER: ModelProfile(
        family=ModelFamily.STARCODER,
        temp_min=0.0,
        temp_max=1.0,
        temp_default=0.2,
        top_p_default=0.95,
        repeat_penalty_default=1.2,
        preferred_ctx=8192,
        max_ctx=16384,
        prefers_low_temp_for_code=True,
        notes="StarCoder. Very low temp, high repeat penalty for clean code.",
    ),
    ModelFamily.NEMOTRON: ModelProfile(
        family=ModelFamily.NEMOTRON,
        temp_min=0.1,
        temp_max=1.0,
        temp_default=0.5,
        top_p_default=0.9,
        repeat_penalty_default=1.05,
        preferred_ctx=8192,
        max_ctx=32768,
        supports_extended_thinking=True,
        notes="Nemotron. Moderate temp, good for instruction following.",
    ),
    ModelFamily.COMMAND_R: ModelProfile(
        family=ModelFamily.COMMAND_R,
        temp_min=0.1,
        temp_max=1.5,
        temp_default=0.7,
        top_p_default=0.9,
        repeat_penalty_default=1.0,
        preferred_ctx=16384,
        max_ctx=131072,
        notes="Command-R. Long context, good RAG performance.",
    ),
    ModelFamily.YI: ModelProfile(
        family=ModelFamily.YI,
        temp_min=0.1,
        temp_max=1.5,
        temp_default=0.7,
        top_p_default=0.9,
        repeat_penalty_default=1.05,
        preferred_ctx=8192,
        max_ctx=32768,
        notes="Yi series. Standard defaults.",
    ),
    ModelFamily.INTERNLM: ModelProfile(
        family=ModelFamily.INTERNLM,
        temp_min=0.1,
        temp_max=1.5,
        temp_default=0.7,
        top_p_default=0.9,
        repeat_penalty_default=1.05,
        preferred_ctx=8192,
        max_ctx=32768,
        notes="InternLM. Standard defaults.",
    ),
}

# Default profile for unknown models
DEFAULT_PROFILE = ModelProfile(
    family=ModelFamily.UNKNOWN,
    temp_min=0.1,
    temp_max=1.5,
    temp_default=0.7,
    top_p_default=0.9,
    repeat_penalty_default=1.05,
    preferred_ctx=8192,
    max_ctx=32768,
    notes="Unknown model. Using conservative defaults.",
)


def detect_model_family(model_name: str) -> ModelFamily:
    """Detect model family from model name string."""
    name = model_name.lower()

    # Order matters - check more specific patterns first
    if "deepseek-r1" in name or "deepseek:r1" in name or "r1-" in name:
        return ModelFamily.DEEPSEEK_R1
    if "deepseek-coder" in name or "deepseek:coder" in name:
        return ModelFamily.DEEPSEEK_CODER
    if "deepseek" in name:
        return ModelFamily.DEEPSEEK

    if "qwq" in name:
        return ModelFamily.QWQ
    if "qwen3" in name or "qwen:3" in name or "qwen2.5" in name:
        return ModelFamily.QWEN3
    if "qwen" in name:
        return ModelFamily.QWEN

    if "codellama" in name or "code-llama" in name:
        return ModelFamily.CODE_LLAMA
    if "llama" in name:
        return ModelFamily.LLAMA

    if "mixtral" in name:
        return ModelFamily.MIXTRAL
    if "mistral" in name:
        return ModelFamily.MISTRAL

    if "codegemma" in name or "code-gemma" in name:
        return ModelFamily.CODE_GEMMA
    if "gemma" in name:
        return ModelFamily.GEMMA

    if "phi" in name:
        return ModelFamily.PHI

    if "starcoder" in name:
        return ModelFamily.STARCODER

    if "nemotron" in name:
        return ModelFamily.NEMOTRON

    if "command-r" in name or "command_r" in name:
        return ModelFamily.COMMAND_R

    if "yi-" in name or "yi:" in name:
        return ModelFamily.YI

    if "internlm" in name:
        return ModelFamily.INTERNLM

    return ModelFamily.UNKNOWN


def get_model_profile(model_name: str) -> ModelProfile:
    """Get the tuning profile for a model."""
    family = detect_model_family(model_name)
    return MODEL_PROFILES.get(family, DEFAULT_PROFILE)


@dataclass
class TuningParameters:
    """Recommended inference parameters for a request."""

    temperature: float = 0.7
    max_tokens: int | None = None
    num_ctx: int | None = None  # Context window size
    top_p: float | None = None  # Nucleus sampling
    top_k: int | None = None  # Top-k sampling
    min_p: float | None = None  # Min-p sampling (Qwen3, etc.)
    repeat_penalty: float | None = None  # Repetition penalty
    frequency_penalty: float | None = None  # Token frequency penalty
    presence_penalty: float | None = None  # Token presence penalty

    # Model-specific hints
    model_family: ModelFamily | None = None
    enable_thinking: bool = False
    thinking_budget: int | None = None

    # Metadata about why these were chosen
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict for passing to LLM call."""
        result = {"temperature": self.temperature}
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.num_ctx is not None:
            result["num_ctx"] = self.num_ctx
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.top_k is not None:
            result["top_k"] = self.top_k
        if self.min_p is not None:
            result["min_p"] = self.min_p
        if self.repeat_penalty is not None:
            result["repeat_penalty"] = self.repeat_penalty
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty
        return result

    def to_ollama_options(self) -> dict:
        """Convert to Ollama options format."""
        options = {"temperature": self.temperature}
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.min_p is not None:
            options["min_p"] = self.min_p
        if self.repeat_penalty is not None:
            options["repeat_penalty"] = self.repeat_penalty
        if self.frequency_penalty is not None:
            options["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            options["presence_penalty"] = self.presence_penalty
        return options


# Task type to base temperature mapping
TASK_TEMPERATURES = {
    # Low temperature for precision tasks
    "quick": 0.3,
    "summarize": 0.4,
    "review": 0.4,
    "analyze": 0.5,

    # Medium temperature for balanced tasks
    "generate": 0.7,
    "plan": 0.6,
    "critique": 0.5,

    # Higher temperature for creative tasks
    "brainstorm": 0.9,
    "creative": 0.95,
}

# Task type to recommended max tokens
TASK_MAX_TOKENS = {
    "quick": 512,
    "summarize": 1024,
    "review": 2048,
    "analyze": 2048,
    "generate": 4096,
    "plan": 4096,
    "critique": 2048,
}

# Stakes-based temperature adjustments
STAKES_TEMPERATURE_MODIFIERS = {
    "critical": -0.3,  # High stakes → lower temp for precision
    "high": -0.2,
    "moderate": -0.1,
    "low": 0.0,
    "minimal": 0.1,  # Low stakes → can be more creative
}


class TuningAdvisor:
    """
    Determines optimal model tuning parameters.

    Uses a rule-based system with adjustments for:
    - Task type (generation vs analysis vs summary)
    - Stakes assessment (high stakes → precision mode)
    - Content length (longer content → larger context)
    - Orchestration mode (thinking → extended output)
    - Model family (DeepSeek, Qwen, Llama quirks)
    """

    def __init__(self):
        self.default_temperature = 0.7
        self.default_max_tokens = 2048

    def advise(
        self,
        task_type: str,
        content_length: int = 0,
        stakes: "StakesAssessment | None" = None,
        orchestration_mode: str = "none",
        tier: str = "coder",
        model_name: str | None = None,
    ) -> TuningParameters:
        """
        Get recommended tuning parameters for a request.

        Args:
            task_type: Type of task (quick, generate, review, etc.)
            content_length: Length of input content in characters
            stakes: Optional stakes assessment from StakesAnalyzer
            orchestration_mode: Current orchestration mode
            tier: Selected model tier (quick, coder, moe, thinking)
            model_name: Optional model name for family-specific tuning

        Returns:
            TuningParameters with recommended settings
        """
        params = TuningParameters()

        # 0. Get model profile if model name provided
        profile: ModelProfile | None = None
        if model_name:
            profile = get_model_profile(model_name)
            params.model_family = profile.family
            params.reasoning.append(f"Model family: {profile.family.value}")

        # 1. Base temperature from task type
        base_temp = TASK_TEMPERATURES.get(task_type.lower(), self.default_temperature)

        # Apply model-specific temperature default if available
        if profile:
            # Use model's default as starting point, but respect task requirements
            if task_type.lower() in ("review", "analyze") and profile.prefers_low_temp_for_code:
                base_temp = min(base_temp, profile.temp_default * 0.7)
                params.reasoning.append(f"Model prefers low temp for code: {base_temp:.2f}")
            else:
                # Blend task temp with model default (70% task, 30% model)
                base_temp = base_temp * 0.7 + profile.temp_default * 0.3

        params.temperature = base_temp
        params.reasoning.append(f"Base temp {base_temp:.2f} for task={task_type}")

        # 2. Adjust for stakes
        if stakes:
            stakes_level = self._stakes_to_level(stakes.score)
            modifier = STAKES_TEMPERATURE_MODIFIERS.get(stakes_level, 0.0)
            params.temperature = max(0.1, min(1.5, params.temperature + modifier))
            if modifier != 0:
                params.reasoning.append(f"Stakes adjustment {modifier:+.1f} for {stakes_level} stakes")

        # 3. Adjust for tier and model capabilities
        if tier == "thinking":
            # Check if model supports extended thinking
            if profile and profile.supports_extended_thinking:
                params.enable_thinking = True
                params.thinking_budget = profile.thinking_token_budget
                params.reasoning.append(f"Thinking enabled, budget={profile.thinking_token_budget}")

            # Thinking models need lower temp for coherent extended reasoning
            max_thinking_temp = profile.temp_max if profile else 0.6
            params.temperature = min(params.temperature, max_thinking_temp * 0.8)
            params.reasoning.append(f"Capped temp at {params.temperature:.2f} for thinking tier")

        elif tier == "moe":
            # MoE models handle slightly higher temps well
            params.temperature = min(params.temperature + 0.1, 1.0)
            params.reasoning.append("Slight temp boost for MoE tier")

        # 4. Enforce model temperature bounds
        if profile:
            params.temperature = max(profile.temp_min, min(profile.temp_max, params.temperature))

        # 5. Set max tokens based on task and mode
        base_max_tokens = TASK_MAX_TOKENS.get(task_type.lower(), self.default_max_tokens)

        if orchestration_mode in ("deep_thinking", "tree_of_thoughts"):
            # Extended reasoning needs more tokens
            if profile and profile.supports_extended_thinking:
                params.max_tokens = max(base_max_tokens * 2, profile.thinking_token_budget)
            else:
                params.max_tokens = base_max_tokens * 2
            params.reasoning.append(f"Extended max_tokens={params.max_tokens} for {orchestration_mode}")
        elif orchestration_mode == "voting":
            # Voting responses should be concise
            params.max_tokens = min(base_max_tokens, 1024)
            params.reasoning.append("Capped max_tokens for voting mode")
        else:
            params.max_tokens = base_max_tokens

        # 6. Set context size based on content length and model limits
        if content_length > 0:
            # Rule: context should be ~3x content length for adequate processing
            estimated_tokens = content_length // 4  # rough char to token ratio
            recommended_ctx = max(4096, estimated_tokens * 3)

            # Respect model's max context
            max_ctx = profile.max_ctx if profile else 32768
            recommended_ctx = min(recommended_ctx, max_ctx)

            # Round to nearest 2048
            params.num_ctx = ((recommended_ctx + 1024) // 2048) * 2048
            params.reasoning.append(f"Context size {params.num_ctx} for {content_length} chars")

        # 7. Apply model-specific sampling parameters
        if profile:
            # Use model's preferred sampling parameters
            if profile.top_p_default is not None:
                params.top_p = profile.top_p_default
            if profile.top_k_default is not None:
                params.top_k = profile.top_k_default
            if profile.min_p_default is not None:
                params.min_p = profile.min_p_default
            if profile.repeat_penalty_default is not None:
                params.repeat_penalty = profile.repeat_penalty_default
            if profile.frequency_penalty is not None:
                params.frequency_penalty = profile.frequency_penalty
            if profile.presence_penalty is not None:
                params.presence_penalty = profile.presence_penalty

            params.reasoning.append(f"Applied {profile.family.value} sampling defaults")

        # 8. Task-specific overrides
        if task_type.lower() == "review":
            # Code review needs precision - increase repeat penalty
            if params.repeat_penalty is None or params.repeat_penalty < 1.1:
                params.repeat_penalty = 1.1
                params.reasoning.append("Boosted repeat_penalty for review precision")

        if task_type.lower() in ("generate", "plan"):
            # Generation benefits from nucleus sampling if not already set
            if params.top_p is None:
                params.top_p = 0.9
                params.reasoning.append("Added top_p=0.9 for generation diversity")

        # 9. Handle code-focused models for code tasks
        is_code_task = task_type.lower() in ("generate", "review", "analyze")
        if profile and profile.prefers_low_temp_for_code and is_code_task:
            # Further reduce temp for code tasks on code-specialized models
            params.temperature = min(params.temperature, 0.4)
            params.reasoning.append("Low temp for code-focused model on code task")

        log.debug(
            "tuning_advice",
            task_type=task_type,
            tier=tier,
            model_family=profile.family.value if profile else "unknown",
            temperature=round(params.temperature, 3),
            max_tokens=params.max_tokens,
            num_ctx=params.num_ctx,
            top_p=params.top_p,
            top_k=params.top_k,
            min_p=params.min_p,
            repeat_penalty=params.repeat_penalty,
            reasoning_count=len(params.reasoning),
        )

        return params

    def _stakes_to_level(self, score: float) -> str:
        """Convert numeric stakes score to level string."""
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "moderate"
        elif score >= 0.3:
            return "low"
        return "minimal"

    def advise_for_dispatcher(self) -> TuningParameters:
        """
        Get tuning parameters optimized for the dispatcher model.

        The dispatcher needs:
        - Low temperature for consistent tool selection
        - Minimal tokens (just needs to output tool call)
        - Small context (dispatcher prompts are short)
        """
        return TuningParameters(
            temperature=0.1,  # Very low for deterministic routing
            max_tokens=256,   # Only needs to output tool call
            num_ctx=2048,     # Minimal context for dispatcher
            reasoning=["Dispatcher-optimized: low temp, minimal tokens"],
        )

    def advise_for_embeddings(self) -> dict:
        """
        Get settings for embedding model.

        Returns dict with model configuration hints.
        """
        return {
            "batch_size": 32,
            "normalize": True,
            "force_cpu": True,  # Preserve VRAM
        }


# Singleton instance
_tuning_advisor: TuningAdvisor | None = None


def get_tuning_advisor() -> TuningAdvisor:
    """Get the singleton TuningAdvisor instance."""
    global _tuning_advisor
    if _tuning_advisor is None:
        _tuning_advisor = TuningAdvisor()
    return _tuning_advisor


def advise_tuning(
    task_type: str,
    content_length: int = 0,
    stakes_score: float | None = None,
    orchestration_mode: str = "none",
    tier: str = "coder",
) -> TuningParameters:
    """
    Convenience function to get tuning advice.

    Args:
        task_type: Type of task
        content_length: Length of input content
        stakes_score: Optional stakes score (0.0-1.0)
        orchestration_mode: Current orchestration mode
        tier: Selected model tier

    Returns:
        TuningParameters with recommended settings
    """
    advisor = get_tuning_advisor()

    # Create minimal stakes assessment if score provided
    stakes = None
    if stakes_score is not None:
        from .stakes import StakesAssessment
        stakes = StakesAssessment(score=stakes_score, is_high_stakes=stakes_score >= 0.7)

    return advisor.advise(
        task_type=task_type,
        content_length=content_length,
        stakes=stakes,
        orchestration_mode=orchestration_mode,
        tier=tier,
    )
