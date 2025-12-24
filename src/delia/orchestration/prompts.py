# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Orchestration-specific prompt generation.

Wraps the unified prompt system with orchestration-specific logic.
All base prompts come from src/delia/prompts.py - no duplication.
"""

from __future__ import annotations

from ..prompts import (
    ModelRole,
    OrchestrationMode,
    build_system_prompt,
    ROLE_PROMPTS,
)
from ..language import detect_language
from .result import DetectedIntent


class SystemPromptGenerator:
    """
    Generates task-specific system prompts for orchestrated operations.
    
    Thin wrapper around the unified prompt system that handles
    DetectedIntent objects from the orchestration layer.
    """
    
    def generate(
        self,
        intent: DetectedIntent,
        user_message: str | None = None,
        model_name: str | None = None,
        backend_name: str | None = None,
    ) -> str:
        """
        Generate a task-specific system prompt from a DetectedIntent.
        
        Args:
            intent: The detected intent from the user message
            user_message: Optional - the original message for context
            model_name: Optional - current model for self-reference
            backend_name: Optional - current backend (unused, for API compat)
            
        Returns:
            Complete system prompt for the model
        """
        # Detect language if this is a coding task
        language = None
        if intent.task_type == "coder" and user_message:
            language = detect_language(user_message)
        
        # Map orchestration mode
        orch_mode = self._map_orchestration_mode(intent.orchestration_mode)
        
        # Use new signature for build_system_prompt
        prompt = build_system_prompt(
            role=intent.model_role,
            orchestration_mode=orch_mode.value,
            include_persona=True,
        )

        # Add additional metadata that used to be in build_system_prompt but isn't legacy/bloat
        if model_name:
            prompt += f"\n\nCurrently running as: {model_name}"
        
        if language:
            prompt += f"\nPrimary language: {language}"

        # Add voting context if applicable
        if orch_mode == OrchestrationMode.VOTING and getattr(intent, 'k_votes', None):
            prompt += f"\n\nNote: Your response will be validated for consistency. Target consensus: {intent.k_votes} matching responses needed."

        # Add Strategic Tool Guidance (Modular Injection)
        guidance = self._get_strategic_guidance(intent.model_role)
        if guidance:
            prompt += f"\n\n### STRATEGIC TOOL GUIDANCE\n{guidance}"

        return prompt

    def _get_strategic_guidance(self, role: ModelRole) -> str | None:
        """Get role-specific tool usage strategies (Professional Agent patterns)."""
        guidance = {
            ModelRole.ARCHITECT: "- Use `lsp_get_symbols` on entry points to map out the system architecture.\n- Use `find_file` to locate configuration and manifest files before planning migrations.",
            ModelRole.DEBUGGER: "- Use `lsp_find_references` to track how an error-producing variable propagates through the system.\n- Use `search_for_pattern` to find similar error-handling blocks in other modules.",
            ModelRole.CODE_REVIEWER: "- Use `lsp_get_symbols` to ensure the new code follows the structural patterns of the existing file.\n- Use `lsp_hover` to verify type safety and documentation completeness.",
            ModelRole.EXECUTOR: "- ALWAYS use `lsp_find_references` before renaming or deleting symbols.\n- Use `think_about_task_adherence()` before committing code changes.",
        }
        return guidance.get(role)
    
    def _map_orchestration_mode(self, mode) -> OrchestrationMode:
        """Map from result.OrchestrationMode to prompts.OrchestrationMode."""
        # Handle both string and enum
        mode_str = mode.value if hasattr(mode, 'value') else str(mode)
        
        # ADR-008: COMPARISON removed
        mapping = {
            "none": OrchestrationMode.NONE,
            "voting": OrchestrationMode.VOTING,
            "deep_thinking": OrchestrationMode.DEEP_THINKING,
            "batch": OrchestrationMode.BATCH,
        }
        return mapping.get(mode_str, OrchestrationMode.NONE)
    
    def get_role_description(self, role: ModelRole) -> str:
        """Get human-readable description of a role."""
        descriptions = {
            ModelRole.ASSISTANT: "General Assistant",
            ModelRole.CODE_REVIEWER: "Code Reviewer",
            ModelRole.CODE_GENERATOR: "Code Generator",
            ModelRole.ARCHITECT: "Software Architect",
            ModelRole.EXPLAINER: "Teacher/Explainer",
            ModelRole.DEBUGGER: "Debugger",
            ModelRole.ANALYST: "Technical Analyst",
            ModelRole.SUMMARIZER: "Summarizer",
        }
        return descriptions.get(role, str(role.value))


# Singleton instance
_generator: SystemPromptGenerator | None = None


def get_prompt_generator() -> SystemPromptGenerator:
    """Get the singleton prompt generator."""
    global _generator
    if _generator is None:
        _generator = SystemPromptGenerator()
    return _generator


def generate_system_prompt(intent: DetectedIntent, **kwargs) -> str:
    """Generate a system prompt from a DetectedIntent."""
    return get_prompt_generator().generate(intent, **kwargs)


# Re-export for convenience
__all__ = [
    "SystemPromptGenerator",
    "get_prompt_generator",
    "generate_system_prompt",
    "ModelRole",
    "OrchestrationMode",
]
