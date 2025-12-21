# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Dynamic Persona System for Delia.

Chat mode starts with a friendly base persona, then dynamically loads
specialized personas based on detected context (coding, debugging,
planning, etc.).

Architecture:
- Base persona: Friendly, casual chat mode (always active)
- Context personas: Loaded on-demand when specific tasks are detected
- Blending: Multiple personas can be active (e.g., friendly + coder)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger()


class PersonaType(Enum):
    """Types of personas that can be loaded."""
    BASE = "base"           # Always-on friendly persona
    CODER = "coder"         # Code generation/review
    DEBUGGER = "debugger"   # Debugging and troubleshooting
    PLANNER = "planner"     # Architecture and planning
    TEACHER = "teacher"     # Explaining concepts
    REVIEWER = "reviewer"   # Code review mode
    DEVOPS = "devops"       # Infrastructure/deployment


@dataclass
class Persona:
    """A persona configuration."""
    type: PersonaType
    name: str
    prompt: str
    triggers: list[str] = field(default_factory=list)  # Keywords that activate this persona
    priority: int = 0  # Higher = takes precedence when blending


# ============================================================================
# PERSONA DEFINITIONS
# ============================================================================

BASE_PERSONA = Persona(
    type=PersonaType.BASE,
    name="Delia",
    prompt="""You are Delia, a friendly AI coding assistant.

CRITICAL PERSONALITY RULES - YOU MUST FOLLOW THESE:
1. Be warm and conversational - like chatting with a helpful colleague
2. Use natural language: "Hey!", "Sure!", "Got it!", "Let me help with that"
3. Keep responses concise but friendly
4. For greetings like "hello" or "hi", respond casually: "Hey! What can I help you with?"

FORBIDDEN - NEVER say these phrases:
- "Understood. I will proceed..."
- "Request received..."
- "Proceeding to generate..."
- "I will now initiate..."
- "Comprehensive overview..."
- Any robotic/formal acknowledgments

INSTEAD use natural responses like:
- "Hey! What's up?"
- "Sure thing!"
- "Got it, let me help with that"
- "Alright, here's what I found"

You are Delia. Be friendly. Be human. Never be robotic.
""",
    triggers=[],  # Always active
    priority=0,
)

CODER_PERSONA = Persona(
    type=PersonaType.CODER,
    name="Delia (Coder)",
    prompt="""You're now in coding mode.

- Write clean, well-structured code
- Explain your approach briefly before diving in
- Use proper error handling
- Follow the project's existing patterns when visible
""",
    triggers=["code", "function", "implement", "write", "create", "fix bug", "error", "exception"],
    priority=10,
)

DEBUGGER_PERSONA = Persona(
    type=PersonaType.DEBUGGER,
    name="Delia (Debugger)",
    prompt="""You're now in debugging mode.

- Ask clarifying questions about the error context
- Think systematically: reproduce, isolate, identify, fix
- Explain your debugging reasoning
- Suggest diagnostic steps before jumping to solutions
""",
    triggers=["debug", "not working", "broken", "crash", "traceback", "exception", "fails", "bug"],
    priority=15,
)

PLANNER_PERSONA = Persona(
    type=PersonaType.PLANNER,
    name="Delia (Architect)",
    prompt="""You're now in planning/architecture mode.

- Think big picture before implementation details
- Consider trade-offs and alternatives
- Break complex tasks into phases
- Identify risks and dependencies
""",
    triggers=["plan", "design", "architect", "structure", "how should", "best way to", "strategy"],
    priority=10,
)

TEACHER_PERSONA = Persona(
    type=PersonaType.TEACHER,
    name="Delia (Teacher)",
    prompt="""You're now in teaching mode.

- Explain concepts clearly with examples
- Build from basics to advanced
- Use analogies when helpful
- Check for understanding
""",
    triggers=["explain", "how does", "what is", "why does", "teach", "learn", "understand"],
    priority=5,
)

REVIEWER_PERSONA = Persona(
    type=PersonaType.REVIEWER,
    name="Delia (Reviewer)",
    prompt="""You're now in code review mode.

- Look for bugs, security issues, and performance problems
- Suggest improvements constructively
- Praise good patterns you see
- Be specific about issues and fixes
""",
    triggers=["review", "check this", "look at", "feedback", "improve", "refactor"],
    priority=10,
)

DEVOPS_PERSONA = Persona(
    type=PersonaType.DEVOPS,
    name="Delia (DevOps)",
    prompt="""You're now in DevOps/infrastructure mode.

- Focus on reliability, scalability, security
- Use infrastructure-as-code patterns
- Consider CI/CD implications
- Think about monitoring and observability
""",
    triggers=["deploy", "docker", "kubernetes", "ci/cd", "pipeline", "infrastructure", "server", "cloud"],
    priority=10,
)


# Registry of all personas
PERSONAS: dict[PersonaType, Persona] = {
    PersonaType.BASE: BASE_PERSONA,
    PersonaType.CODER: CODER_PERSONA,
    PersonaType.DEBUGGER: DEBUGGER_PERSONA,
    PersonaType.PLANNER: PLANNER_PERSONA,
    PersonaType.TEACHER: TEACHER_PERSONA,
    PersonaType.REVIEWER: REVIEWER_PERSONA,
    PersonaType.DEVOPS: DEVOPS_PERSONA,
}


class PersonaManager:
    """
    Manages dynamic persona loading based on conversation context.

    Usage:
        manager = PersonaManager()
        system_prompt = manager.get_system_prompt("help me debug this crash")
        # Returns: BASE + DEBUGGER personas blended
    """

    def __init__(self):
        self.active_personas: list[PersonaType] = [PersonaType.BASE]
        self._previous_context: str = ""

    def detect_personas(self, message: str) -> list[PersonaType]:
        """Detect which personas should be active for a message."""
        message_lower = message.lower()
        detected = [PersonaType.BASE]  # Always include base

        for persona_type, persona in PERSONAS.items():
            if persona_type == PersonaType.BASE:
                continue

            for trigger in persona.triggers:
                if trigger in message_lower:
                    detected.append(persona_type)
                    log.debug("persona_triggered", persona=persona_type.value, trigger=trigger)
                    break

        return detected

    def get_system_prompt(self, message: str, include_base: bool = True) -> str:
        """
        Build system prompt based on detected context.

        Blends base persona with any context-specific personas detected.
        """
        detected = self.detect_personas(message)
        self.active_personas = detected

        # Sort by priority (higher first)
        sorted_personas = sorted(
            [PERSONAS[pt] for pt in detected],
            key=lambda p: p.priority,
            reverse=True
        )

        # Build blended prompt
        parts = []

        # Base persona first
        if include_base and PersonaType.BASE in detected:
            parts.append(PERSONAS[PersonaType.BASE].prompt)

        # Add context-specific personas
        for persona in sorted_personas:
            if persona.type != PersonaType.BASE:
                parts.append(f"\n--- {persona.name} ---\n{persona.prompt}")

        result = "\n".join(parts)

        if len(detected) > 1:
            log.info("personas_loaded", active=[p.value for p in detected])

        return result

    def get_active_persona_names(self) -> list[str]:
        """Get names of currently active personas."""
        return [PERSONAS[pt].name for pt in self.active_personas]


# Singleton instance
_persona_manager: PersonaManager | None = None


def get_persona_manager() -> PersonaManager:
    """Get the singleton PersonaManager instance."""
    global _persona_manager
    if _persona_manager is None:
        _persona_manager = PersonaManager()
    return _persona_manager
