# Copyright (C) 2023 the project owner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Content detection and routing utilities for Delia."""

import re
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from .backend_manager import BackendManager
    from .config import Config

log = structlog.get_logger()


# Code indicators with weights for confidence scoring
# Pre-compiled regex patterns for performance (avoids recompilation on each call)
CODE_INDICATORS = {
    # Strong indicators (weight 3) - almost certainly code
    "strong": [
        re.compile(r"\bdef\s+\w+\s*\(", re.MULTILINE),  # Python function
        re.compile(r"\bclass\s+\w+[\s:(]", re.MULTILINE),  # Class definition
        re.compile(r"\bimport\s+\w+", re.MULTILINE),  # Import statement
        re.compile(r"\bfrom\s+\w+\s+import", re.MULTILINE),  # From import
        re.compile(r"\bfunction\s+\w+\s*\(", re.MULTILINE),  # JS function
        re.compile(r"\bconst\s+\w+\s*=", re.MULTILINE),  # JS const
        re.compile(r"\blet\s+\w+\s*=", re.MULTILINE),  # JS let
        re.compile(r"\bexport\s+(default\s+)?", re.MULTILINE),  # JS export
        re.compile(r"^\s*@\w+", re.MULTILINE),  # Decorator
        re.compile(r"\basync\s+(def|function)", re.MULTILINE),  # Async
        re.compile(r"\bawait\s+\w+", re.MULTILINE),  # Await
        re.compile(r"\breturn\s+[\w{(\[]", re.MULTILINE),  # Return statement
        re.compile(r"if\s*\(.+\)\s*{", re.MULTILINE),  # C-style if
        re.compile(r"for\s*\(.+\)\s*{", re.MULTILINE),  # C-style for
        re.compile(r"\bwhile\s*\(.+\)", re.MULTILINE),  # While loop
        re.compile(r"\btry\s*[:{]", re.MULTILINE),  # Try block
        re.compile(r"\bcatch\s*\(", re.MULTILINE),  # Catch block
        re.compile(r"\bexcept\s+\w*:", re.MULTILINE),  # Python except
        re.compile(r"=>\s*{", re.MULTILINE),  # Arrow function
        re.compile(r"\.map\(|\.filter\(|\.reduce\(", re.MULTILINE),  # Array methods
    ],
    # Medium indicators (weight 2) - likely code
    "medium": [
        re.compile(r"\bself\.", re.MULTILINE),  # Python self
        re.compile(r"\bthis\.", re.MULTILINE),  # JS this
        re.compile(r"===|!==", re.MULTILINE),  # Strict equality
        re.compile(r"&&|\|\|", re.MULTILINE),  # Logical operators
        re.compile(r"\bnull\b|\bundefined\b", re.MULTILINE),  # Null/undefined
        re.compile(r"\bTrue\b|\bFalse\b|\bNone\b", re.MULTILINE),  # Python booleans
        re.compile(r":\s*\w+\s*[,)\]]", re.MULTILINE),  # Type annotations
        re.compile(r"\[\w+\]", re.MULTILINE),  # Array indexing
        re.compile(r"\{\s*\w+:\s*", re.MULTILINE),  # Object literal
        re.compile(r"console\.|print\(|logger\.", re.MULTILINE),  # Logging
        re.compile(r"\braise\s+\w+", re.MULTILINE),  # Python raise
        re.compile(r"\bthrow\s+new", re.MULTILINE),  # JS throw
        re.compile(r"`[^`]+\$\{", re.MULTILINE),  # Template literal
        re.compile(r'f"[^"]*\{', re.MULTILINE),  # Python f-string
    ],
    # Weak indicators (weight 1) - could be code
    "weak": [
        re.compile(r";$", re.MULTILINE),  # Semicolon ending
        re.compile(r"\{|\}", re.MULTILINE),  # Braces
        re.compile(r"\[|\]", re.MULTILINE),  # Brackets
        re.compile(r"==|!=", re.MULTILINE),  # Equality
        re.compile(r"->", re.MULTILINE),  # Arrow (type hints, etc)
        re.compile(r"\bint\b|\bstr\b|\bbool\b", re.MULTILINE),  # Type names
        re.compile(r"\bvar\b", re.MULTILINE),  # Var keyword
    ],
}


def detect_code_content(content: str) -> tuple[bool, float, str]:
    """
    Detect if content is primarily code or text.

    Returns:
        (is_code, confidence, reasoning)
        - is_code: True if content appears to be code
        - confidence: 0.0-1.0 score
        - reasoning: Brief explanation
    """
    if not content or len(content.strip()) < 20:
        return False, 0.0, "Content too short"

    lines = content.strip().split("\n")

    # Count code indicators
    strong_matches = 0
    medium_matches = 0
    weak_matches = 0

    for pattern in CODE_INDICATORS["strong"]:
        matches = len(pattern.findall(content))  # Use pre-compiled pattern
        strong_matches += min(matches, 5)  # Cap per pattern

    for pattern in CODE_INDICATORS["medium"]:
        matches = len(pattern.findall(content))  # Use pre-compiled pattern
        medium_matches += min(matches, 5)

    for pattern in CODE_INDICATORS["weak"]:
        matches = len(pattern.findall(content))  # Use pre-compiled pattern
        weak_matches += min(matches, 5)

    # Weighted score
    score = strong_matches * 3 + medium_matches * 2 + weak_matches * 1

    # Normalize by content length (per 1000 chars)
    normalized = score / max(1, len(content) / 1000)

    # Additional heuristics
    avg_line_length = sum(len(line) for line in lines) / max(1, len(lines))
    indent_lines = sum(1 for line in lines if line.startswith("  ") or line.startswith("\t"))
    indent_ratio = indent_lines / max(1, len(lines))

    # Adjust score based on structure
    if indent_ratio > 0.3:  # Lots of indentation = code
        normalized *= 1.3
    if avg_line_length < 100:  # Code lines tend to be shorter
        normalized *= 1.1

    # Determine threshold
    if normalized > 3.0:
        return True, min(1.0, normalized / 5), f"Strong code signals (score={normalized:.1f})"
    elif normalized > 1.5:
        return True, normalized / 4, f"Likely code (score={normalized:.1f})"
    elif normalized > 0.8:
        return False, 0.4, f"Mixed content (score={normalized:.1f})"
    else:
        return False, max(0, 0.3 - normalized / 3), f"Primarily text (score={normalized:.1f})"


def parse_model_override(model_hint: str | None, content: str) -> str | None:
    """Parse explicit model request from content or hint.

    Recognizes tier keywords (moe, coder, quick, thinking) and natural language model references.
    Supports size-based hints (7b, 14b, 30b), capability hints (coder, reasoning), and descriptive terms.
    """
    # Check explicit hint first (tier names and natural language)
    if model_hint:
        hint = model_hint.lower().strip()

        # Direct tier names
        if "moe" in hint or "30b" in hint or "large" in hint or "complex" in hint or "reasoning" in hint:
            return "moe"
        if "coder" in hint or "code" in hint or "programming" in hint or "14b" in hint:
            return "coder"
        if "quick" in hint or "7b" in hint or "small" in hint or "fast" in hint:
            return "quick"
        if "thinking" in hint or "think" in hint or "chain" in hint:
            return "thinking"

        # Pass through other hints as-is (might be specific model name)
        return model_hint

    # Scan content for tier keywords and natural language using word boundaries
    if content:
        content_lower = content.lower()

        # Size and capability-based patterns
        if re.search(r"\b(30b|large|big|complex|reasoning|deep|planning|critique)\b", content_lower) or re.search(
            r"\bmoe\b", content_lower
        ):
            log.info("model_override_detected", tier="moe", source="content", pattern="size/capability")
            return "moe"

        if re.search(r"\b(14b|coder|code|programming|development|review|analyze)\b", content_lower):
            log.info("model_override_detected", tier="coder", source="content", pattern="size/capability")
            return "coder"

        if re.search(r"\b(7b|small|fast|quick|simple|basic|summarize)\b", content_lower):
            log.info("model_override_detected", tier="quick", source="content", pattern="size/capability")
            return "quick"

        if re.search(r"\b(thinking|think|chain|reason|step.*by.*step)\b", content_lower):
            log.info("model_override_detected", tier="thinking", source="content", pattern="capability")
            return "thinking"

    return None


class ModelRouter:
    """Intelligent model and backend selection for Delia."""

    def __init__(self, config: "Config", backend_manager: "BackendManager"):
        """Initialize the model router with configuration and backend manager.

        Args:
            config: Configuration object with model tiers and task mappings
            backend_manager: Backend manager for retrieving active/enabled backends
        """
        self.config = config
        self.backend_manager = backend_manager

    async def select_model(
        self, task_type: str, content_size: int = 0, model_override: str | None = None, content: str = ""
    ) -> str:
        """
        Select the best model for the task with intelligent code-aware routing.

        Tiers (configured in settings.json):
        - quick: Fast general tasks, text analysis, summarize
        - coder: Code generation, review, analysis
        - moe: Complex reasoning - plan, critique, large text

        Strategy:
        1. Honor explicit overrides
        2. MoE tasks (plan, critique) always use MoE model
        3. Detect if content is CODE or TEXT:
           - Large CODE → coder (specialized for programming)
           - Large TEXT → moe (better reasoning for prose)
        4. Code-focused tasks on code content → coder
        5. Default to quick for everything else
        """
        # Get models from active backend
        backend = self.backend_manager.get_active_backend()
        if backend:
            model_quick = backend.models.get("quick", self.config.model_quick.ollama_model)
            model_coder = backend.models.get("coder", self.config.model_coder.ollama_model)
            model_moe = backend.models.get("moe", self.config.model_moe.ollama_model)
            model_thinking = backend.models.get("thinking", self.config.model_thinking.ollama_model)
        else:
            # Fallback to config defaults
            model_quick = self.config.model_quick.ollama_model
            model_coder = self.config.model_coder.ollama_model
            model_moe = self.config.model_moe.ollama_model
            model_thinking = self.config.model_thinking.ollama_model

        # Helper to resolve tier name to model name
        def resolve_tier(tier_name):
            if tier_name == "quick":
                return model_quick
            if tier_name == "coder":
                return model_coder
            if tier_name == "moe":
                return model_moe
            if tier_name == "thinking":
                return model_thinking
            return tier_name  # Assume it's a model name if not a tier

        # Priority 1: Explicit override
        if model_override:
            resolved = resolve_tier(model_override)
            log.info("model_selected", source="override", tier=model_override, model=resolved)
            return resolved

        # Priority 2: Tasks that REQUIRE MoE (complex multi-step reasoning)
        if task_type in self.config.moe_tasks:
            log.info("model_selected", source="moe_task", task=task_type, tier="moe")
            return model_moe

        # Detect code content once (cache result for reuse below)
        code_detection = None
        if content and (content_size > self.config.large_content_threshold or task_type in self.config.coder_tasks):
            code_detection = detect_code_content(content)

        # Priority 3: Large content - detect if code or text
        if content_size > self.config.large_content_threshold and code_detection:
            is_code, confidence, reasoning = code_detection
            if is_code and confidence > 0.5:
                log.info(
                    "model_selected",
                    source="large_code",
                    content_kb=content_size // 1000,
                    confidence=f"{confidence:.0%}",
                    tier="coder",
                    reasoning=reasoning,
                )
                return model_coder
            else:
                # Large text content benefits from MoE's reasoning
                log.info(
                    "model_selected",
                    source="large_text",
                    content_kb=content_size // 1000,
                    confidence=f"{1 - confidence:.0%}",
                    tier="moe",
                    reasoning=reasoning,
                )
                return model_moe

        # Priority 4: Code-focused tasks - check if content is actually code (reuse cached detection)
        if task_type in self.config.coder_tasks and code_detection:
            is_code, confidence, reasoning = code_detection
            if is_code or confidence > 0.3:
                log.info("model_selected", source="coder_task_code", task=task_type, tier="coder", reasoning=reasoning)
                return model_coder
            else:
                # Task like "analyze" on text should use quick/moe, not coder
                log.info("model_selected", source="coder_task_text", task=task_type, tier="quick", reasoning=reasoning)
                # Fall through to default

        # Priority 5: Default to quick (fastest model)
        log.info("model_selected", source="default", task=task_type, tier="quick")
        return model_quick

    async def select_optimal_backend(
        self,
        content: str,
        file_path: str | None = None,
        task_type: str = "quick",
        backend_type: str | None = None,
    ) -> tuple[str | None, Any | None]:
        """
        Select optimal backend.

        If backend_type is specified ("local" or "remote"), tries to find a matching backend.
        Otherwise uses the active backend.

        Args:
            content: The content to process
            file_path: Optional file path for context
            task_type: Type of task being performed
            backend_type: Optional backend type constraint ("local" or "remote")

        Returns:
            Tuple of (backend_provider, backend_obj) where backend_provider may be None
        """
        if backend_type:
            # Try to find a backend of the requested type
            for b in self.backend_manager.get_enabled_backends():
                if b.type == backend_type:
                    return (None, b)

        # Default to active backend
        active_backend = self.backend_manager.get_active_backend()
        return (None, active_backend)
