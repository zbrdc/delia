#!/usr/bin/env python3
# Copyright (C) 2024 Delia Contributors
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

"""
Language detection and system prompt generation for Delia.

This module provides language and framework detection from code content,
file extensions, and hints. It also generates optimized system prompts
for different programming languages and task types.
"""

import re
from pathlib import Path

from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound

import structlog

log = structlog.get_logger()


# ============================================================
# LANGUAGE CONFIGURATIONS
# ============================================================

LANGUAGE_CONFIGS = {
    "pytorch": {
        "extensions": [".py"],
        "keywords": ["torch", "nn.Module", "cuda", "tensor", "backward()", "optimizer"],
        "system_prompt": """Role: Expert PyTorch ML engineer
Style: Efficient GPU code, proper tensor ops
Patterns: Training loops, model architecture
Output: Optimized, trainable models""",
    },
    "sklearn": {
        "extensions": [".py"],
        "keywords": ["sklearn", "fit(", "predict(", "Pipeline", "cross_val", "train_test_split"],
        "system_prompt": """Role: Expert ML engineer (scikit-learn)
Style: Clean pipelines, proper preprocessing
Patterns: Cross-validation, hyperparameter tuning
Output: Validated, reproducible ML code""",
    },
    "react": {
        "extensions": [".jsx", ".tsx"],
        "keywords": ["useState", "useEffect", "import React", "export default", "<div", "className="],
        "system_prompt": """Role: Expert React developer
Style: Functional components, hooks, TypeScript
Patterns: Custom hooks, proper state management
Output: Type-safe, performant components""",
    },
    "react-native": {
        "extensions": [".jsx", ".tsx"],
        "keywords": ["react-native", "StyleSheet", "View", "Text", "TouchableOpacity", "Animated"],
        "system_prompt": """Role: Expert React Native developer
Style: Mobile-first, platform-aware
Patterns: Performance optimization, proper styling
Output: Cross-platform compatible code""",
    },
    "nextjs": {
        "extensions": [".jsx", ".tsx", ".js", ".ts"],
        "keywords": ["next/", "getServerSideProps", "getStaticProps", "useRouter", "app/page", "layout.tsx"],
        "system_prompt": """Role: Expert Next.js developer
Style: App Router, Server Components
Patterns: RSC, data fetching, caching
Output: Optimized full-stack code""",
    },
    "rust": {
        "extensions": [".rs"],
        "keywords": ["fn ", "impl ", "use std::", "println!", "let mut", "struct ", "enum "],
        "system_prompt": """Role: Expert Rust developer
Style: Memory-safe, zero-cost abstractions
Patterns: Ownership, borrowing, lifetimes
Output: Safe, performant systems code""",
    },
    "go": {
        "extensions": [".go"],
        "keywords": ["func ", "package ", "import ", "go ", "defer ", "goroutine", "chan "],
        "system_prompt": """Role: Expert Go developer
Style: Simple, concurrent, efficient
Patterns: Goroutines, channels, interfaces
Output: Scalable, concurrent systems""",
    },
    "java": {
        "extensions": [".java"],
        "keywords": ["public class", "import java", "System.out", "public static", "ArrayList", "HashMap"],
        "system_prompt": """Role: Expert Java developer
Style: OOP, JVM ecosystem
Patterns: Design patterns, collections, concurrency
Output: Robust, enterprise-grade applications""",
    },
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
        "keywords": ["#include", "std::", "class ", "template", "virtual ", "override", "auto "],
        "system_prompt": """Role: Expert C++ developer
Style: Modern C++17/20, RAII, templates
Patterns: STL, smart pointers, exceptions
Output: High-performance, memory-efficient code""",
    },
    "csharp": {
        "extensions": [".cs"],
        "keywords": ["using System", "public class", "Console.Write", "async ", "Task<", "IEnumerable"],
        "system_prompt": """Role: Expert C# developer
Style: .NET, LINQ, async/await
Patterns: Dependency injection, SOLID principles
Output: Maintainable, scalable applications""",
    },
    "nodejs": {
        "extensions": [".js", ".ts", ".mjs"],
        "keywords": ["require(", "module.exports", "express", "async/await", "Buffer", "process."],
        "system_prompt": """Role: Expert Node.js developer
Style: Async/await, proper error handling
Patterns: Scalable architecture, streams
Output: Production-ready backend code""",
    },
    "python": {
        "extensions": [".py", ".pyx", ".pyi"],
        "keywords": ["def ", "import ", "class ", "async def", "from ", "if __name__"],
        "system_prompt": """Role: Expert Python developer
Style: PEP8, type hints, docstrings
Version: Python 3.10+
Output: Clean, production-ready code""",
    },
}

# Pygments lexer name -> our language key mapping
PYGMENTS_LANGUAGE_MAP = {
    "python": "python",
    "python 3": "python",
    "python3": "python",
    "javascript": "nodejs",
    "typescript": "nodejs",
    "jsx": "react",
    "tsx": "react",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "c": "c",
    "c++": "cpp",
    "c#": "csharp",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "sql": "sql",
    "bash": "bash",
    "shell": "bash",
    "yaml": "yaml",
    "json": "json",
    "html": "html",
    "css": "css",
}


# ============================================================
# LANGUAGE DETECTION
# ============================================================


def detect_language(content: str, file_path: str = "", hint: str | None = None) -> str:
    """
    Detect programming language/framework from content, file path, and optional hint.

    Priority:
    1. Explicit hint if provided
    2. Framework keyword detection (React, Next.js, PyTorch, etc. - ≥2 keyword matches)
    3. Pygments get_lexer_for_filename() when file_path is available
    4. Simple keyword fallback for content-only detection

    Args:
        content: Source code content to analyze
        file_path: Optional file path for extension-based detection
        hint: Optional explicit language/framework hint

    Returns:
        Detected language key (e.g., "python", "react", "rust")
    """
    # Priority 0: Use explicit hint if provided
    if hint:
        hint_lower = hint.lower()
        # Direct match
        if hint_lower in LANGUAGE_CONFIGS:
            log.debug("lang_hint_used", hint=hint_lower)
            return hint_lower
        # Fuzzy matching for common aliases
        hint_map = {
            "py": "python",
            "js": "nodejs",
            "ts": "nodejs",
            "typescript": "nodejs",
            "javascript": "nodejs",
            "node": "nodejs",
            "c++": "cpp",
            "c#": "csharp",
            "dotnet": "csharp",
            "golang": "go",
            "next": "nextjs",
        }
        if hint_lower in hint_map:
            detected = hint_map[hint_lower]
            log.debug("lang_hint_mapped", hint=hint_lower, detected=detected)
            return detected

    content_lower = content.lower()

    # Priority 1: Detect frameworks by keyword density (most specific first)
    # This catches React, Next.js, PyTorch, etc. which need content analysis
    for lang, lang_config in LANGUAGE_CONFIGS.items():
        matches = sum(1 for kw in lang_config["keywords"] if kw.lower() in content_lower)
        if matches >= 2:
            log.debug("lang_keyword_detected", language=lang, matches=matches)
            return lang

    # Priority 2: Use Pygments for reliable extension-based detection
    if file_path:
        ext = Path(file_path).suffix
        if ext:
            try:
                # Only pass extension to Pygments (privacy: don't expose full paths)
                lexer = get_lexer_for_filename(f"file{ext}", content)
                pygments_name = lexer.name.lower()
                # Map Pygments name to our language keys
                if pygments_name in PYGMENTS_LANGUAGE_MAP:
                    detected = PYGMENTS_LANGUAGE_MAP[pygments_name]
                    log.debug("lang_pygments_extension", pygments_name=pygments_name, detected=detected)
                    return detected
                # Direct name matching for common languages
                for our_lang in [
                    "python",
                    "go",
                    "rust",
                    "java",
                    "cpp",
                    "csharp",
                    "ruby",
                    "php",
                    "swift",
                    "kotlin",
                    "scala",
                ]:
                    if our_lang in pygments_name:
                        log.debug("lang_pygments_name", pygments_name=pygments_name, our_lang=our_lang)
                        return our_lang
                # JavaScript/TypeScript family
                if "script" in pygments_name or "type" in pygments_name:
                    log.debug("lang_pygments_script", pygments_name=pygments_name, detected="nodejs")
                    return "nodejs"
            except ClassNotFound:
                pass  # Extension not recognized - fall through to keyword fallback
            except Exception as e:
                log.debug("lang_pygments_error", error=str(e))

    # Priority 3: Simple keyword fallback for content-only detection
    if "fn " in content or "impl " in content or "use std::" in content:
        return "rust"
    if "func " in content or "package " in content:
        return "go"
    if "public class" in content or "import java" in content:
        return "java"
    if "#include" in content or "std::" in content:
        return "cpp"
    if "using System" in content or "Console.Write" in content:
        return "csharp"
    if "require(" in content or "module.exports" in content:
        return "nodejs"
    if "def " in content or "import " in content or "class " in content:
        return "python"
    if "function " in content or "const " in content or "let " in content:
        return "nodejs"
    if "React" in content or "useState" in content or "jsx" in content_lower:
        return "react"

    return "python"  # Default fallback


# ============================================================
# SYSTEM PROMPT GENERATION
# ============================================================


def get_system_prompt(language: str, task_type: str) -> str:
    """
    Get structured system prompt optimized for LLM-to-LLM communication.

    Args:
        language: Programming language/framework key
        task_type: Type of task (review, generate, analyze, etc.)

    Returns:
        Optimized system prompt string
    """
    base: str = str(LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["python"])["system_prompt"])

    # Aggressive LLM-to-LLM optimization - eliminate all fluff
    llm_prefix = """CRITICAL: Your output is consumed by another LLM (Claude/Copilot), NOT a human.
RULES:
- NO preamble ("Sure", "I'd be happy to", "Let me", "Here's")
- NO sign-offs ("Hope this helps", "Let me know", "Feel free to ask")
- NO filler phrases or pleasantries
- NO restating the question
- START with the answer/content directly
- Be DENSE: every token must add information
- Use structured format (bullets, numbered lists) for clarity
"""

    task_instructions = {
        "review": """
TASK: CODE REVIEW
OUTPUT FORMAT:
[CRITICAL] issue description (if any)
[MAJOR] issue description (if any)
[MINOR] issue description (if any)
[GOOD] positive aspects (brief)
[SUGGEST] improvements (actionable)""",
        "generate": """
TASK: CODE GENERATION
OUTPUT: Code only. Minimal inline comments. No explanation unless critical.""",
        "analyze": """
TASK: ANALYSIS
OUTPUT FORMAT:
PURPOSE: one line
FINDINGS:
- finding 1
- finding 2
RECOMMENDATIONS:
- recommendation 1""",
        "summarize": """
TASK: SUMMARIZE
OUTPUT: 3-5 bullet points maximum. No introduction.""",
        "critique": """
TASK: CRITIQUE
OUTPUT FORMAT:
STRENGTHS: bullet list
WEAKNESSES: bullet list
PRIORITY FIXES: numbered list""",
        "plan": """
TASK: PLANNING
OUTPUT FORMAT:
OVERVIEW: one paragraph max
STEPS:
1. step
2. step
RISKS: bullet list (if any)""",
        "quick": """
TASK: QUICK ANSWER
OUTPUT: Direct answer first. One sentence explanation if needed.""",
    }

    return llm_prefix + base + task_instructions.get(task_type, "")


# ============================================================
# PROMPT OPTIMIZATION
# ============================================================


def optimize_prompt(content: str, task_type: str) -> str:
    """
    Strip natural language triggers and structure prompt for LLM consumption.

    Removes conversational phrases and formats for optimal LLM processing.

    Args:
        content: Raw prompt content
        task_type: Type of task (affects optimization strategy)

    Returns:
        Cleaned and optimized prompt
    """
    # Remove trigger phrases (case insensitive)
    triggers = [
        r"\s*,?\s*locally\s*$",
        r"\s*,?\s*ask\s+(ollama|locally|local|coder|moe|qwen)\s*$",
        r"\s*,?\s*use\s+(ollama|local|coder|moe|quick)\s*$",
        r"\s*,?\s*on\s+my\s+(gpu|machine|device)\s*$",
        r"\s*,?\s*(privately|offline)\s*$",
        r"\s*,?\s*without\s+(api|cloud)\s*$",
        r"\s*,?\s*via\s+ollama\s*$",
        r"\s*,?\s*no\s+cloud\s*$",
        r"^\s*(please|can you|could you|i want you to|i need you to)\s+",
        r"^\s*(hey|hi|hello),?\s*",
    ]

    cleaned = content
    for pattern in triggers:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip()

    # For certain tasks, add structure if not already present
    # Ensure questions have question marks
    if task_type == "quick" and "?" not in cleaned and cleaned and not cleaned.endswith("."):
        cleaned = cleaned.rstrip(".,!;") + "?"

    return cleaned


def create_enhanced_prompt(
    task_type: str,
    content: str,
    file: str | None = None,
    language: str | None = None,
    symbols: list[str] | None = None,
    context_files: list[str] | None = None,
    user_instructions: str | None = None,
) -> str:
    """
    Create an enhanced, structured prompt using templates and context.

    This function optimizes prompts for specific tasks and languages,
    incorporating file context, symbols, and user instructions.

    Args:
        task_type: Type of task (review, generate, analyze, etc.)
        content: Main prompt content
        file: Optional file path for context
        language: Optional language hint
        symbols: Optional list of code symbols to focus on
        context_files: Optional list of additional context files
        user_instructions: Optional additional instructions

    Returns:
        Enhanced, structured prompt ready for LLM consumption

    Note:
        This function requires additional imports and dependencies from mcp_server.py:
        - read_file_safe() for file reading
        - create_structured_prompt() for template-based generation
        These are intentionally not included here to keep this module standalone.
        When used in practice, this function should be called from mcp_server.py
        where these dependencies are available.
    """
    # First clean the content using the existing optimization
    cleaned_content = optimize_prompt(content, task_type)

    # Detect language if not provided
    detected_language = language or detect_language(cleaned_content, file or "")

    # Build a simple enhanced prompt
    # Note: Full implementation requires imports from mcp_server.py
    prompt_parts = []

    # Add task context
    if task_type:
        prompt_parts.append(f"Task: {task_type}")

    # Add language context
    if detected_language:
        prompt_parts.append(f"Language: {detected_language}")

    # Add symbols if provided
    if symbols:
        prompt_parts.append(f"Focus on symbols: {', '.join(symbols)}")

    # Add user instructions if provided
    if user_instructions:
        prompt_parts.append(f"Additional instructions: {user_instructions}")

    # Add main content
    prompt_parts.append(f"\n{cleaned_content}")

    # Add file reference if provided
    if file:
        prompt_parts.append(f"\nFile: {file}")

    # Add context files if provided
    if context_files:
        prompt_parts.append(f"Context files: {', '.join(context_files)}")

    return "\n".join(prompt_parts)
