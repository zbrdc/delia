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
