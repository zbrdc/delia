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

"""Text processing utilities for LLM responses."""

import re


def strip_thinking_tags(response: str) -> str:
    """Remove thinking content from LLM response.

    Handles both tagged (<think>) and untagged reasoning.
    """
    # First try standard tag extraction
    result = extract_answer(response)

    # If no tags found, check for untagged thinking patterns
    if result == response:
        result = strip_untagged_thinking(result)

    return result


def strip_untagged_thinking(response: str) -> str:
    """Remove untagged chain-of-thought reasoning.

    Detects common thinking patterns and returns only the final answer.
    """
    # Patterns that indicate internal reasoning (case-insensitive)
    thinking_indicators = [
        r'okay,?\s*(let me|so)',
        r'let me (think|process|check|analyze)',
        r'^hmm\.{0,3}',
        r'\*[^*]+\*',  # *actions in asterisks*
        r'\bi (need to|should|recall|remember)\b',
        r'^(first|now),?\s*(i|let)',
        r'previous response',
        r'double-?checking',
        r'my (first |next )?response should',
        r'per delia\'?s rules',
        r'must avoid robotic',
        r'user greeted',
        r'this is on-topic',
        r'feels right',
    ]

    patterns = [re.compile(p, re.IGNORECASE) for p in thinking_indicators]

    # Split into paragraphs
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]

    if not paragraphs:
        return response

    # Find last paragraph that doesn't contain thinking patterns
    for i in range(len(paragraphs) - 1, -1, -1):
        para = paragraphs[i]
        has_thinking = any(p.search(para) for p in patterns)

        if not has_thinking and len(para) > 5:
            # This looks like the actual answer
            return para

    # If all paragraphs have thinking patterns, try last line approach
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        # Return last substantial line that isn't meta-commentary
        for line in reversed(lines):
            has_thinking = any(p.search(line) for p in patterns)
            if not has_thinking and len(line) > 10:
                return line

    return response


def extract_thinking(response: str) -> str | None:
    """Extract content between <think> tags.
    
    Handles:
    - Multiple <think> blocks (concatenates them)
    - Unclosed <think> tags (extracts until end of string)
    """
    if "<think>" not in response:
        return None
        
    # Find all <think> blocks
    thinking_parts = []
    
    # Simple regex for matching blocks
    pattern = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)
    for match in pattern.finditer(response):
        content = match.group(1).strip()
        if content:
            thinking_parts.append(content)
            
    return "\n\n".join(thinking_parts) if thinking_parts else None


def extract_answer(response: str) -> str:
    """Extract the actual answer (content outside <think> tags).

    If thinking blocks exist, returns everything AFTER the last </think> tag.
    If no thinking tags exist, returns the original response.
    If only <think> exists but no </think>, tries to find content after thinking.
    """
    if "</think>" in response:
        # Split by the last </think> and take the remainder
        parts = response.split("</think>")
        return parts[-1].strip()

    if "<think>" in response:
        # Thinking started but not closed - try to find where answer starts
        # Common patterns that indicate end of thinking:
        import re

        # Try to find code blocks, which often follow thinking
        code_block_match = re.search(r'```\w*\n', response)
        if code_block_match:
            # Return from the code block onwards
            return response[code_block_match.start():].strip()

        # Try to find "Here's", "Here is", "Output:", etc.
        answer_patterns = [
            r'\n\s*(Here\'?s?|Output|Answer|Result|Solution|Code)[:\s]',
            r'\n\s*(?:The\s+)?(?:Python|code|script|program)\s+(?:is|would be|looks like)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return response[match.start():].strip()

        # Last resort: if thinking is very long, it might contain the answer
        # Return everything after the first few lines of thinking
        lines = response.split('\n')
        if len(lines) > 10:
            # Skip the thinking preamble, return rest
            for i, line in enumerate(lines[5:], start=5):
                if '```' in line or line.strip().startswith('print'):
                    return '\n'.join(lines[i:]).strip()

        # Can't find answer, return the thinking content with the tag stripped
        think_content = response.split('<think>', 1)[-1]
        return f"[Model thinking only - no final answer]\n{think_content.strip()}"

    return response.strip()
