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
    """Remove <think> tags from LLM response.
    
    DEPRECATED: Use extract_answer() for better logic.
    """
    return extract_answer(response)


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
    If only <think> exists but no </think>, returns empty string (answer not started).
    """
    if "</think>" in response:
        # Split by the last </think> and take the remainder
        parts = response.split("</think>")
        return parts[-1].strip()
        
    if "<think>" in response:
        # Thinking started but not finished, so answer hasn't started
        return ""
        
    return response.strip()
