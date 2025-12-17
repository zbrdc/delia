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

    Some models (e.g., Qwen3, DeepSeek-R1, nemotron) output thinking in 
    <think>...</think> tags. This function extracts the useful content:

    1. If there's content after </think>, return that (the "answer")
    2. If </think> is at the end but <think> exists, return thinking content
    3. If </think> exists without <think> (malformed), strip everything before it
    4. If no thinking tags, return the original response

    Args:
        response: Raw LLM response text

    Returns:
        Cleaned response with thinking tags removed
    """
    if "</think>" not in response:
        return response

    # Try to get content after the thinking block
    after_think = response.split("</think>")[-1].strip()
    if after_think:
        return after_think

    # No content after </think> - check if <think> tag exists
    if "<think>" in response:
        # Extract thinking content between tags
        match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Malformed: </think> exists but no <think> and no content after
    # This shouldn't normally happen, but return empty or original
    # Remove everything up to and including </think>
    return response.replace("</think>", "").strip()
