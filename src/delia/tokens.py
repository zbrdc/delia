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
"""Token counting utilities for Delia."""

import structlog
import tiktoken

log = structlog.get_logger()

_tiktoken_encoder: tiktoken.Encoding | None = None
_tiktoken_failed: bool = False  # Track if loading failed to avoid repeated attempts


def get_tiktoken_encoder() -> tiktoken.Encoding | None:
    """
    Get or initialize tiktoken encoder (cl100k_base works for most models).

    Returns cached encoder, or None if loading failed.
    """
    global _tiktoken_encoder, _tiktoken_failed

    # Don't retry if we already failed
    if _tiktoken_failed:
        return None

    if _tiktoken_encoder is None:
        try:
            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            _tiktoken_failed = True
            log.warning("tiktoken_load_failed", error=str(e), fallback="estimate")
            return None

    return _tiktoken_encoder


def count_tokens(text: str) -> int:
    """
    Count tokens accurately using tiktoken, with fallback to estimation.

    Uses tiktoken's cl100k_base encoding (compatible with GPT-4, Claude, etc.)
    Falls back to ~4 chars per token estimate if tiktoken unavailable.

    Args:
        text: The text to count tokens for

    Returns:
        Token count (accurate if tiktoken available, else estimated)
    """
    if not text:
        return 0

    encoder = get_tiktoken_encoder()
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception:  # noqa: S110 - Fall through to estimate below
            pass

    # Fallback: ~4 chars per token (rough estimate for modern models)
    return len(text) // 4


def estimate_tokens(text: str) -> int:
    """
    Quick token estimation without tiktoken.

    Use this for non-critical estimates where speed matters more than accuracy.
    """
    if not text:
        return 0
    return len(text) // 4
