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
Conversation Compaction System.

Provides Claude-like conversation summarization to prevent context overflow
while preserving key information across long sessions.

Key features:
- LLM-based summarization of older messages
- Preservation of tool calls, file modifications, and decisions
- Incremental compaction (only process new messages)
- Configurable thresholds and preservation rules
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

import structlog

from .tokens import count_tokens, estimate_tokens
from .config import config

if TYPE_CHECKING:
    from .session_manager import SessionMessage, SessionState

log = structlog.get_logger()

# Compaction configuration
DEFAULT_COMPACTION_THRESHOLD_TOKENS = 12000  # Trigger compaction at this point
DEFAULT_PRESERVE_RECENT_MESSAGES = 6  # Always keep last N message pairs
DEFAULT_SUMMARY_MAX_TOKENS = 2000  # Max tokens for summary


@dataclass
class CompactionMetadata:
    """Tracks compaction history for a session."""

    # When compaction was performed
    compacted_at: str = ""

    # Number of messages that were summarized
    messages_summarized: int = 0

    # Original token count before compaction
    original_tokens: int = 0

    # Token count after compaction
    compacted_tokens: int = 0

    # Version for incremental compaction (tracks last compacted message index)
    last_compacted_index: int = 0

    # Compression ratio achieved
    compression_ratio: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "compacted_at": self.compacted_at,
            "messages_summarized": self.messages_summarized,
            "original_tokens": self.original_tokens,
            "compacted_tokens": self.compacted_tokens,
            "last_compacted_index": self.last_compacted_index,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompactionMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    success: bool = False
    summary: str = ""
    messages_compacted: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    error: str | None = None

    # Key information extracted during compaction
    key_decisions: list[str] = field(default_factory=list)
    file_modifications: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


# Patterns to detect important information that should be preserved
TOOL_CALL_PATTERNS = [
    r"\b(created|wrote|edited|modified|deleted)\s+(?:file\s+)?[`'\"]?([^`'\"]+\.[a-z]+)[`'\"]?",
    r"\b(running|ran|executed)\s+(?:command\s+)?[`'\"]?([^`'\"]+)[`'\"]?",
    r"\b(pip|npm|yarn|git|docker)\s+\w+",
    r"```(?:bash|shell|sh)\n([^`]+)```",
]

DECISION_PATTERNS = [
    r"(?:decided|choosing|will use|going with|selected)\s+(.+?)(?:\.|$)",
    r"(?:approach|strategy|plan):\s*(.+?)(?:\.|$)",
    r"(?:instead of|rather than)\s+(.+?),\s*(?:we|I)",
]

FILE_MOD_PATTERNS = [
    r"(?:created|updated|modified|wrote to)\s+[`'\"]?([^`'\"]+\.[a-z]+)[`'\"]?",
    r"```(?:python|javascript|typescript|rust|go|java|cpp|c)\n[\s\S]*?```",
]


class ConversationCompactor:
    """
    Handles conversation compaction using LLM summarization.

    Follows Claude Code's approach of preserving recent context while
    summarizing older messages to prevent context overflow.
    """

    def __init__(
        self,
        call_llm_fn: Callable[..., Any] | None = None,
        threshold_tokens: int = DEFAULT_COMPACTION_THRESHOLD_TOKENS,
        preserve_recent: int = DEFAULT_PRESERVE_RECENT_MESSAGES,
        summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
    ):
        """
        Initialize the compactor.

        Args:
            call_llm_fn: Async function to call LLM for summarization
            threshold_tokens: Token count to trigger automatic compaction
            preserve_recent: Number of recent message pairs to always preserve
            summary_max_tokens: Maximum tokens for the summary
        """
        self._call_llm = call_llm_fn
        self.threshold_tokens = threshold_tokens
        self.preserve_recent = preserve_recent
        self.summary_max_tokens = summary_max_tokens
        self._lock = asyncio.Lock()

    def _get_call_llm(self) -> Callable[..., Any]:
        """Get or initialize the LLM call function."""
        if self._call_llm is None:
            from .llm import call_llm
            self._call_llm = call_llm
        return self._call_llm

    def needs_compaction(self, session: "SessionState") -> bool:
        """
        Check if a session needs compaction based on token count.

        Args:
            session: The session to check

        Returns:
            True if compaction should be triggered
        """
        if not session.messages:
            return False

        # Count tokens in all messages
        total_tokens = sum(
            count_tokens(msg.content) for msg in session.messages
        )

        return total_tokens > self.threshold_tokens

    def get_compaction_stats(self, session: "SessionState") -> dict[str, Any]:
        """
        Get current compaction statistics for a session.

        Returns token counts, message counts, and compaction recommendations.
        """
        if not session.messages:
            return {
                "total_messages": 0,
                "total_tokens": 0,
                "needs_compaction": False,
                "threshold_tokens": self.threshold_tokens,
                "preserve_recent": self.preserve_recent,
                "compactable_messages": 0,
                "last_compaction": None,
                "compression_ratio": 0.0,
            }

        total_tokens = sum(count_tokens(msg.content) for msg in session.messages)

        # Check for existing compaction metadata
        compaction_meta = session.metadata.get("compaction")

        return {
            "total_messages": len(session.messages),
            "total_tokens": total_tokens,
            "needs_compaction": total_tokens > self.threshold_tokens,
            "threshold_tokens": self.threshold_tokens,
            "preserve_recent": self.preserve_recent,
            "compactable_messages": max(0, len(session.messages) - (self.preserve_recent * 2)),
            "last_compaction": compaction_meta.get("compacted_at") if compaction_meta else None,
            "compression_ratio": compaction_meta.get("compression_ratio", 0.0) if compaction_meta else 0.0,
        }

    def _extract_key_info(self, messages: list["SessionMessage"]) -> dict[str, list[str]]:
        """
        Extract important information from messages that should be highlighted in summary.

        Returns:
            Dict with 'decisions', 'file_mods', and 'tool_calls' lists
        """
        decisions = []
        file_mods = []
        tool_calls = []

        for msg in messages:
            content = msg.content

            # Extract tool calls
            for pattern in TOOL_CALL_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        tool_calls.append(f"{match[0]} {match[1]}")
                    else:
                        tool_calls.append(match)

            # Extract decisions
            for pattern in DECISION_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                decisions.extend(matches[:2])  # Limit per message

            # Extract file modifications
            for pattern in FILE_MOD_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], str) and not matches[0].startswith("```"):
                        file_mods.extend(matches[:3])

        return {
            "decisions": list(set(decisions))[:10],  # Dedupe and limit
            "file_mods": list(set(file_mods))[:10],
            "tool_calls": list(set(tool_calls))[:10],
        }

    async def compact(
        self,
        session: "SessionState",
        force: bool = False,
    ) -> CompactionResult:
        """
        Compact a session's conversation history.

        Args:
            session: The session to compact
            force: Force compaction even if below threshold

        Returns:
            CompactionResult with summary and statistics
        """
        async with self._lock:
            if not session.messages:
                return CompactionResult(
                    success=False,
                    error="No messages to compact"
                )

            # Check if compaction is needed
            stats = self.get_compaction_stats(session)
            if not force and not stats["needs_compaction"]:
                return CompactionResult(
                    success=False,
                    error=f"Token count ({stats['total_tokens']}) below threshold ({self.threshold_tokens})"
                )

            # Determine which messages to compact
            # Keep last N message pairs (user + assistant)
            preserve_count = self.preserve_recent * 2
            if len(session.messages) <= preserve_count:
                return CompactionResult(
                    success=False,
                    error=f"Not enough messages to compact (have {len(session.messages)}, need >{preserve_count})"
                )

            # Split messages
            messages_to_compact = session.messages[:-preserve_count]
            messages_to_preserve = session.messages[-preserve_count:]

            # Calculate tokens in messages to compact
            original_tokens = sum(count_tokens(m.content) for m in messages_to_compact)

            # Extract key information
            key_info = self._extract_key_info(messages_to_compact)

            # Generate summary using LLM
            try:
                summary = await self._generate_summary(
                    messages_to_compact,
                    key_info,
                )
            except Exception as e:
                log.error("compaction_summary_failed", error=str(e))
                return CompactionResult(
                    success=False,
                    error=f"Summary generation failed: {e}"
                )

            summary_tokens = count_tokens(summary)
            tokens_saved = original_tokens - summary_tokens
            compression_ratio = 1.0 - (summary_tokens / original_tokens) if original_tokens > 0 else 0.0

            # Create compaction metadata
            compaction_meta = CompactionMetadata(
                compacted_at=datetime.now().isoformat(),
                messages_summarized=len(messages_to_compact),
                original_tokens=original_tokens,
                compacted_tokens=summary_tokens,
                last_compacted_index=len(messages_to_compact),
                compression_ratio=compression_ratio,
            )

            # Update session with compacted messages
            # Replace compacted messages with a single summary message
            from .session_manager import SessionMessage

            summary_message = SessionMessage(
                role="system",
                content=f"[CONVERSATION SUMMARY]\n{summary}",
                timestamp=datetime.now().isoformat(),
                tokens=summary_tokens,
                model="compactor",
                task_type="compaction",
            )

            # New message list: summary + preserved recent messages
            session.messages = [summary_message] + list(messages_to_preserve)

            # Store compaction metadata
            session.metadata["compaction"] = compaction_meta.to_dict()

            log.info(
                "session_compacted",
                session_id=session.session_id,
                messages_compacted=len(messages_to_compact),
                tokens_saved=tokens_saved,
                compression_ratio=f"{compression_ratio:.1%}",
            )

            return CompactionResult(
                success=True,
                summary=summary,
                messages_compacted=len(messages_to_compact),
                tokens_saved=tokens_saved,
                compression_ratio=compression_ratio,
                key_decisions=key_info["decisions"],
                file_modifications=key_info["file_mods"],
                tool_calls=key_info["tool_calls"],
            )

    async def _generate_summary(
        self,
        messages: list["SessionMessage"],
        key_info: dict[str, list[str]],
    ) -> str:
        """
        Generate a summary of the conversation using LLM.

        Uses the 'quick' tier for fast, efficient summarization.
        """
        # Build conversation text
        conversation_parts = []
        for msg in messages:
            role_prefix = "User" if msg.role == "user" else "Assistant"
            # Truncate very long messages
            content = msg.content[:2000] + "..." if len(msg.content) > 2000 else msg.content
            conversation_parts.append(f"{role_prefix}: {content}")

        conversation_text = "\n\n".join(conversation_parts)

        # Build key info section
        key_info_text = ""
        if key_info["file_mods"]:
            key_info_text += f"\n\nFILES MODIFIED: {', '.join(key_info['file_mods'][:5])}"
        if key_info["tool_calls"]:
            key_info_text += f"\n\nTOOL CALLS: {', '.join(key_info['tool_calls'][:5])}"
        if key_info["decisions"]:
            key_info_text += f"\n\nKEY DECISIONS: {', '.join(key_info['decisions'][:5])}"

        prompt = f"""Summarize this conversation history concisely, preserving:
1. The main task/goal being worked on
2. Key decisions made and their reasoning
3. Important file modifications or code changes
4. Any errors encountered and how they were resolved
5. Current state and what was accomplished

Be concise but comprehensive. This summary will replace the original messages.
{key_info_text}

CONVERSATION:
{conversation_text}

SUMMARY:"""

        system_prompt = """You are a conversation summarizer. Create a concise summary that captures:
- Main objectives and tasks
- Key decisions and their rationale
- File/code modifications made
- Problems solved and approaches used
- Current progress and state

Format the summary in clear sections. Be direct and factual."""

        call_llm = self._get_call_llm()

        result = await call_llm(
            model=config.model_quick.default_model,
            prompt=prompt,
            system=system_prompt,
            task_type="summarize",
            temperature=0.3,  # Slightly creative for better summaries
            max_tokens=self.summary_max_tokens,
        )

        if result.get("success"):
            return result.get("response", "").strip()
        else:
            raise RuntimeError(f"LLM call failed: {result.get('error', 'Unknown error')}")


# Module-level singleton
_compactor: ConversationCompactor | None = None


def get_compactor() -> ConversationCompactor:
    """Get the global ConversationCompactor instance."""
    global _compactor
    if _compactor is None:
        _compactor = ConversationCompactor()
    return _compactor


async def compact_session(
    session: "SessionState",
    force: bool = False,
) -> CompactionResult:
    """
    Convenience function to compact a session.

    Args:
        session: The session to compact
        force: Force compaction even if below threshold

    Returns:
        CompactionResult with summary and statistics
    """
    return await get_compactor().compact(session, force=force)


def needs_compaction(session: "SessionState") -> bool:
    """
    Check if a session needs compaction.

    Args:
        session: The session to check

    Returns:
        True if compaction should be triggered
    """
    return get_compactor().needs_compaction(session)
