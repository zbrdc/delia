# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Conversation Compression System.

Provides LLM-based conversation summarization to prevent context overflow
while preserving key information across long sessions.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

import structlog

from ..tokens import count_tokens
from ..config import config

if TYPE_CHECKING:
    from ..session_manager import SessionMessage, SessionState

log = structlog.get_logger()

# Compression configuration
DEFAULT_COMPACTION_THRESHOLD_TOKENS = 12000
DEFAULT_PRESERVE_RECENT_MESSAGES = 6
DEFAULT_SUMMARY_MAX_TOKENS = 2000

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


@dataclass
class CompressionMetadata:
    """Tracks compression history for a session."""
    compacted_at: str = ""
    messages_summarized: int = 0
    original_tokens: int = 0
    compacted_tokens: int = 0
    last_compacted_index: int = 0
    compression_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "compacted_at": self.compacted_at,
            "messages_summarized": self.messages_summarized,
            "original_tokens": self.original_tokens,
            "compacted_tokens": self.compacted_tokens,
            "last_compacted_index": self.last_compacted_index,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompressionMetadata":
        return cls(**data)


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    success: bool = False
    summary: str = ""
    messages_compacted: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    error: str | None = None
    key_decisions: list[str] = field(default_factory=list)
    file_modifications: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


class ConversationCompressor:
    """
    Handles conversation compression using LLM summarization.
    """

    def __init__(
        self,
        call_llm_fn: Callable[..., Any] | None = None,
        threshold_tokens: int = DEFAULT_COMPACTION_THRESHOLD_TOKENS,
        preserve_recent: int = DEFAULT_PRESERVE_RECENT_MESSAGES,
        summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
    ):
        self._call_llm = call_llm_fn
        self.threshold_tokens = threshold_tokens
        self.preserve_recent = preserve_recent
        self.summary_max_tokens = summary_max_tokens
        self._lock = asyncio.Lock()

    def _get_call_llm(self) -> Callable[..., Any]:
        if self._call_llm is None:
            from ..llm import call_llm
            self._call_llm = call_llm
        return self._call_llm

    def needs_compaction(self, session: "SessionState") -> bool:
        if not session.messages:
            return False
        total_tokens = sum(count_tokens(msg.content) for msg in session.messages)
        return total_tokens > self.threshold_tokens

    def get_compaction_stats(self, session: "SessionState") -> dict[str, Any]:
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

    async def compact(self, session: "SessionState", force: bool = False) -> CompressionResult:
        async with self._lock:
            if not session.messages:
                return CompressionResult(success=False, error="No messages to compact")

            stats = self.get_compaction_stats(session)
            if not force and not stats["needs_compaction"]:
                return CompressionResult(
                    success=False, 
                    error=f"Token count ({stats['total_tokens']}) below threshold ({self.threshold_tokens})"
                )

            preserve_count = self.preserve_recent * 2
            if len(session.messages) <= preserve_count:
                return CompressionResult(
                    success=False, 
                    error=f"Not enough messages to compact (have {len(session.messages)}, need >{preserve_count})"
                )

            messages_to_compact = session.messages[:-preserve_count]
            messages_to_preserve = session.messages[-preserve_count:]
            original_tokens = sum(count_tokens(m.content) for m in messages_to_compact)

            # Extract info
            key_info = self._extract_key_info(messages_to_compact)

            try:
                summary = await self._generate_summary(messages_to_compact, key_info)
            except Exception as e:
                log.error("compression_summary_failed", error=str(e))
                return CompressionResult(success=False, error=f"Summary generation failed: {e}")

            summary_tokens = count_tokens(summary)
            tokens_saved = original_tokens - summary_tokens
            ratio = 1.0 - (summary_tokens / original_tokens) if original_tokens > 0 else 0.0

            meta = CompressionMetadata(
                compacted_at=datetime.now().isoformat(),
                messages_summarized=len(messages_to_compact),
                original_tokens=original_tokens,
                compacted_tokens=summary_tokens,
                last_compacted_index=len(messages_to_compact),
                compression_ratio=ratio,
            )

            from ..session_manager import SessionMessage
            summary_message = SessionMessage(
                role="system",
                content=f"[CONVERSATION SUMMARY]\n{summary}",
                timestamp=datetime.now().isoformat(),
                tokens=summary_tokens,
                model="compressor",
                task_type="compaction",
            )

            session.messages = [summary_message] + list(messages_to_preserve)
            session.metadata["compaction"] = meta.to_dict()

            return CompressionResult(
                success=True,
                summary=summary,
                messages_compacted=len(messages_to_compact),
                tokens_saved=tokens_saved,
                compression_ratio=ratio,
                key_decisions=key_info["decisions"],
                file_modifications=key_info["file_mods"],
                tool_calls=key_info["tool_calls"],
            )

    def _extract_key_info(self, messages: list["SessionMessage"]) -> dict[str, list[str]]:
        decisions = []
        file_mods = []
        tool_calls = []

        for msg in messages:
            content = msg.content
            for pattern in TOOL_CALL_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        tool_calls.append(f"{match[0]} {match[1]}")
                    else:
                        tool_calls.append(match)

            for pattern in DECISION_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                decisions.extend(matches[:2])

            for pattern in FILE_MOD_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], str) and not matches[0].startswith("```"):
                        file_mods.extend(matches[:3])

        return {
            "decisions": list(set(decisions))[:10],
            "file_mods": list(set(file_mods))[:10],
            "tool_calls": list(set(tool_calls))[:10],
        }

    async def _generate_summary(self, messages: list["SessionMessage"], key_info: dict) -> str:
        prompt = "Summarize the following conversation history concisely:\n\n"
        for msg in messages:
            prompt += f"{msg.role.upper()}: {msg.content[:2000]}\n"
        
        call_llm = self._get_call_llm()
        res = await call_llm(model="quick", prompt=prompt, task_type="summarize")
        if not res.get("success"):
            raise RuntimeError(res.get("error", "LLM call failed"))
        return res.get("response", "").strip()


# Singleton
_compressor: ConversationCompressor | None = None

def get_conversation_compressor() -> ConversationCompressor:
    global _compressor
    if _compressor is None:
        _compressor = ConversationCompressor()
    return _compressor

# Compatibility Aliases for Compaction logic
ConversationCompactor = ConversationCompressor
CompactionMetadata = CompressionMetadata
CompactionResult = CompressionResult
get_compactor = get_conversation_compressor
compact_session = lambda session, force=False: get_conversation_compressor().compact(session, force)

def needs_compaction(session: "SessionState") -> bool:
    return get_conversation_compressor().needs_compaction(session)