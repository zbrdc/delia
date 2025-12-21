# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the conversation compaction system."""

import pytest
from datetime import datetime

from delia.session_manager import SessionState, SessionMessage
from delia.semantic.compression import (
    ConversationCompressor as ConversationCompactor,
    CompressionResult as CompactionResult,
    CompressionMetadata as CompactionMetadata,
    get_conversation_compressor as get_compactor,
    needs_compaction,
    DEFAULT_COMPACTION_THRESHOLD_TOKENS,
)


class TestCompactionMetadata:
    """Test CompactionMetadata dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        meta = CompactionMetadata(
            compacted_at="2024-01-01T00:00:00",
            messages_summarized=10,
            original_tokens=5000,
            compacted_tokens=1000,
            last_compacted_index=10,
            compression_ratio=0.8,
        )
        d = meta.to_dict()
        assert d["messages_summarized"] == 10
        assert d["compression_ratio"] == 0.8

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "compacted_at": "2024-01-01T00:00:00",
            "messages_summarized": 10,
            "original_tokens": 5000,
            "compacted_tokens": 1000,
            "last_compacted_index": 10,
            "compression_ratio": 0.8,
        }
        meta = CompactionMetadata.from_dict(d)
        assert meta.messages_summarized == 10
        assert meta.compression_ratio == 0.8


class TestConversationCompactor:
    """Test ConversationCompactor class."""

    def test_needs_compaction_empty_session(self):
        """Empty session doesn't need compaction."""
        session = SessionState(session_id="test-123")
        compactor = ConversationCompactor()
        assert not compactor.needs_compaction(session)

    def test_needs_compaction_below_threshold(self):
        """Session below threshold doesn't need compaction."""
        session = SessionState(session_id="test-123")
        # Add a few short messages
        for i in range(5):
            session.add_message("user", f"Message {i}")
            session.add_message("assistant", f"Response {i}")

        compactor = ConversationCompactor()
        assert not compactor.needs_compaction(session)

    def test_needs_compaction_above_threshold(self):
        """Session above threshold needs compaction."""
        session = SessionState(session_id="test-123")
        # Add many long messages to exceed threshold
        long_content = "x" * 1000  # ~250 tokens per message
        for i in range(60):  # 60 * 250 * 2 = 30,000 tokens
            session.add_message("user", f"{long_content} {i}")
            session.add_message("assistant", f"Response: {long_content} {i}")

        compactor = ConversationCompactor(threshold_tokens=10000)
        assert compactor.needs_compaction(session)

    def test_get_compaction_stats(self):
        """Test compaction stats retrieval."""
        session = SessionState(session_id="test-123")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")

        compactor = ConversationCompactor()
        stats = compactor.get_compaction_stats(session)

        assert "total_messages" in stats
        assert "total_tokens" in stats
        assert "needs_compaction" in stats
        assert "threshold_tokens" in stats
        assert stats["total_messages"] == 2
        assert not stats["needs_compaction"]

    def test_extract_key_info_tool_calls(self):
        """Test extraction of tool calls from messages."""
        session = SessionState(session_id="test-123")
        session.add_message("assistant", "I created file src/main.py")
        session.add_message("assistant", "Running npm install")

        compactor = ConversationCompactor()
        key_info = compactor._extract_key_info(session.messages)

        assert len(key_info["tool_calls"]) > 0

    def test_extract_key_info_file_mods(self):
        """Test extraction of file modifications."""
        session = SessionState(session_id="test-123")
        session.add_message("assistant", "I modified the file config.json")
        session.add_message("assistant", "Updated settings.py with new values")

        compactor = ConversationCompactor()
        key_info = compactor._extract_key_info(session.messages)

        assert len(key_info["file_mods"]) > 0


class TestSessionStateCompaction:
    """Test SessionState compaction integration."""

    def test_session_get_compaction_stats(self):
        """Test SessionState.get_compaction_stats method."""
        session = SessionState(session_id="test-123")
        session.add_message("user", "Test message")

        stats = session.get_compaction_stats()
        assert "total_messages" in stats
        assert stats["total_messages"] == 1

    def test_session_needs_compaction(self):
        """Test SessionState.needs_compaction method."""
        session = SessionState(session_id="test-123")
        assert not session.needs_compaction()


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_get_compactor_singleton(self):
        """Test that get_compactor returns singleton."""
        c1 = get_compactor()
        c2 = get_compactor()
        assert c1 is c2

    def test_needs_compaction_function(self):
        """Test needs_compaction convenience function."""
        session = SessionState(session_id="test-123")
        assert not needs_compaction(session)


@pytest.mark.asyncio
class TestAsyncCompaction:
    """Test async compaction operations."""

    async def test_compact_empty_session(self):
        """Test compacting empty session fails gracefully."""
        session = SessionState(session_id="test-123")
        compactor = ConversationCompactor()

        result = await compactor.compact(session)

        assert not result.success
        assert "No messages" in result.error

    async def test_compact_insufficient_messages(self):
        """Test compacting with insufficient messages (force=True to bypass threshold)."""
        session = SessionState(session_id="test-123")
        for i in range(5):
            session.add_message("user", f"Message {i}")
            session.add_message("assistant", f"Response {i}")

        compactor = ConversationCompactor(preserve_recent=6)
        # Force to bypass threshold check and hit message count check
        result = await compactor.compact(session, force=True)

        assert not result.success
        assert "Not enough messages" in result.error

    async def test_compact_below_threshold_without_force(self):
        """Test that compaction below threshold requires force=True."""
        session = SessionState(session_id="test-123")
        for i in range(10):
            session.add_message("user", f"Message {i}")
            session.add_message("assistant", f"Response {i}")

        # Low threshold but not forced
        compactor = ConversationCompactor(
            threshold_tokens=100000,  # High threshold
            preserve_recent=2
        )

        result = await compactor.compact(session, force=False)
        assert not result.success
        assert "below threshold" in result.error.lower()
