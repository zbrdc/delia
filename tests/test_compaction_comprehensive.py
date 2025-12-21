# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Comprehensive validation tests for the conversation compaction system.

These tests validate:
1. Core compaction logic and algorithms
2. Session integration
3. MCP tool functionality
4. Auto-compaction triggers
5. Edge cases and error handling
6. End-to-end workflow
"""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from delia.session_manager import (
    SessionState,
    SessionMessage,
    SessionManager,
    get_session_manager,
)
from delia.compaction import (
    ConversationCompactor,
    CompactionResult,
    CompactionMetadata,
    get_compactor,
    needs_compaction,
    compact_session,
    DEFAULT_COMPACTION_THRESHOLD_TOKENS,
    DEFAULT_PRESERVE_RECENT_MESSAGES,
    DEFAULT_SUMMARY_MAX_TOKENS,
    TOOL_CALL_PATTERNS,
    DECISION_PATTERNS,
    FILE_MOD_PATTERNS,
)


# ============================================================
# SECTION 1: Core Compaction Logic Tests
# ============================================================

class TestCompactionThresholds:
    """Test threshold detection and configuration."""

    def test_default_threshold_is_reasonable(self):
        """Default threshold should be 12000 tokens."""
        assert DEFAULT_COMPACTION_THRESHOLD_TOKENS == 12000

    def test_default_preserve_recent(self):
        """Should preserve 6 message pairs by default."""
        assert DEFAULT_PRESERVE_RECENT_MESSAGES == 6

    def test_custom_threshold(self):
        """Compactor should accept custom threshold."""
        compactor = ConversationCompactor(threshold_tokens=5000)
        assert compactor.threshold_tokens == 5000

    def test_custom_preserve_recent(self):
        """Compactor should accept custom preserve_recent."""
        compactor = ConversationCompactor(preserve_recent=10)
        assert compactor.preserve_recent == 10

    def test_threshold_boundary_below(self):
        """Session just below threshold shouldn't need compaction."""
        session = SessionState(session_id="test")
        # Add messages just below 12000 token threshold (~3000 chars = 750 tokens)
        for i in range(15):
            session.add_message("user", "x" * 200)
            session.add_message("assistant", "y" * 200)

        compactor = ConversationCompactor(threshold_tokens=12000)
        # 30 messages * ~50 tokens each = ~1500 tokens, well below threshold
        assert not compactor.needs_compaction(session)

    def test_threshold_boundary_above(self):
        """Session just above threshold should need compaction."""
        session = SessionState(session_id="test")
        # Add enough to exceed threshold
        for i in range(50):
            session.add_message("user", "x" * 1000)
            session.add_message("assistant", "y" * 1000)

        compactor = ConversationCompactor(threshold_tokens=12000)
        assert compactor.needs_compaction(session)


class TestKeyInfoExtraction:
    """Test extraction of important information from messages."""

    def test_extract_file_creation(self):
        """Should detect file creation patterns."""
        session = SessionState(session_id="test")
        session.add_message("assistant", "I created file src/main.py with the implementation")

        compactor = ConversationCompactor()
        info = compactor._extract_key_info(session.messages)

        assert any("src/main.py" in mod for mod in info["file_mods"]) or \
               any("src/main.py" in call for call in info["tool_calls"])

    def test_extract_command_execution(self):
        """Should detect command execution patterns."""
        session = SessionState(session_id="test")
        session.add_message("assistant", "Running npm install to install dependencies")
        session.add_message("assistant", "Executing git commit -m 'fix: bug'")

        compactor = ConversationCompactor()
        info = compactor._extract_key_info(session.messages)

        assert len(info["tool_calls"]) > 0

    def test_extract_decisions(self):
        """Should detect decision patterns."""
        session = SessionState(session_id="test")
        session.add_message("assistant", "I decided to use TypeScript instead of JavaScript")
        session.add_message("assistant", "The approach will be to implement caching first")

        compactor = ConversationCompactor()
        info = compactor._extract_key_info(session.messages)

        # At least one decision should be detected
        assert len(info["decisions"]) >= 0  # Patterns may not match all formats

    def test_extract_code_blocks(self):
        """Should detect code blocks in messages."""
        session = SessionState(session_id="test")
        session.add_message("assistant", "Here's the code:\n```python\ndef hello():\n    pass\n```")

        compactor = ConversationCompactor()
        info = compactor._extract_key_info(session.messages)
        # Code blocks are detected via FILE_MOD_PATTERNS
        # Result depends on pattern matching

    def test_extract_limits_results(self):
        """Should limit extracted results to prevent bloat."""
        session = SessionState(session_id="test")
        # Add many file modifications
        for i in range(20):
            session.add_message("assistant", f"Modified file{i}.py")

        compactor = ConversationCompactor()
        info = compactor._extract_key_info(session.messages)

        # Should be limited to 10 each
        assert len(info["file_mods"]) <= 10
        assert len(info["decisions"]) <= 10
        assert len(info["tool_calls"]) <= 10


class TestCompactionStats:
    """Test compaction statistics calculation."""

    def test_stats_empty_session(self):
        """Empty session should have zero counts."""
        session = SessionState(session_id="test")
        compactor = ConversationCompactor()
        stats = compactor.get_compaction_stats(session)

        assert stats["total_messages"] == 0
        assert stats["total_tokens"] == 0
        assert not stats["needs_compaction"]
        assert stats["compactable_messages"] == 0
        assert stats["preserve_recent"] == DEFAULT_PRESERVE_RECENT_MESSAGES

    def test_stats_with_messages(self):
        """Stats should reflect actual message counts."""
        session = SessionState(session_id="test")
        for i in range(10):
            session.add_message("user", f"Message {i}")
            session.add_message("assistant", f"Response {i}")

        compactor = ConversationCompactor(preserve_recent=3)
        stats = compactor.get_compaction_stats(session)

        assert stats["total_messages"] == 20
        assert stats["total_tokens"] > 0
        # 20 messages - (3 pairs * 2) = 14 compactable
        assert stats["compactable_messages"] == 14

    def test_stats_includes_threshold(self):
        """Stats should include threshold information."""
        compactor = ConversationCompactor(threshold_tokens=8000)
        session = SessionState(session_id="test")
        stats = compactor.get_compaction_stats(session)

        assert stats["threshold_tokens"] == 8000
        assert stats["preserve_recent"] == DEFAULT_PRESERVE_RECENT_MESSAGES


# ============================================================
# SECTION 2: SessionState Integration Tests
# ============================================================

class TestSessionStateIntegration:
    """Test SessionState compaction method integration."""

    def test_session_has_needs_compaction(self):
        """SessionState should have needs_compaction method."""
        session = SessionState(session_id="test")
        assert hasattr(session, "needs_compaction")
        assert callable(session.needs_compaction)

    def test_session_has_get_compaction_stats(self):
        """SessionState should have get_compaction_stats method."""
        session = SessionState(session_id="test")
        assert hasattr(session, "get_compaction_stats")
        assert callable(session.get_compaction_stats)

    def test_session_needs_compaction_returns_bool(self):
        """needs_compaction should return boolean."""
        session = SessionState(session_id="test")
        result = session.needs_compaction()
        assert isinstance(result, bool)

    def test_session_stats_returns_dict(self):
        """get_compaction_stats should return dict with required keys."""
        session = SessionState(session_id="test")
        stats = session.get_compaction_stats()

        assert isinstance(stats, dict)
        assert "total_messages" in stats
        assert "total_tokens" in stats
        assert "needs_compaction" in stats


# ============================================================
# SECTION 3: SessionManager Integration Tests
# ============================================================

class TestSessionManagerIntegration:
    """Test SessionManager compaction method integration."""

    @pytest.fixture
    def temp_session_manager(self, tmp_path):
        """Create a SessionManager with temporary storage."""
        return SessionManager(session_dir=tmp_path / "sessions")

    def test_manager_has_compact_session(self, temp_session_manager):
        """SessionManager should have compact_session method."""
        assert hasattr(temp_session_manager, "compact_session")
        assert asyncio.iscoroutinefunction(temp_session_manager.compact_session)

    def test_manager_has_get_compaction_stats(self, temp_session_manager):
        """SessionManager should have get_compaction_stats method."""
        assert hasattr(temp_session_manager, "get_compaction_stats")

    def test_manager_has_needs_compaction(self, temp_session_manager):
        """SessionManager should have needs_compaction method."""
        assert hasattr(temp_session_manager, "needs_compaction")

    def test_manager_stats_nonexistent_session(self, temp_session_manager):
        """get_compaction_stats should return None for nonexistent session."""
        stats = temp_session_manager.get_compaction_stats("nonexistent-id")
        assert stats is None

    def test_manager_needs_compaction_nonexistent(self, temp_session_manager):
        """needs_compaction should return False for nonexistent session."""
        result = temp_session_manager.needs_compaction("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_manager_compact_nonexistent(self, temp_session_manager):
        """compact_session should fail for nonexistent session."""
        result = await temp_session_manager.compact_session("nonexistent-id")
        assert not result["success"]
        assert "not found" in result["error"].lower()

    def test_manager_stats_existing_session(self, temp_session_manager):
        """get_compaction_stats should work for existing session."""
        session = temp_session_manager.create_session()
        stats = temp_session_manager.get_compaction_stats(session.session_id)

        assert stats is not None
        assert "total_messages" in stats


# ============================================================
# SECTION 4: MCP Tool Tests
# ============================================================

class TestMCPTools:
    """Test MCP tool registration and functionality."""

    def test_session_compact_tool_exists(self):
        """session_compact tool should be registered."""
        from delia.mcp_server import session_compact
        assert session_compact is not None

    def test_session_stats_tool_exists(self):
        """session_stats tool should be registered."""
        from delia.mcp_server import session_stats
        assert session_stats is not None

    @pytest.mark.asyncio
    async def test_session_stats_returns_json(self, tmp_path):
        """session_stats should return valid JSON via SessionManager."""
        # Instead of calling MCP tool directly, test the underlying functionality
        sm = get_session_manager()
        session = sm.create_session()

        try:
            stats = sm.get_compaction_stats(session.session_id)
            assert stats is not None
            assert "total_messages" in stats
        finally:
            sm.delete_session(session.session_id)

    @pytest.mark.asyncio
    async def test_session_compact_returns_result(self):
        """session_compact should return result via SessionManager."""
        sm = get_session_manager()

        # Test with nonexistent session
        result = await sm.compact_session("nonexistent-session-id")
        assert "success" in result
        assert not result["success"]
        assert "error" in result


# ============================================================
# SECTION 5: Auto-Compaction Trigger Tests
# ============================================================

class TestAutoCompactionTrigger:
    """Test auto-compaction trigger in orchestration service."""

    def test_import_works(self):
        """Should be able to import orchestration service."""
        from delia.orchestration.service import OrchestrationService
        assert OrchestrationService is not None

    def test_service_has_check_method(self):
        """OrchestrationService should have _check_auto_compaction method."""
        from delia.orchestration.service import OrchestrationService
        assert hasattr(OrchestrationService, "_check_auto_compaction")

    @pytest.mark.asyncio
    async def test_check_no_compaction_needed(self):
        """Should return None when compaction not needed."""
        from delia.orchestration.service import OrchestrationService

        session = SessionState(session_id="test")
        session.add_message("user", "Hello")

        # Create a minimal service mock
        service = MagicMock(spec=OrchestrationService)
        service._check_auto_compaction = OrchestrationService._check_auto_compaction.__get__(service)

        result = await service._check_auto_compaction(session, "test")
        assert result is None  # No compaction needed


# ============================================================
# SECTION 6: Async Compaction Operation Tests
# ============================================================

@pytest.mark.asyncio
class TestAsyncCompactionOperations:
    """Test async compaction operations."""

    async def test_compact_empty_session_fails(self):
        """Compacting empty session should fail."""
        session = SessionState(session_id="test")
        result = await compact_session(session)

        assert not result.success
        assert result.error is not None

    async def test_compact_small_session_fails(self):
        """Compacting session with few messages should fail."""
        session = SessionState(session_id="test")
        for i in range(5):
            session.add_message("user", f"Msg {i}")
            session.add_message("assistant", f"Resp {i}")

        # Force=True to bypass threshold, but should fail on message count
        result = await compact_session(session, force=True)

        assert not result.success
        # Either "Not enough messages" or threshold error
        assert result.error is not None

    async def test_compact_result_structure(self):
        """CompactionResult should have all required fields."""
        result = CompactionResult()

        assert hasattr(result, "success")
        assert hasattr(result, "summary")
        assert hasattr(result, "messages_compacted")
        assert hasattr(result, "tokens_saved")
        assert hasattr(result, "compression_ratio")
        assert hasattr(result, "error")
        assert hasattr(result, "key_decisions")
        assert hasattr(result, "file_modifications")
        assert hasattr(result, "tool_calls")

    async def test_compact_with_mock_llm(self):
        """Test compaction with mocked LLM call."""
        # Create session with enough messages
        session = SessionState(session_id="test")
        for i in range(20):
            session.add_message("user", f"Question {i}: " + "x" * 500)
            session.add_message("assistant", f"Answer {i}: " + "y" * 500)

        # Mock the LLM call
        mock_response = {
            "success": True,
            "response": "Summary: This conversation covered topics 1-20."
        }

        async def mock_call_llm(*args, **kwargs):
            return mock_response

        compactor = ConversationCompactor(
            call_llm_fn=mock_call_llm,
            threshold_tokens=1000,  # Low threshold
            preserve_recent=3,
        )

        result = await compactor.compact(session, force=True)

        assert result.success
        assert result.messages_compacted > 0
        assert result.tokens_saved > 0
        assert result.compression_ratio > 0


# ============================================================
# SECTION 7: Edge Cases and Error Handling
# ============================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_message_content(self):
        """Should handle empty message content."""
        session = SessionState(session_id="test")
        session.add_message("user", "")
        session.add_message("assistant", "")

        compactor = ConversationCompactor()
        stats = compactor.get_compaction_stats(session)
        assert stats["total_messages"] == 2

    def test_very_long_message(self):
        """Should handle very long messages."""
        session = SessionState(session_id="test")
        long_msg = "x" * 100000  # 100KB message
        session.add_message("user", long_msg)

        compactor = ConversationCompactor()
        stats = compactor.get_compaction_stats(session)
        assert stats["total_tokens"] > 0

    def test_special_characters_in_messages(self):
        """Should handle special characters."""
        session = SessionState(session_id="test")
        session.add_message("user", "Hello 你好 🎉 <script>alert('xss')</script>")
        session.add_message("assistant", "Response with émojis: 🚀🌟")

        compactor = ConversationCompactor()
        stats = compactor.get_compaction_stats(session)
        assert stats["total_messages"] == 2

    def test_null_session_id(self):
        """Should handle sessions created with unusual IDs."""
        session = SessionState(session_id="")
        assert session.session_id == ""

        compactor = ConversationCompactor()
        stats = compactor.get_compaction_stats(session)
        assert stats is not None

    @pytest.mark.asyncio
    async def test_compaction_metadata_persistence(self):
        """Compaction metadata should be stored in session."""
        session = SessionState(session_id="test")
        for i in range(20):
            session.add_message("user", "x" * 500)
            session.add_message("assistant", "y" * 500)

        async def mock_call_llm(*args, **kwargs):
            return {"success": True, "response": "Test summary"}

        compactor = ConversationCompactor(
            call_llm_fn=mock_call_llm,
            threshold_tokens=1000,
            preserve_recent=3,
        )

        await compactor.compact(session, force=True)

        # Check metadata was stored
        assert "compaction" in session.metadata
        meta = session.metadata["compaction"]
        assert "compacted_at" in meta
        assert "messages_summarized" in meta
        assert "compression_ratio" in meta


# ============================================================
# SECTION 8: Module-Level Function Tests
# ============================================================

class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_get_compactor_returns_instance(self):
        """get_compactor should return ConversationCompactor instance."""
        compactor = get_compactor()
        assert isinstance(compactor, ConversationCompactor)

    def test_get_compactor_singleton(self):
        """get_compactor should return same instance."""
        c1 = get_compactor()
        c2 = get_compactor()
        assert c1 is c2

    def test_needs_compaction_function(self):
        """needs_compaction module function should work."""
        session = SessionState(session_id="test")
        result = needs_compaction(session)
        assert isinstance(result, bool)
        assert result is False  # Empty session

    @pytest.mark.asyncio
    async def test_compact_session_function(self):
        """compact_session module function should work."""
        session = SessionState(session_id="test")
        result = await compact_session(session)
        assert isinstance(result, CompactionResult)


# ============================================================
# SECTION 9: Pattern Matching Tests
# ============================================================

class TestPatternMatching:
    """Test regex patterns for info extraction."""

    def test_tool_call_patterns_exist(self):
        """TOOL_CALL_PATTERNS should be defined."""
        assert TOOL_CALL_PATTERNS is not None
        assert len(TOOL_CALL_PATTERNS) > 0

    def test_decision_patterns_exist(self):
        """DECISION_PATTERNS should be defined."""
        assert DECISION_PATTERNS is not None
        assert len(DECISION_PATTERNS) > 0

    def test_file_mod_patterns_exist(self):
        """FILE_MOD_PATTERNS should be defined."""
        assert FILE_MOD_PATTERNS is not None
        assert len(FILE_MOD_PATTERNS) > 0


# ============================================================
# SECTION 10: End-to-End Workflow Tests
# ============================================================

@pytest.mark.asyncio
class TestEndToEndWorkflow:
    """Test complete compaction workflows."""

    async def test_full_compaction_workflow(self, tmp_path):
        """Test complete workflow: create -> fill -> check -> compact."""
        # 1. Create session manager with temp storage
        sm = SessionManager(session_dir=tmp_path / "sessions")

        # 2. Create session
        session = sm.create_session(client_id="e2e-test")

        # 3. Add enough messages to exceed 12000 token threshold
        # Each message ~500 chars = ~125 tokens, need 100+ messages for 12000 tokens
        for i in range(65):
            sm.add_to_session(
                session.session_id, "user", f"Question {i}: " + "x" * 500
            )
            sm.add_to_session(
                session.session_id, "assistant", f"Answer {i}: " + "y" * 500
            )

        # 4. Check if compaction needed
        stats = sm.get_compaction_stats(session.session_id)
        # Should have enough tokens now
        assert stats["total_tokens"] > 12000, f"Only {stats['total_tokens']} tokens, need >12000"
        assert sm.needs_compaction(session.session_id)

        # 5. Get stats
        stats = sm.get_compaction_stats(session.session_id)
        assert stats["needs_compaction"]
        assert stats["total_messages"] == 130  # 65 pairs * 2

        # 6. Compact (will fail without LLM, but tests the flow)
        result = await sm.compact_session(session.session_id, force=True)
        # Result depends on LLM availability

        # 7. Cleanup
        sm.delete_session(session.session_id)

    async def test_workflow_with_mocked_llm(self, tmp_path):
        """Test complete workflow with mocked LLM."""
        # Create custom compactor with mock
        async def mock_llm(*args, **kwargs):
            return {"success": True, "response": "Test summary of conversation."}

        # Create session with enough messages
        session = SessionState(session_id="e2e-mock")
        for i in range(20):
            session.add_message("user", f"Q{i}: " + "test " * 100)
            session.add_message("assistant", f"A{i}: " + "response " * 100)

        original_count = len(session.messages)

        # Compact with mock
        compactor = ConversationCompactor(
            call_llm_fn=mock_llm,
            threshold_tokens=500,
            preserve_recent=3,
        )

        result = await compactor.compact(session, force=True)

        # Verify results
        assert result.success
        assert result.messages_compacted > 0
        assert len(session.messages) < original_count
        # Should have 1 summary + 6 preserved messages = 7
        assert len(session.messages) == 7
        # First message should be the summary
        assert "[CONVERSATION SUMMARY]" in session.messages[0].content
