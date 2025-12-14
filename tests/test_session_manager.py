# Delia - Local LLM Orchestration
# Copyright (C) 2024 Dan Yishai
# Licensed under GPL-3.0

"""Comprehensive tests for session manager module."""

import json
import os
import pytest
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

# Set test data dir before imports
os.environ["DELIA_DATA_DIR"] = "/tmp/delia-test-sessions"


class TestSessionMessage:
    """Tests for SessionMessage dataclass."""

    def test_create_message(self):
        from delia.session_manager import SessionMessage
        msg = SessionMessage(
            role="user",
            content="Hello",
            timestamp=datetime.now().isoformat(),
        )
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tokens == 0

    def test_message_with_metadata(self):
        from delia.session_manager import SessionMessage
        msg = SessionMessage(
            role="assistant",
            content="Response",
            timestamp=datetime.now().isoformat(),
            tokens=50,
            model="qwen2.5:14b",
            task_type="review",
        )
        assert msg.tokens == 50
        assert msg.model == "qwen2.5:14b"

    def test_to_dict_from_dict(self):
        from delia.session_manager import SessionMessage
        msg = SessionMessage(
            role="user",
            content="Test",
            timestamp=datetime.now().isoformat(),
            tokens=10,
            model="test-model",
        )
        data = msg.to_dict()
        restored = SessionMessage.from_dict(data)

        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.tokens == msg.tokens


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_create_session_state(self):
        from delia.session_manager import SessionState
        state = SessionState(
            session_id="test-123",
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            messages=[],
            metadata={},
        )
        assert state.session_id == "test-123"
        assert state.total_tokens == 0

    def test_add_message(self):
        from delia.session_manager import SessionState
        state = SessionState(
            session_id="test-123",
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            messages=[],
            metadata={},
        )
        state.add_message("user", "Hello", tokens=10, model="test", task_type="quick")
        assert len(state.messages) == 1
        assert state.total_tokens == 10
        assert state.total_calls == 0  # Only assistant messages count

    def test_add_assistant_message(self):
        from delia.session_manager import SessionState
        state = SessionState(
            session_id="test-123",
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            messages=[],
            metadata={},
        )
        state.add_message("assistant", "Response", tokens=50, model="test", task_type="quick")
        assert len(state.messages) == 1
        assert state.total_tokens == 50
        assert state.total_calls == 1
        assert "test" in state.models_used

    def test_get_context_window_empty(self):
        from delia.session_manager import SessionState
        state = SessionState(
            session_id="test-123",
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            messages=[],
            metadata={},
        )
        context = state.get_context_window()
        assert context == ""

    def test_get_context_window_with_messages(self):
        from delia.session_manager import SessionState
        state = SessionState(
            session_id="test-123",
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            messages=[],
            metadata={},
        )
        state.add_message("user", "Question 1")
        state.add_message("assistant", "Answer 1")
        state.add_message("user", "Question 2")

        context = state.get_context_window()
        assert "[user]:" in context
        assert "[assistant]:" in context
        assert "Question 1" in context
        assert "Answer 1" in context

    def test_is_expired(self):
        from delia.session_manager import SessionState
        # Create session with old timestamp
        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        state = SessionState(
            session_id="test-123",
            created_at=old_time,
            last_accessed=old_time,
            messages=[],
            metadata={},
        )
        assert state.is_expired(ttl_seconds=86400) is True

        # Recent session
        state.last_accessed = datetime.now().isoformat()
        assert state.is_expired(ttl_seconds=86400) is False

    def test_to_dict_from_dict(self):
        from delia.session_manager import SessionState
        state = SessionState(
            session_id="test-123",
            client_id="client-1",
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            messages=[],
            metadata={"key": "value"},
        )
        state.add_message("user", "Test")

        data = state.to_dict()
        restored = SessionState.from_dict(data)

        assert restored.session_id == state.session_id
        assert restored.client_id == state.client_id
        assert len(restored.messages) == 1
        assert isinstance(restored.models_used, set)


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment."""
        os.environ["DELIA_DATA_DIR"] = str(tmp_path)
        # Clear any cached instances
        import delia.session_manager as sm
        sm._global_session_manager = None

        # Clear cached modules to ensure fresh import
        import sys
        modules_to_clear = ["delia.session_manager", "delia.paths", "delia"]
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) or mod == m for m in modules_to_clear):
                del sys.modules[mod]

    def test_create_session(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions")

        session = manager.create_session()
        assert session is not None
        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID format

    def test_create_session_with_client_id(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions")

        session = manager.create_session(client_id="user-123")
        assert session.client_id == "user-123"

    def test_get_session(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions")

        created = manager.create_session()
        retrieved = manager.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions")

        result = manager.get_session("nonexistent-id")
        assert result is None

    def test_add_to_session(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions")

        session = manager.create_session()
        success = manager.add_to_session(
            session.session_id, "user", "Hello!", tokens=5
        )

        assert success is True
        retrieved = manager.get_session(session.session_id)
        assert len(retrieved.messages) == 1

    def test_list_sessions(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions")

        manager.create_session(client_id="client-1")
        manager.create_session(client_id="client-1")
        manager.create_session(client_id="client-2")

        all_sessions = manager.list_sessions()
        assert len(all_sessions) == 3

        client1_sessions = manager.list_sessions(client_id="client-1")
        assert len(client1_sessions) == 2

    def test_delete_session(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions")

        session = manager.create_session()
        manager.delete_session(session.session_id)

        assert manager.get_session(session.session_id) is None

    def test_clear_expired_sessions(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions", ttl_seconds=1)

        session = manager.create_session()
        time.sleep(1.5)  # Wait for expiration

        cleared = manager.clear_expired_sessions()
        assert cleared >= 1

    def test_persistence(self, tmp_path):
        from delia.session_manager import SessionManager

        # Create manager and add data
        manager1 = SessionManager(session_dir=tmp_path / "sessions")
        session = manager1.create_session()
        manager1.add_to_session(session.session_id, "user", "Test message")

        # Create new manager and load from disk
        manager2 = SessionManager(session_dir=tmp_path / "sessions")
        loaded = manager2.get_session(session.session_id)

        assert loaded is not None
        assert len(loaded.messages) == 1

    def test_max_messages_limit(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(
            session_dir=tmp_path / "sessions",
            max_messages_per_session=5
        )

        session = manager.create_session()
        for i in range(10):
            manager.add_to_session(session.session_id, "user", f"Message {i}")

        retrieved = manager.get_session(session.session_id)
        assert len(retrieved.messages) <= 5  # Should be capped

    def test_get_stats(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions")

        manager.create_session()
        manager.create_session()

        stats = manager.get_stats()
        assert stats["sessions_in_memory"] == 2
        assert stats["sessions_created"] == 2


class TestThreadSafety:
    """Tests for thread safety."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        os.environ["DELIA_DATA_DIR"] = str(tmp_path)
        import sys
        import delia.session_manager as sm
        sm._global_session_manager = None

        modules_to_clear = ["delia.session_manager", "delia.paths", "delia"]
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) or mod == m for m in modules_to_clear):
                del sys.modules[mod]

    def test_concurrent_session_creation(self, tmp_path):
        from delia.session_manager import SessionManager
        manager = SessionManager(session_dir=tmp_path / "sessions")

        sessions = []
        errors = []

        def create_session():
            try:
                session = manager.create_session()
                sessions.append(session)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_session) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(sessions) == 10
        # All session IDs should be unique
        ids = [s.session_id for s in sessions]
        assert len(set(ids)) == 10


class TestGlobalSingleton:
    """Tests for global session manager singleton."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        os.environ["DELIA_DATA_DIR"] = str(tmp_path)
        import sys
        import delia.session_manager as sm
        sm._global_session_manager = None

        modules_to_clear = ["delia.session_manager", "delia.paths", "delia"]
        for mod in list(sys.modules.keys()):
            if any(mod.startswith(m) or mod == m for m in modules_to_clear):
                del sys.modules[mod]

    def test_get_session_manager(self, tmp_path):
        from delia.session_manager import get_session_manager

        manager1 = get_session_manager()
        manager2 = get_session_manager()

        assert manager1 is manager2  # Same instance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
