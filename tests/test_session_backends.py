# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for session storage backends."""

import tempfile
from pathlib import Path

import pytest

from delia.session_backends import (
    JSONSessionBackend,
    SQLiteSessionBackend,
    migrate_json_to_sqlite,
)


@pytest.fixture
def sample_session():
    """Sample session data for testing backend storage/retrieval."""
    return {
        "session_id": "test-session-123",
        "client_id": "user-1",
        "created_at": "2024-12-16T12:00:00",
        "last_accessed": "2024-12-16T12:30:00",
        "total_tokens": 150,
        "total_calls": 3,
        "models_used": ["model-a", "model-b"],  # Arbitrary model names
        "metadata": {"project": "test-project"},
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": "2024-12-16T12:00:00",
                "tokens": 5,
                "model": "",
                "task_type": "",
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you!",
                "timestamp": "2024-12-16T12:00:01",
                "tokens": 10,
                "model": "model-a",  # Just testing string storage
                "task_type": "quick",
            },
            {
                "role": "user",
                "content": "Can you review this code?",
                "timestamp": "2024-12-16T12:01:00",
                "tokens": 8,
                "model": "",
                "task_type": "",
            },
        ],
    }


class TestJSONBackend:
    """Tests for JSON file backend."""

    def test_save_and_load(self, sample_session):
        """Test basic save and load operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONSessionBackend(Path(tmpdir))
            
            backend.save("test-123", sample_session)
            loaded = backend.load("test-123")
            
            assert loaded is not None
            assert loaded["session_id"] == sample_session["session_id"]
            assert len(loaded["messages"]) == 3

    def test_load_nonexistent(self):
        """Test loading a nonexistent session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONSessionBackend(Path(tmpdir))
            
            result = backend.load("does-not-exist")
            assert result is None

    def test_delete(self, sample_session):
        """Test deleting a session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONSessionBackend(Path(tmpdir))
            
            backend.save("test-delete", sample_session)
            assert backend.exists("test-delete")
            
            result = backend.delete("test-delete")
            assert result is True
            assert not backend.exists("test-delete")

    def test_list_sessions(self, sample_session):
        """Test listing all sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONSessionBackend(Path(tmpdir))
            
            backend.save("session-1", sample_session)
            backend.save("session-2", sample_session)
            backend.save("session-3", sample_session)
            
            sessions = backend.list_sessions()
            assert len(sessions) == 3
            assert "session-1" in sessions
            assert "session-2" in sessions


class TestSQLiteBackend:
    """Tests for SQLite backend."""

    def test_save_and_load(self, sample_session):
        """Test basic save and load operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteSessionBackend(Path(tmpdir) / "test.db")
            
            session_id = sample_session["session_id"]
            backend.save(session_id, sample_session)
            loaded = backend.load(session_id)
            
            assert loaded is not None
            assert loaded["session_id"] == session_id
            assert len(loaded["messages"]) == 3
            assert loaded["total_tokens"] == 150

    def test_load_nonexistent(self):
        """Test loading a nonexistent session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteSessionBackend(Path(tmpdir) / "test.db")
            
            result = backend.load("does-not-exist")
            assert result is None

    def test_delete(self, sample_session):
        """Test deleting a session (with cascade to messages)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteSessionBackend(Path(tmpdir) / "test.db")
            
            backend.save("test-delete", sample_session)
            assert backend.exists("test-delete")
            
            result = backend.delete("test-delete")
            assert result is True
            assert not backend.exists("test-delete")

    def test_query_by_date(self, sample_session):
        """Test querying sessions by date range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteSessionBackend(Path(tmpdir) / "test.db")
            
            # Create sessions with different dates
            session_old = {**sample_session, "session_id": "old", "created_at": "2024-01-01T12:00:00"}
            session_new = {**sample_session, "session_id": "new", "created_at": "2024-12-15T12:00:00"}
            
            backend.save("old", session_old)
            backend.save("new", session_new)
            
            # Query recent sessions
            results = backend.query(since="2024-12-01")
            assert len(results) == 1
            assert results[0]["session_id"] == "new"

    def test_query_by_client(self, sample_session):
        """Test querying sessions by client ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteSessionBackend(Path(tmpdir) / "test.db")
            
            session_user1 = {**sample_session, "session_id": "user1-session", "client_id": "user-1"}
            session_user2 = {**sample_session, "session_id": "user2-session", "client_id": "user-2"}
            
            backend.save("user1-session", session_user1)
            backend.save("user2-session", session_user2)
            
            results = backend.query(client_id="user-1")
            assert len(results) == 1
            assert results[0]["client_id"] == "user-1"

    def test_stats(self, sample_session):
        """Test getting database statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteSessionBackend(Path(tmpdir) / "test.db")
            
            backend.save("session-1", sample_session)
            backend.save("session-2", sample_session)
            
            stats = backend.get_stats()
            
            assert stats["total_sessions"] == 2
            assert stats["total_messages"] == 6  # 3 messages * 2 sessions
            assert stats["db_size_bytes"] > 0

    def test_cleanup_expired(self, sample_session):
        """Test cleaning up expired sessions."""
        from datetime import datetime
        
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteSessionBackend(Path(tmpdir) / "test.db")
            
            now = datetime.now().isoformat()
            
            # Create an "old" session
            old_session = {
                **sample_session,
                "session_id": "old-session",
                "last_accessed": "2020-01-01T12:00:00",
            }
            # Create a "new" session with recent timestamp
            new_session = {
                **sample_session,
                "session_id": "new-session",
                "last_accessed": now,
            }
            
            backend.save("old-session", old_session)
            backend.save("new-session", new_session)
            
            # Cleanup with 1 hour TTL (old session should be deleted)
            deleted = backend.cleanup_expired(ttl_seconds=3600)
            
            assert deleted == 1
            assert not backend.exists("old-session")
            assert backend.exists("new-session")


class TestMigration:
    """Tests for JSON to SQLite migration."""

    def test_migration(self, sample_session):
        """Test migrating sessions from JSON to SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            json_dir = tmpdir / "json_sessions"
            sqlite_path = tmpdir / "migrated.db"
            
            # Create JSON sessions
            json_backend = JSONSessionBackend(json_dir)
            for i in range(5):
                session = {**sample_session, "session_id": f"session-{i}"}
                json_backend.save(f"session-{i}", session)
            
            # Migrate
            count = migrate_json_to_sqlite(
                json_dir=json_dir,
                sqlite_path=sqlite_path,
                delete_json=False,
            )
            
            assert count == 5
            
            # Verify SQLite has all sessions
            sqlite_backend = SQLiteSessionBackend(sqlite_path)
            assert len(sqlite_backend.list_sessions()) == 5
            
            # Verify JSON files still exist (delete_json=False)
            assert len(json_backend.list_sessions()) == 5

    def test_migration_with_delete(self, sample_session):
        """Test migration with JSON file deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            json_dir = tmpdir / "json_sessions"
            sqlite_path = tmpdir / "migrated.db"
            
            # Create JSON sessions
            json_backend = JSONSessionBackend(json_dir)
            json_backend.save("to-migrate", sample_session)
            
            # Migrate with deletion
            count = migrate_json_to_sqlite(
                json_dir=json_dir,
                sqlite_path=sqlite_path,
                delete_json=True,
            )
            
            assert count == 1
            
            # Verify JSON files deleted
            assert len(json_backend.list_sessions()) == 0

