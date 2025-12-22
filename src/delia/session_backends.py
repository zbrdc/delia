# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Session Storage Backends for Delia.

Provides pluggable session storage with support for:
- JSON files (default, backwards compatible)
- SQLite (single file, queryable)
- Redis (future: distributed deployments)

Usage:
    from delia.session_backends import get_session_backend
    
    # Auto-selects based on settings.json
    backend = get_session_backend()
    
    # Or explicitly:
    backend = SQLiteSessionBackend(db_path)
    
    session = backend.load("session-id")
    backend.save(session)
    
    # Query sessions (SQLite only)
    recent = backend.query(since="2024-12-01", task_type="review")
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import structlog

from . import paths

log = structlog.get_logger()


# =============================================================================
# Abstract Backend
# =============================================================================

class SessionBackend(ABC):
    """Abstract base class for session storage backends."""
    
    @abstractmethod
    def save(self, session_id: str, data: dict) -> None:
        """Save a session to storage."""
        pass
    
    @abstractmethod
    def load(self, session_id: str) -> dict | None:
        """Load a session from storage. Returns None if not found."""
        pass
    
    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""
        pass
    
    @abstractmethod
    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        pass
    
    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        pass
    
    def query(
        self,
        since: str | None = None,
        until: str | None = None,
        client_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Query sessions with filters. Default implementation loads all and filters.
        Backends like SQLite override for efficient querying.
        """
        results = []
        for session_id in self.list_sessions():
            if len(results) >= limit:
                break
            session = self.load(session_id)
            if session is None:
                continue
            
            # Apply filters
            if since and session.get("created_at", "") < since:
                continue
            if until and session.get("created_at", "") > until:
                continue
            if client_id and session.get("client_id") != client_id:
                continue
            
            results.append(session)
        
        return results
    
    def cleanup_expired(self, ttl_seconds: int) -> int:
        """
        Remove expired sessions. Returns count of deleted sessions.
        """
        cutoff = (datetime.now() - timedelta(seconds=ttl_seconds)).isoformat()
        deleted = 0
        
        for session_id in self.list_sessions():
            session = self.load(session_id)
            if session and session.get("last_accessed", "") < cutoff:
                if self.delete(session_id):
                    deleted += 1
        
        return deleted


# =============================================================================
# JSON File Backend (Original Implementation)
# =============================================================================

class JSONSessionBackend(SessionBackend):
    """
    JSON file-based session storage.
    
    Each session is stored as a separate JSON file.
    Backwards compatible with existing Delia sessions.
    """

    def __init__(self, session_dir: Path | None = None):
        # PER-PROJECT ISOLATION: Default to project-specific sessions
        if session_dir is None:
            session_dir = Path.cwd() / ".delia" / "sessions"
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        log.debug("json_backend_initialized", path=str(self.session_dir))
    
    def _session_path(self, session_id: str) -> Path:
        return self.session_dir / f"{session_id}.json"
    
    def save(self, session_id: str, data: dict) -> None:
        with self._lock:
            session_file = self._session_path(session_id)
            temp_file = session_file.with_suffix(".tmp")
            
            try:
                temp_file.write_text(json.dumps(data, indent=2, default=str))
                temp_file.replace(session_file)  # Atomic on POSIX
            except Exception as e:
                log.error("json_save_failed", session_id=session_id, error=str(e))
                if temp_file.exists():
                    temp_file.unlink()
                raise
    
    def load(self, session_id: str) -> dict | None:
        session_file = self._session_path(session_id)
        
        if not session_file.exists():
            return None
        
        try:
            return json.loads(session_file.read_text())
        except json.JSONDecodeError as e:
            log.error("json_load_failed", session_id=session_id, error=str(e))
            return None
    
    def delete(self, session_id: str) -> bool:
        session_file = self._session_path(session_id)
        
        if session_file.exists():
            session_file.unlink()
            return True
        return False
    
    def list_sessions(self) -> list[str]:
        return [f.stem for f in self.session_dir.glob("*.json")]
    
    def exists(self, session_id: str) -> bool:
        return self._session_path(session_id).exists()


# =============================================================================
# SQLite Backend
# =============================================================================

class SQLiteSessionBackend(SessionBackend):
    """
    SQLite-based session storage.
    
    Benefits over JSON:
    - Single file instead of many
    - Efficient queries by date, client, etc.
    - ACID compliance
    - Built-in Python, no dependencies
    
    Schema:
    - sessions: id, client_id, created_at, last_accessed, metadata_json
    - messages: session_id, idx, role, content, timestamp, tokens, model, task_type
    """
    
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (paths.DATA_DIR / "sessions.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        
        # Initialize schema
        self._init_schema()
        
        log.info("sqlite_backend_initialized", path=str(self.db_path))
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode for better concurrency
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn.execute("PRAGMA journal_mode = WAL")
        return self._local.conn
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                client_id TEXT,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                total_tokens INTEGER DEFAULT 0,
                total_calls INTEGER DEFAULT 0,
                models_used TEXT DEFAULT '[]',
                metadata_json TEXT DEFAULT '{}'
            );
            
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,
                model TEXT DEFAULT '',
                task_type TEXT DEFAULT '',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            
            CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_client ON sessions(client_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_accessed ON sessions(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        """)
        
        conn.commit()
    
    def save(self, session_id: str, data: dict) -> None:
        """Save session to SQLite."""
        conn = self._get_connection()
        
        try:
            # Convert models_used set to list for JSON
            models_used = list(data.get("models_used", []))
            metadata = data.get("metadata", {})
            
            # Upsert session
            conn.execute("""
                INSERT INTO sessions (
                    session_id, client_id, created_at, last_accessed,
                    total_tokens, total_calls, models_used, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    client_id = excluded.client_id,
                    last_accessed = excluded.last_accessed,
                    total_tokens = excluded.total_tokens,
                    total_calls = excluded.total_calls,
                    models_used = excluded.models_used,
                    metadata_json = excluded.metadata_json
            """, (
                session_id,
                data.get("client_id"),
                data.get("created_at", datetime.now().isoformat()),
                data.get("last_accessed", datetime.now().isoformat()),
                data.get("total_tokens", 0),
                data.get("total_calls", 0),
                json.dumps(models_used),
                json.dumps(metadata),
            ))
            
            # Delete existing messages and re-insert
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            
            messages = data.get("messages", [])
            for idx, msg in enumerate(messages):
                conn.execute("""
                    INSERT INTO messages (
                        session_id, idx, role, content, timestamp, tokens, model, task_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    idx,
                    msg.get("role", "user"),
                    msg.get("content", ""),
                    msg.get("timestamp", datetime.now().isoformat()),
                    msg.get("tokens", 0),
                    msg.get("model", ""),
                    msg.get("task_type", ""),
                ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            log.error("sqlite_save_failed", session_id=session_id, error=str(e))
            raise
    
    def load(self, session_id: str) -> dict | None:
        """Load session from SQLite."""
        conn = self._get_connection()
        
        # Get session
        cursor = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # Get messages
        msg_cursor = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY idx",
            (session_id,)
        )
        
        messages = []
        for msg_row in msg_cursor:
            messages.append({
                "role": msg_row["role"],
                "content": msg_row["content"],
                "timestamp": msg_row["timestamp"],
                "tokens": msg_row["tokens"],
                "model": msg_row["model"],
                "task_type": msg_row["task_type"],
            })
        
        return {
            "session_id": row["session_id"],
            "client_id": row["client_id"],
            "created_at": row["created_at"],
            "last_accessed": row["last_accessed"],
            "total_tokens": row["total_tokens"],
            "total_calls": row["total_calls"],
            "models_used": set(json.loads(row["models_used"])),
            "metadata": json.loads(row["metadata_json"]),
            "messages": messages,
        }
    
    def delete(self, session_id: str) -> bool:
        """Delete session from SQLite."""
        conn = self._get_connection()
        
        cursor = conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        conn.commit()
        
        return cursor.rowcount > 0
    
    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        conn = self._get_connection()
        
        cursor = conn.execute("SELECT session_id FROM sessions ORDER BY last_accessed DESC")
        return [row["session_id"] for row in cursor]
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        conn = self._get_connection()
        
        cursor = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        return cursor.fetchone() is not None
    
    def query(
        self,
        since: str | None = None,
        until: str | None = None,
        client_id: str | None = None,
        task_type: str | None = None,
        model: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Query sessions with filters (efficient SQL implementation).
        
        Args:
            since: Filter sessions created after this ISO timestamp
            until: Filter sessions created before this ISO timestamp
            client_id: Filter by client ID
            task_type: Filter by task type (searches messages)
            model: Filter by model used (searches messages or models_used)
            limit: Maximum results to return
            
        Returns:
            List of session dicts matching criteria
        """
        conn = self._get_connection()
        
        conditions = []
        params = []
        
        if since:
            conditions.append("s.created_at >= ?")
            params.append(since)
        if until:
            conditions.append("s.created_at <= ?")
            params.append(until)
        if client_id:
            conditions.append("s.client_id = ?")
            params.append(client_id)
        if model:
            conditions.append("s.models_used LIKE ?")
            params.append(f"%{model}%")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # If task_type filter, need to join with messages
        if task_type:
            query = f"""
                SELECT DISTINCT s.session_id FROM sessions s
                JOIN messages m ON s.session_id = m.session_id
                WHERE {where_clause} AND m.task_type = ?
                ORDER BY s.last_accessed DESC
                LIMIT ?
            """
            params.extend([task_type, limit])
        else:
            query = f"""
                SELECT session_id FROM sessions s
                WHERE {where_clause}
                ORDER BY last_accessed DESC
                LIMIT ?
            """
            params.append(limit)
        
        cursor = conn.execute(query, params)
        
        results = []
        for row in cursor:
            session = self.load(row["session_id"])
            if session:
                results.append(session)
        
        return results
    
    def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove expired sessions efficiently."""
        conn = self._get_connection()
        
        cutoff = (datetime.now() - timedelta(seconds=ttl_seconds)).isoformat()
        
        cursor = conn.execute(
            "DELETE FROM sessions WHERE last_accessed < ?",
            (cutoff,)
        )
        conn.commit()
        
        deleted = cursor.rowcount
        if deleted > 0:
            log.info("sqlite_cleanup", deleted=deleted, cutoff=cutoff)
        
        return deleted
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        conn = self._get_connection()
        
        stats = {}
        
        # Session count
        cursor = conn.execute("SELECT COUNT(*) as count FROM sessions")
        stats["total_sessions"] = cursor.fetchone()["count"]
        
        # Message count
        cursor = conn.execute("SELECT COUNT(*) as count FROM messages")
        stats["total_messages"] = cursor.fetchone()["count"]
        
        # Total tokens
        cursor = conn.execute("SELECT SUM(total_tokens) as total FROM sessions")
        row = cursor.fetchone()
        stats["total_tokens"] = row["total"] or 0
        
        # Unique clients
        cursor = conn.execute("SELECT COUNT(DISTINCT client_id) as count FROM sessions WHERE client_id IS NOT NULL")
        stats["unique_clients"] = cursor.fetchone()["count"]
        
        # Most used models
        cursor = conn.execute("""
            SELECT model, COUNT(*) as count FROM messages
            WHERE model != '' GROUP BY model ORDER BY count DESC LIMIT 5
        """)
        stats["top_models"] = {row["model"]: row["count"] for row in cursor}
        
        # Database file size
        stats["db_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return stats
    
    def vacuum(self) -> None:
        """Reclaim unused space in database."""
        conn = self._get_connection()
        conn.execute("VACUUM")
        log.info("sqlite_vacuum_complete")


# =============================================================================
# Migration Tool
# =============================================================================

def migrate_json_to_sqlite(
    json_dir: Path | None = None,
    sqlite_path: Path | None = None,
    delete_json: bool = False,
) -> int:
    """
    Migrate existing JSON sessions to SQLite.

    Args:
        json_dir: Source JSON directory (default: <project>/.delia/sessions/)
        sqlite_path: Target SQLite database (default: paths.DATA_DIR / "sessions.db")
        delete_json: Whether to delete JSON files after migration
        
    Returns:
        Number of sessions migrated
    """
    json_backend = JSONSessionBackend(json_dir)
    sqlite_backend = SQLiteSessionBackend(sqlite_path)
    
    session_ids = json_backend.list_sessions()
    migrated = 0
    
    log.info("migration_starting", total=len(session_ids))
    
    for session_id in session_ids:
        data = json_backend.load(session_id)
        if data is None:
            log.warning("migration_skip_invalid", session_id=session_id)
            continue
        
        try:
            sqlite_backend.save(session_id, data)
            migrated += 1
            
            if delete_json:
                json_backend.delete(session_id)
                
        except Exception as e:
            log.error("migration_failed", session_id=session_id, error=str(e))
    
    log.info("migration_complete", migrated=migrated, total=len(session_ids))
    
    return migrated


# =============================================================================
# Factory Function
# =============================================================================

_backend_instance: SessionBackend | None = None


def get_session_backend(backend_type: str | None = None) -> SessionBackend:
    """
    Get the configured session backend.
    
    Args:
        backend_type: Override backend type ("json", "sqlite")
                     If None, reads from settings.json or defaults to "json"
    
    Returns:
        Configured SessionBackend instance
    """
    global _backend_instance
    
    if _backend_instance is not None and backend_type is None:
        return _backend_instance
    
    # Determine backend type
    if backend_type is None:
        # Try to read from settings
        try:
            from .backend_manager import get_settings
            settings = get_settings()
            backend_type = settings.get("session_backend", "json")
        except Exception:
            backend_type = "json"
    
    # Create appropriate backend
    if backend_type == "sqlite":
        _backend_instance = SQLiteSessionBackend()
    else:
        _backend_instance = JSONSessionBackend()
    
    log.info("session_backend_selected", type=backend_type)
    
    return _backend_instance

