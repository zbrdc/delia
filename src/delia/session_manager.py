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
Delia Session Manager

Hybrid in-memory + file-based session management with TTL expiration and LRU eviction.
Thread-safe implementation for concurrent access.
"""

import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING

import structlog

from . import paths
from .tokens import count_tokens

if TYPE_CHECKING:
    from .session_backends import SessionBackend

log = structlog.get_logger()

# Session configuration constants
SESSION_TTL_SECONDS = 86400  # 24 hours
MAX_SESSIONS = 1000
MAX_MESSAGES_PER_SESSION = 500


@dataclass
class SessionMessage:
    """
    Represents a single message in a conversation session.

    Attributes:
        role: Message role (user, assistant, or system)
        content: Message content text
        timestamp: ISO timestamp when message was created
        tokens: Number of tokens used (for assistant messages)
        model: Model name/tier used (for assistant messages)
        task_type: Task type identifier (for assistant messages)
    """

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str  # ISO format
    tokens: int = 0
    model: str = ""
    task_type: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMessage":
        """Create SessionMessage from dictionary."""
        return cls(**data)


@dataclass
class SessionState:
    """
    Represents the complete state of a conversation session.

    Attributes:
        session_id: Unique session identifier (UUID4)
        client_id: Optional client/user identifier
        created_at: ISO timestamp when session was created
        last_accessed: ISO timestamp when session was last accessed
        messages: List of conversation messages
        metadata: Additional session metadata
        total_tokens: Total tokens used in this session
        total_calls: Total number of LLM calls in this session
        models_used: Set of model names/tiers used in this session
        original_task: The original user request for "Am I on task?" checks.
                       Cached until explicitly overwritten by a new primary task.
        task_history: List of previous tasks for context.
    """

    session_id: str
    client_id: str | None = None
    created_at: str = ""
    last_accessed: str = ""
    messages: list[SessionMessage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    total_tokens: int = 0
    total_calls: int = 0
    models_used: set[str] = field(default_factory=set)
    original_task: str = ""  # Current primary task for focus checks
    task_history: list[str] = field(default_factory=list)  # Previous tasks

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_accessed:
            self.last_accessed = now

    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        tokens: int = 0,
        model: str = "",
        task_type: str = "",
    ) -> None:
        """
        Add a message to the session.

        Args:
            role: Message role (user, assistant, or system)
            content: Message content text
            tokens: Number of tokens used (for assistant messages)
            model: Model name/tier used (for assistant messages)
            task_type: Task type identifier (for assistant messages)
        """
        message = SessionMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            tokens=tokens,
            model=model,
            task_type=task_type,
        )

        self.messages.append(message)
        self.last_accessed = datetime.now().isoformat()

        # Update session statistics
        if tokens > 0:
            self.total_tokens += tokens
        if role == "assistant":
            self.total_calls += 1
        
        # Defensive check: only add strings to models_used set
        if model and isinstance(model, str):
            self.models_used.add(model)
        elif model:
            # Handle cases where model might be a list or other object
            log.warning("session_add_message_invalid_model_type", type=type(model).__name__, value=str(model))
            if isinstance(model, list) and model:
                self.models_used.add(str(model[0]))
            else:
                self.models_used.add(str(model))

    def get_context_window(self, max_tokens: int = 8000) -> str:
        """
        Get conversation history formatted as context, most recent first.

        Returns messages up to max_tokens, formatted as:
        [user]: Previous question...
        [assistant]: Previous answer...
        [user]: Current question...

        Args:
            max_tokens: Maximum tokens to include (approximate)

        Returns:
            Formatted conversation history string
        """
        if not self.messages:
            return ""

        # Build context from most recent messages, working backwards
        # Using accurate token counting for optimal window utilization
        lines = []
        current_tokens = 0

        for message in reversed(self.messages):
            # Format message
            line = f"[{message.role}]: {message.content}"
            
            # Use real token count if possible
            msg_tokens = message.tokens if message.tokens > 0 else count_tokens(line)

            # Check if adding this message would exceed limit
            # Always allow at least one message (the most recent)
            if current_tokens + msg_tokens > max_tokens and lines:
                break

            lines.insert(0, line)  # Prepend to maintain chronological order
            current_tokens += msg_tokens

        return "\n".join(lines)

    def set_task(self, task: str, is_new_primary: bool = True) -> None:
        """
        Set the current task for "Am I on task?" focus checks.

        Args:
            task: The user's request/task description
            is_new_primary: If True, archives current task and sets new primary.
                           If False, updates the current task without archiving.
        """
        if is_new_primary and self.original_task:
            # Archive the previous task
            self.task_history.append(self.original_task)
            # Keep only last 5 tasks in history
            self.task_history = self.task_history[-5:]
        self.original_task = task
        self.last_accessed = datetime.now().isoformat()
        log.debug("session_task_set", task=task[:100], history_len=len(self.task_history))

    def get_task_focus_prompt(self) -> str:
        """
        Generate a focus reminder for the system prompt.

        Returns:
            Formatted string to inject into system prompt for task focus.
        """
        if not self.original_task:
            return ""

        focus = f"""
### TASK FOCUS (Ground Truth)
**Original Request:** {self.original_task}

Before responding, verify: "Am I addressing this task?"
If you've drifted off-topic, acknowledge it and refocus.
"""
        if self.task_history:
            focus += f"\n**Previous Tasks:** {', '.join(self.task_history[-3:])}\n"
        return focus

    def get_compaction_stats(self) -> dict:
        """
        Get compaction statistics for this session.

        Returns:
            Dict with token counts, message counts, and compaction info.
        """
        from .compaction import get_compactor
        return get_compactor().get_compaction_stats(self)

    def needs_compaction(self) -> bool:
        """
        Check if this session needs compaction.

        Returns:
            True if token count exceeds compaction threshold.
        """
        from .compaction import needs_compaction
        return needs_compaction(self)

    def is_expired(self, ttl_seconds: int = SESSION_TTL_SECONDS) -> bool:
        """
        Check if this session has expired based on last access time.

        Args:
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if session is expired, False otherwise
        """
        last_access_time = datetime.fromisoformat(self.last_accessed)
        expiry_time = last_access_time + timedelta(seconds=ttl_seconds)
        return datetime.now() > expiry_time

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert set to list for JSON serialization
        data["models_used"] = list(self.models_used)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        """Create SessionState from dictionary."""
        # Convert messages list to SessionMessage objects
        if "messages" in data:
            data["messages"] = [SessionMessage.from_dict(msg) for msg in data["messages"]]
        
        # Convert models_used list back to set, ensuring all items are strings
        if "models_used" in data:
            raw_models = data["models_used"]
            clean_models = []
            for m in raw_models:
                if isinstance(m, list):
                    # Flatten list if present
                    clean_models.extend([str(item) for item in m if item])
                elif m:
                    clean_models.append(str(m))
            data["models_used"] = set(clean_models)
            
        return cls(**data)


class SessionManager:
    """
    Thread-safe hybrid session manager for conversation state.

    Features:
    - In-memory dict for fast lookups with lazy loading from disk
    - Individual JSON files per session for scalability
    - TTL expiration (default 24 hours)
    - LRU eviction when exceeding max sessions
    - Thread-safe operations with threading.Lock
    - Per-session message limits
    """

    def __init__(
        self,
        session_dir: Path | None = None,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        max_sessions: int = MAX_SESSIONS,
        max_messages_per_session: int = MAX_MESSAGES_PER_SESSION,
        backend: "SessionBackend | None" = None,
    ):
        """
        Initialize the session manager.

        Args:
            session_dir: Optional session directory path. Defaults to paths.SESSIONS_DIR
            ttl_seconds: Time-to-live for sessions in seconds (default 24 hours)
            max_sessions: Maximum number of sessions before LRU eviction
            max_messages_per_session: Maximum messages per session
            backend: Optional SessionBackend (SQLite, JSON, etc.). If None, uses JSON.
        """
        self.session_dir = session_dir or paths.SESSIONS_DIR
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session

        # Storage backend (None = use legacy JSON file operations)
        self._backend = backend
        
        # In-memory session cache (lazy loaded)
        self._memory_cache: dict[str, SessionState] = {}

        # Statistics tracking
        self._sessions_created = 0
        self._sessions_loaded = 0
        self._evictions = 0

        # Thread safety
        self._lock = threading.Lock()

        # Ensure session directory exists
        paths.ensure_directories()
        
        backend_type = type(backend).__name__ if backend else "legacy_json"

        log.info(
            "session_manager_initialized",
            ttl_seconds=ttl_seconds,
            max_sessions=max_sessions,
            max_messages_per_session=max_messages_per_session,
            session_dir=str(self.session_dir),
            backend=backend_type,
        )

    def create_session(
        self,
        client_id: str | None = None,
        metadata: dict | None = None,
    ) -> SessionState:
        """
        Create a new session.

        Args:
            client_id: Optional client/user identifier
            metadata: Optional session metadata

        Returns:
            Newly created SessionState
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        session = SessionState(
            session_id=session_id,
            client_id=client_id,
            created_at=now,
            last_accessed=now,
            metadata=metadata or {},
        )

        with self._lock:
            self._memory_cache[session_id] = session
            self._sessions_created += 1

            # Check if eviction is needed
            if len(self._memory_cache) > self.max_sessions:
                self._evict_lru()

            # Save to disk
            self._save_session(session_id)

            log.info(
                "session_created",
                session_id=session_id,
                client_id=client_id,
                cache_size=len(self._memory_cache),
            )

        return session

    def get_session(self, session_id: str) -> SessionState | None:
        """
        Get a session by ID. Lazy loads from disk if not in memory.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found and not expired, None otherwise
        """
        with self._lock:
            # Check memory cache first
            session = self._memory_cache.get(session_id)

            if session is None:
                # Try lazy loading from disk
                session = self._load_session(session_id)
                if session is None:
                    log.debug("session_not_found", session_id=session_id)
                    return None

                # Add to memory cache
                self._memory_cache[session_id] = session
                self._sessions_loaded += 1

            # Check expiration
            if session.is_expired(self.ttl_seconds):
                del self._memory_cache[session_id]
                # Delete from disk
                session_file = self.session_dir / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
                log.debug("session_expired", session_id=session_id)
                return None

            # Update last accessed time
            session.last_accessed = datetime.now().isoformat()

            log.debug("session_retrieved", session_id=session_id, messages=len(session.messages))
            return session

    def add_to_session(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        tokens: int = 0,
        model: str = "",
        task_type: str = "",
    ) -> bool:
        """
        Add a message to an existing session.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, or system)
            content: Message content text
            tokens: Number of tokens used (for assistant messages)
            model: Model name/tier used (for assistant messages)
            task_type: Task type identifier (for assistant messages)

        Returns:
            True if message was added, False if session not found or limit exceeded
        """
        session = self.get_session(session_id)
        if session is None:
            log.warning("add_to_session_failed", reason="session_not_found", session_id=session_id)
            return False

        with self._lock:
            # Check message limit
            if len(session.messages) >= self.max_messages_per_session:
                log.warning(
                    "add_to_session_failed",
                    reason="message_limit_exceeded",
                    session_id=session_id,
                    max_messages=self.max_messages_per_session,
                )
                return False

            # Add message
            session.add_message(
                role=role,
                content=content,
                tokens=tokens,
                model=model,
                task_type=task_type,
            )

            # Save to disk
            self._save_session(session_id)

            log.debug(
                "message_added_to_session",
                session_id=session_id,
                role=role,
                tokens=tokens,
                total_messages=len(session.messages),
            )

        return True

    def list_sessions(self, client_id: str | None = None) -> list[dict]:
        """
        List all active sessions, optionally filtered by client_id.

        Returns summary information for each session (not full message history).

        Args:
            client_id: Optional client ID to filter by

        Returns:
            List of session summary dictionaries
        """
        # Load all sessions from disk to ensure complete list
        self.load_all_from_disk()

        with self._lock:
            sessions = []
            for session_id, session in self._memory_cache.items():
                # Filter by client_id if specified
                if client_id is not None and session.client_id != client_id:
                    continue

                # Check expiration
                if session.is_expired(self.ttl_seconds):
                    continue

                # Return summary (not full message history)
                sessions.append(
                    {
                        "session_id": session_id,
                        "client_id": session.client_id,
                        "created_at": session.created_at,
                        "last_accessed": session.last_accessed,
                        "message_count": len(session.messages),
                        "total_tokens": session.total_tokens,
                        "total_calls": session.total_calls,
                        "models_used": list(session.models_used),
                        "metadata": session.metadata,
                    }
                )

            log.debug("sessions_listed", count=len(sessions), client_id=client_id)
            return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from memory and disk.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            # Remove from memory
            if session_id in self._memory_cache:
                del self._memory_cache[session_id]

            # Delete from disk
            session_file = self.session_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                log.info("session_deleted", session_id=session_id)
                return True
            else:
                log.debug("session_delete_skipped", reason="not_found", session_id=session_id)
                return False

    def clear_expired_sessions(self) -> int:
        """
        Clear all expired sessions from memory and disk.

        Returns:
            Number of sessions cleared
        """
        # Load all sessions from disk first
        self.load_all_from_disk()

        with self._lock:
            expired_sessions = []

            for session_id, session in self._memory_cache.items():
                if session.is_expired(self.ttl_seconds):
                    expired_sessions.append(session_id)

            # Remove expired sessions
            for session_id in expired_sessions:
                del self._memory_cache[session_id]

                # Delete from disk
                session_file = self.session_dir / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()

            if expired_sessions:
                log.info("expired_sessions_cleared", count=len(expired_sessions))

            return len(expired_sessions)

    def _save_session(self, session_id: str) -> None:
        """
        Save a session to storage (backend or legacy JSON).

        Must be called while holding self._lock.

        Args:
            session_id: Session identifier
        """
        session = self._memory_cache.get(session_id)
        if session is None:
            return

        try:
            if self._backend:
                # Use pluggable backend
                self._backend.save(session_id, session.to_dict())
            else:
                # Legacy JSON file storage
                session_file = self.session_dir / f"{session_id}.json"
                temp_file = session_file.with_suffix(".tmp")
                temp_file.write_text(json.dumps(session.to_dict(), indent=2))
                temp_file.replace(session_file)  # Atomic on POSIX

            log.debug("session_saved", session_id=session_id, messages=len(session.messages))
        except Exception as e:
            log.warning(
                "session_save_failed",
                error=str(e),
                session_id=session_id,
            )

    def _load_session(self, session_id: str) -> SessionState | None:
        """
        Load a session from storage (backend or legacy JSON).

        Must be called while holding self._lock.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found, None otherwise
        """
        try:
            if self._backend:
                # Use pluggable backend
                session_data = self._backend.load(session_id)
                if session_data is None:
                    return None
            else:
                # Legacy JSON file storage
                session_file = self.session_dir / f"{session_id}.json"
                if not session_file.exists():
                    return None
                session_data = json.loads(session_file.read_text())

            session = SessionState.from_dict(session_data)

            log.debug(
                "session_loaded",
                session_id=session_id,
                messages=len(session.messages),
                from_storage=True,
            )
            return session

        except json.JSONDecodeError as e:
            log.warning(
                "session_load_failed",
                error=str(e),
                reason="invalid_json",
                session_id=session_id,
            )
            return None
        except Exception as e:
            log.warning(
                "session_load_failed",
                error=str(e),
                session_id=session_id,
            )
            return None

    async def save_all_async(self) -> None:
        """
        Save all in-memory sessions to disk asynchronously.

        Note: This is a synchronous implementation for now.
        Can be made truly async if needed in the future.
        """
        with self._lock:
            session_ids = list(self._memory_cache.keys())

        for session_id in session_ids:
            with self._lock:
                self._save_session(session_id)

        log.info("all_sessions_saved", count=len(session_ids))

    def load_all_from_disk(self) -> int:
        """
        Load all sessions from storage into memory.

        Filters out expired sessions during load.

        Returns:
            Number of sessions loaded
        """
        with self._lock:
            # Get list of session IDs from backend or filesystem
            if self._backend:
                session_ids = self._backend.list_sessions()
            else:
                if not self.session_dir.exists():
                    log.debug("session_load_all_skipped", reason="dir_not_found")
                    return 0
                session_ids = [
                    f.stem for f in self.session_dir.glob("*.json")
                    if f.suffix != ".tmp"
                ]
            
            loaded_count = 0
            expired_count = 0

            for session_id in session_ids:
                # Skip if already in memory
                if session_id in self._memory_cache:
                    continue

                session = self._load_session(session_id)
                if session is None:
                    continue

                # Check expiration
                if session.is_expired(self.ttl_seconds):
                    expired_count += 1
                    # Delete expired session
                    if self._backend:
                        self._backend.delete(session_id)
                    else:
                        session_file = self.session_dir / f"{session_id}.json"
                        if session_file.exists():
                            session_file.unlink()
                    continue

                self._memory_cache[session_id] = session
                loaded_count += 1

            if loaded_count > 0 or expired_count > 0:
                log.info(
                    "sessions_loaded_from_disk",
                    loaded=loaded_count,
                    expired_filtered=expired_count,
                    total_in_memory=len(self._memory_cache),
                )

            return loaded_count

    def _evict_lru(self) -> None:
        """
        Evict least recently used sessions (oldest 10% by last_accessed).

        This method should be called when the cache exceeds max_sessions.
        Must be called while holding self._lock.
        """
        if len(self._memory_cache) <= self.max_sessions:
            return

        # Calculate number of entries to evict (10%)
        evict_count = max(1, len(self._memory_cache) // 10)

        # Score entries by last access time (older is worse)
        scored_entries = []
        for session_id, session in self._memory_cache.items():
            last_access_time = datetime.fromisoformat(session.last_accessed)
            # Score: higher is better (more recent)
            score = last_access_time.timestamp()
            scored_entries.append((score, session_id))

        # Sort by score (lowest first) and take bottom evict_count entries
        scored_entries.sort()
        sessions_to_evict = [session_id for _, session_id in scored_entries[:evict_count]]

        # Remove evicted entries
        for session_id in sessions_to_evict:
            del self._memory_cache[session_id]
            self._evictions += 1

            # Delete from disk
            session_file = self.session_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

        log.debug(
            "session_eviction",
            evicted=len(sessions_to_evict),
            remaining=len(self._memory_cache),
            total_evictions=self._evictions,
        )

    def get_stats(self) -> dict:
        """
        Get session manager statistics.

        Thread-safe operation.

        Returns:
            Dictionary with:
            - sessions_in_memory: Current number of sessions in memory
            - sessions_created: Total number of sessions created
            - sessions_loaded: Total number of sessions loaded from disk
            - evictions: Total number of evictions
            - max_sessions: Maximum allowed sessions
            - ttl_seconds: TTL in seconds
            - max_messages_per_session: Maximum messages per session
        """
        with self._lock:
            return {
                "sessions_in_memory": len(self._memory_cache),
                "sessions_created": self._sessions_created,
                "sessions_loaded": self._sessions_loaded,
                "evictions": self._evictions,
                "max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl_seconds,
                "max_messages_per_session": self.max_messages_per_session,
            }

    async def compact_session(
        self,
        session_id: str,
        force: bool = False,
    ) -> dict:
        """
        Compact a session's conversation history using LLM summarization.

        This reduces token count by summarizing older messages while
        preserving recent context and key information (tool calls, decisions).

        Args:
            session_id: Session identifier
            force: Force compaction even if below threshold

        Returns:
            Dict with compaction result:
            - success: Whether compaction was successful
            - messages_compacted: Number of messages summarized
            - tokens_saved: Tokens reduced
            - compression_ratio: Compression efficiency
            - error: Error message if failed
        """
        session = self.get_session(session_id)
        if session is None:
            return {
                "success": False,
                "error": f"Session not found: {session_id}",
            }

        from .compaction import compact_session as do_compact
        result = await do_compact(session, force=force)

        if result.success:
            # Save the compacted session to disk
            with self._lock:
                self._save_session(session_id)

        return {
            "success": result.success,
            "messages_compacted": result.messages_compacted,
            "tokens_saved": result.tokens_saved,
            "compression_ratio": result.compression_ratio,
            "summary_preview": result.summary[:200] + "..." if len(result.summary) > 200 else result.summary,
            "key_decisions": result.key_decisions,
            "file_modifications": result.file_modifications,
            "error": result.error,
        }

    def get_compaction_stats(self, session_id: str) -> dict | None:
        """
        Get compaction statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with compaction stats or None if session not found
        """
        session = self.get_session(session_id)
        if session is None:
            return None
        return session.get_compaction_stats()

    def needs_compaction(self, session_id: str) -> bool:
        """
        Check if a session needs compaction.

        Args:
            session_id: Session identifier

        Returns:
            True if session needs compaction, False otherwise
        """
        session = self.get_session(session_id)
        if session is None:
            return False
        return session.needs_compaction()


# Global session manager instance
_global_session_manager: SessionManager | None = None


def get_session_manager(backend_type: str | None = None) -> SessionManager:
    """
    Get or create the global session manager instance with lazy initialization.
    
    Args:
        backend_type: Optional backend type override ("json", "sqlite").
                     If None, uses settings.json configuration or defaults to "json".

    Returns:
        Global SessionManager instance
    """
    global _global_session_manager
    if _global_session_manager is None:
        # Get configured backend
        from .session_backends import get_session_backend
        backend = get_session_backend(backend_type)
        _global_session_manager = SessionManager(backend=backend)
    return _global_session_manager


def create_sqlite_session_manager() -> SessionManager:
    """
    Create a SessionManager with SQLite backend.
    
    Convenience function for explicit SQLite usage.
    
    Returns:
        SessionManager with SQLiteSessionBackend
    """
    from .session_backends import SQLiteSessionBackend
    backend = SQLiteSessionBackend()
    return SessionManager(backend=backend)
