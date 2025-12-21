# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Security Policy and Audit System for Delia.

Provides:
- Centralized security policy configuration
- Audit logging for all tool operations
- Undo stack for file modifications
- Approval workflow for dangerous operations
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import structlog

from . import paths

log = structlog.get_logger()


class PermissionLevel(str, Enum):
    """Permission levels for operations."""

    READ = "read"  # Safe, no modifications
    WRITE = "write"  # File modifications
    EXEC = "exec"  # Shell execution
    ADMIN = "admin"  # System-level (sudo, etc.)


class ApprovalMode(str, Enum):
    """How to handle dangerous operations."""

    AUTO = "auto"  # Auto-approve safe ops, prompt for dangerous
    PROMPT = "prompt"  # Always prompt for write/exec
    YOLO = "yolo"  # Auto-approve everything (dangerous!)
    DENY = "deny"  # Block all dangerous operations


@dataclass
class SecurityPolicy:
    """Centralized security policy configuration."""

    # Approval mode
    approval_mode: ApprovalMode = ApprovalMode.AUTO

    # Workspace bounds
    workspace_root: Path | None = None
    allow_parent_traversal: bool = False

    # Allowed paths (beyond workspace)
    allowed_paths: list[str] = field(default_factory=list)

    # Blocked patterns (always denied)
    blocked_patterns: list[str] = field(
        default_factory=lambda: [
            "/.ssh",
            "/.aws",
            "/.gnupg",
            "/.config/gcloud",
            "/etc/passwd",
            "/etc/shadow",
            "/etc/sudoers",
        ]
    )

    # Command allowlist (if set, only these are allowed)
    command_allowlist: list[str] | None = None

    # Command blocklist (always denied)
    command_blocklist: list[str] = field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf /*",
            "sudo ",
            "su ",
            "doas ",
            "chmod 777",
            "chmod -R 777",
            ":(){",  # Fork bomb
            "dd if=",
            "mkfs",
            "format ",
            "> /dev/",
            "curl | bash",
            "wget | bash",
            "curl | sh",
            "wget | sh",
        ]
    )

    # Safe commands (auto-approved even in prompt mode)
    safe_commands: list[str] = field(
        default_factory=lambda: [
            "git status",
            "git diff",
            "git log",
            "git branch",
            "ls",
            "pwd",
            "cat",
            "head",
            "tail",
            "grep",
            "find",
            "wc",
            "pytest",
            "python -m pytest",
            "npm test",
            "npm run test",
            "cargo test",
            "go test",
            "make test",
        ]
    )

    # File extensions that are safe to modify
    safe_extensions: list[str] = field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".md",
            ".txt",
            ".html",
            ".css",
            ".scss",
            ".rs",
            ".go",
            ".java",
            ".kt",
            ".swift",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".sh",
            ".bash",
            ".zsh",
        ]
    )

    # Dangerous extensions (require approval)
    dangerous_extensions: list[str] = field(
        default_factory=lambda: [
            ".sql",
            ".db",
            ".sqlite",
            ".env",
            ".pem",
            ".key",
            ".p12",
            ".exe",
            ".dll",
            ".so",
            ".bin",
        ]
    )

    # Audit settings
    audit_enabled: bool = True
    audit_file: Path = field(default_factory=lambda: paths.DATA_DIR / "audit.jsonl")

    # Undo settings
    undo_enabled: bool = True
    undo_dir: Path = field(default_factory=lambda: paths.DATA_DIR / "undo")
    max_undo_entries: int = 100

    @classmethod
    def from_dict(cls, data: dict) -> "SecurityPolicy":
        """Create policy from settings dict."""
        return cls(
            approval_mode=ApprovalMode(data.get("approval_mode", "auto")),
            allowed_paths=data.get("allowed_paths", []),
            blocked_patterns=data.get("blocked_patterns", cls().blocked_patterns),
            command_allowlist=data.get("command_allowlist"),
            command_blocklist=data.get("command_blocklist", cls().command_blocklist),
            safe_commands=data.get("safe_commands", cls().safe_commands),
            safe_extensions=data.get("safe_extensions", cls().safe_extensions),
            dangerous_extensions=data.get(
                "dangerous_extensions", cls().dangerous_extensions
            ),
            audit_enabled=data.get("audit_enabled", True),
            undo_enabled=data.get("undo_enabled", True),
            max_undo_entries=data.get("max_undo_entries", 100),
        )


@dataclass
class AuditEntry:
    """Single audit log entry."""

    timestamp: str
    operation: str
    tool_name: str
    arguments: dict
    permission_level: str
    approved: bool
    approval_method: str  # "auto", "user", "policy", "denied"
    result: str  # "success", "error", "denied"
    error_message: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "permission_level": self.permission_level,
            "approved": self.approved,
            "approval_method": self.approval_method,
            "result": self.result,
            "error_message": self.error_message,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "duration_ms": self.duration_ms,
        }


@dataclass
class UndoEntry:
    """Entry in the undo stack for file operations."""

    timestamp: str
    operation: str  # "write", "delete", "modify"
    path: str
    backup_path: str | None  # Path to backup file
    original_hash: str | None  # SHA256 of original content
    session_id: str | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "path": self.path,
            "backup_path": self.backup_path,
            "original_hash": self.original_hash,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UndoEntry":
        """Deserialize from dict."""
        return cls(
            timestamp=data["timestamp"],
            operation=data["operation"],
            path=data["path"],
            backup_path=data.get("backup_path"),
            original_hash=data.get("original_hash"),
            session_id=data.get("session_id"),
        )


class SecurityManager:
    """
    Manages security policy, audit logging, and undo operations.

    This is the central point for all security decisions in Delia.
    """

    def __init__(self, policy: SecurityPolicy | None = None):
        self.policy = policy or SecurityPolicy()
        self._undo_stack: list[UndoEntry] = []
        self._approval_callback: Callable[[str, dict], bool] | None = None
        self._load_undo_stack()

    def set_approval_callback(
        self, callback: Callable[[str, dict], bool] | None
    ) -> None:
        """
        Set callback for interactive approval prompts.

        The callback receives (message, context) and returns True to approve.
        If no callback is set, operations requiring approval are denied.
        """
        self._approval_callback = callback

    # =========================================================================
    # Permission Checking
    # =========================================================================

    def check_path(self, path: str) -> tuple[bool, str]:
        """
        Check if a path is allowed by the security policy.

        Returns:
            (allowed, reason)
        """
        path_lower = path.lower()

        # Check blocked patterns
        for pattern in self.policy.blocked_patterns:
            if pattern.lower() in path_lower:
                return False, f"Path matches blocked pattern: {pattern}"

        # Check dangerous extensions
        for ext in self.policy.dangerous_extensions:
            if path_lower.endswith(ext):
                return False, f"Dangerous file extension: {ext}"

        # Check workspace bounds
        if self.policy.workspace_root:
            try:
                resolved = Path(path).resolve()
                workspace = self.policy.workspace_root.resolve()
                if not str(resolved).startswith(str(workspace)):
                    # Check allowed paths
                    allowed = any(
                        str(resolved).startswith(str(Path(p).resolve()))
                        for p in self.policy.allowed_paths
                    )
                    if not allowed:
                        return False, f"Path outside workspace: {path}"
            except Exception as e:
                return False, f"Invalid path: {e}"

        return True, ""

    def check_command(self, command: str) -> tuple[bool, str, bool]:
        """
        Check if a shell command is allowed.

        Returns:
            (allowed, reason, is_safe)
            is_safe means it can be auto-approved
        """
        cmd_lower = command.lower().strip()

        # Check blocklist first
        for blocked in self.policy.command_blocklist:
            if blocked.lower() in cmd_lower:
                return False, f"Command matches blocklist: {blocked}", False

        # Check allowlist if set
        if self.policy.command_allowlist is not None:
            allowed = any(
                cmd_lower.startswith(allow.lower())
                for allow in self.policy.command_allowlist
            )
            if not allowed:
                return False, "Command not in allowlist", False

        # Check if it's a safe command
        is_safe = any(
            cmd_lower.startswith(safe.lower()) for safe in self.policy.safe_commands
        )

        return True, "", is_safe

    def needs_approval(
        self, tool_name: str, permission_level: str, is_dangerous: bool
    ) -> bool:
        """
        Check if an operation needs user approval.
        """
        mode = self.policy.approval_mode

        if mode == ApprovalMode.YOLO:
            return False
        if mode == ApprovalMode.DENY and is_dangerous:
            return True  # Will be denied
        if mode == ApprovalMode.PROMPT:
            return permission_level in ("write", "exec", "admin")
        if mode == ApprovalMode.AUTO:
            return is_dangerous or permission_level == "admin"

        return False

    async def request_approval(
        self,
        tool_name: str,
        arguments: dict,
        permission_level: str,
        description: str | None = None,
    ) -> tuple[bool, str]:
        """
        Request approval for an operation.

        Returns:
            (approved, method) where method is "auto", "user", "policy", "denied"
        """
        # Check policy mode
        if self.policy.approval_mode == ApprovalMode.YOLO:
            return True, "policy"
        if self.policy.approval_mode == ApprovalMode.DENY:
            return False, "policy"

        # Build approval message
        desc = description or f"{tool_name}({json.dumps(arguments)[:100]})"
        message = f"Delia wants to execute: {desc}\nPermission level: {permission_level}"

        # Try callback
        if self._approval_callback:
            try:
                approved = self._approval_callback(message, {"tool": tool_name, "args": arguments})
                return approved, "user"
            except Exception as e:
                log.warning("approval_callback_error", error=str(e))
                return False, "error"

        # No callback - deny by default for dangerous ops
        log.warning(
            "approval_required_no_callback",
            tool=tool_name,
            permission=permission_level,
        )
        return False, "denied"

    # =========================================================================
    # Audit Logging
    # =========================================================================

    def audit(
        self,
        operation: str,
        tool_name: str,
        arguments: dict,
        permission_level: str,
        approved: bool,
        approval_method: str,
        result: str,
        error_message: str | None = None,
        session_id: str | None = None,
        duration_ms: int = 0,
    ) -> None:
        """Log an operation to the audit log."""
        if not self.policy.audit_enabled:
            return

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            operation=operation,
            tool_name=tool_name,
            arguments=self._sanitize_arguments(arguments),
            permission_level=permission_level,
            approved=approved,
            approval_method=approval_method,
            result=result,
            error_message=error_message,
            session_id=session_id,
            duration_ms=duration_ms,
        )

        try:
            self.policy.audit_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.policy.audit_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            log.warning("audit_write_failed", error=str(e))

    def _sanitize_arguments(self, arguments: dict) -> dict:
        """Sanitize arguments for audit logging (remove sensitive data)."""
        sanitized = {}
        sensitive_keys = {"password", "secret", "token", "key", "credential", "auth"}

        for k, v in arguments.items():
            if any(s in k.lower() for s in sensitive_keys):
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, str) and len(v) > 1000:
                sanitized[k] = v[:500] + f"...[truncated {len(v)} chars]"
            else:
                sanitized[k] = v

        return sanitized

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Get recent audit entries."""
        if not self.policy.audit_file.exists():
            return []

        entries = []
        try:
            with open(self.policy.audit_file) as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except Exception as e:
            log.warning("audit_read_failed", error=str(e))

        return entries[-limit:]

    # =========================================================================
    # Undo Stack
    # =========================================================================

    def backup_file(self, path: str, session_id: str | None = None) -> UndoEntry | None:
        """
        Backup a file before modification.

        Returns the undo entry or None if backup failed.
        """
        if not self.policy.undo_enabled:
            return None

        try:
            file_path = Path(path)
            if not file_path.exists():
                # New file - no backup needed, but track for undo
                entry = UndoEntry(
                    timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    operation="create",
                    path=str(file_path.resolve()),
                    backup_path=None,
                    original_hash=None,
                    session_id=session_id,
                )
                self._add_undo_entry(entry)
                return entry

            # Calculate hash of original
            content = file_path.read_bytes()
            original_hash = hashlib.sha256(content).hexdigest()

            # Create backup
            self.policy.undo_dir.mkdir(parents=True, exist_ok=True)
            backup_name = f"{original_hash[:16]}_{file_path.name}"
            backup_path = self.policy.undo_dir / backup_name

            if not backup_path.exists():
                shutil.copy2(file_path, backup_path)

            entry = UndoEntry(
                timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                operation="modify",
                path=str(file_path.resolve()),
                backup_path=str(backup_path),
                original_hash=original_hash,
                session_id=session_id,
            )
            self._add_undo_entry(entry)
            return entry

        except Exception as e:
            log.warning("backup_failed", path=path, error=str(e))
            return None

    def undo_last(self, session_id: str | None = None) -> tuple[bool, str]:
        """
        Undo the last file operation.

        Args:
            session_id: If provided, only undo operations from this session

        Returns:
            (success, message)
        """
        if not self._undo_stack:
            return False, "Nothing to undo"

        # Find the last entry (optionally filtered by session)
        for i in range(len(self._undo_stack) - 1, -1, -1):
            entry = self._undo_stack[i]
            if session_id is None or entry.session_id == session_id:
                return self._perform_undo(entry, i)

        return False, "No matching undo entries"

    def _perform_undo(self, entry: UndoEntry, index: int) -> tuple[bool, str]:
        """Perform the actual undo operation."""
        try:
            file_path = Path(entry.path)

            if entry.operation == "create":
                # File was created - delete it
                if file_path.exists():
                    file_path.unlink()
                    self._undo_stack.pop(index)
                    self._save_undo_stack()
                    return True, f"Deleted created file: {entry.path}"
                return False, f"File not found: {entry.path}"

            elif entry.operation == "modify" and entry.backup_path:
                # File was modified - restore from backup
                backup_path = Path(entry.backup_path)
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    self._undo_stack.pop(index)
                    self._save_undo_stack()
                    return True, f"Restored: {entry.path}"
                return False, f"Backup not found: {entry.backup_path}"

            elif entry.operation == "delete" and entry.backup_path:
                # File was deleted - restore from backup
                backup_path = Path(entry.backup_path)
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    self._undo_stack.pop(index)
                    self._save_undo_stack()
                    return True, f"Restored deleted file: {entry.path}"
                return False, f"Backup not found: {entry.backup_path}"

            return False, f"Unknown operation: {entry.operation}"

        except Exception as e:
            return False, f"Undo failed: {e}"

    def get_undo_stack(self, session_id: str | None = None) -> list[dict]:
        """Get the undo stack, optionally filtered by session."""
        entries = self._undo_stack
        if session_id:
            entries = [e for e in entries if e.session_id == session_id]
        return [e.to_dict() for e in entries]

    def _add_undo_entry(self, entry: UndoEntry) -> None:
        """Add entry to undo stack with size limit."""
        self._undo_stack.append(entry)

        # Trim if over limit
        while len(self._undo_stack) > self.policy.max_undo_entries:
            old_entry = self._undo_stack.pop(0)
            # Clean up old backup
            if old_entry.backup_path:
                try:
                    Path(old_entry.backup_path).unlink(missing_ok=True)
                except Exception:
                    pass

        self._save_undo_stack()

    def _load_undo_stack(self) -> None:
        """Load undo stack from disk."""
        stack_file = self.policy.undo_dir / "stack.json"
        if stack_file.exists():
            try:
                data = json.loads(stack_file.read_text())
                self._undo_stack = [UndoEntry.from_dict(e) for e in data]
            except Exception as e:
                log.warning("undo_stack_load_failed", error=str(e))
                self._undo_stack = []

    def _save_undo_stack(self) -> None:
        """Save undo stack to disk."""
        try:
            self.policy.undo_dir.mkdir(parents=True, exist_ok=True)
            stack_file = self.policy.undo_dir / "stack.json"
            data = [e.to_dict() for e in self._undo_stack]
            stack_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.warning("undo_stack_save_failed", error=str(e))


# =============================================================================
# Global Instance
# =============================================================================

_security_manager: SecurityManager | None = None


def get_security_manager() -> SecurityManager:
    """Get or create the global security manager."""
    global _security_manager
    if _security_manager is None:
        policy = _load_security_policy()
        _security_manager = SecurityManager(policy)
    return _security_manager


def _load_security_policy() -> SecurityPolicy:
    """Load security policy from settings.json."""
    try:
        from .backend_manager import get_backend_manager

        manager = get_backend_manager()
        if hasattr(manager, "security_config") and manager.security_config:
            return SecurityPolicy.from_dict(manager.security_config)
    except Exception as e:
        log.debug("security_policy_load_fallback", error=str(e))

    return SecurityPolicy()


def reset_security_manager() -> None:
    """Reset the global security manager (for testing)."""
    global _security_manager
    _security_manager = None
