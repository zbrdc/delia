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
"""Input validation functions for Delia MCP tools."""

VALID_TASKS = frozenset({"review", "analyze", "generate", "summarize", "critique", "quick", "plan", "think"})
VALID_MODELS = frozenset({"quick", "coder", "moe", "thinking"})
VALID_BACKENDS = frozenset({"ollama", "llamacpp"})
MAX_CONTENT_LENGTH = 500_000  # 500KB max content
MAX_FILE_PATH_LENGTH = 1000


def validate_task(task: str) -> tuple[bool, str]:
    """Validate task type. Returns (is_valid, error_message)."""
    if not task:
        return False, "Task type is required"
    if task not in VALID_TASKS:
        return False, f"Invalid task type: '{task}'. Valid types: {', '.join(sorted(VALID_TASKS))}"
    return True, ""


def validate_content(content: str) -> tuple[bool, str]:
    """Validate content byte length. Returns (is_valid, error_message)."""
    if content is None:
        return False, "Content is required"
    if not isinstance(content, str):
        return False, f"Content must be a string, got {type(content).__name__}"
    if not content:
        return False, "Content is required"
    # Use byte length (UTF-8) not character count to enforce accurate size limit
    byte_length = len(content.encode("utf-8"))
    if byte_length > MAX_CONTENT_LENGTH:
        return False, f"Content too large: {byte_length} bytes (max: {MAX_CONTENT_LENGTH})"
    return True, ""


def validate_file_path(file_path: str | None) -> tuple[bool, str]:
    """Validate file path if provided. Returns (is_valid, error_message)."""
    if file_path is None:
        return True, ""  # Optional field (None is allowed)
    if file_path == "":
        return False, "File path cannot be empty string"
    if len(file_path) > MAX_FILE_PATH_LENGTH:
        return False, f"File path too long: {len(file_path)} chars (max: {MAX_FILE_PATH_LENGTH})"
    # Security: Reject path traversal attempts
    if ".." in file_path:
        return False, "File path cannot contain '..' (path traversal not allowed)"
    # Note: ~ is allowed and will be resolved safely by Path.expanduser() in read_file_safe
    return True, ""


def validate_model_hint(model: str | None) -> tuple[bool, str]:
    """Validate model hint if provided. Returns (is_valid, error_message)."""
    if not model:
        return True, ""  # Optional field
    if model not in VALID_MODELS:
        return False, f"Invalid model hint: '{model}'. Valid models: {', '.join(sorted(VALID_MODELS))}"
    return True, ""
