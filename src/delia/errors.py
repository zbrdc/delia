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

"""Structured error types for Delia.

This module provides typed exceptions for better error handling and debugging.
All Delia-specific errors inherit from DeliaError for easy catching.

Usage:
    from delia.errors import ConfigError, BackendError, InitError

    try:
        ...
    except DeliaError as e:
        log.error("delia_error", type=type(e).__name__, msg=str(e))
"""

from __future__ import annotations


class DeliaError(Exception):
    """Base exception for all Delia errors.

    Catch this to handle any Delia-specific error.
    """

    pass


class InitError(DeliaError):
    """Raised when module initialization fails.

    Examples:
        - LLM module not initialized before use
        - Required callbacks not provided
        - Missing configuration files
    """

    pass


class ConfigError(DeliaError):
    """Raised for configuration and validation errors.

    Examples:
        - Invalid settings.json
        - Unknown backend ID
        - Invalid model tier
        - Malformed workflow/chain definitions
    """

    pass


class BackendError(DeliaError):
    """Raised for backend/provider errors.

    Examples:
        - No active backend available
        - Backend health check failed
        - Model loading timeout
        - Unsupported provider
    """

    def __init__(self, message: str, backend_id: str | None = None, provider: str | None = None):
        super().__init__(message)
        self.backend_id = backend_id
        self.provider = provider


class ValidationError(DeliaError):
    """Raised for input validation failures.

    Examples:
        - Content exceeds max length
        - Invalid file path
        - Unknown task type
    """

    def __init__(self, message: str, field: str | None = None, value: str | None = None):
        super().__init__(message)
        self.field = field
        self.value = value


class QueueError(DeliaError):
    """Raised for model queue errors.

    Examples:
        - Queue timeout waiting for model
        - Queue full (max depth exceeded)
    """

    def __init__(self, message: str, model: str | None = None, wait_seconds: float | None = None):
        super().__init__(message)
        self.model = model
        self.wait_seconds = wait_seconds
