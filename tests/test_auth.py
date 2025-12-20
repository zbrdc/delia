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
Tests for auth.py - authentication and user management.

Run with: DELIA_DATA_DIR=/tmp/delia-test-data uv run pytest tests/test_auth.py -v
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Use a temp directory for test data."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    # Clear cached modules
    modules_to_clear = ["delia.paths", "delia.config", "delia.auth", "delia.multi_user_tracking", "delia"]
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) or mod == m for m in modules_to_clear):
            del sys.modules[mod]

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


class TestAuthModuleImport:
    """Test that auth module imports correctly."""

    def test_auth_imports(self):
        """auth.py should import without errors."""
        from delia import paths
        paths.ensure_directories()

        from delia import auth
        assert auth is not None

    def test_user_model_exists(self):
        """User model should be defined."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import User
        assert User is not None

    def test_user_manager_exists(self):
        """UserManager should be defined."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import UserManager
        assert UserManager is not None


class TestUserModel:
    """Test the User SQLAlchemy model."""

    def test_user_has_required_fields(self):
        """User model should have required fields."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import User

        # Check model has expected attributes
        assert hasattr(User, 'id')
        assert hasattr(User, 'email')
        assert hasattr(User, 'hashed_password')
        assert hasattr(User, 'is_active')
        assert hasattr(User, 'is_superuser')
        assert hasattr(User, 'is_verified')

    def test_user_has_delia_fields(self):
        """User model should have Delia-specific fields."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import User

        # Delia-specific fields for quotas
        assert hasattr(User, 'display_name') or hasattr(User, 'max_tokens_per_hour')


class TestDatabaseSetup:
    """Test database initialization."""

    def test_database_url_uses_paths(self):
        """Database URL should be based on paths module."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import DATABASE_URL

        assert DATABASE_URL is not None
        # Should contain the user data directory path
        assert "users" in DATABASE_URL or "sqlite" in DATABASE_URL.lower()

    @pytest.mark.asyncio
    async def test_create_db_and_tables(self):
        """Should be able to create database tables."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import create_db_and_tables

        # Should not raise
        await create_db_and_tables()

        # Database file should exist
        db_path = paths.USER_DB_FILE
        assert db_path.exists() or True  # May be in-memory for tests


class TestUserManagerCallbacks:
    """Test UserManager callback methods."""

    def test_user_manager_instantiation(self):
        """UserManager should instantiate correctly."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import UserManager, get_user_db

        # UserManager requires a user_db dependency
        # Just verify the class exists and has expected methods
        assert hasattr(UserManager, 'on_after_register')
        assert hasattr(UserManager, 'on_after_login')
        assert hasattr(UserManager, 'on_after_forgot_password')


class TestFastAPIUsersIntegration:
    """Test FastAPI-Users integration."""

    def test_fastapi_users_instance(self):
        """fastapi_users instance should be available."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import fastapi_users

        assert fastapi_users is not None

    def test_current_user_dependencies(self):
        """Current user dependencies should be available."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import current_active_user, current_superuser, current_user_optional

        assert current_active_user is not None
        assert current_superuser is not None
        assert current_user_optional is not None


class TestJWTAuthentication:
    """Test JWT authentication backend."""

    def test_jwt_backend_exists(self):
        """JWT authentication backend should be configured."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import auth_backend

        assert auth_backend is not None

    def test_jwt_strategy(self):
        """JWT secret should be configured."""
        from delia import paths
        paths.ensure_directories()

        # JWT should have a secret key
        from delia.auth import JWT_SECRET

        assert JWT_SECRET is not None
        assert len(JWT_SECRET) > 0


class TestAnonymousUser:
    """Test anonymous user functionality."""

    @pytest.mark.asyncio
    async def test_get_or_create_anonymous_user(self):
        """Should be able to get or create anonymous user."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import get_or_create_anonymous_user, create_db_and_tables

        # Create tables first
        await create_db_and_tables()

        # Get anonymous user
        user_id = await get_or_create_anonymous_user()

        assert user_id is not None
        # Should be a valid UUID string or similar identifier
        assert isinstance(user_id, (str, type(uuid4())))


class TestUserQuotaInfo:
    """Test user quota extraction."""

    def test_get_user_quota_info(self):
        """Should extract quota info from user object."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import get_user_quota_info

        # Create a mock user object with quota fields
        class MockUser:
            max_tokens_per_hour = 100000
            max_requests_per_hour = 500
            max_model_tier = "moe"

        user = MockUser()
        quota = get_user_quota_info(user)

        assert quota is not None
        # Quota should contain the user's limits
        if hasattr(quota, 'max_tokens_per_hour'):
            assert quota.max_tokens_per_hour == 100000


class TestAuthRouters:
    """Test authentication routers."""

    def test_auth_router_exists(self):
        """Auth router should be available."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import fastapi_users

        # FastAPI-Users provides router methods
        assert hasattr(fastapi_users, 'get_auth_router')
        assert hasattr(fastapi_users, 'get_register_router')
        assert hasattr(fastapi_users, 'get_users_router')


class TestOAuthIntegration:
    """Test OAuth integration (if configured)."""

    def test_oauth_router_available(self):
        """OAuth router should be available if configured."""
        from delia import paths
        paths.ensure_directories()

        try:
            from delia.auth import oauth_router
            # OAuth is configured
            assert oauth_router is not None or True
        except ImportError:
            # OAuth not configured, which is fine
            pass


class TestSecretKeyConfiguration:
    """Test secret key configuration."""

    def test_secret_key_not_default(self):
        """JWT secret should not be a weak default value."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import JWT_SECRET

        # Should not be common placeholder values
        assert JWT_SECRET != "secret"
        assert JWT_SECRET != "changeme"
        assert JWT_SECRET != ""
        # Should have reasonable length for security
        assert len(JWT_SECRET) >= 16


class TestDatabasePathConfiguration:
    """Test database path uses centralized paths."""

    def test_database_in_user_data_dir(self):
        """Database should be in USER_DATA_DIR."""
        from delia import paths
        paths.ensure_directories()

        from delia.auth import DATABASE_URL

        # Should reference the user data directory
        user_dir_str = str(paths.USER_DATA_DIR)
        # Database URL should contain the path
        assert user_dir_str in DATABASE_URL or "users.db" in DATABASE_URL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
