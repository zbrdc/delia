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
"""
Delia Authentication Module

Provides JWT-based authentication using FastAPI-Users.
Supports user registration, login, and token-based access.
Includes Microsoft 365 OAuth integration for enterprise users.

Usage:
    from auth import fastapi_users, auth_backend, current_active_user

    @app.get("/protected")
    async def protected_route(user: User = Depends(current_active_user)):
        return {"email": user.email}
"""

import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    CookieTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from fastapi_users.router import get_oauth_router

# OAuth imports
from httpx_oauth.clients.microsoft import MicrosoftGraphOAuth2
from sqlalchemy import String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from . import paths


# Get logger lazily each time to ensure we use the current configuration
# (important for STDIO transport where logging is reconfigured after import)
def _get_log() -> structlog.stdlib.BoundLogger:
    return structlog.get_logger()


log = None  # Will be set to actual logger on first use

# ============================================================
# CONFIGURATION
# ============================================================

# Database path (SQLite for simplicity)
DATA_DIR = paths.USER_DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite+aiosqlite:///{paths.USER_DB_FILE}"

# JWT Secret (generate a secure one in production!)
# You can override with DELIA_JWT_SECRET environment variable
JWT_SECRET = os.environ.get("DELIA_JWT_SECRET", "CHANGE_ME_IN_PRODUCTION_" + str(uuid.uuid4()))
JWT_LIFETIME_SECONDS = 3600 * 24  # 24 hours

# ============================================================
# MICROSOFT 365 OAUTH CONFIGURATION
# ============================================================

# Microsoft OAuth2 client
microsoft_oauth_client = MicrosoftGraphOAuth2(
    client_id=os.environ.get("MICROSOFT_CLIENT_ID", ""),
    client_secret=os.environ.get("MICROSOFT_CLIENT_SECRET", ""),
)

# OAuth callback URL (must match Azure app registration)
MICROSOFT_REDIRECT_URL = os.environ.get("MICROSOFT_REDIRECT_URL", "http://localhost:8000/auth/microsoft/callback")

# Cookie transport for OAuth (required for OAuth flows)
cookie_transport = CookieTransport(cookie_secure=False)  # Set to True in production with HTTPS


# ============================================================
# DATABASE MODELS
# ============================================================


class Base(DeclarativeBase):
    """SQLAlchemy base class."""

    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    """
    User model with additional Delia-specific fields.

    Inherits from FastAPI-Users base which includes:
    - id: UUID
    - email: str
    - hashed_password: str
    - is_active: bool
    - is_superuser: bool
    - is_verified: bool
    """

    __tablename__ = "users"

    # Custom fields for Delia
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    max_tokens_per_hour: Mapped[int] = mapped_column(default=1_000_000)
    max_requests_per_hour: Mapped[int] = mapped_column(default=1000)
    max_model_tier: Mapped[str] = mapped_column(String(20), default="moe")


# ============================================================
# DATABASE ENGINE & SESSION
# ============================================================

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def create_db_and_tables() -> None:
    """Create database tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _get_log().info("database_initialized", path=str(DATA_DIR / "users.db"))


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session_maker() as session:
        yield session


async def get_user_db(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncGenerator[SQLAlchemyUserDatabase[User, uuid.UUID], None]:
    """Dependency to get user database."""
    yield SQLAlchemyUserDatabase(session, User)


# Context manager versions for use outside of FastAPI dependency injection
from contextlib import asynccontextmanager


@asynccontextmanager
async def get_async_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager to get database session (for use outside DI)."""
    async with async_session_maker() as session:
        yield session


@asynccontextmanager
async def get_user_db_context(session: AsyncSession) -> AsyncGenerator[SQLAlchemyUserDatabase[User, uuid.UUID], None]:
    """Context manager to get user database (for use outside DI)."""
    yield SQLAlchemyUserDatabase(session, User)


@asynccontextmanager
async def get_user_manager_context(
    user_db: SQLAlchemyUserDatabase[User, uuid.UUID],
) -> AsyncGenerator["UserManager", None]:
    """Context manager to get user manager (for use outside DI)."""
    yield UserManager(user_db)


# ============================================================
# USER MANAGER
# ============================================================


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """
    User manager with custom callbacks.

    Handles user lifecycle events like registration, login, etc.
    """

    reset_password_token_secret = JWT_SECRET
    verification_token_secret = JWT_SECRET

    async def on_after_register(self, user: User, request: Request | None = None) -> None:
        """Called after successful registration."""
        _get_log().info("user_registered", user_id=str(user.id), email=user.email, log_type="auth")

    async def on_after_login(
        self,
        user: User,
        request: Request | None = None,
        response: Any = None,
    ) -> None:
        """Called after successful login."""
        _get_log().info("user_logged_in", user_id=str(user.id), email=user.email, log_type="auth")

    async def on_after_forgot_password(self, user: User, token: str, request: Request | None = None) -> None:
        """Called when password reset is requested."""
        _get_log().info("password_reset_requested", user_id=str(user.id), log_type="auth")
        # In production: send email with reset link
        # For now, just log the token (don't do this in production!)
        _get_log().debug("reset_token", token=token[:20] + "...")

    async def on_after_request_verify(self, user: User, token: str, request: Request | None = None) -> None:
        """Called when email verification is requested."""
        _get_log().info("verification_requested", user_id=str(user.id), log_type="auth")


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase[User, uuid.UUID] = Depends(get_user_db),
) -> AsyncGenerator[UserManager, None]:
    """Dependency to get user manager."""
    yield UserManager(user_db)


# ============================================================
# AUTHENTICATION BACKEND (JWT)
# ============================================================

# Bearer transport: tokens passed in Authorization header
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy[User, uuid.UUID]:
    """Get JWT strategy with configured secret and lifetime."""
    return JWTStrategy(secret=JWT_SECRET, lifetime_seconds=JWT_LIFETIME_SECONDS)


# Authentication backend combining transport + strategy
auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

# ============================================================
# OAUTH BACKEND (Microsoft 365)
# ============================================================


def get_oauth_strategy() -> JWTStrategy[User, uuid.UUID]:
    """Get OAuth strategy (reuses JWT for simplicity)."""
    return JWTStrategy(secret=JWT_SECRET, lifetime_seconds=JWT_LIFETIME_SECONDS)


oauth_backend = AuthenticationBackend(
    name="microsoft",
    transport=cookie_transport,
    get_strategy=get_oauth_strategy,
)

# OAuth router for Microsoft 365
oauth_router = get_oauth_router(
    microsoft_oauth_client,
    oauth_backend,
    get_user_manager,  # get_user_manager dependency
    JWT_SECRET,  # state_secret
    redirect_url=MICROSOFT_REDIRECT_URL,
    associate_by_email=True,  # Link accounts by email if user exists
)


# ============================================================
# FASTAPI-USERS INSTANCE
# ============================================================

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend, oauth_backend],
)

# Dependency to get current active user
current_active_user = fastapi_users.current_user(active=True)

# Dependency to get current superuser
current_superuser = fastapi_users.current_user(active=True, superuser=True)

# Optional: get current user (may be None if not authenticated)
current_user_optional = fastapi_users.current_user(optional=True)


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

from fastapi_users import schemas


class UserRead(schemas.BaseUser[uuid.UUID]):
    """Schema for reading user data."""

    display_name: str | None = None
    max_tokens_per_hour: int = 1_000_000
    max_requests_per_hour: int = 1000
    max_model_tier: str = "moe"


class UserCreate(schemas.BaseUserCreate):
    """Schema for creating a new user."""

    display_name: str | None = None


class UserUpdate(schemas.BaseUserUpdate):
    """Schema for updating user data."""

    display_name: str | None = None
    max_tokens_per_hour: int | None = None
    max_requests_per_hour: int | None = None
    max_model_tier: str | None = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_user_quota_info(user: User) -> dict[str, Any]:
    """Extract quota info from user for multi_user_tracking integration."""
    return {
        "max_tokens_per_hour": user.max_tokens_per_hour,
        "max_requests_per_hour": user.max_requests_per_hour,
        "max_model_tier": user.max_model_tier,
    }


async def get_or_create_anonymous_user() -> str:
    """
    For MCP stdio mode where no auth is possible,
    return a default anonymous user ID.
    """
    return "anonymous-local-user"
