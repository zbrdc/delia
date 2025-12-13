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
Authentication routes for Delia MCP server.

These routes are only active when AUTH_ENABLED=true and provide:
- User registration (/auth/register)
- JWT login (/auth/jwt/login)
- User info (/auth/me)
- User listing (/auth/users) - superuser only
- Usage stats (/auth/stats, /auth/stats/all)
- Microsoft OAuth (/auth/microsoft/authorize, /auth/microsoft/callback)
"""
from __future__ import annotations

import secrets
import uuid
from typing import TYPE_CHECKING, cast

import structlog
from pydantic import BaseModel as PydanticBaseModel
from pydantic import EmailStr
from starlette.requests import Request
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi.responses import RedirectResponse
    from fastapi_users.authentication import JWTStrategy

    from .auth import User

log = structlog.get_logger()


class LoginRequest(PydanticBaseModel):
    """Login request body."""

    username: str  # Actually email
    password: str


class RegisterRequest(PydanticBaseModel):
    """User registration request."""

    email: EmailStr
    password: str
    display_name: str | None = None


def register_auth_routes(mcp, tracker):
    """Register authentication routes on the MCP server.

    Args:
        mcp: FastMCP instance to register routes on
        tracker: SimpleTracker instance for usage tracking
    """
    # Import auth dependencies here to avoid circular imports and ensure they're
    # only loaded when auth is actually enabled
    from .auth import (
        User,
        UserCreate,
        auth_backend,
        decode_jwt_token,
        get_async_session_context,
        get_user_db_context,
        get_user_manager_context,
    )
    from fastapi_users.authentication import JWTStrategy

    @mcp.custom_route("/auth/register", methods=["POST"])
    async def auth_register(request: Request) -> JSONResponse:
        """
        Register a new user.

        POST /auth/register
        Body: {"email": "...", "password": "...", "display_name": "..."}
        Returns: User data with JWT token
        """
        try:
            body = await request.json()
            reg = RegisterRequest(**body)

            # Use context managers for dependency injection outside FastAPI
            async with (
                get_async_session_context() as session,
                get_user_db_context(session) as user_db,
                get_user_manager_context(user_db) as user_manager,
            ):
                # Create user
                user_create = UserCreate(email=reg.email, password=reg.password, display_name=reg.display_name)
                user = await user_manager.create(user_create)

                # Commit the session to persist the user
                await session.commit()

                # Generate token
                strategy = cast(JWTStrategy[User, uuid.UUID], auth_backend.get_strategy())
                token = await strategy.write_token(user)

                return JSONResponse(
                    {
                        "access_token": token,
                        "token_type": "bearer",
                        "user": {
                            "id": str(user.id),
                            "email": user.email,
                            "display_name": user.display_name,
                            "is_active": user.is_active,
                            "is_superuser": user.is_superuser,
                        },
                    },
                    status_code=201,
                )
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            log.error("registration_failed", error=str(e), traceback=tb)
            return JSONResponse({"detail": str(e) or repr(e), "traceback": tb}, status_code=400)

    @mcp.custom_route("/auth/jwt/login", methods=["POST"])
    async def auth_login(request: Request) -> JSONResponse:
        """
        Login and get JWT token.

        POST /auth/jwt/login
        Body (JSON): {"username": "email@example.com", "password": "..."}
        Body (Form): username=email@example.com&password=...
        Returns: {"access_token": "...", "token_type": "bearer"}
        """
        try:
            # Support both JSON and form data (OAuth2 password flow uses form)
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body = await request.json()
                login = LoginRequest(**body)
            else:
                # Form data
                form = await request.form()
                login = LoginRequest(username=form.get("username", ""), password=form.get("password", ""))

            async with (
                get_async_session_context() as session,
                get_user_db_context(session) as user_db,
                get_user_manager_context(user_db) as user_manager,
            ):
                # Authenticate user
                user = await user_manager.authenticate(
                    credentials=type("Creds", (), {"username": login.username, "password": login.password})()
                )

                if user is None:
                    return JSONResponse({"detail": "Invalid credentials"}, status_code=401)

                if not user.is_active:
                    return JSONResponse({"detail": "User is inactive"}, status_code=401)

                # Generate token
                strategy = cast(JWTStrategy[User, uuid.UUID], auth_backend.get_strategy())
                token = await strategy.write_token(user)

                log.info("user_logged_in", user_id=str(user.id), email=user.email)

                return JSONResponse({"access_token": token, "token_type": "bearer"})
        except Exception as e:
            log.error("login_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=400)

    @mcp.custom_route("/auth/me", methods=["GET"])
    async def auth_me(request: Request) -> JSONResponse:
        """
        Get current user info.

        GET /auth/me
        Headers: Authorization: Bearer <token>
        Returns: User data
        """
        try:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)

            token = auth_header.replace("Bearer ", "")

            # Decode token using helper
            payload = decode_jwt_token(token)
            if not payload:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            user_id = payload.get("sub")
            if not user_id:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            # Get user from database
            async with get_async_session_context() as session, get_user_db_context(session) as user_db:
                user = await user_db.get(uuid.UUID(user_id))

                if not user:
                    return JSONResponse({"detail": "User not found"}, status_code=404)

                return JSONResponse(
                    {
                        "id": str(user.id),
                        "email": user.email,
                        "display_name": user.display_name,
                        "is_active": user.is_active,
                        "is_superuser": user.is_superuser,
                        "max_tokens_per_hour": user.max_tokens_per_hour,
                        "max_requests_per_hour": user.max_requests_per_hour,
                        "max_model_tier": user.max_model_tier,
                    }
                )

        except Exception as e:
            log.error("auth_me_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=500)

    @mcp.custom_route("/auth/users", methods=["GET"])
    async def list_users(request: Request) -> JSONResponse:
        """
        List all users (superuser only).

        GET /auth/users
        Headers: Authorization: Bearer <superuser_token>
        Returns: List of users
        """
        try:
            # Validate superuser
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)

            token = auth_header.replace("Bearer ", "")

            # Decode token using helper
            payload = decode_jwt_token(token)
            if not payload:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            user_id = payload.get("sub")
            if not user_id:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            async with get_async_session_context() as session, get_user_db_context(session) as user_db:
                admin = await user_db.get(uuid.UUID(user_id))

                if not admin or not admin.is_superuser:
                    return JSONResponse({"detail": "Superuser access required"}, status_code=403)

                # Get all users
                from sqlalchemy import select

                result = await session.execute(select(User))
                users = result.scalars().all()

                return JSONResponse(
                    {
                        "users": [
                            {
                                "id": str(u.id),
                                "email": u.email,
                                "display_name": u.display_name,
                                "is_active": u.is_active,
                                "is_superuser": u.is_superuser,
                                "max_tokens_per_hour": u.max_tokens_per_hour,
                                "max_requests_per_hour": u.max_requests_per_hour,
                            }
                            for u in users
                        ]
                    }
                )

        except Exception as e:
            log.error("list_users_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=500)

    @mcp.custom_route("/auth/stats", methods=["GET"])
    async def auth_user_stats(request: Request) -> JSONResponse:
        """
        Get usage stats for authenticated user.

        GET /auth/stats
        Headers: Authorization: Bearer <token>
        Returns: User's usage statistics
        """
        try:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)

            token = auth_header.replace("Bearer ", "")

            # Decode token using helper
            payload = decode_jwt_token(token)
            if not payload:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            user_id = payload.get("sub")
            if not user_id:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            # Get user from database
            async with get_async_session_context() as session, get_user_db_context(session) as user_db:
                user = await user_db.get(uuid.UUID(user_id))

                if not user:
                    return JSONResponse({"detail": "User not found"}, status_code=404)

                # Get tracking stats for this user
                stats = tracker.get_user_stats(user.email)

                return JSONResponse(
                    {
                        "user_id": str(user.id),
                        "email": user.email,
                        "quotas": {
                            "max_tokens_per_hour": user.max_tokens_per_hour,
                            "max_requests_per_hour": user.max_requests_per_hour,
                            "max_model_tier": user.max_model_tier,
                        },
                        "usage": {
                            "total_requests": stats.total_requests if stats else 0,
                            "total_tokens": stats.total_tokens if stats else 0,
                            "requests_this_hour": stats.requests_this_hour if stats else 0,
                            "tokens_this_hour": stats.tokens_this_hour if stats else 0,
                        },
                        "quota_remaining": {
                            "requests_remaining": user.max_requests_per_hour
                            - (stats.requests_this_hour if stats else 0),
                            "tokens_remaining": user.max_tokens_per_hour - (stats.tokens_this_hour if stats else 0),
                        },
                    }
                )

        except Exception as e:
            log.error("user_stats_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=500)

    @mcp.custom_route("/auth/stats/all", methods=["GET"])
    async def auth_all_stats(request: Request) -> JSONResponse:
        """
        Get usage stats for all users (superuser only).

        GET /auth/stats/all
        Headers: Authorization: Bearer <superuser_token>
        Returns: All users' usage statistics
        """
        try:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)

            token = auth_header.replace("Bearer ", "")

            # Decode token using helper
            payload = decode_jwt_token(token)
            if not payload:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            user_id = payload.get("sub")
            if not user_id:
                return JSONResponse({"detail": "Invalid token"}, status_code=401)

            async with get_async_session_context() as session, get_user_db_context(session) as user_db:
                admin = await user_db.get(uuid.UUID(user_id))

                if not admin or not admin.is_superuser:
                    return JSONResponse({"detail": "Superuser access required"}, status_code=403)

                # Get all user stats from tracker
                all_stats = tracker.get_all_users()

                return JSONResponse(
                    {
                        "users": [
                            {
                                "client_id": stats.client_id,
                                "total_requests": stats.total_requests,
                                "total_tokens": stats.total_tokens,
                                "requests_this_hour": stats.requests_this_hour,
                                "tokens_this_hour": stats.tokens_this_hour,
                                "first_request": stats.first_request.isoformat() if stats.first_request else None,
                                "last_request": stats.last_request.isoformat() if stats.last_request else None,
                            }
                            for stats in all_stats
                        ]
                    }
                )

        except Exception as e:
            log.error("all_stats_failed", error=str(e))
            return JSONResponse({"detail": str(e)}, status_code=500)

    log.info("auth_routes_registered", endpoints=["/auth/register", "/auth/jwt/login", "/auth/me", "/auth/stats"])


def register_oauth_routes(mcp):
    """Register Microsoft OAuth routes on the MCP server.

    Args:
        mcp: FastMCP instance to register routes on
    """
    from fastapi.responses import RedirectResponse
    from fastapi_users.authentication import JWTStrategy

    from .auth import (
        JWT_SECRET,
        MICROSOFT_REDIRECT_URL,
        User,
        UserCreate,
        auth_backend,
        get_async_session_context,
        get_user_db_context,
        get_user_manager_context,
        microsoft_oauth_client,
    )

    @mcp.custom_route("/auth/microsoft/authorize", methods=["GET"])
    async def oauth_authorize(request: Request):
        """Initiate Microsoft OAuth login flow."""
        try:
            # Generate state for CSRF protection
            state = secrets.token_urlsafe(32)

            # Get authorization URL from OAuth client
            authorization_url = await microsoft_oauth_client.get_authorization_url(
                redirect_uri=MICROSOFT_REDIRECT_URL,
                state=state,
            )

            # Store state in session (simplified - in production use proper session storage)
            # For now, we'll skip state validation for simplicity

            return RedirectResponse(authorization_url)
        except Exception as e:
            log.error("oauth_authorize_failed", error=str(e))
            return JSONResponse({"detail": "OAuth authorization failed"}, status_code=500)

    @mcp.custom_route("/auth/microsoft/callback", methods=["GET"])
    async def oauth_callback(request: Request) -> JSONResponse:
        """Handle Microsoft OAuth callback."""
        try:
            # Get authorization code from query params
            code = request.query_params.get("code")
            if not code:
                return JSONResponse({"detail": "Authorization code missing"}, status_code=400)

            # Exchange code for tokens
            tokens = await microsoft_oauth_client.get_access_token(code, MICROSOFT_REDIRECT_URL)

            # Get user info from Microsoft Graph API
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://graph.microsoft.com/v1.0/me", headers={"Authorization": f"Bearer {tokens['access_token']}"}
                )
                user_info = resp.json()

            user_email = user_info.get("mail") or user_info.get("userPrincipalName", "")
            user_name = user_info.get("displayName", "")
            _microsoft_id = user_info.get("id", "")  # Stored for potential future use

            # Create or get user in database
            async with (
                get_async_session_context() as session,
                get_user_db_context(session) as user_db,
                get_user_manager_context(user_db) as user_manager,
            ):
                # Try to find existing user by email
                try:
                    user = await user_manager.get_by_email(user_email)
                except Exception:
                    user = None  # User not found or database error

                if not user:
                    # Create new user with OAuth info
                    user_create = UserCreate(
                        email=user_email,
                        password=secrets.token_urlsafe(32),  # Random password for OAuth users
                        display_name=user_name,
                    )
                    user = await user_manager.create(user_create)
                    await session.commit()

                # Generate JWT token
                strategy = cast(JWTStrategy[User, uuid.UUID], auth_backend.get_strategy())
                token = await strategy.write_token(user)

                # Return JSON response with token
                return JSONResponse(
                    {
                        "access_token": token,
                        "token_type": "bearer",
                        "user": {
                            "id": str(user.id),
                            "email": user.email,
                            "name": user_name,
                        },
                    }
                )

        except Exception as e:
            log.error("oauth_callback_failed", error=str(e))
            return JSONResponse({"detail": "OAuth callback failed"}, status_code=500)

    log.info(
        "oauth_routes_registered",
        provider="microsoft",
        endpoints=["/auth/microsoft/authorize", "/auth/microsoft/callback"],
    )
