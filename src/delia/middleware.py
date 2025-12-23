# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
User Tracking Middleware for Delia.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from fastmcp.server.dependencies import get_http_headers, get_http_request
from fastmcp.server.middleware import Middleware, MiddlewareContext

from .container import get_container

log = structlog.get_logger()


class UserTrackingMiddleware(Middleware):
    """
    Middleware to extract authenticated user from JWT token and track requests.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Track tool calls per user."""
        container = get_container()
        if not container.config.tracking_enabled:
            return await call_next()

        start_time = time.time()
        ctx = context.fastmcp_context

        user_id = None
        username = "anonymous"
        ip_address = ""
        api_key = None
        transport = "stdio"

        try:
            request = get_http_request()
            if request:
                transport = "http"
                ip_address = request.client.host if request.client else ""
                forwarded = request.headers.get("x-forwarded-for", "")
                if forwarded: ip_address = forwarded.split(",")[0].strip()
        except Exception: pass

        if container.config.auth_enabled and transport == "http":
            try:
                headers = get_http_headers()
                auth_header = headers.get("authorization", "")
                if auth_header.startswith("Bearer "):
                    token = auth_header.replace("Bearer ", "")
                    api_key = token
                    from .mcp_server import decode_jwt_token, get_async_session_context, get_user_db_context
                    payload = decode_jwt_token(token)
                    if payload and payload.get("sub"):
                        user_id = payload["sub"]
                        async with get_async_session_context() as session, get_user_db_context(session) as user_db:
                            user = await user_db.get(uuid.UUID(user_id))
                            if user: username = user.email
            except Exception: pass

        if not user_id and ctx and ctx.request_context:
            user_id = ctx.session_id
            username = f"session:{ctx.session_id[:8]}"

        client = None
        if user_id:
            client = container.user_tracker.get_or_create_client(username=username, ip_address=ip_address, api_key=api_key, transport=transport)
            if ctx:
                ctx.set_state("user_id", user_id)
                ctx.set_state("username", username)
                ctx.set_state("client_id", client.client_id if client else None)
            
            from .mcp_server import current_client_id, current_username
            current_client_id.set(client.client_id if client else None)
            current_username.set(username)

            if client:
                can_proceed, msg = container.user_tracker.check_quota(client.client_id)
                if not can_proceed:
                    from mcp.types import CallToolResult, TextContent
                    return CallToolResult(content=[TextContent(type="text", text=f"Quota exceeded: {msg}")], isError=True)
                container.user_tracker.start_request(client.client_id)

        try:
            result = await call_next(context)
            success, error_msg = True, ""
        except Exception as e:
            success, error_msg = False, str(e); raise
        finally:
            elapsed_ms = int((time.time() - start_time) * 1000)
            tool_name = context.method or getattr(context.message, "name", None) or "unknown"

            # Record to stats service (always, for dashboard metrics)
            container.stats_service.record_tool_call(
                tool_name=tool_name,
                elapsed_ms=elapsed_ms,
                success=success,
                error=error_msg if not success else None,
            )

            if client:
                container.user_tracker.record_request(client_id=client.client_id, task_type=tool_name, model_tier="", tokens=0, elapsed_ms=elapsed_ms, success=success, error=error_msg)
        return result
