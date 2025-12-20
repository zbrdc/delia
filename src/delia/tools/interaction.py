# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Interaction tools for human-in-the-loop (HITL) communication.
"""

from __future__ import annotations
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML

async def ask_user(prompt: str) -> str:
    """
    Prompt the user for input asynchronously.
    Used for security gating and clarifications.
    """
    session = PromptSession()
    # Ensure we use the async version to play nice with the TUI
    result = await session.prompt_async(HTML(f"<ansired><b>{prompt}</b></ansired> > "))
    return result.strip()