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
Interaction tools for Delia agents.

Allows agents to request input or confirmation from the user.
"""

from __future__ import annotations

import structlog

try:
    from rich.prompt import Prompt
    from rich.console import Console
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

log = structlog.get_logger()


async def ask_user(question: str) -> str:
    """Ask the user a question and get their input.

    Use this tool when you need clarification, confirmation, or missing information
    that you cannot find in the codebase or docs.

    Args:
        question: The question to ask the user

    Returns:
        The user's response
    """
    log.info("asking_user", question=question)

    if RICH_AVAILABLE and console:
        # Use Rich prompt for better UX
        console.print()
        response = Prompt.ask(f"[bold yellow]Agent asks:[/bold yellow] {question}")
        console.print()
    else:
        # Fallback to standard input
        print(f"\nAgent asks: {question}")
        response = input("> ")
        print()
    
    return response
