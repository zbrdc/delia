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
Text User Interface (TUI) for Delia using Rich.

Provides a clean, linear streaming interface for agent interactions.
Similar to standard CLIs (Gemini/Claude) with spinners and clear status updates.
"""

from __future__ import annotations

import datetime
from typing import Any, List

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from rich.theme import Theme
    from rich.status import Status
    from rich.padding import Padding
    
    # Custom theme for Delia
    delia_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "task": "bold blue",
        "tool": "bold magenta",
        "reflection": "italic purple",
        "dim": "dim",
        "thought": "italic cyan",
    })
    
    console = Console(theme=delia_theme)
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None


class RichAgentUI:
    """Linear Streaming TUI for Delia Agent."""

    def __init__(self):
        self.console = console
        self._status: Status | None = None
        self.model: str = "Unknown"
        self.backend: str = "Unknown"

    def is_available(self) -> bool:
        return RICH_AVAILABLE and self.console is not None

    def start(self) -> None:
        """Start the interaction (no-op for linear UI)."""
        pass

    def stop(self) -> None:
        """Stop any active spinners."""
        if self._status:
            self._status.stop()
            self._status = None

    def _set_status(self, message: str, spinner: str = "dots") -> None:
        """Update or create the status spinner."""
        if not self.is_available():
            print(f"Status: {message}")
            return

        if self._status:
            self._status.update(message, spinner=spinner)
        else:
            self._status = self.console.status(message, spinner=spinner)
            self._status.start()

    # --- Public API ---

    def print_header(self, task: str, subtitle: str = "") -> None:
        """Print the task header."""
        if not self.is_available():
            print(f"\nTask: {task}\n{'='*40}")
            return

        self.console.print()
        title = Text(" 🍈 DELIA ", style="bold magenta reverse")
        title.append(f" {task}", style="bold white")
        
        if subtitle:
            self.console.print(Panel(
                Text(subtitle, style="dim italic", justify="center"),
                title=title,
                border_style="magenta",
                padding=(0, 2)
            ))
        else:
            self.console.print(Panel(
                Text("Processing...", style="dim"),
                title=title,
                border_style="magenta"
            ))

    def update_metadata(self, model: str, backend: str) -> None:
        """Update model info (displayed in footer/logs)."""
        self.model = model
        self.backend = backend

    def print_planning_start(self) -> None:
        """Indicate planning phase."""
        self._set_status("[bold blue]Generating execution plan...[/bold blue]", spinner="earth")

    def print_plan(self, plan: str) -> None:
        """Display the plan."""
        self.stop()
        if not self.is_available():
            print(f"Plan: {plan}")
            return
            
        self.console.print(Panel(
            Markdown(plan),
            title="[bold blue]Execution Plan[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        ))

    def print_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Log a tool call."""
        self.stop() # Stop previous spinner
        
        # Format args for display
        import json
        args_str = json.dumps(args)
        if len(args_str) > 120:
            args_str = args_str[:117] + "..."
            
        if self.is_available():
            self.console.print(f"[bold magenta]🛠️  {name}[/bold magenta] [dim]{args_str}[/dim]")
            self._set_status(f"Running {name}...", spinner="bouncingBall")
        else:
            print(f"[Tool] {name}: {args_str}")

    def print_tool_result(self, name: str, result: str, success: bool = True) -> None:
        """Log a tool result."""
        self.stop()
        
        # Truncate long results for display
        preview = result.strip()
        lines = preview.split('\n')
        if len(lines) > 5:
            preview = "\n".join(lines[:5]) + f"\n... ({len(lines)-5} more lines)"
        elif len(preview) > 300:
            preview = preview[:300] + "..."

        if not self.is_available():
            status = "OK" if success else "FAIL"
            print(f"[{status}] {preview}")
            return

        if success:
            self.console.print(Padding(Text(f"✓ {preview}", style="dim green"), (0, 2)))
        else:
            self.console.print(Padding(Text(f"✗ {preview}", style="red"), (0, 2)))

    def print_reflection_start(self) -> None:
        """Indicate reflection."""
        self._set_status("[italic purple]Reflecting on results...[/italic purple]", spinner="moon")

    def print_reflection_feedback(self, feedback: str) -> None:
        """Log critique."""
        self.stop()
        if self.is_available():
            self.console.print(f"[yellow]⚠️  Critique:[/yellow] [dim yellow]{feedback}[/dim yellow]")
        else:
            print(f"Critique: {feedback}")

    def print_reflection_success(self) -> None:
        """Log verification."""
        self.stop()
        if self.is_available():
            self.console.print("[bold green]✨ Verified.[/bold green]")

    def print_final_response(self, response: str) -> None:
        """Display final response."""
        self.stop()
        
        if not self.is_available():
            print(f"\nResponse:\n{response}")
            return

        self.console.print()
        self.console.print(Panel(
            Markdown(response),
            title="[bold green]Final Response[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

    def print_footer(self, model: str, backend: str, iterations: int, elapsed_ms: int) -> None:
        """Print final stats."""
        self.stop()
        if not self.is_available():
            print(f"Stats: {model} | {iterations} steps | {elapsed_ms}ms")
            return
            
        stats = f"[dim]Model:[/dim] [cyan]{model}[/cyan]  [dim]Time:[/dim] [cyan]{elapsed_ms}ms[/cyan]  [dim]Steps:[/dim] [cyan]{iterations}[/cyan]"
        self.console.print(Padding(stats, (1, 0)))