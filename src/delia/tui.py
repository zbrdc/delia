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
    from rich.rule import Rule
    
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
        self.model: str = "auto"
        self.backend: str = "local"

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

        # Modern status: message followed by a subtle model tag
        full_msg = f"{message} [dim]({self.model})[/dim]"

        if self._status:
            self._status.update(full_msg, spinner=spinner)
        else:
            self._status = self.console.status(full_msg, spinner=spinner)
            self._status.start()

    # --- Public API ---

    def print_header(self, task: str, subtitle: str = "", console_override: Any = None) -> None:
        """Print the task header."""
        con = console_override or self.console
        if not con:
            print(f"\nTask: {task}\n{'='*40}")
            return

        con.print()
        # Sleek top-aligned header
        header = Text(" 🍈 ", style="green")
        header.append("DELIA", style="bold white")
        header.append(" │ ", style="dim")
        header.append(task, style="bold cyan")
        
        con.print(header)
        if subtitle:
            con.print(f" [dim italic]{subtitle}[/dim italic]")
        con.print(Rule(style="dim"))

    def update_metadata(self, model: str, backend: str) -> None:
        """Update model info (displayed in status line)."""
        self.model = model
        self.backend = backend
        # Refresh status if active
        if self._status:
            current = self._status.status
            if " [dim]" in str(current):
                base = str(current).split(" [dim]")[0]
                self._set_status(base)

    def print_planning_start(self) -> None:
        """Indicate planning phase."""
        self._set_status("[bold blue]Planning...[/bold blue]", spinner="simpleDots")

    def print_plan(self, plan: str, console_override: Any = None) -> None:
        """Display the plan."""
        self.stop()
        con = console_override or self.console
        if not con:
            print(f"Plan: {plan}")
            return
            
        con.print(Panel(
            Markdown(plan),
            title="[bold blue]Strategic Plan[/bold blue]",
            border_style="blue",
            padding=(0, 2)
        ))

    def print_tool_call(self, name: str, args: dict[str, Any], console_override: Any = None) -> None:
        """Log a tool call."""
        self.stop() 
        con = console_override or self.console
        
        # Format args for display
        import json
        args_str = json.dumps(args)
        if len(args_str) > 100:
            args_str = args_str[:97] + "..."
            
        if con:
            con.print(f" [magenta]⚙[/magenta] [white]{name}[/white] [dim]{args_str}[/dim]")
            self._set_status(f"Running {name}", spinner="simpleDotsScrolling")
        else:
            print(f"[Tool] {name}: {args_str}")

    def print_tool_result(self, name: str, result: str, success: bool = True, console_override: Any = None) -> None:
        """Log a tool result."""
        self.stop()
        con = console_override or self.console
        
        preview = str(result).strip()
        lines = preview.split('\n')
        if len(lines) > 3:
            preview = "\n".join(lines[:3]) + f" [dim](+{len(lines)-3} more)[/dim]"
        elif len(preview) > 200:
            preview = preview[:200] + "..."

        if not con:
            return

        if success:
            con.print(Padding(Text(f"✓ {preview}", style="dim green"), (0, 4)))
        else:
            con.print(Padding(Text(f"✗ {preview}", style="bold red"), (0, 4)))

    def print_reflection_start(self) -> None:
        """Indicate reflection."""
        self._set_status("[italic purple]Refining...[/italic purple]", spinner="point")

    def print_reflection_feedback(self, feedback: str, console_override: Any = None) -> None:
        """Log critique."""
        self.stop()
        con = console_override or self.console
        if con:
            con.print(f" [yellow]![/yellow] [dim yellow]Self-Correction:[/dim yellow] [yellow]{feedback}[/yellow]")

    def print_reflection_success(self, console_override: Any = None) -> None:
        """Log verification."""
        self.stop()
        con = console_override or self.console
        if con:
            con.print(" [green]✨ Output verified.[/green]")

    def print_final_response(self, response: str, console_override: Any = None) -> None:
        """Display final response."""
        self.stop()
        con = console_override or self.console
        
        if not con:
            print(f"\nResponse:\n{response}")
            return

        con.print()
        con.print(Markdown(response))
        con.print(Rule(style="dim"))

    def print_footer(self, model: str, backend: str, iterations: int, elapsed_ms: int) -> None:
        """Print final stats."""
        self.stop()
        if not self.is_available():
            print(f"Stats: {model} | {iterations} steps | {elapsed_ms}ms")
            return
            
        stats = f"[dim]Model:[/dim] [cyan]{model}[/cyan] [dim]•[/dim] [dim]Time:[/dim] [cyan]{elapsed_ms}ms[/cyan] [dim]•[/dim] [dim]Steps:[/dim] [cyan]{iterations}[/cyan]"
        self.console.print(Padding(stats, (0, 0)))