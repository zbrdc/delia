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

Provides a centralized, styled interface for agent interactions.
Includes panels for tasks, planning, tool execution, and final results.
"""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.theme import Theme
    
    # Custom theme for Delia
    delia_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "task": "bold blue",
        "tool": "bold magenta",
        "reflection": "italic purple",
    })
    
    console = Console(theme=delia_theme)
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None


class RichAgentUI:
    """Rich-based TUI for Delia Agent."""

    def __init__(self):
        self.console = console

    def is_available(self) -> bool:
        """Check if Rich is available."""
        return RICH_AVAILABLE and self.console is not None

    def print_header(self, task: str) -> None:
        """Print the task header."""
        if not self.is_available():
            print(f"\nTask: {task}\n{'='*40}")
            return

        self.console.print()
        self.console.print(Panel(
            task,
            title="[task]Current Task[/task]",
            border_style="blue",
            padding=(1, 2),
        ))

    def print_planning_start(self) -> None:
        """Indicate planning phase has started."""
        if not self.is_available():
            print("\n[Planning] Generating plan...")
            return
            
        self.console.print(Panel(
            "[dim]Analyzing request and generating execution plan...[/dim]",
            title="[task]Planning Phase[/task]",
            border_style="blue",
        ))

    def print_plan(self, plan: str) -> None:
        """Display the approved plan."""
        if not self.is_available():
            print(f"\n[Plan]\n{plan}\n")
            return

        self.console.print(Panel(
            Markdown(plan),
            title="[success]Execution Plan[/success]",
            border_style="green",
        ))

    def print_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Display a tool call."""
        if not self.is_available():
            print(f"\n[Tool] {name}: {args}")
            return

        # Format args nicely
        import json
        args_str = json.dumps(args, indent=2)
        syntax = Syntax(args_str, "json", theme="monokai", background_color="default")
        
        self.console.print(Panel(
            syntax,
            title=f"[tool]🛠️  {name}[/tool]",
            border_style="magenta",
            expand=False,
        ))

    def print_tool_result(self, name: str, result: str, success: bool = True) -> None:
        """Display a tool result."""
        if not self.is_available():
            status = "OK" if success else "FAIL"
            print(f"[{status}] Result: {result[:200]}...")
            return

        status_color = "green" if success else "red"
        status_icon = "✅" if success else "❌"
        
        # Truncate very long results for UI, but keep enough context
        if len(result) > 2000:
            display_result = result[:2000] + "\n... [truncated]"
        else:
            display_result = result

        self.console.print(Panel(
            display_result,
            title=f"{status_icon} Result: {name}",
            border_style=status_color,
            border_style="dim",
        ))

    def print_reflection_start(self) -> None:
        """Indicate reflection/critique has started."""
        if not self.is_available():
            print("\n[Reflection] Critiquing response...")
            return

        self.console.print(Panel(
            "[dim]Reviewing work against requirements...[/dim]",
            title="[reflection]🤔 Reflection[/reflection]",
            border_style="purple",
        ))

    def print_reflection_feedback(self, feedback: str) -> None:
        """Display critique feedback."""
        if not self.is_available():
            print(f"Feedback: {feedback}")
            return

        self.console.print(Panel(
            Markdown(feedback),
            title="[warning]Critique Feedback[/warning]",
            border_style="yellow",
        ))

    def print_reflection_success(self) -> None:
        """Display successful reflection."""
        if not self.is_available():
            print("Response Verified.")
            return

        self.console.print("[success]✨ Verified. Quality checks passed.[/success]")

    def print_final_response(self, response: str) -> None:
        """Display the final agent response."""
        if not self.is_available():
            print(f"\nResponse:\n{response}")
            return

        self.console.print()
        self.console.print(Panel(
            Markdown(response),
            title="[success]Final Response[/success]",
            border_style="green",
            padding=(1, 2),
        ))
    
    def print_footer(self, model: str, backend: str, iterations: int, elapsed_ms: int) -> None:
        """Print execution stats."""
        if not self.is_available():
            print(f"\nStats: {model} on {backend} | {iterations} steps | {elapsed_ms}ms")
            return
            
        text = Text()
        text.append("Model: ", style="dim")
        text.append(f"{model} ", style="cyan")
        text.append("| Backend: ", style="dim")
        text.append(f"{backend} ", style="cyan")
        text.append("| Steps: ", style="dim")
        text.append(f"{iterations} ", style="cyan")
        text.append("| Time: ", style="dim")
        text.append(f"{elapsed_ms}ms", style="cyan")
        
        self.console.print(Panel(text, border_style="dim", expand=False))
