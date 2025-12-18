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

Provides a centralized, LIVE Dashboard interface for agent interactions.
Includes panels for tasks, planning, tool execution, and final results.
"""

from __future__ import annotations

import datetime
from typing import Any, List

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.theme import Theme
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.progress_bar import ProgressBar
    from rich.style import Style
    
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
    })
    
    console = Console(theme=delia_theme)
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None


class RichAgentUI:
    """Live Dashboard TUI for Delia Agent."""

    def __init__(self):
        self.console = console
        self._live: Live | None = None
        
        # State
        self.task: str = ""
        self.plan: str = ""
        self.logs: List[Any] = []  # List of renderables
        self.status: str = "Initializing..."
        self.model: str = "Unknown"
        self.backend: str = "Unknown"
        self.is_thinking: bool = False
        
        # Layout components
        self.layout = None

    def is_available(self) -> bool:
        """Check if Rich is available."""
        return RICH_AVAILABLE and self.console is not None

    def start(self) -> None:
        """Start the Live Dashboard."""
        if not self.is_available():
            return
            
        self.layout = self._make_layout()
        self._live = Live(
            self.layout,
            console=self.console,
            refresh_per_second=4,
            screen=True  # Full screen mode
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the Live Dashboard."""
        if self._live:
            self._live.stop()
            self._live = None

    def _make_layout(self) -> Layout:
        """Create the main layout structure."""
        layout = Layout(name="root")
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )
        
        self._update_layout(layout)
        return layout

    def _update_layout(self, layout: Layout) -> None:
        """Update layout contents from state."""
        # Header
        header_text = Text()
        header_text.append(" 🍈 DELIA AGENT ", style="bold magenta reverse")
        header_text.append(f"  Task: {self.task[:60]}...", style="bold white")
        if len(self.task) > 60:
            header_text.append("...", style="bold white")
            
        meta_table = Table.grid(padding=(0, 2))
        meta_table.add_row(
            Text(f"Model: {self.model}", style="cyan"),
            Text(f"Backend: {self.backend}", style="green"),
            Text(datetime.datetime.now().strftime("%H:%M:%S"), style="dim")
        )
        
        layout["header"].update(
            Panel(
                meta_table,
                title=header_text,
                border_style="magenta",
                padding=(0, 1)
            )
        )
        
        # Left: Plan
        if self.plan:
            plan_renderable = Markdown(self.plan)
        else:
            plan_renderable = Text("No plan generated yet.", style="dim italic")
            
        layout["left"].update(
            Panel(
                plan_renderable,
                title="[bold blue]Execution Plan[/bold blue]",
                border_style="blue",
            )
        )
        
        # Right: Activity Log
        # Keep last 10 logs
        visible_logs = self.logs[-10:] if self.logs else [Text("Waiting for activity...", style="dim")]
        log_group = Group(*visible_logs)
        
        layout["right"].update(
            Panel(
                log_group,
                title="[bold green]Activity Log[/bold green]",
                border_style="green",
            )
        )
        
        # Footer: Status
        status_text = Text()
        if self.is_thinking:
            status_text.append("🧠 ", style="bold")
        status_text.append(self.status, style="bold white")
        
        layout["footer"].update(
            Panel(
                status_text,
                border_style="dim",
                padding=(0, 1)
            )
        )

    def refresh(self) -> None:
        """Trigger a UI refresh."""
        if self._live and self.layout:
            self._update_layout(self.layout)
            self._live.refresh()

    # --- Public API (Updates State) ---

    def print_header(self, task: str) -> None:
        """Set the task header."""
        self.task = task.replace("\n", " ").strip()
        if not self.is_available():
            print(f"\nTask: {task}\n{'='*40}")

    def update_metadata(self, model: str, backend: str) -> None:
        """Update model and backend info."""
        self.model = model
        self.backend = backend
        self.refresh()

    def print_planning_start(self) -> None:
        """Indicate planning phase has started."""
        self.status = "Generating plan..."
        self.is_thinking = True
        self.logs.append(Text("📝 Generating execution plan...", style="cyan"))
        self.refresh()
        
        if not self.is_available():
            print("\n[Planning] Generating plan...")

    def print_plan(self, plan: str) -> None:
        """Set the approved plan."""
        self.plan = plan
        self.status = "Plan approved."
        self.is_thinking = False
        self.logs.append(Text("✅ Plan generated.", style="green"))
        self.refresh()
        
        if not self.is_available():
            print(f"\n[Plan]\n{plan}\n")

    def print_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Log a tool call."""
        self.status = f"Executing tool: {name}..."
        self.is_thinking = False
        
        # Format args
        import json
        args_str = json.dumps(args)
        if len(args_str) > 100:
            args_str = args_str[:100] + "..."
            
        log_entry = Text()
        log_entry.append("🛠️  ", style="bold magenta")
        log_entry.append(f"{name}", style="bold magenta")
        log_entry.append(f"({args_str})", style="dim")
        
        self.logs.append(log_entry)
        self.refresh()
        
        if not self.is_available():
            print(f"\n[Tool] {name}: {args}")

    def print_tool_result(self, name: str, result: str, success: bool = True) -> None:
        """Log a tool result."""
        status_icon = "✅" if success else "❌"
        style = "green" if success else "red"
        
        preview = result.replace("\n", " ")[:100]
        if len(result) > 100:
            preview += "..."
            
        log_entry = Text()
        log_entry.append(f"  {status_icon} ", style=style)
        log_entry.append(f" {preview}", style="dim")
        
        self.logs.append(log_entry)
        self.refresh()
        
        if not self.is_available():
            status = "OK" if success else "FAIL"
            print(f"[{status}] Result: {result[:200]}...")

    def print_reflection_start(self) -> None:
        """Indicate reflection started."""
        self.status = "Reflecting on results..."
        self.is_thinking = True
        self.logs.append(Text("🤔 Critiquing response...", style="purple"))
        self.refresh()
        
        if not self.is_available():
            print("\n[Reflection] Critiquing response...")

    def print_reflection_feedback(self, feedback: str) -> None:
        """Log reflection feedback."""
        self.logs.append(Text("⚠️  Critique: Feedback received", style="yellow"))
        # Add feedback as a dim block
        self.logs.append(Text(f"  > {feedback[:100]}...", style="dim yellow"))
        self.refresh()
        
        if not self.is_available():
            print(f"Feedback: {feedback}")

    def print_reflection_success(self) -> None:
        """Log verification success."""
        self.status = "Verified."
        self.logs.append(Text("✨ Verified.", style="bold green"))
        self.refresh()
        
        if not self.is_available():
            print("Response Verified.")

    def print_final_response(self, response: str) -> None:
        """Display final response."""
        # Stop live mode to print final result cleanly
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