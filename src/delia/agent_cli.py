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
Premium Agent CLI for Delia.
High-fidelity TUI with pinned-bottom input and scrollable ANSI history.
"""

from __future__ import annotations

import asyncio
import time
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

import structlog

from .tools.agent import AgentConfig, AgentResult, run_agent_loop
from .tools.builtins import get_default_tools
from .tools.registry import ToolRegistry, TrustLevel
from .types import Workspace
from .delegation import DelegateContext, delegate_impl
from .multi_user_tracking import tracker
from .tui import RichAgentUI, RICH_AVAILABLE, console

from prompt_toolkit.application import Application
from prompt_toolkit.layout.containers import HSplit, Window, VSplit, ConditionalContainer, FloatContainer, Float
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML, ANSI
from prompt_toolkit.widgets import Frame, TextArea, Dialog, Button, Label
from prompt_toolkit.filters import Condition
from pygments.lexers.markup import MarkdownLexer

try:
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.console import Console
    from rich.text import Text
except ImportError:
    Panel = None  # type: ignore
    Markdown = None # type: ignore
    Console = None # type: ignore

log = structlog.get_logger()

# Initialize UI
ui = RichAgentUI()

@dataclass
class AgentCLIConfig:
    model: str | None = None
    workspace: str | None = None
    max_iterations: int = 10
    tools: list[str] | None = None
    backend_type: str | None = None
    verbose: bool = False
    reflection_enabled: bool = True
    planning_enabled: bool = True

class ChatState:
    def __init__(self, backend_name: str):
        self.messages: list[dict[str, Any]] = []
        self.history_ansi = ""
        self.plan_markdown = ""
        self.show_plan = False
        self.allow_all = False
        self.backend_name = backend_name
        self.current_model = "auto"
        self.is_processing = False
        self.pending_confirm = False
        self.confirm_text = ""
        self.confirm_future: asyncio.Future[str] | None = None

async def run_chat_cli(config: AgentCLIConfig) -> None:
    from .orchestration.executor import get_orchestration_executor
    from .orchestration.intent import detect_intent
    from .mcp_server import get_active_backend
    from .paths import DATA_DIR

    # 1. Initialization
    active_backend = get_active_backend()
    state = ChatState(active_backend.name if active_backend else "local")
    executor = get_orchestration_executor()
    history_file = DATA_DIR / "chat_history.txt"
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2. Rendering Helpers
    def render_rich(obj) -> str:
        """Render a Rich object to ANSI string."""
        with io.StringIO() as buf:
            c = Console(file=buf, force_terminal=True, color_system="truecolor", width=100)
            c.print(obj)
            return buf.getvalue()

    def append_to_history(obj):
        state.history_ansi += render_rich(obj)
        # Update layout
        history_window.content.text = ANSI(state.history_ansi)
        # Scroll to bottom
        history_window.vertical_scroll = 1000000 

    # 3. Layout Components
    history_window = Window(
        content=FormattedTextControl(ANSI("")),
        wrap_lines=True,
        always_hide_cursor=True,
    )

    plan_window = Window(
        content=FormattedTextControl(lambda: ANSI(render_rich(Panel(Markdown(state.plan_markdown or "No active plan."), title="Strategic Plan", border_style="blue")))),
        width=40,
    )

    def get_status_text():
        color = "cyan" if state.is_processing else "green"
        status = "THINKING" if state.is_processing else "READY"
        return HTML(
            f' <{color}>● {status}</{color}> | ' 
            f' <b>{state.backend_name}</b> | ' 
            f' <b>{ui.model}</b> | ' 
            f' <b>F2</b> Plan ' 
        )

    status_bar = Window(content=FormattedTextControl(get_status_text), height=1, style="reverse")

    input_buffer = Buffer(history=FileHistory(str(history_file)))
    input_field = Window(
        content=BufferControl(buffer=input_buffer, lexer=PygmentsLexer(MarkdownLexer)),
        height=3,
    )

    # 4. Security Gate (Inline)
    async def gated_security(tc, reg):
        tool_def = reg.get(tc.name)
        if not tool_def or tool_def.trust_level == TrustLevel.READ_ONLY: return True
        
        is_destructive = "delete" in tc.name or "rm" in str(tc.arguments)
        if not is_destructive and state.allow_all: return True
        
        state.pending_confirm = True
        state.confirm_text = f"Allow {tc.name}?"
        state.confirm_future = asyncio.Future()
        
        append_to_history(Panel(
            f"[bold yellow]Security Gate:[/bold yellow] Agent wants to run [cyan]{tc.name}[/cyan]\n"
            f"[dim]Args: {str(tc.arguments)}[/dim]\n\n"
            f"Press [bold green]y[/bold green] to allow, [bold red]n[/bold red] to deny, [bold blue]a[/bold blue] for all session",
            border_style="yellow"
        ))
        
        res = await state.confirm_future
        state.pending_confirm = False
        
        if res == "a":
            state.allow_all = True
            return True
        return res == "y"

    # 5. Intent and Execution
    async def process_input():
        user_input = input_buffer.text.strip()
        if not user_input: return
        input_buffer.reset()
        
        if user_input.lower() in ("exit", "quit"):
            app.exit()
            return

        append_to_history(Text(f"\n❯ {user_input}", style="bold green"))
        
        state.is_processing = True
        intent = detect_intent(user_input)
        
        # Override UI methods to pipe to our history
        original_print_plan = ui.print_plan
        ui.print_plan = lambda p, **kw: setattr(state, 'plan_markdown', p) or setattr(state, 'show_plan', True)
        
        full_response = ""
        try:
            async for ev in executor.execute_stream(
                intent=intent, message=user_input, messages=state.messages,
                backend_type=config.backend_type, model_override=config.model,
                on_tool_call=gated_security
            ):
                if ev.event_type == "status": ui.model = ev.message
                elif ev.event_type == "thinking": pass
                elif ev.event_type == "tool_call":
                    append_to_history(f" [magenta]⚙[/magenta] [dim]Running {ev.details['name']}...[/dim]")
                elif ev.event_type == "response":
                    full_response = ev.message
            
            if full_response:
                append_to_history(Panel(Markdown(full_response), title="Assistant", border_style="green"))
                state.messages.append({"role": "user", "content": user_input})
                state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            append_to_history(f" [bold red]Error:[/bold red] {e}")
        finally:
            state.is_processing = False
            ui.print_plan = original_print_plan

    # 6. Key Bindings
    kb = KeyBindings()
    @kb.add("c-c")
    def _(event): app.exit()
    
    @kb.add("f2")
    def _(event): state.show_plan = not state.show_plan

    @kb.add("y", filter=Condition(lambda: state.pending_confirm))
    def _(event): state.confirm_future.set_result("y")

    @kb.add("n", filter=Condition(lambda: state.pending_confirm))
    def _(event): state.confirm_future.set_result("n")

    @kb.add("a", filter=Condition(lambda: state.pending_confirm))
    def _(event): state.confirm_future.set_result("a")

    @kb.add("enter", filter=Condition(lambda: not state.pending_confirm))
    def _(event):
        asyncio.create_task(process_input())

    # 7. Layout Assembler
    root_container = HSplit([
        VSplit([
            history_window,
            ConditionalContainer(content=plan_window, filter=Condition(lambda: state.show_plan)),
        ]),
        status_bar,
        input_field,
    ])

    app = Application(
        layout=Layout(root_container, focused_element=input_field),
        key_bindings=kb, full_screen=True, mouse_support=True,
    )

    append_to_history(Text("🍈 DELIA Native Chat session started.", style="italic green"))
    await app.run_async()

def run_agent_sync(task: str, config: AgentCLIConfig) -> AgentCLIResult:
    return asyncio.run(run_agent_cli(task, config))
