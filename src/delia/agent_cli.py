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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

# Suppress torchao deprecation warnings (third-party library noise)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchao")
warnings.filterwarnings("ignore", message="builtin type Swig.*has no __module__")

import structlog

from .tools.agent import AgentConfig, AgentResult, run_agent_loop
from .tools.builtins import get_default_tools
from .tools.registry import ToolRegistry, TrustLevel
from .types import Workspace
from .delegation import DelegateContext, delegate_impl
from .multi_user_tracking import tracker
from .tui import RichAgentUI, RICH_AVAILABLE, console
from .llm import init_llm_module
from .queue import ModelQueue
from .session_manager import get_session_manager
from .security import get_security_manager

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
        self.app: Application | None = None  # Set after app creation for refresh
        self.session_id: str | None = None  # For task focus tracking
        self.watchdog_timeout: float = 30.0  # Seconds before watchdog intervenes

async def run_chat_cli(config: AgentCLIConfig) -> None:
    from .orchestration.executor import get_orchestration_executor
    from .orchestration.intent import detect_intent
    from .mcp_server import get_active_backend
    from .paths import DATA_DIR

    # 1. Initialization - LLM module must be initialized before using the executor
    def _cli_stats_callback(model_tier, task_type, original_task, tokens, elapsed_ms, content_preview, enable_thinking, backend="ollama"):
        log.debug("cli_stats", tier=model_tier, tokens=tokens, elapsed_ms=elapsed_ms)

    def _cli_save_stats_callback():
        pass  # CLI doesn't persist stats

    model_queue = ModelQueue()
    init_llm_module(
        stats_callback=_cli_stats_callback,
        save_stats_callback=_cli_save_stats_callback,
        model_queue=model_queue,
    )

    active_backend = get_active_backend()
    state = ChatState(active_backend.name if active_backend else "local")
    executor = get_orchestration_executor()
    history_file = DATA_DIR / "chat_history.txt"
    os.makedirs(DATA_DIR, exist_ok=True)

    # Create a session for task focus tracking
    session_mgr = get_session_manager()
    session = session_mgr.create_session(client_id="cli-user")
    state.session_id = session.session_id

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
        # Trigger screen refresh if app is running
        if state.app:
            state.app.invalidate() 

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

    # 4. Security Gate (Inline) - Uses SecurityManager policy
    security_manager = get_security_manager()

    async def gated_security(tc, reg):
        tool_def = reg.get(tc.name)
        if not tool_def:
            return True

        # Check if security policy allows this automatically
        perm_level = getattr(tool_def, 'permission_level', 'read')
        is_dangerous = getattr(tool_def, 'dangerous', False)

        # Read-only tools always pass
        if perm_level == 'read' and not is_dangerous:
            return True

        # Check command blocklist for shell_exec
        if tc.name == 'shell_exec' and 'command' in tc.arguments:
            allowed, reason, is_safe = security_manager.check_command(tc.arguments['command'])
            if not allowed:
                append_to_history(Panel(
                    f"[bold red]Blocked:[/bold red] {reason}",
                    border_style="red"
                ))
                return False
            if is_safe and not is_dangerous:
                return True  # Safe commands auto-approve

        # Check path security for file operations
        if 'path' in tc.arguments:
            allowed, reason = security_manager.check_path(tc.arguments['path'])
            if not allowed:
                append_to_history(Panel(
                    f"[bold red]Blocked:[/bold red] {reason}",
                    border_style="red"
                ))
                return False

        # If allow_all is set and not destructive, auto-approve
        is_destructive = "delete" in tc.name or "rm -rf" in str(tc.arguments)
        if not is_destructive and state.allow_all:
            return True

        # Check if approval is needed per security policy
        if not security_manager.needs_approval(tc.name, perm_level, is_dangerous):
            return True

        # Interactive approval prompt
        state.pending_confirm = True
        state.confirm_text = f"Allow {tc.name}?"
        state.confirm_future = asyncio.Future()

        # Show detailed info
        args_str = str(tc.arguments)[:200]
        if len(str(tc.arguments)) > 200:
            args_str += "..."

        append_to_history(Panel(
            f"[bold yellow]Security Gate:[/bold yellow] Agent wants to run [cyan]{tc.name}[/cyan]\n"
            f"[dim]Permission: {perm_level}[/dim]\n"
            f"[dim]Args: {args_str}[/dim]\n\n"
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
        start_time = time.time()

        async def run_with_watchdog():
            """Execute stream with timeout watchdog."""
            nonlocal full_response
            streaming_buffer = ""
            async for ev in executor.execute_stream(
                intent=intent, message=user_input, messages=state.messages,
                backend_type=config.backend_type, model_override=config.model,
                session_id=state.session_id,  # Pass session for task focus
                on_tool_call=gated_security
            ):
                if ev.event_type == "status": ui.model = ev.message
                elif ev.event_type == "thinking": pass
                elif ev.event_type == "token":
                    # Accumulate and show streaming tokens
                    if ev.message:
                        streaming_buffer += ev.message
                        # Update status to show we're receiving
                        if len(streaming_buffer) % 20 == 0:  # Throttle updates
                            if state.app:
                                state.app.invalidate()
                elif ev.event_type == "tool_call":
                    append_to_history(f" [magenta]⚙[/magenta] [dim]Running {ev.details['name']}...[/dim]")
                elif ev.event_type == "response":
                    full_response = ev.message

        try:
            # Watchdog: If no response in 60s, provide fallback
            await asyncio.wait_for(run_with_watchdog(), timeout=60.0)

            if full_response:
                elapsed = time.time() - start_time
                append_to_history(Panel(Markdown(full_response), title=f"Assistant ({elapsed:.1f}s)", border_style="green"))
                state.messages.append({"role": "user", "content": user_input})
                state.messages.append({"role": "assistant", "content": full_response})
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            append_to_history(Panel(
                f"[yellow]Watchdog triggered after {elapsed:.0f}s.[/yellow]\n"
                "The model is taking too long. You can:\n"
                "• Wait for it to complete\n"
                "• Press Ctrl+C to cancel\n"
                "• Try a simpler question",
                title="⏱️ Timeout", border_style="yellow"
            ))
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
    state.app = app  # Enable screen refresh from append_to_history

    append_to_history(Text("🍈 DELIA Native Chat session started.", style="italic green"))
    await app.run_async()

def run_agent_sync(task: str, config: AgentCLIConfig) -> AgentCLIResult:
    return asyncio.run(run_agent_cli(task, config))
