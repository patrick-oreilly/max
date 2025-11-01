from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from rich.syntax import Syntax
from .vectorstore import build_or_load_vectorstore
from .chain import make_code_chain

class CharlieApp(App):
    TITLE = "Charlie - Local assistant for Your Codebase"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+d", "clear_log", "Clear"),
        ("escape", "focus_input", "Focus Input"),
    ]
    CSS = """
    Screen {
        background: transparent;
    }
    #main-container {
        padding: 1 4;
        align: center middle;
        background: transparent;
    }
    #agent-output {
        height: 100%;
        width: 100%;
        padding:1;
        border: solid #666;
        }
    RichLog {
        height: 1fr;
        background: transparent;
        border: none;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    Input {
        dock: bottom;
        border: tall $primary;
    }
    """

    def __init__(self, project_path: str):
        super().__init__()
        self.project_path = project_path
        self.chat_history = []  # Store chat messages for redrawing

    def compose(self) -> ComposeResult:
        yield Header()
        self.chat_log = RichLog(highlight=True, markup=True, wrap=True)
        yield self.chat_log
        self.input = Input(placeholder="Ask about your code...")
        yield self.input
        yield Footer()

    async def on_mount(self):
        self.chat_log.write("[bold magenta]Initializing Charlie...[/bold magenta]\n")
        self.vectorstore = build_or_load_vectorstore(self.project_path)
        self.chain, self.memory = make_code_chain(self.vectorstore)
        self.chat_log.write("[bold green]Ready! Ask about your codebase.[/bold green]\n")

    def action_clear_log(self):
        """Clear the chat log."""
        self.chat_history = []
        self.chat_log.clear()
        self.chat_log.write("[bold green]Chat cleared. Ready for new questions.[/bold green]")

    def action_focus_input(self):
        """Focus the input field."""
        self.input.focus()

    async def on_input_submitted(self, event: Input.Submitted):
        query = event.value.strip()
        if not query:
            return

        self.input.value = ""

        # Add user message to history
        self.chat_history.append(("user", query))

        # Stream the response token by token
        response = ""
        async for chunk in self.chain.astream(query):
            response += chunk
            # Redraw the entire chat with the streaming response
            self._redraw_chat(streaming_response=response)

        # Finalize the response in history
        self.chat_history.append(("ai", response))
        self._redraw_chat()

        self.memory.save_context({"input": query}, {"output": response})

        # Syntax highlight code blocks
        if "```" in response:
            self.highlight_code_in_log()

    def _redraw_chat(self, streaming_response=None):
        """Redraw the entire chat log with optional streaming response."""
        self.chat_log.clear()

        # Write all historical messages
        for role, message in self.chat_history:
            if role == "user":
                self.chat_log.write(f"[bold cyan]You:[/bold cyan] {message}")
            else:
                self.chat_log.write(f"[bold green]AI:[/bold green] {message}")

        # If streaming, show the current partial response
        if streaming_response:
            self.chat_log.write(f"[bold green]AI:[/bold green] {streaming_response}")

    def highlight_code_in_log(self):
        lines = self.chat_log.text.splitlines()
        in_code = False
        lang = "text"
        code_lines = []

        for line in lines:
            if line.strip().startswith("```"):
                if in_code:
                    # End code block
                    code = "\n".join(code_lines)
                    syntax = Syntax(code, lang, theme="monokai", line_numbers=False)
                    self.chat_log.clear()
                    self.chat_log.write(syntax)
                    in_code = False
                else:
                    lang = line.strip("`").strip() or "text"
                    code_lines = []
                    in_code = True
            elif in_code:
                code_lines.append(line)
            else:
                self.chat_log.write(line + "\n")