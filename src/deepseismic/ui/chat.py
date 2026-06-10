"""Terminal chat interface for the DeepSeismic Analyst agent.

Provides a readline-based interactive session that connects to the agent,
streams responses token-by-token, and shows abbreviated tool-call markers
inline so the analyst can follow the agent's reasoning steps.

Usage
-----
Run directly::

    python -m deepseismic.ui.chat

Or with mock mode for offline iteration::

    MOCK_LLM=true python -m deepseismic.ui.chat

Command shortcuts
-----------------
``/help``       Show available commands.
``/status``     Ask the agent for current run status.
``/interpret``  Trigger a full end-to-end workflow analysis.
``/wells``      Show well inventory for the current survey.
``/persona``    Show or set the active domain perspective.
``/state``      Show current session state (dataset, run, result IDs).
``/clear``      Clear the terminal.
``/exit``       Quit the session.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# readline with Windows fallback
# ---------------------------------------------------------------------------

try:
    import readline as _readline

    _readline.set_completer_delims(" \t\n")
    _READLINE_AVAILABLE = True
except ImportError:
    _READLINE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

# ANSI colour codes — suppressed when not writing to a TTY
_IS_TTY = sys.stdout.isatty()

_RESET = "\033[0m" if _IS_TTY else ""
_BOLD = "\033[1m" if _IS_TTY else ""
_DIM = "\033[2m" if _IS_TTY else ""
_CYAN = "\033[36m" if _IS_TTY else ""
_GREEN = "\033[32m" if _IS_TTY else ""
_YELLOW = "\033[33m" if _IS_TTY else ""
_BLUE = "\033[34m" if _IS_TTY else ""
_GREY = "\033[90m" if _IS_TTY else ""


def _print_header() -> None:
    width = 72
    print()
    print(f"{_CYAN}{_BOLD}{'═' * width}{_RESET}")
    print(f"{_CYAN}{_BOLD}  DeepSeismic Analyst  —  Volve Field PoC{_RESET}")
    print(f"{_GREY}  Azure AI Foundry Agent  |  Type /help for commands{_RESET}")
    mock = os.environ.get("MOCK_LLM", "").lower() in ("true", "1", "yes")
    if mock:
        print(f"{_YELLOW}  ⚠  MOCK MODE  (MOCK_LLM=true)  —  no live Azure calls{_RESET}")
    print(f"{_CYAN}{_BOLD}{'═' * width}{_RESET}")
    print()


def _print_help() -> None:
    commands = [
        ("/help",       "Show this help message"),
        ("/status",     "Check current run and dataset status"),
        ("/interpret",  "Run a full end-to-end Volve workflow analysis"),
        ("/wells",      "Show well inventory for the current survey"),
        ("/persona",    "Show active perspective or switch it"),
        ("            ", "  Usage: /persona geophysics  (or geology / geoengineering)"),
        ("/state",      "Show current session state (IDs, tool call count)"),
        ("/clear",      "Clear the terminal screen"),
        ("/exit",       "Exit the session"),
    ]
    print(f"\n{_CYAN}{_BOLD}Commands:{_RESET}")
    for cmd, desc in commands:
        print(f"  {_GREEN}{cmd:<16}{_RESET} {desc}")
    print()


def _print_divider() -> None:
    print(f"{_GREY}{'─' * 72}{_RESET}")


def _print_agent_prefix() -> None:
    print(f"\n{_BLUE}{_BOLD}Agent:{_RESET} ", end="", flush=True)


def _print_user_prompt() -> str:
    """Display the prompt and return the user's input."""
    prompt = f"\n{_GREEN}{_BOLD}You:{_RESET} "
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return "/exit"


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

_COMMAND_MESSAGES: dict[str, str] = {
    "/status": "What is the current status of the latest preprocessing and inference run?",
    "/interpret": (
        "Analyze the latest Volve run end-to-end: check data, QC, results, "
        "and give me a handoff note."
    ),
    "/wells": "Show me the well inventory linked to the current survey.",
}


def _handle_command(cmd: str, agent: object) -> bool:
    """Process a slash command. Returns True to continue, False to exit."""
    from deepseismic.agent.agent import DeepSeismicAgent

    assert isinstance(agent, DeepSeismicAgent)

    parts = cmd.strip().split()
    command = parts[0].lower()

    if command == "/exit":
        print(f"\n{_DIM}Session ended. Goodbye.{_RESET}\n")
        return False

    if command == "/help":
        _print_help()
        return True

    if command == "/clear":
        os.system("cls" if sys.platform == "win32" else "clear")
        _print_header()
        return True

    if command == "/state":
        state = agent.get_state_summary()
        print(f"\n{_CYAN}{_BOLD}Session State:{_RESET}")
        for key, val in state.items():
            print(f"  {_GREEN}{key:<18}{_RESET} {val!r}")
        print()
        return True

    if command == "/persona":
        if len(parts) == 1:
            current = agent.persona or "auto (context-dependent)"
            print(f"\n{_CYAN}Active perspective:{_RESET} {current}")
            print(
                f"  Change with: {_GREEN}/persona geophysics{_RESET} | "
                f"{_GREEN}/persona geology{_RESET} | "
                f"{_GREEN}/persona geoengineering{_RESET}\n"
            )
            return True
        new_persona = parts[1].lower()
        try:
            agent.set_persona(new_persona)
            print(f"\n{_CYAN}Perspective set to:{_RESET} {new_persona}\n")
        except ValueError as exc:
            print(f"\n{_YELLOW}⚠ {exc}{_RESET}\n")
        return True

    if command in _COMMAND_MESSAGES:
        message = _COMMAND_MESSAGES[command]
        print(f"{_DIM}  → {message}{_RESET}")
        _stream_response(agent, message)
        return True

    print(f"\n{_YELLOW}Unknown command: {command}  (type /help){_RESET}\n")
    return True


# ---------------------------------------------------------------------------
# Response streaming
# ---------------------------------------------------------------------------

def _stream_response(agent: object, message: str) -> None:
    """Send a message to the agent and stream its response to stdout."""
    from deepseismic.agent.agent import DeepSeismicAgent

    assert isinstance(agent, DeepSeismicAgent)

    _print_agent_prefix()
    try:
        for chunk in agent.chat(message):
            # Tool call markers already include a newline; print as-is
            if chunk.startswith("\n> 🔧"):
                print(f"\n{_GREY}{chunk.strip()}{_RESET}", end="", flush=True)
            else:
                print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n{_YELLOW}[interrupted]{_RESET}", end="")
    except Exception as exc:  # noqa: BLE001
        print(f"\n{_YELLOW}⚠ Error: {exc}{_RESET}", end="")

    print("\n")
    _print_divider()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Start the interactive terminal chat session."""
    from deepseismic.agent.agent import DeepSeismicAgent

    _print_header()

    try:
        agent = DeepSeismicAgent()
    except Exception as exc:  # noqa: BLE001
        print(f"{_YELLOW}⚠ Failed to initialise agent: {exc}{_RESET}")
        print(
            f"{_DIM}  Tip: set MOCK_LLM=true to run offline without Azure credentials.{_RESET}\n"
        )
        sys.exit(1)

    if _READLINE_AVAILABLE:
        # Basic tab-completion for slash commands
        _commands = list(_COMMAND_MESSAGES.keys()) + [
            "/help", "/state", "/persona", "/clear", "/exit"
        ]

        def _completer(text: str, state: int) -> str | None:
            matches = [c for c in _commands if c.startswith(text)]
            return matches[state] if state < len(matches) else None

        _readline.set_completer(_completer)
        _readline.parse_and_bind("tab: complete")

    print(f"{_DIM}Type a question or a /command. Press Ctrl+C to interrupt a response.{_RESET}\n")

    while True:
        user_input = _print_user_prompt()

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not _handle_command(user_input, agent):
                break
        else:
            _stream_response(agent, user_input)


if __name__ == "__main__":
    run()
