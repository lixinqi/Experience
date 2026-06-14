"""
execute_tmux — Replay keystroke DSL in a tmux session.

Generated from execute_tmux.viba:
    main := void
        <- ArgParse[$input TmuxActionSequence]
        <- ArgParse[$tmux_session_name str]
        <- Import[design.viba]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import libtmux


def load_statements(path: Path) -> list[str]:
    """Load keystroke DSL statements from file (one per line)."""
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.rstrip("\n")]


def execute_keystroke(pane, stmt: str) -> None:
    """Execute a single keystroke DSL statement against a tmux pane.

    Supported statements:
        tmux-text <value>   — send literal text
        tmux-ctrl <key>     — send control key
    """
    low = stmt.lower()
    if low.startswith("tmux-text "):
        value = stmt[10:]
        # Handle embedded newlines
        if "\\n" in value:
            parts = value.split("\\n")
            for i, part in enumerate(parts):
                if part:
                    pane.send_keys(part, literal=True, enter=False)
                if i < len(parts) - 1:
                    pane.send_keys("Enter", literal=False, enter=False)
        elif "\n" in value:
            parts = value.split("\n")
            for i, part in enumerate(parts):
                if part:
                    pane.send_keys(part, literal=True, enter=False)
                if i < len(parts) - 1:
                    pane.send_keys("Enter", literal=False, enter=False)
        else:
            pane.send_keys(value, literal=True, enter=False)
    elif low.startswith("tmux-ctrl "):
        key = stmt[10:].strip()
        if key:
            pane.send_keys(key, literal=False, enter=False)
    else:
        # Unknown statement — treat as literal text
        pane.send_keys(stmt, literal=True, enter=False)


def execute_statements(
    statements: list[str],
    session_name: str,
    *,
    settle_delay: float = 0.05,
) -> None:
    """Execute keystroke DSL statements in a tmux session.

    Args:
        statements: List of keystroke DSL statements.
        session_name: Tmux session name to execute in.
        settle_delay: Seconds to wait after each statement.
    """
    server = libtmux.Server()
    session = None
    for s in server.sessions:
        if s.session_name == session_name:
            session = s
            break

    if session is None:
        raise RuntimeError(f"Tmux session '{session_name}' not found")

    pane = session.active_window.active_pane

    for stmt in statements:
        execute_keystroke(pane, stmt)
        time.sleep(settle_delay)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay keystroke DSL statements in a tmux session."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to keystroke DSL file (one statement per line).",
    )
    parser.add_argument(
        "--tmux-session-name",
        type=str,
        required=True,
        help="Name of the tmux session to execute in.",
    )
    parser.add_argument(
        "--settle-delay",
        type=float,
        default=0.05,
        help="Seconds to wait after each keystroke (default: 0.05).",
    )
    args = parser.parse_args()

    statements = load_statements(args.input)

    if not statements:
        print("No statements to execute.", file=sys.stderr)
        sys.exit(1)

    print(f"Executing {len(statements)} keystroke statements in '{args.tmux_session_name}'")
    execute_statements(
        statements,
        args.tmux_session_name,
        settle_delay=args.settle_delay,
    )
    print(f"  Done.")


if __name__ == "__main__":
    main()
