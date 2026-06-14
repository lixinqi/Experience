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


def _send_text(pane, text: str):
    """Send text, splitting on embedded newlines into Enter key presses."""
    parts = text.split("\n")
    for i, part in enumerate(parts):
        if part:
            pane.send_keys(part, literal=True, enter=False)
        if i < len(parts) - 1:
            pane.send_keys("Enter", literal=False, enter=False)


def execute_keystroke(pane, stmt: str) -> None:
    """Execute a single keystroke DSL statement against a tmux pane."""
    low = stmt.lower()
    if low.startswith("tmux-text "):
        _send_text(pane, stmt[10:])
    elif low.startswith("tmux-ctrl "):
        key = stmt[10:].strip()
        if key:
            pane.send_keys(key, literal=False, enter=False)
    else:
        pane.send_keys(stmt, literal=True, enter=False)


def _get_session_pane(session_name: str):
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == session_name:
            return s.active_window.active_pane
    return None


def execute_statements(statements: list[str], session_name: str, *,
                       settle_delay: float = 0.05) -> None:
    """Execute keystroke DSL statements in a tmux session."""
    pane = _get_session_pane(session_name)
    if pane is None:
        raise RuntimeError(f"Tmux session '{session_name}' not found")
    for stmt in statements:
        execute_keystroke(pane, stmt)
        time.sleep(settle_delay)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay keystroke DSL in a tmux session.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tmux-session-name", type=str, required=True)
    p.add_argument("--settle-delay", type=float, default=0.05)
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    statements = load_statements(args.input)
    if not statements:
        print("No statements to execute.", file=sys.stderr)
        sys.exit(1)
    print(f"Executing {len(statements)} keystroke statements in '{args.tmux_session_name}'")
    execute_statements(statements, args.tmux_session_name, settle_delay=args.settle_delay)
    print("  Done.")


if __name__ == "__main__":
    main()
