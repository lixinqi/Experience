"""
test_ls: Integration test for LsEngine + ReActLoop.

Types "ls" + Enter, validates that directory listing appears in terminal output.
"""

import os
import subprocess
import sys

import libtmux

from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.react.ls_engine import engine_step
from experience.react.react_loop import react_loop
from experience.react.types import ReactConfig


# ─── Source LLM env ───

result = subprocess.run(
    ["bash", "-c", "source ~/.anthropic.sh && env"],
    capture_output=True, text=True,
)
for line in result.stdout.splitlines():
    if "=" in line:
        key, _, val = line.partition("=")
        os.environ[key] = val
os.environ.pop("CLAUDECODE", None)


INSTANCE_ID = "react_ls_test"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"


def ls_validator(capture: str, iteration: int) -> bool:
    """Validate that ls output appeared (look for common files/dirs)."""
    if iteration < 1:
        return False
    lines = [l.strip() for l in capture.split("\n") if l.strip()]
    for line in lines:
        if "experience" in line or "setup.py" in line or "README" in line:
            return True
    return False


def _print_terminal():
    """Print current terminal content from the test session."""
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == SESSION_NAME:
            lines = s.active_window.active_pane.capture_pane()
            screen = "\n".join(lines) if isinstance(lines, list) else str(lines)
            for line in screen.split("\n"):
                if line.strip():
                    print(f"  | {line}")
            return


def main():
    """Run the ls integration test."""
    print("=" * 60)
    print("ReAct Engine: ls Test")
    print("=" * 60)
    config = ReactConfig(max_iterations=10, step_budget=8)
    task = "Type 'ls' and press Enter to list the current directory."

    print(f"\nRunning react_loop (instance_id={INSTANCE_ID})...")
    success = react_loop(
        instance_id=INSTANCE_ID,
        engine_step_fn=engine_step,
        task=task,
        validator_fn=ls_validator,
        config=config,
    )

    print("\nVerifying terminal content...")
    _print_terminal()
    status = "\u2713 SUCCESS" if success else "\u2717 FAILED"
    print(f"\n  {status}: directory listing {'appeared' if success else 'not found'} in output")
    print(f"\n  Session left alive: {SESSION_NAME}")
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
