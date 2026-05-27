"""
test_echo_hello: Integration test for EchoHelloWorldEngine + ReActLoop.

Composes static DAG, engine provides keystroke plans,
validator checks "hello world" in terminal output.
"""

import os
import subprocess
import sys

import libtmux

from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.react.echo_hello_world_engine import engine_step
from experience.react.react_loop import react_loop
from experience.react.react_types import ReactConfig


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


INSTANCE_ID = "react_echo_test"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"


def echo_validator(capture: str, iteration: int) -> bool:
    """Validate that 'hello world' appears as command output in terminal."""
    if iteration < 1:
        return False
    return any(l.strip() == "hello world" for l in capture.split("\n"))


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
    """Run the echo hello world integration test."""
    print("=" * 60)
    print("ReAct Engine: Echo Hello World Test")
    print("=" * 60)
    config = ReactConfig(max_iterations=10, step_budget=8)
    task = "Type 'echo hello world' and press Enter in the terminal."

    print(f"\nRunning react_loop (instance_id={INSTANCE_ID})...")
    success = react_loop(
        instance_id=INSTANCE_ID,
        engine_step_fn=engine_step,
        task=task,
        validator_fn=echo_validator,
        config=config,
    )

    print("\nVerifying terminal content...")
    _print_terminal()
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"\n  {status}: 'hello world' {'appeared' if success else 'not found'} in output")
    print(f"\n  Session left alive: {SESSION_NAME}")
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
