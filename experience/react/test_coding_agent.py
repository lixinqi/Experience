"""
test_coding_agent: Integration test for CodingAgentEngine + react_loop.

Launches a coding agent in an inner tmux session, tasks it with typing 'ls' + Enter,
and validates that directory listing appears in the workspace terminal.
"""

import os
import subprocess
import sys

import libtmux

from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.react.coding_agent_engine import CodingAgentEngine
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


INSTANCE_ID = "react_coding_agent_test"
INNER_SESSION_ID = "coding_agent_brain"
WORK_DIR = "/tmp/coding_agent_work"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"


def echo_validator(capture: str, iteration: int) -> bool:
    """Validate that 'hello world' appeared in output."""
    if iteration < 1:
        return False
    for line in capture.split("\n"):
        if line.strip() == "hello world":
            return True
    return False


def _print_terminal():
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
    print("=" * 60)
    print("ReAct Engine: CodingAgentEngine Test")
    print("=" * 60)

    engine = CodingAgentEngine(
        agent_type="ducc",
        inner_session_id=INNER_SESSION_ID,
        work_dir=WORK_DIR,
    )

    config = ReactConfig(max_iterations=10, step_budget=8)
    task = "Type 'echo hello world' and press Enter."

    print(f"\nRunning react_loop (instance_id={INSTANCE_ID})...")
    print(f"  Inner brain session: {engine.session_name}")
    print(f"  IPC work_dir: {WORK_DIR}")

    success = react_loop(
        instance_id=INSTANCE_ID,
        engine_step_fn=engine,
        task=task,
        validator_fn=echo_validator,
        config=config,
    )

    print("\nVerifying terminal content...")
    _print_terminal()
    status = "\u2713 SUCCESS" if success else "\u2717 FAILED"
    print(f"\n  {status}: 'hello world' {'appeared' if success else 'not found'}")
    print(f"\n  Workspace session: {SESSION_NAME}")
    print(f"  Brain session: {engine.session_name}")
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
