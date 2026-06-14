"""
test_engine_multistep (MVP): Multi-step CodingAgentEngine test.

Task requires two steps: echo MARKER_ONE, then echo MARKER_TWO.
Validator requires both markers.  No fixation, no foveals — just
capture → think → act → repeat.
"""

import os
import subprocess
import sys

import libtmux

from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.react.coding_agent_engine import CodingAgentEngine
from experience.react.react_loop import react_loop
from experience.react.react_types import ReactConfig


result = subprocess.run(
    ["bash", "-c", "source ~/.anthropic.sh && env"],
    capture_output=True, text=True,
)
for line in result.stdout.splitlines():
    if "=" in line:
        key, _, val = line.partition("=")
        os.environ[key] = val
os.environ.pop("CLAUDECODE", None)


INSTANCE_ID = "react_multistep_test"
INNER_SESSION_ID = "multistep_brain"
WORK_DIR = "/tmp/react_multistep_workdir"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"


def multistep_validator(capture: str, iteration: int) -> bool:
    return "MARKER_ONE" in capture and "MARKER_TWO" in capture


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
    print("ReAct Engine: Multi-step Coding Agent Test (MVP)")
    print("=" * 60)

    task = "Step 1: echo MARKER_ONE, Enter. Step 2: echo MARKER_TWO, Enter."
    config = ReactConfig(max_iterations=10, step_budget=4)
    engine = CodingAgentEngine(
        agent_type="claude",
        inner_session_id=INNER_SESSION_ID,
        work_dir=WORK_DIR,
    )

    print(f"\nRunning react_loop (instance_id={INSTANCE_ID})...")
    print(f"  Inner brain session: {engine.session_name}")
    print(f"  IPC work_dir: {WORK_DIR}")

    success = react_loop(
        instance_id=INSTANCE_ID,
        engine_step_fn=engine,
        task=task,
        validator_fn=multistep_validator,
        config=config,
    )

    print("\nVerifying terminal content...")
    _print_terminal()
    status = "SUCCESS" if success else "FAILED"
    print(f"\n  {status}: BOTH markers {'found' if success else 'not found'} in output")
    print(f"\n  Workspace session: {SESSION_NAME}")
    print(f"  Brain session: {engine.session_name}")
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
