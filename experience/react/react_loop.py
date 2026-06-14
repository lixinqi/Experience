"""
react_loop (MVP) — Simple ReAct loop: capture → think → act → repeat.

No FutureTensor DAG, no fixation, no foveals, no KeystrokeNode.
The engine receives (capture, task) and returns list[keystroke_statement].
"""

import time
from typing import Callable, List, Optional

import libtmux

from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.react.react_types import ReactConfig, EngineStepFn, ValidatorFn

# Called after each iteration: (index, capture, statements, post_capture).
StepCallback = Callable[[int, str, List[str], str], None]


def _get_or_create_pane(session_name: str):
    """Get or create a tmux pane for the given session name."""
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == session_name:
            return s.active_window.active_pane
    session = server.new_session(session_name=session_name, attach=False)
    time.sleep(0.5)
    return session.active_window.active_pane


def _capture(pane) -> str:
    """Capture pane content as a string."""
    lines = pane.capture_pane()
    return "\n".join(lines) if isinstance(lines, list) else str(lines)


def _execute_keystroke(pane, stmt: str):
    """Execute a single keystroke DSL statement against a tmux pane.

    ``tmux-text <value>`` sends literal text.
    ``tmux-ctrl <key>``  sends a control key by name.

    A trailing ``\\n`` on a tmux-text value triggers an extra Enter
    (the engine auto-appends ``\\n`` when a shell prompt is detected).
    """
    low = stmt.lower()
    if low.startswith("tmux-text "):
        value = stmt[10:]
        enter_after = value.endswith("\n")
        value = value.rstrip("\n")
        if value:
            pane.send_keys(value, literal=True, enter=False)
        if enter_after:
            pane.send_keys("Enter", literal=False, enter=False)
    elif low.startswith("tmux-ctrl "):
        key = stmt[10:].strip()
        if key:
            pane.send_keys(key, literal=False, enter=False)


def react_loop(
    instance_id: str,
    engine_step_fn: EngineStepFn,
    task: str,
    validator_fn: ValidatorFn,
    config: Optional[ReactConfig] = None,
    skip_setup: bool = False,
    on_step: Optional[StepCallback] = None,
) -> bool:
    """Run a simple ReAct loop: capture → think → act → repeat.

    Args:
        instance_id: Unique ID for the tmux session (workspace).
        engine_step_fn: ``(capture: str, task: str) -> list[str]``.
            Returns a list of keystroke DSL statements to execute.
        task: Natural-language task description.
        validator_fn: ``(capture: str, iteration: int) -> bool``.
            Returns True when the task is complete.
        config: Loop configuration (max_iterations, settle_delay, …).
        skip_setup: If True, assume the tmux session already exists.
        on_step: Optional callback called after each iteration with
            ``(index, capture, statements, post_capture)``.  Useful for
            recording experience packs.
    """
    if config is None:
        config = ReactConfig()

    session_name = f"{tmux_session_prefix}{instance_id}"

    # ── setup ──────────────────────────────────────────────────────────
    if not skip_setup:
        # Kill any existing session with this name, then create a fresh one.
        server = libtmux.Server()
        for s in server.sessions:
            if s.session_name == session_name:
                s.kill()
                break
        server.new_session(session_name=session_name, attach=False)
        time.sleep(1.0)

    pane = _get_or_create_pane(session_name)

    # ── loop ───────────────────────────────────────────────────────────
    for i in range(config.max_iterations):
        # 1. Capture the current screen.
        screen = _capture(pane)

        # 2. Engine thinks — returns keystroke statements to execute.
        statements = engine_step_fn(screen, task)
        if not statements:
            print(f"  react_loop step {i}: no keystrokes — skipping")
            continue

        # 3. Execute each statement.
        for stmt in statements:
            _execute_keystroke(pane, stmt)

        # 4. Wait for the shell / REPL to settle.
        time.sleep(config.settle_delay)

        # 5. Capture post-execution screen.
        post_screen = _capture(pane)

        # 6. Notify recorder (if any).
        if on_step is not None:
            on_step(i, screen, statements, post_screen)

        # 7. Validate.
        if validator_fn(post_screen, i):
            print(f"  react_loop step {i}: completed")
            return True

        print(f"  react_loop step {i}: continuing")

    return False
