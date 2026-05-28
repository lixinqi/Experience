"""
test_flex_coder_simulator: flexible coder simulator using ft_recurrent + ft_switch.

Architecture:
  ft_recurrent(
    check_terminator_last_line(
      ft_tmux_capture_pane(
        ft_switch(get_action, [ft_tmux_send_text, ft_tmux_send_ctrl])
      )
    )
  )

  - get_action:  condition that returns action symbol for current step
  - ft_switch:   routes to ft_tmux_send_text / ft_tmux_send_ctrl
  - ft_tmux_capture_pane: observe terminal state after each action
  - check_terminator_last_line: terminate when shell prompt reappears after output

The validator is decoupled from the action plan — it observes the terminal
and terminates when the command has finished executing (fresh prompt visible).
"""

import os
import sys
import tempfile

import sympy
import libtmux

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.function.ft_make_forwarded import ft_make_forwarded
from experience.future_tensor.function.ft_tmux_create_session import ft_tmux_create_session
from experience.future_tensor.function.ft_tmux_send_text import ft_tmux_send_text
from experience.future_tensor.function.ft_tmux_send_ctrl import ft_tmux_send_ctrl
from experience.future_tensor.function.ft_tmux_capture_pane import ft_tmux_capture_pane
from experience.future_tensor.function.ft_sleep import ft_sleep
from experience.future_tensor.function.ft_sequential import ft_sequential
from experience.future_tensor.function.ft_switch import ft_switch
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor


# ─── Termination check ───

def check_terminator_last_line(captured_text: str) -> bool:
    """Check if terminal shows command completion: fresh prompt on last line.

    After a command executes, the shell shows:
      prompt> command
      output...
      prompt>          <- fresh prompt (termination signal)

    Returns True when there are at least 3 non-empty lines and the last one
    is a prompt (contains the prompt indicator).
    """
    lines = [l for l in captured_text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    last_line = lines[-1]
    return "λ" in last_line or "$ " in last_line or last_line.rstrip().endswith("$")


# ─── Action plan ───

def make_action_plan(code: str) -> list:
    """Convert code into action plan: list of (symbol, payload).

    No "done" marker — termination is decided by the validator observing the terminal.
    """
    actions = []
    for ch in code:
        if ch == "\n":
            actions.append(("ctrl", "Enter"))
        else:
            actions.append(("text", ch))
    actions.append(("ctrl", "Enter"))  # execute the command
    return actions


# ─── Pipeline builder ───

def build_flex_pipeline(id_ft: FutureTensor, action_plan: list, tmpdir: str):
    """Build ft_recurrent(check_terminator(capture_pane(ft_switch(get_action, branches)))).

    Returns (output, prompt_tensor) from ft_recurrent.
    """
    plan_len = len(action_plan)
    # Extra iterations for the terminal to show output + prompt
    max_iters = plan_len + 10
    relative_to = id_ft.ft_initial_static_tensor.st_relative_to

    # Shared step counter
    step = [0]

    # ── get_action: condition FutureTensor for ft_switch ──
    async def get_action_fn(coords, prompt):
        i = step[0]
        if i < plan_len:
            return (action_plan[i][0], Status.confidence(1.0))
        # Past end of plan: send noop text (empty string)
        return ("text", Status.confidence(1.0))

    get_action = FutureTensor(relative_to, get_action_fn, [sympy.Integer(1)])
    get_action.ft_capacity_shape = [1]

    # ── Input FutureTensors: provide text/ctrl payload per step ──
    async def text_payload_fn(coords, prompt):
        i = step[0]
        payload = action_plan[i][1] if i < plan_len else ""
        return (payload, Status.confidence(1.0))

    text_input = FutureTensor(relative_to, text_payload_fn, [sympy.Integer(1)])
    text_input.ft_capacity_shape = [1]

    async def ctrl_payload_fn(coords, prompt):
        i = step[0]
        payload = action_plan[i][1] if i < plan_len else ""
        return (payload, Status.confidence(1.0))

    ctrl_input = FutureTensor(relative_to, ctrl_payload_fn, [sympy.Integer(1)])
    ctrl_input.ft_capacity_shape = [1]

    # ── Branches: real ops (session_name_ft=id_ft broadcasts to input shape) ──
    send_text_branch = ft_tmux_send_text(text_input, id_ft)
    send_ctrl_branch = ft_tmux_send_ctrl(ctrl_input, id_ft)

    # ── ft_switch: route action ──
    switched = ft_switch(get_action, [
        ("text", "type", "type character via send_text", send_text_branch),
        ("ctrl", "ctrl", "send control key via send_ctrl", send_ctrl_branch),
    ])

    # ── capture_pane: observe terminal after each action ──
    capture_op = ft_tmux_capture_pane(id_ft)

    # ── Validator: switch → capture → check_terminator_last_line ──
    async def validator_fn(coords, prompt):
        import asyncio

        i = step[0]

        # Execute action via ft_switch
        await switched.ft_async_get([0], prompt)
        await asyncio.sleep(0.05)

        step[0] += 1

        # Capture pane to observe terminal state
        captured_text, _ = await capture_op.ft_async_get([0], prompt)

        # check_terminator_last_line: fresh prompt visible after output?
        if i >= plan_len - 1 and check_terminator_last_line(captured_text):
            return (captured_text, Status.confidence(1.0))
        else:
            return ("", Status.self_confidence_but_failed(0.9))

    validator = FutureTensor(relative_to, validator_fn, [sympy.Integer(max_iters)])
    validator.ft_capacity_shape = [max_iters]

    return ft_recurrent(validator)


# ─── Helpers ───

def read_ft_element(ft, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        ft.ft_initial_static_tensor.st_relative_to, ft.ft_initial_static_tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


# ─── Test ───

print("Running test_flex_coder_simulator...\n")

server = libtmux.Server()
INSTANCE_ID = "flex_coder_test"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"

with tempfile.TemporaryDirectory() as tmpdir:
    id_ft = ft_make_forwarded(tmpdir, [1], [INSTANCE_ID])

    # Setup: create session
    setup = ft_sequential(ft_tmux_create_session(id_ft), ft_sleep(id_ft, 0.3))
    setup.ft_forward(st_make_tensor(["go"], tmpdir))

    # Build: ft_recurrent(check_terminator(capture_pane(ft_switch(action, [text, ctrl]))))
    plan = make_action_plan("echo hello")
    output = build_flex_pipeline(id_ft, plan, tmpdir)
    output.ft_forward(st_make_tensor("execute", tmpdir))

    # Verify
    content = read_ft_element(output, 0)
    ok = content is not None and "hello" in content
    print(f"  {'✓' if ok else '✗'} pane contains 'hello'")
    print(f"\n  Pane:\n{content}\n")

if not ok:
    sys.exit(1)
print("test_flex_coder_simulator passed.")
