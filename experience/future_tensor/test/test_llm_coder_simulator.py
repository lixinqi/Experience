"""
test_llm_coder_simulator: LLM-powered coder simulator using ft_legacy_expert + ft_recurrent.

Architecture (all composed at pipeline level):

  workspace_ft[1]             — tmux instance ID
  capture_op[1, 30]          = ft_tmux_capture_pane(ft_expand(workspace_ft, [1, 30]))
  decision_expert[1, 30]     = ft_legacy_expert(capture_op, decision_exp)   → "text:hint"|"ctrl:hint"
  cmd_expert[1, 30]          = ft_legacy_expert(decision_expert, cmd_exp)   → "echo hello"|"Enter"
  send_text_op[1, 30]        = ft_tmux_send_text(cmd_expert, workspace_expanded)
  send_ctrl_op[1, 30]        = ft_tmux_send_ctrl(cmd_expert, workspace_expanded)
  switched_send[1, 30]       = ft_switch(decision_expert, [("text",send_text_op),("ctrl",send_ctrl_op)])
  validator[1, 30]           = ft_coder_validator(ft_sequential(switched_send, sleep, capture))
  output[1]                  = ft_recurrent(validator)

  Expert 1 decides WHAT (action_type + hint). Expert 2 knows HOW (hint → real cmd).
  Extensible: new action types don't require changing cmd_expert.
  No shared mutable state — coordination through tensor coordinates.
"""

import os
import sys
import subprocess
import tempfile

import sympy

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
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_legacy_expert import ft_legacy_expert
from experience.future_tensor.function.ft_switch import ft_switch
from experience.future_tensor.function.ft_expand import ft_expand
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor


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


MAX_ITERS = 30


# ─── Termination check ───

def check_terminator_last_line(captured_text: str) -> bool:
    """Check if terminal shows command completion: fresh EMPTY prompt on last line."""
    lines = [l for l in captured_text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    last_line = lines[-1]
    if "λ" in last_line:
        after_lambda = last_line.split("λ", 1)[1].strip()
        parts = after_lambda.split()
        path_idx = -1
        for idx, p in enumerate(parts):
            if p.startswith("/") or p.startswith("~"):
                path_idx = idx
                break
        if path_idx == -1:
            return False
        return len(parts) == path_idx + 1
    if "$ " in last_line:
        return last_line.split("$ ", 1)[1].strip() == ""
    if last_line.rstrip().endswith("$"):
        return True
    return False


# ─── Experience ───

def make_decision_experience(tmpdir: str):
    """Experience for decision expert: input is raw terminal text, output is action_type:hint."""
    return st_make_tensor([
        ["(base) λ hostname /workspace\n提示符后面没有其他文字",
         "提示符后面没有命令文字，命令行为空",
         "text:输入shell命令"],
        ["(base) λ hostname /workspace echo hello world\n提示符后面有echo命令",
         "提示符后面有命令文字，命令行已有内容",
         "ctrl:按回车执行"],
    ], tmpdir)


def make_cmd_experience(tmpdir: str):
    """Experience for cmd expert: input is decision hint, output is actual command."""
    return st_make_tensor([
        ["text:输入shell命令\n任务是列出目录",
         "需要ls",
         "ls"],
        ["text:输入shell命令\n任务是输出hello world",
         "需要echo",
         "echo hello world"],
        ["ctrl:按回车执行",
         "按Enter",
         "Enter"],
    ], tmpdir)


# ─── ft_coder_validator ───

def ft_coder_validator(
    iteration_body: FutureTensor,
    max_iters: int = 30,
) -> FutureTensor:
    """Validator: run iteration body, check if terminal shows command completion."""
    relative_to = iteration_body.ft_initial_static_tensor.st_relative_to

    async def _validator_async_get(coords, trajactory):
        i = coords[-1]
        captured_text, _ = await iteration_body.ft_async_get(coords, trajactory)

        if i >= 1 and check_terminator_last_line(captured_text):
            return (captured_text, Status.confidence(1.0))
        else:
            return ("", Status.self_confidence_but_failed(0.9))

    validator = FutureTensor(relative_to, _validator_async_get,
                             [sympy.Integer(1), sympy.Integer(max_iters)])
    validator.ft_capacity_shape = [1, max_iters]
    return validator


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

print("Running test_llm_coder_simulator...\n")

INSTANCE_ID = "llm_coder_test"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"

with tempfile.TemporaryDirectory() as tmpdir:
    workspace_ft = ft_make_forwarded(tmpdir, [1], [INSTANCE_ID])  # [1]

    # Setup: create session + delay
    setup = ft_sequential(
        ft_tmux_create_session(workspace_ft),  # [1]
        ft_sleep(workspace_ft, 0.5))           # [1]
    setup.ft_forward(st_make_tensor(["启动终端会话"], tmpdir))

    # ── Compose entire pipeline ──

    # 1. Expand workspace_ft[1] → [1, 30]
    workspace_expanded = ft_expand(workspace_ft, [1, MAX_ITERS])  # [1, 30]

    # 2. capture_op[1, 30]: live terminal observation
    capture_op = ft_tmux_capture_pane(workspace_expanded)  # [1, 30]

    # 3. decision_expert[1, 30]: reads terminal → "text:hint" or "ctrl:hint"
    decision_experience = make_decision_experience(tmpdir)
    decision_expert = ft_legacy_expert(
        capture_op, decision_experience,
        task_prompt="观察终端最后一行。如果提示符(λ或$)后面只有路径没有其他文字，输出：text:输入shell命令。如果提示符后面有命令文字（如echo、ls等），输出：ctrl:按回车执行。只输出一行，格式必须是text:或ctrl:开头。",
        topk=2,
    )

    # 4. cmd_expert[1, 30]: reads decision_expert output → real command
    cmd_experience = make_cmd_experience(tmpdir)
    cmd_expert = ft_legacy_expert(
        decision_expert, cmd_experience,
        task_prompt="根据输入的动作描述，输出实际要执行的内容。如果输入以text:开头，输出对应的shell命令（如echo hello world）。如果输入以ctrl:开头，输出键名Enter。只输出命令本身一行，不要加text:或ctrl:前缀，不要解释。",
        topk=2,
    )

    # 5. send ops[1, 30]: cmd_expert provides payload
    send_text_op = ft_tmux_send_text(cmd_expert, workspace_expanded)   # [1, 30]
    send_ctrl_op = ft_tmux_send_ctrl(cmd_expert, workspace_expanded)   # [1, 30]

    # 6. switched_send[1, 30]: routes based on decision_expert prefix
    switched_send = ft_switch(decision_expert, [
        ("text", "type", "send text to terminal", send_text_op),
        ("ctrl", "ctrl", "send control key to terminal", send_ctrl_op),
    ])

    # 7. sleep_op[1, 30]
    sleep_op = ft_sleep(workspace_expanded, 0.5)  # [1, 30]

    # 8. iteration_body = ft_sequential(switched_send, sleep, capture)
    iteration_body = ft_sequential(switched_send, sleep_op, capture_op)  # [1, 30]

    # 9. validator + recurrent
    validator = ft_coder_validator(iteration_body, max_iters=MAX_ITERS)
    output = ft_recurrent(validator)  # output[1]

    # ONE ft_forward — the prompt IS the task
    output.ft_forward(st_make_tensor(["在终端中输出问候语hello world"], tmpdir))

    # Verify
    content = read_ft_element(output, 0)
    ok = content is not None and "hello" in content.lower()
    print(f"\n  {'✓' if ok else '✗'} pane contains 'hello'")
    print(f"\n  Pane:\n{content}\n")

if not ok:
    sys.exit(1)
print("test_llm_coder_simulator passed.")
