"""
test_repl_coder_simulator: Online Stage 1.

Harness model composes the static DAG and drives the REPL loop.
ClaudeCodeMock is the mock human: observes terminal, writes experience.
It knows nothing about the pipeline.
"""

import os
import shutil
import sys
import subprocess
import tempfile

import libtmux

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.function.ft_make_forwarded import ft_make_forwarded
from experience.future_tensor.function.ft_tmux_create_session import ft_tmux_create_session
from experience.future_tensor.function.ft_tmux_send_keys import ft_tmux_send_keys
from experience.future_tensor.function.ft_tmux_capture_pane import ft_tmux_capture_pane
from experience.future_tensor.function.ft_sleep import ft_sleep
from experience.future_tensor.function.ft_sequential import ft_sequential
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_expert import ft_expert
from experience.future_tensor.function.ft_switch import ft_switch
from experience.future_tensor.function.ft_expand import ft_expand
from experience.future_tensor.function.ft_mean import ft_mean
from experience.future_tensor.function.ft_terminal_idle_gate import ft_terminal_idle_gate
from experience.future_tensor.function.ft_tmux_speculative_complete import ft_tmux_speculative_complete
from experience.future_tensor.function.ft_ps1_shows_up import ft_ps1_shows_up
from experience.future_tensor.backward_dispatch import (
    dispatch_policy,
    need_reflection,
    ft_reflection_starter,
    TracePolicy,
)
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
from experience.symbolic_tensor.tensor_util.st_setitem import st_setitem


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
N_EXPERIENCE = 8


# ─── ClaudeCodeMock: the mock human ───

class ClaudeCodeMock:
    """Mock human operator. Observes a tmux session, writes experience.

    Knows nothing about the pipeline, tensors, autograd, or ft_forward.
    Only provides teach_all() and observe_and_update() to be called from the harness loop.

    NEVER interacts with the terminal directly — only observes and writes
    experience entries. All terminal actions come through the pipeline.
    """

    def __init__(self, session_name, decision_exp, cmd_exp, ps1_exp,
                 decision_task, cmd_task, ps1_task, n_experience=8):
        self.session_name = session_name
        self.decision_exp = decision_exp
        self.cmd_exp = cmd_exp
        self.ps1_exp = ps1_exp
        self.decision_task = decision_task
        self.cmd_task = cmd_task
        self.ps1_task = ps1_task
        self.n_experience = n_experience
        self._di = 0
        self._ci = 0
        self._pi = 0
        self._task_taught = False

        server = libtmux.Server()
        self._pane = None
        for s in server.sessions:
            if s.session_name == session_name:
                self._pane = s.active_window.active_pane
                break

    def teach_all(self):
        """Pre-teach all experience before the REPL loop starts."""
        # Task prompts
        st_setitem(self.decision_task, [0],
            "输出交互模式名。只有bash_builtin。只输出模式名。")
        st_setitem(self.cmd_task, [0],
            "输出带前缀的按键序列。每行一个动作。"
            "T 前缀=发送文字到终端。C 前缀=发送控制键。"
            "必须输出可执行的shell命令，不是命令的参数。"
            "正确：T echo hello world\\nC Enter"
            "错误：T hello world（这不是shell命令）")
        st_setitem(self.ps1_task, [0],
            "根据终端输出，输出一个能匹配PS1提示符的正则表达式。"
            "只输出正则表达式，不要解释。")
        self._task_taught = True

        # Decision experience
        self._teach_decision("$ ", "空提示符\n没有命令", "bash_builtin")
        self._teach_decision("$ ls", "命令已输入\n提示符后有命令", "bash_builtin")

        # Cmd experience
        self._teach_cmd("bash_builtin\n空提示符需要输入命令并执行",
            "输入echo命令\n然后回车", "T echo hello world\nC Enter")
        self._teach_cmd("bash_builtin\n命令已输入需要执行",
            "已有命令\n只需回车", "C Enter")

        # PS1 experience
        self._teach_ps1("(base) λ b6048c6be068 /workspace/Experience",
            "conda环境+lambda提示符", r"[λ$]\s*$")
        self._teach_ps1("user@host:~$ ",
            "标准bash提示符", r"\$\s*$")
        self._teach_ps1("(base) λ host /tmp echo hello\nhello\n(base) λ host /tmp",
            "命令执行后回到提示符", r"[λ$]\s*$")

    def observe_and_update(self):
        """Look at the terminal. Write experience based on what we see.

        Only writes experience entries — NEVER sends keys or interacts
        with the terminal directly.
        """
        captured = self._pane.capture_pane()
        lines = [l for l in captured if l.strip()]
        if not lines:
            return
        last_line = lines[-1]
        has_prompt, has_cmd = self._parse(last_line)

        if has_cmd:
            # Terminal has command typed — need ctrl Enter to execute
            for _ in range(min(3, self.n_experience - self._di)):
                self._teach_decision(last_line, "命令已输入\n提示符后有命令", "bash_builtin")
            for _ in range(min(3, self.n_experience - self._ci)):
                self._teach_cmd("bash_builtin\n命令已输入需要执行",
                    "已有命令\n只需回车", "C Enter")
        elif has_prompt:
            # Empty prompt — need to type command + press Enter
            self._teach_decision(last_line, "空提示符\n没有命令", "bash_builtin")
            self._teach_cmd("bash_builtin\n空提示符需要输入命令并执行",
                "输入echo命令\n然后回车", "T echo hello world\nC Enter")
            self._teach_ps1(last_line, "当前提示符", r"[λ$]\s*$")

        print(f"    cc: prompt={has_prompt} cmd={has_cmd} | {last_line}")

    def _parse(self, line):
        if "λ" in line:
            after = line.split("λ", 1)[1].strip()
            parts = after.split()
            pi = next((i for i, p in enumerate(parts) if p.startswith(("/", "~"))), -1)
            if pi == -1:
                return True, len(parts) > 0
            return True, len(parts) > pi + 1
        if "$ " in line:
            return True, line.split("$ ", 1)[1].strip() != ""
        if line.rstrip().endswith("$"):
            return True, False
        return False, False

    def _teach_decision(self, q, k, v):
        if self._di < self.n_experience:
            self._write(self.decision_exp, self._di, q, k, v)
            self._di += 1

    def _teach_cmd(self, q, k, v):
        if self._ci < self.n_experience:
            self._write(self.cmd_exp, self._ci, q, k, v)
            self._ci += 1

    def _teach_ps1(self, q, k, v):
        if self._pi < self.n_experience:
            self._write(self.ps1_exp, self._pi, q, k, v)
            self._pi += 1

    def _write(self, tensor, idx, q, k, v):
        st_setitem(tensor, [idx, 0], q)
        st_setitem(tensor, [idx, 1], k)
        st_setitem(tensor, [idx, 2], v)
        cache = os.path.join(tensor.st_relative_to, tensor.st_tensor_uid, "qkv_data_view")
        if os.path.isdir(cache):
            shutil.rmtree(cache)


# ─── Terminal parsing (for validator) ───

def parse_terminal(text):
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False, False
    last = lines[-1]
    if "λ" in last:
        after = last.split("λ", 1)[1].strip()
        parts = after.split()
        pi = next((i for i, p in enumerate(parts) if p.startswith(("/", "~"))), -1)
        if pi == -1:
            return True, len(parts) > 0
        return True, len(parts) > pi + 1
    if "$ " in last:
        return True, last.split("$ ", 1)[1].strip() != ""
    if last.rstrip().endswith("$"):
        return True, False
    return False, False


# ─── Validator ───

def ft_coder_validator(body, max_iters=30):
    relative_to = body.ft_initial_static_tensor.st_relative_to

    async def _get(coords, trajactory):
        i = coords[-1]
        text, _ = await body.ft_async_get(coords, trajactory)
        has_prompt, has_cmd = parse_terminal(text)
        lines = [l for l in text.split("\n") if l.strip()]
        # Check that "hello" appears in a non-prompt output line (not garbage in prompt)
        output_lines = [l for l in lines if "λ" not in l and "$ " not in l and not l.rstrip().endswith("$")]
        has_hello_output = any("hello" in l.lower() for l in output_lines)
        if i >= 1 and has_prompt and not has_cmd and len(lines) >= 3 and has_hello_output:
            return (text, Status.confidence(1.0))
        return ("", Status.self_confidence_but_failed(0.9))

    v = FutureTensor(relative_to, _get,
                     [sympy.Integer(1), sympy.Integer(max_iters)])
    v.ft_capacity_shape = [1, max_iters]
    v.requires_grad_(True)
    return v


def read_ft_element(ft):
    path = os.path.join(
        ft.ft_initial_static_tensor.st_relative_to, ft.ft_initial_static_tensor.st_tensor_uid,
        "storage", "0", "data",
    )
    return open(path).read() if os.path.isfile(path) else None


# ─── Inline output_prompt: experience and input arrive as text from upstream ops ───


def inline_output_prompt(task_prompt, exp_text, input_text, prompt):
    """Build LLM prompt with experience and input content inline.

    Called by ft_build_expert_context with 4 args:
        task_prompt: high-level task description
        exp_text: formatted experience text (from ft_retrieve_experience)
        input_text: input content string
        prompt: serialized trajectory from ft_async_get

    Supports bunch-style output: multiple actions separated by blank lines.
    """
    return (
        f"{prompt}\n\n"
        f"{task_prompt}\n\n"
        f"Examples of correct input→output:\n{exp_text}\n\n"
        f"Now for this input:\n{input_text}\n\n"
        "Reply with the output. Multiple actions separated by blank lines are OK. "
        "No explanation. No prefix."
    )


# ─── Harness model + REPL loop ───

print("Running test_repl_coder_simulator (online Stage 1)...\n")

INSTANCE_ID = "repl_coder_test"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"

with tempfile.TemporaryDirectory() as tmpdir:
    # Workspace
    workspace = ft_make_forwarded(tmpdir, [1], [INSTANCE_ID])
    setup = ft_sequential(ft_tmux_create_session(workspace), ft_sleep(workspace, 0.5))
    setup.ft_forward(st_make_tensor(["启动终端会话"], tmpdir))

    # Experience (TODO — cold-start)
    decision_exp = st_make_tensor([["", "", ""]] * N_EXPERIENCE, tmpdir)
    cmd_exp = st_make_tensor([["", "", ""]] * N_EXPERIENCE, tmpdir)
    ps1_exp = st_make_tensor([["", "", ""]] * N_EXPERIENCE, tmpdir)

    # Task prompts (trainable 0D symbolic tensors — cold-start, taught by ClaudeCodeMock)
    decision_task = st_make_tensor([""], tmpdir)
    decision_task.requires_grad_(True)
    cmd_task = st_make_tensor([""], tmpdir)
    cmd_task.requires_grad_(True)
    ps1_task = st_make_tensor([""], tmpdir)
    ps1_task.requires_grad_(True)

    # Stage 1 operator (knows nothing about the pipeline above)
    cc = ClaudeCodeMock(SESSION_NAME, decision_exp, cmd_exp, ps1_exp,
                        decision_task, cmd_task, ps1_task, N_EXPERIENCE)

    # Teach all experience before REPL loop
    cc.teach_all()

    # PS1 detection: one-shot capture → expert → materialize regexp at setup time
    setup_capture = ft_tmux_capture_pane(workspace)
    ps1_raw = ft_expert(setup_capture, ps1_exp, ps1_task,
        output_prompt=inline_output_prompt, topk=2, skip_query_gen=True)
    ps1_raw.ft_forward(st_make_tensor(["检测PS1提示符模式"], tmpdir))
    # Broadcast materialized PS1 regexp to [1, MAX_ITERS]
    ps1_expanded = ft_expand(ps1_raw, [1, MAX_ITERS])

    expanded = ft_expand(workspace, [1, MAX_ITERS])
    reflection_starter = ft_reflection_starter()
    capture = need_reflection(ft_tmux_capture_pane(expanded), reflection_starter)

    # Expert chain: retrieve → build context → LLM call (composed via ft_expert)
    # No ft_first_line between experts — speculative complete handles multi-action dispatch
    decision_raw = ft_expert(capture, decision_exp, decision_task,
        output_prompt=inline_output_prompt, topk=2, skip_query_gen=True)

    cmd_raw = ft_expert(decision_raw, cmd_exp, cmd_task,
        output_prompt=inline_output_prompt, topk=2, skip_query_gen=True)

    # Speculative complete: consume actions one at a time, confirming via screen
    decision_spec = ft_tmux_speculative_complete(decision_raw, capture)
    cmd_spec = ft_tmux_speculative_complete(cmd_raw, capture)

    cmd_gated = ft_terminal_idle_gate(cmd_spec, expanded)

    # Unified send op — no switch needed for text/ctrl dispatch.
    # The expert generates prefixed keys: "T echo hello world" / "C Enter"
    send_keys = ft_tmux_send_keys(cmd_gated, expanded)

    # Post-send capture + PS1 polling: skip sleep when prompt already back
    post_capture = ft_tmux_capture_pane(expanded)
    ps1_check = ft_ps1_shows_up(ps1_expanded, post_capture)

    conditional_wait = ft_switch(ps1_check, [
        ("true", "done", "command finished", post_capture),
        ("false", "wait", "still running", ft_sequential(ft_sleep(expanded, 0.5), ft_tmux_capture_pane(expanded))),
    ])

    # ft_switch at HIGH level for mode routing (bash_builtin only for now)
    body_inner = ft_sequential(send_keys, conditional_wait)
    switched = ft_switch(decision_spec, [
        ("bash_builtin", "bash", "execute shell commands", body_inner),
    ])
    body = switched
    validator = ft_coder_validator(body, max_iters=MAX_ITERS)

    output = ft_recurrent(validator, step_budget=8)
    prompt = st_make_tensor(["在终端中输出问候语hello world"], tmpdir)

    # REPL loop
    for step in range(MAX_ITERS):
        output.ft_forward(prompt)
        if output.ft_forwarded:
            print(f"  step {step}: completed")
            break

        loss = ft_mean(output)
        records = []
        with dispatch_policy(TracePolicy(records)):
            loss.backward(create_graph=True, retain_graph=True)
        print(f"  step {step}: trace={len(records)}")
        records.clear()

        cc.observe_and_update()

    # Verify — require "hello" in command output, not just anywhere in pane
    content = read_ft_element(output)
    if content:
        content_lines = [l for l in content.split("\n") if l.strip()]
        output_lines = [l for l in content_lines if "λ" not in l and "$ " not in l and not l.rstrip().endswith("$")]
        ok = any("hello" in l.lower() for l in output_lines)
    else:
        ok = False
    print(f"\n  {'✓' if ok else '✗'} output lines contain 'hello' (exp: d={cc._di} c={cc._ci})")
    if content:
        print(f"  {content.strip()}")

if not ok:
    sys.exit(1)
print("\ntest_repl_coder_simulator passed.")
