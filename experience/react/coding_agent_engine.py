"""
CodingAgentEngine: Keeps a Claude REPL warm with --add-dir.
Each react_loop iteration captures the screen, sends it to Claude via
query_interactive_repl_coding_agent, Claude outputs ONE keystroke
based on the current screen.
The warm REPL keeps KV cache across calls.
"""

import os, time
from pathlib import Path
from typing import List, Optional

import libtmux
from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.react.react_types import KeystrokeNode
from experience.react.fixation import extract_foveal
from experience.react.raw_llm_engine import _predict
from experience.keystroke_dsl.parse_and_expand import parse_and_expand, ParseOk
from experience.react.query_interactive_repl_coding_agent import (
    query_interactive_repl_coding_agent,
)


async def _read_ft(ft, coordinates):
    if ft.ft_forwarded:
        try:
            _, f = ft.ft_get_materialized_value(coordinates)
            return open(f).read()
        except Exception:
            pass
    text, _ = await ft.ft_async_get(coordinates, "")
    return text


def _first_valid_cmd(raw: str) -> Optional[str]:
    """Extract the first valid keystroke DSL command from Claude output.

    The DSL grammar supports ``;`` as a statement separator, so chained
    commands like ``tmux-text a; tmux-text b`` parse as two statements.
    Quoted strings (``tmux-text ";"``) produce a literal semicolon.
    """
    for line in raw.strip().strip("`").split("\n"):
        line = line.strip()
        if line.startswith("```") or not line:
            continue
        # Strip surrounding quotes from tmux-text arg (Claude sometimes
        # wraps the text value in quotes).
        if line.startswith("tmux-text "):
            arg = line[10:]
            if (arg.startswith('"') and arg.endswith('"')) or \
               (arg.startswith("'") and arg.endswith("'")):
                arg = arg[1:-1]
                line = "tmux-text " + arg
        if line.startswith("tmux-text ") or line.startswith("tmux-ctrl "):
            result = parse_and_expand(line)
            if isinstance(result, ParseOk) and result.ok:
                return result.ok[0]
    return None


def _cmd_to_node(cmd: str, capture_text: str = "",
                 row: int = 0, col: int = 0) -> KeystrokeNode:
    """Convert a single DSL command to a KeystrokeNode with foveal prediction."""
    low = cmd.lower()
    if low.startswith("tmux-ctrl "):
        key = cmd[10:].strip()
        is_ctrl, text = True, key
    elif low.startswith("tmux-text "):
        text = cmd[10:].strip()
        is_ctrl = False
    else:
        return KeystrokeNode()

    enter_after = text.endswith("\n")
    text = text.rstrip("\n")
    fov = extract_foveal(capture_text, (row, col))
    nr, nc, nf = _predict(capture_text, row, col, text, is_ctrl)
    ks = ("TMUX-CTRL-" + text) if is_ctrl else ("TMUX-TEXT-" + text)
    if enter_after:
        ks += "\n"

    return KeystrokeNode(
        current_fixation=(row, col),
        current_foveal=fov,
        keystrokes=ks,
        predicted_next_fixation=(nr, nc),
        predicted_next_foveal=nf,
    )


class CodingAgentEngine:
    """Reactive engine: warm Claude REPL, one keystroke per screen capture."""

    def __init__(self, agent_type: str = "claude", inner_session_id: str = "",
                 work_dir: str = "", launch_cmd: Optional[str] = None):
        self.inner_session_id = inner_session_id
        self.session_name = tmux_session_prefix + inner_session_id
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._launched = False
        self._claude_launched = False

    def __call__(self, capture, fixation, mail, task,
                 llm_model=None) -> FutureTensor:
        shape = list(capture.ft_capacity_shape)
        schema = list(capture.ft_shape_schema)

        async def _engine_get(coordinates, trajectory):
            self._ensure_launched()

            import glob as _g
            for f in _g.glob("/tmp/.*.sw*") + _g.glob("/tmp/.*.swo"):
                try: os.remove(f)
                except OSError: pass

            cap = await _read_ft(capture, coordinates)
            fix = await _read_ft(fixation, coordinates)
            try:
                parts = fix.strip().split(",")
                sr, sc = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                sr, sc = 0, 0

            pane = self._pane()
            if pane is None:
                return ("", Status.self_confidence_but_failed(0.3))

            # Build prompt and query interactive Claude via
            # query_interactive_repl_coding_agent.  Its validate_notify
            # handles permission prompts, idle reminders, and output
            # validation — no manual polling needed.
            prompt = (
                "Terminal:\n" + cap + "\n\n"
                "Task: " + task + "\n\n"
                "Reply with keystroke DSL commands. "
                "You can chain multiple commands with ; on one line.\n"
                "tmux-text <text> — type literal text\n"
                "tmux-ctrl <key>  — press a control key\n"
                "Use tmux-text \";\" to type a literal semicolon.\n"
                "Ctrl keys: Enter Escape Tab C-c C-a C-e C-o C-x "
                "Backspace Delete Up Down Left Right Home End F1-F12\n"
                "Do not reply in chat. Use the Write tool to save your answer."
            )

            try:
                raw = query_interactive_repl_coding_agent(
                    online_tmux_session_name=self.session_name,
                    get_query=lambda output_path: prompt + " Save to " + output_path,
                    interval_seconds=0.5,
                )
            except Exception:
                return ("", Status.self_confidence_but_failed(0.3))

            cmd = _first_valid_cmd(raw)
            if cmd is None:
                return ("", Status.self_confidence_but_failed(0.3))

            # Auto-Enter for bash commands
            if cmd.lower().startswith("tmux-text "):
                if "lixinqi@" in cap or "% " in cap or "# " in cap:
                    cmd = cmd + "\n"

            node = _cmd_to_node(cmd, cap, sr, sc)
            if node.keystrokes:
                return (node.serialize(), Status.confidence(1.0))
            return ("", Status.self_confidence_but_failed(0.3))

        ft = FutureTensor(str(self.work_dir), _engine_get, schema)
        ft.ft_capacity_shape = shape
        ft.requires_grad_(True)
        return ft

    def _pane(self):
        try:
            server = libtmux.Server()
            s = server.sessions.get(session_name=self.session_name)
            return s.active_window.active_pane if s else None
        except Exception:
            return None

    def _ensure_launched(self):
        if self._launched and self._claude_launched:
            return
        try:
            server = libtmux.Server()
            pane = None
            for s in server.sessions:
                if s.session_name == self.session_name:
                    pane = s.active_window.active_pane
                    break

            if pane is None:
                session = server.new_session(session_name=self.session_name,
                                             attach=False)
                time.sleep(0.3)
                pane = session.active_window.active_pane

            if not self._launched:
                pane.send_keys("unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT", enter=True)
                time.sleep(0.1)
                flash = os.path.expanduser("~/.llm-config/flash-env.sh")
                if os.path.isfile(flash):
                    pane.send_keys("source " + flash, enter=True)
                    time.sleep(0.1)
                self._launched = True

            if not self._claude_launched:
                pane.send_keys(
                    "claude --add-dir " + str(self.work_dir) +
                    " --allow-dangerously-skip-permissions"
                    " --permission-mode bypassPermissions",
                    enter=True,
                )
                time.sleep(4)  # wait for Claude to start
                self._claude_launched = True
        except Exception:
            pass


def make_engine_step(agent_type: str = "claude", inner_session_id: str = "",
                     work_dir: str = "", launch_cmd: Optional[str] = None):
    return CodingAgentEngine(agent_type, inner_session_id, work_dir, launch_cmd)
