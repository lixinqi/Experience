"""
CodingAgentEngine (MVP): Keeps a Claude REPL warm with --add-dir.
Each react_loop iteration captures the screen, sends it to Claude via
query_interactive_repl_coding_agent, and Claude returns keystroke DSL
statements.  The warm REPL keeps KV cache across calls.

No fixation, no foveals, no KeystrokeNode — just ``list[str]``.
"""

import os, time
from pathlib import Path
from typing import List, Optional

import libtmux

from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.keystroke_dsl.parse_and_expand import parse_and_expand, ParseOk
from experience.react.query_interactive_repl_coding_agent import (
    query_interactive_repl_coding_agent,
)


class CodingAgentEngine:
    """MVP engine: warm Claude REPL, returns list of keystroke statements."""

    def __init__(self, agent_type: str = "claude", inner_session_id: str = "",
                 work_dir: str = "", launch_cmd: Optional[str] = None):
        self.inner_session_id = inner_session_id
        self.session_name = tmux_session_prefix + inner_session_id
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._launched = False
        self._claude_launched = False

    def __call__(self, capture: str, task: str) -> List[str]:
        """Think: given the current screen and task, return keystroke statements.

        Returns:
            A list of keystroke DSL commands (``tmux-text …`` /
            ``tmux-ctrl …``).  The caller executes them in order.
        """
        self._ensure_launched()

        import glob as _g
        for f in _g.glob("/tmp/.*.sw*") + _g.glob("/tmp/.*.swo"):
            try: os.remove(f)
            except OSError: pass

        pane = self._pane()
        if pane is None:
            return []

        # Build prompt and query interactive Claude.
        # validate_notify handles permission prompts, idle reminders,
        # and output validation — no manual polling needed.
        prompt = (
            "Terminal:\n" + capture + "\n\n"
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
            return []

        # Parse the raw output into keystroke statements.
        result = parse_and_expand(raw)
        if not isinstance(result, ParseOk) or not result.ok:
            return []

        statements = result.ok

        # Auto-Enter: when the screen shows a shell prompt, append
        # Enter after tmux-text commands so they actually execute.
        _at_prompt = any(p in capture for p in ("lixinqi@", "% ", "# ", "$ "))
        if _at_prompt:
            out = []
            for stmt in statements:
                low = stmt.lower()
                if low.startswith("tmux-text ") and not stmt.endswith("\n"):
                    out.append(stmt + "\n")
                else:
                    out.append(stmt)
            return out

        return statements

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
                pane.send_keys("unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT",
                               enter=True)
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
