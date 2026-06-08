"""
query_interactive_repl_coding_agent — Query an interactive ducc REPL via
query_interactive_tui with a smart validate_notify that handles permission
prompts, idle reminders, and output validation.

    query_interactive_repl_coding_agent :=
      $query_result str
      <- $online_tmux_session_name str
      <- $get_query ($cur_query str <- $output_file_path str)
      <- $interval_seconds float # default 1.0
      # inline
      <- $parse_and_expand Import[../keystroke_dsl/parse_and_expand]
      <- $validate_notify (... <- ... <- {
        if coding agent blocked with "Do you want to proceed?":
          tmux send Enter.
        elif coding agent idle while no output file saved:
          tmux send file writing notification.
        elif output file saved:
          if parse_and_expand passed:
            return True
          else:
            tmux send error message of parse_and_expand.
        return False
      })
      <- Import[../high_order_fn/query_interactive_tui]
"""

from __future__ import annotations

import os
import subprocess
from typing import Callable

from experience.high_order_fn.query_interactive_tui import query_interactive_tui
from experience.keystroke_dsl.parse_and_expand import parse_and_expand, ParseOk


def _capture_screen(session_name: str) -> str:
    return subprocess.run(
        ["tmux", "capture-pane", "-t", session_name, "-p"],
        capture_output=True, text=True,
    ).stdout


def _tmux_send(session_name: str, text: str):
    subprocess.run(
        ["tmux", "send-keys", "-t", session_name, text], check=True,
    )


def query_interactive_repl_coding_agent(
    online_tmux_session_name: str,
    get_query: Callable[[str], str],
    interval_seconds: float = 1.0,
) -> str:
    """Query an interactive ducc REPL and wait for a valid keystroke command.

    Delegates to :func:`query_interactive_tui` with a smart
    ``validate_notify`` that:

    * Auto-approves "Do you want to proceed?" permission prompts.
    * Reminds ducc to write to the output file when it appears idle.
    * Validates the output with ``parse_and_expand`` and signals
      readiness when a valid keystroke DSL command is detected.

    Args:
        online_tmux_session_name: Name of the tmux session running ducc.
        get_query: Called with ``output_file_path`` to produce the chat
            message sent to ducc.
        interval_seconds: Seconds between polls (default 1.0).

    Returns:
        The validated keystroke DSL command from ducc as a string.
    """
    def validate_notify(output_path: str) -> bool:
        screen = _capture_screen(online_tmux_session_name)

        # Permission prompt? Approve it and keep waiting.
        if "do you want to proceed?" in screen.lower():
            _tmux_send(online_tmux_session_name, "1")
            subprocess.run(
                ["tmux", "send-keys", "-t", online_tmux_session_name, "Enter"],
                check=True,
            )
            return False

        # Output file saved — validate it.
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, "r") as f:
                raw = f.read().strip()

            for line in raw.strip().strip("`").split("\n"):
                line = line.strip()
                if line.startswith("```") or not line:
                    continue
                if line.startswith("tmux-text ") or line.startswith("tmux-ctrl "):
                    result = parse_and_expand(line)
                    if isinstance(result, ParseOk) and result.ok:
                        return True
                    else:
                        err_msg = str(result) if not isinstance(result, ParseOk) else "parse error"
                        _tmux_send(
                            online_tmux_session_name,
                            f"Parse error: {err_msg}. Retry with a valid "
                            f"tmux-text or tmux-ctrl command to {output_path}.",
                        )
                        subprocess.run(
                            ["tmux", "send-keys", "-t", online_tmux_session_name, "Enter"],
                            check=True,
                        )
                        return False
            return False

        # Agent idle, no output — remind to write the file.
        _tmux_send(
            online_tmux_session_name,
            f"Use the Write tool to save your keystroke command to "
            f"{output_path}. Do not reply in chat.",
        )
        subprocess.run(
            ["tmux", "send-keys", "-t", online_tmux_session_name, "Enter"],
            check=True,
        )
        return False

    return query_interactive_tui(
        online_tmux_session_name=online_tmux_session_name,
        get_query=get_query,
        validate_notify=validate_notify,
        interval_seconds=interval_seconds,
    )


# ── mock tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import unittest
    from unittest.mock import Mock, patch, call, mock_open

    _MOD = __name__
    _TMP = "FAKE_TMP_PATH"

    class QueryInteractiveReplCodingAgentTests(unittest.TestCase):

        def test_delegates_to_query_interactive_tui(self):
            with patch(_MOD + ".query_interactive_tui") as fake_tui:
                fake_tui.return_value = "tmux-text echo hello"
                result = query_interactive_repl_coding_agent(
                    online_tmux_session_name="sess",
                    get_query=lambda p: "do work",
                    interval_seconds=0.5,
                )
            self.assertEqual(result, "tmux-text echo hello")
            fake_tui.assert_called_once()
            _, kwargs = fake_tui.call_args
            self.assertEqual(kwargs["online_tmux_session_name"], "sess")
            self.assertEqual(kwargs["interval_seconds"], 0.5)
            self.assertTrue(callable(kwargs["validate_notify"]))

        def test_get_query_forwarded(self):
            with patch(_MOD + ".query_interactive_tui") as fake_tui:
                fake_tui.return_value = "ok"
                my_fn = lambda p: "q"
                query_interactive_repl_coding_agent("s", my_fn)
            self.assertIs(fake_tui.call_args[1]["get_query"], my_fn)

        def test_default_interval(self):
            with patch(_MOD + ".query_interactive_tui") as fake_tui:
                fake_tui.return_value = "ok"
                query_interactive_repl_coding_agent("s", lambda p: "q")
            self.assertEqual(fake_tui.call_args[1]["interval_seconds"], 1.0)

        # ── validate_notify behaviour ─────────────────────────────────

        def _get_validate_notify(self):
            """Capture the validate_notify closure for direct testing."""
            captured = []
            with patch(_MOD + ".query_interactive_tui",
                       side_effect=lambda **kw: captured.append(kw["validate_notify"]) or "ok"):
                query_interactive_repl_coding_agent("s", lambda p: "q")
            return captured[0]

        def test_validate_approves_permission_prompt(self):
            vn = self._get_validate_notify()
            with patch(_MOD + "._capture_screen",
                       return_value="Do you want to proceed?"), \
                 patch(_MOD + "._tmux_send") as fake_send, \
                 patch(_MOD + ".subprocess.run") as fake_run, \
                 patch("os.path.exists", return_value=False):
                result = vn("/tmp/out")
            self.assertFalse(result)
            fake_send.assert_called_with("s", "1")

        def test_validate_detects_valid_keystroke(self):
            vn = self._get_validate_notify()
            with patch(_MOD + "._capture_screen", return_value="❯ "), \
                 patch(_MOD + "._tmux_send"), \
                 patch(_MOD + ".subprocess.run"), \
                 patch("os.path.exists", return_value=True), \
                 patch("os.path.getsize", return_value=20), \
                 patch("builtins.open", mock_open(read_data="tmux-text echo hello")):
                result = vn("/tmp/out")
            self.assertTrue(result)

        def test_validate_sends_error_on_invalid_output(self):
            vn = self._get_validate_notify()
            with patch(_MOD + "._capture_screen", return_value="❯ "), \
                 patch(_MOD + "._tmux_send") as fake_send, \
                 patch(_MOD + ".subprocess.run"), \
                 patch("os.path.exists", return_value=True), \
                 patch("os.path.getsize", return_value=20), \
                 patch("builtins.open", mock_open(read_data="not a keystroke")):
                result = vn("/tmp/out")
            self.assertFalse(result)

        def test_validate_accepts_semicolon_chained_commands(self):
            # ; is a valid statement separator — chained commands parse fine.
            vn = self._get_validate_notify()
            with patch(_MOD + "._capture_screen", return_value="❯ "), \
                 patch(_MOD + "._tmux_send"), \
                 patch(_MOD + ".subprocess.run"), \
                 patch("os.path.exists", return_value=True), \
                 patch("os.path.getsize", return_value=20), \
                 patch("builtins.open", mock_open(
                     read_data="tmux-text echo ONE; tmux-ctrl Enter; tmux-text echo TWO")):
                result = vn("/tmp/out")
            self.assertTrue(result)  # chaining is valid grammar now

        def test_validate_idle_reminds_to_write(self):
            vn = self._get_validate_notify()
            with patch(_MOD + "._capture_screen", return_value="❯ "), \
                 patch(_MOD + "._tmux_send") as fake_send, \
                 patch(_MOD + ".subprocess.run"), \
                 patch("os.path.exists", return_value=False):
                result = vn("/tmp/out")
            self.assertFalse(result)
            self.assertTrue(any(
                "Write tool" in str(c) for c in fake_send.call_args_list
            ))

    unittest.main(verbosity=2)
