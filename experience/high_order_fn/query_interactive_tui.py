"""
query_interactive_tui — Query an online TUI agent via tmux, poll for the output
file, and use screen-capture comparison to detect when the agent is idle
before calling validate_notify.

    query_interactive_tui :=
      $query_result str
      <- $online_tmux_session_name str
      <- $get_query ($cur_query str <- $output_file_path str)
      <- $validate_notify ($output_valid bool <- $output_file_path str)
      <- $interval_seconds float # default 1.0
      # inline
      <- $output_file_path TempFile
      <- $last_screen str
      <- {
        1. tmux send $get_query($output_file_path)
        2. loop: wait $interval_seconds.
        3. if output_file_path filled with new content, return the content.
        4. capture current screen via tmux capture-pane.
        5. if current screen == last_screen:
             # agent is idle — safe to check validate_notify
             if validate_notify(output_file_path) == True:
                return content of output_file_path
        6. else:
             # agent is busy — update last_screen, keep waiting
             last_screen = current screen
        7. goto loop
      }
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from typing import Callable


def _capture_screen(session_name: str) -> str:
    """Capture the current tmux pane content as a string."""
    return subprocess.run(
        ["tmux", "capture-pane", "-t", session_name, "-p"],
        capture_output=True, text=True, check=True,
    ).stdout


def query_interactive_tui(
    online_tmux_session_name: str,
    get_query: Callable[[str], str],
    validate_notify: Callable[[str], bool],
    interval_seconds: float = 1.0,
) -> str:
    """Query an agent running in an online TUI via tmux.

    Sends the query (produced by ``get_query``) to the named tmux session,
    then polls ``output_file_path`` every ``interval_seconds``.  On each
    poll where the output file is not yet ready, the tmux screen is captured
    and compared with the previous capture:

    * If the screen **changed**, the agent is busy — keep waiting silently.
    * If the screen is **stable** (same as last poll), the agent is idle —
      ``validate_notify`` is called to give the caller a chance to signal
      readiness.

    The file's content is returned as soon as it has non-whitespace content
    **or** ``validate_notify`` returns ``True``.

    Args:
        online_tmux_session_name: Name of the tmux session running the agent.
        get_query: Called with ``output_file_path`` (a temp-file path) to
            produce the query string sent to the REPL.
        validate_notify: Called with ``output_file_path`` only when the
            tmux screen is stable (agent appears idle).  If it returns
            ``True`` the current file content is returned immediately.
        interval_seconds: Seconds to sleep between polls (default 1.0).

    Returns:
        The content of the output file once it is considered ready.
    """
    output_file_path = tempfile.mktemp()

    # 1. Send the initial query via tmux.
    cur_query = get_query(output_file_path)
    subprocess.run(
        ["tmux", "send-keys", "-t", online_tmux_session_name, cur_query],
        check=True,
    )
    subprocess.run(
        ["tmux", "send-keys", "-t", online_tmux_session_name, "Enter"],
        check=True,
    )

    # 2–7. Poll loop with screen-change detection.
    last_screen = ""
    while True:
        time.sleep(interval_seconds)

        # 3. Read the file if it exists — non-whitespace content → ready.
        content = ""
        if os.path.exists(output_file_path):
            with open(output_file_path, "r") as f:
                content = f.read()
        if content.strip():
            return content

        # 4. Capture the current screen.
        current_screen = _capture_screen(online_tmux_session_name)

        # 5–6. Only call validate_notify when the agent appears idle.
        if current_screen == last_screen:
            # Agent is idle — safe to check validate_notify.
            if validate_notify(output_file_path):
                return content
        else:
            # Agent is busy (screen still changing) — keep waiting.
            last_screen = current_screen


# ── mock tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import unittest
    from unittest.mock import Mock, patch, call, mock_open

    _MOD = __name__
    _TMP = "FAKE_TMP_PATH_12345"

    class QueryInteractiveTuiTests(unittest.TestCase):
        """Mock-based tests — no real tmux, no real filesystem."""

        # ── helpers ───────────────────────────────────────────────────

        @staticmethod
        def _call(*, query="do work", validate=False, session="sess",
                  interval=0.001, file_contents=None,
                  get_query=None, validate_notify=None,
                  screen_contents=None):
            gq = get_query or (lambda p: query)
            vn = validate_notify or (lambda p: validate)

            with patch(_MOD + ".tempfile.mktemp", return_value=_TMP), \
                 patch(_MOD + ".subprocess.run") as fake_run, \
                 patch(_MOD + ".time.sleep") as fake_sleep, \
                 patch(_MOD + ".os.path.exists", return_value=True), \
                 patch(_MOD + "._capture_screen") as fake_capture, \
                 patch("builtins.open", mock_open(read_data="")) as fake_open:

                if file_contents is not None:
                    fake_open.side_effect = file_contents
                if screen_contents is not None:
                    fake_capture.side_effect = screen_contents
                else:
                    fake_capture.return_value = ""

                result = query_interactive_tui(
                    online_tmux_session_name=session,
                    get_query=gq,
                    validate_notify=vn,
                    interval_seconds=interval,
                )
                return result, fake_run, fake_sleep, fake_capture

        # ── basic success paths ───────────────────────────────────────

        def test_immediate_success_first_poll(self):
            """File has content → return before screen capture."""
            content = "agent output\n"
            result, run_calls, sleep_calls, cap_calls = self._call(
                file_contents=[mock_open(read_data=content).return_value],
            )
            self.assertEqual(result, content)
            self.assertEqual(run_calls.call_count, 2)  # query + Enter
            cap_calls.assert_not_called()
            sleep_calls.assert_called_once_with(0.001)

        def test_success_after_empty_polls(self):
            """Two empty polls → content on third → return."""
            contents = [
                mock_open(read_data="").return_value,
                mock_open(read_data="").return_value,
                mock_open(read_data="done!").return_value,
            ]
            result, run_calls, sleep_calls, cap_calls = self._call(
                file_contents=contents, validate=False,
            )
            self.assertEqual(result, "done!")
            self.assertEqual(run_calls.call_count, 2)
            self.assertEqual(sleep_calls.call_count, 3)
            self.assertEqual(cap_calls.call_count, 2)

        # ── screen-change detection ───────────────────────────────────

        def test_screen_changing_skips_validate(self):
            """Screen keeps changing → validate_notify never called."""
            vn_calls = []
            result, _, sleep_calls, _ = self._call(
                validate_notify=lambda p: vn_calls.append(p) or False,
                file_contents=[
                    mock_open(read_data="").return_value,
                    mock_open(read_data="").return_value,
                    mock_open(read_data="done").return_value,
                ],
                screen_contents=["scr_A", "scr_B", "scr_C"],
                validate=False,
            )
            self.assertEqual(result, "done")
            self.assertEqual(len(vn_calls), 0)

        def test_screen_stable_fires_validate(self):
            """Screen stabilizes → validate_notify fires."""
            vn_calls = []
            result, _, sleep_calls, _ = self._call(
                validate_notify=lambda p: vn_calls.append(p) or True,
                file_contents=[
                    mock_open(read_data="").return_value,
                    mock_open(read_data="").return_value,
                ],
                screen_contents=["v1", "v1"],
                validate=False,
            )
            self.assertEqual(result, "")
            self.assertEqual(len(vn_calls), 1)

        # ── validate_notify ───────────────────────────────────────────

        def test_validate_notify_true_returns_content(self):
            """validate_notify=True + stable screen → returns content."""
            result, _, sleep_calls, _ = self._call(
                validate=True,
                file_contents=[mock_open(read_data="").return_value],
            )
            self.assertEqual(result, "")
            self.assertEqual(sleep_calls.call_count, 1)

        def test_validate_fires_after_idle(self):
            """Stable screen, validate_notify returns True on 2nd idle poll."""
            call_count = [0]
            def cv(path):
                call_count[0] += 1
                return call_count[0] >= 2

            contents = [
                mock_open(read_data="").return_value,
                mock_open(read_data="").return_value,
                mock_open(read_data="   \n").return_value,
            ]
            result, _, sleep_calls, _ = self._call(
                validate_notify=cv,
                file_contents=contents,
                screen_contents=["s", "s", "s"],
                validate=False,
            )
            self.assertEqual(result, "   \n")
            self.assertEqual(call_count[0], 2)
            self.assertEqual(sleep_calls.call_count, 3)

        # ── tmux command structure ────────────────────────────────────

        def test_initial_tmux_sends_query_then_enter(self):
            result, run_calls, _, _ = self._call(
                query="my query text",
                file_contents=[mock_open(read_data="ok").return_value],
            )
            cmd0 = run_calls.call_args_list[0][0][0]
            cmd1 = run_calls.call_args_list[1][0][0]
            self.assertEqual(cmd0, ["tmux", "send-keys", "-t", "sess", "my query text"])
            self.assertEqual(cmd1, ["tmux", "send-keys", "-t", "sess", "Enter"])

        def test_session_name_passed_through(self):
            result, run_calls, _, _ = self._call(
                session="my_agent_session",
                file_contents=[mock_open(read_data="ok").return_value],
            )
            for c in run_calls.call_args_list:
                self.assertEqual(c[0][0][3], "my_agent_session")

        def test_no_send_keys_after_initial(self):
            result, run_calls, _, _ = self._call(
                file_contents=[
                    mock_open(read_data="").return_value,
                    mock_open(read_data="done").return_value,
                ],
                validate=False,
            )
            self.assertEqual(run_calls.call_count, 2)

        # ── callbacks ─────────────────────────────────────────────────

        def test_get_query_receives_temp_file_path(self):
            paths = []
            self._call(
                get_query=lambda p: paths.append(p) or "q",
                file_contents=[mock_open(read_data="ok").return_value],
            )
            self.assertEqual(paths, [_TMP])

        def test_get_query_called_exactly_once(self):
            cnt = [0]
            def cq(p):
                cnt[0] += 1
                return "q"
            self._call(get_query=cq, file_contents=[
                mock_open(read_data="").return_value,
                mock_open(read_data="final").return_value,
            ], validate=False)
            self.assertEqual(cnt[0], 1)

        # ── interval ──────────────────────────────────────────────────

        def test_default_interval_is_1_second(self):
            mod = _MOD + "."
            with patch(mod + "tempfile.mktemp", return_value=_TMP), \
                 patch(mod + "subprocess.run"), \
                 patch(mod + "time.sleep") as fake_sleep, \
                 patch(mod + "_capture_screen", return_value=""), \
                 patch(mod + "os.path.exists", return_value=True), \
                 patch("builtins.open", mock_open(read_data="done!")):
                query_interactive_tui("s", lambda p: "q", lambda p: False)
            fake_sleep.assert_called_with(1.0)

        def test_custom_interval_is_used(self):
            result, _, sleep_calls, _ = self._call(
                interval=0.25,
                file_contents=[mock_open(read_data="ok").return_value],
            )
            sleep_calls.assert_called_once_with(0.25)

        # ── whitespace ────────────────────────────────────────────────

        def test_whitespace_not_ready_without_validate(self):
            contents = [
                mock_open(read_data="   \t \n  ").return_value,
                mock_open(read_data="real content").return_value,
            ]
            result, _, sleep_calls, _ = self._call(
                file_contents=contents, validate=False,
            )
            self.assertEqual(result, "real content")
            self.assertEqual(sleep_calls.call_count, 2)

        def test_whitespace_ready_if_validate_true(self):
            contents = [mock_open(read_data="   \t \n  ").return_value]
            result, _, sleep_calls, _ = self._call(
                file_contents=contents, validate=True,
            )
            self.assertEqual(result, "   \t \n  ")
            self.assertEqual(sleep_calls.call_count, 1)

        # ── content fidelity ──────────────────────────────────────────

        def test_multiline_returned_verbatim(self):
            content = "line1\nline2\nline3\n"
            result, _, _, _ = self._call(
                file_contents=[mock_open(read_data=content).return_value],
            )
            self.assertEqual(result, content)

        def test_unicode_preserved(self):
            content = "result: \U0001f680 中文\n"
            result, _, _, _ = self._call(
                file_contents=[mock_open(read_data=content).return_value],
            )
            self.assertEqual(result, content)

        # ── errors ────────────────────────────────────────────────────

        def test_subprocess_error_propagates(self):
            mod = _MOD + "."
            with patch(mod + "tempfile.mktemp", return_value=_TMP), \
                 patch(mod + "subprocess.run",
                       side_effect=subprocess.CalledProcessError(1, "tmux")), \
                 patch(mod + "time.sleep"):
                with self.assertRaises(subprocess.CalledProcessError):
                    query_interactive_tui("s", lambda p: "q", lambda p: False)

        # ── file appears after being absent ───────────────────────────

        def test_file_absent_then_appears(self):
            mod = _MOD + "."
            with patch(mod + "tempfile.mktemp", return_value=_TMP), \
                 patch(mod + "subprocess.run"), \
                 patch(mod + "time.sleep") as fake_sleep, \
                 patch(mod + "_capture_screen", return_value=""), \
                 patch(mod + "os.path.exists", side_effect=[False, True]) as fake_exists, \
                 patch("builtins.open", mock_open(read_data="found!")):

                result = query_interactive_tui(
                    "s", lambda p: "q", lambda p: False, interval_seconds=0.01,
                )
            self.assertEqual(result, "found!")
            self.assertEqual(fake_exists.call_count, 2)
            self.assertEqual(fake_sleep.call_count, 2)

    # ── run ───────────────────────────────────────────────────────────

    unittest.main(verbosity=2)
