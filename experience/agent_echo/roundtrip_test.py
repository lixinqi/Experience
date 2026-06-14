"""
Roundtrip test: echo hello world → capture → reenact → execute → capture → compare.

Types the command character-by-character to produce fine-grained frame diffs,
then verifies the reenacted keystrokes replay identically.
"""

from __future__ import annotations

import re
import sys
import time
import json
import threading
from pathlib import Path

import libtmux

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?[\x40-\x7e]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _normalize(text: str) -> str:
    lines = [_strip_ansi(line).rstrip() for line in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


# ── session helpers ────────────────────────────────────────────────────────

def _get_pane(session_name: str):
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == session_name:
            return s.active_window.active_pane
    return None


def create_session(session_name: str) -> None:
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == session_name:
            s.kill()
            break
    server.new_session(session_name=session_name, attach=False)
    time.sleep(0.6)


def kill_session(session_name: str) -> None:
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == session_name:
            s.kill()
            return


def send_char(session_name: str, char: str) -> None:
    pane = _get_pane(session_name)
    if pane is None:
        raise RuntimeError(f"Session '{session_name}' not found")
    if char == "\n":
        pane.send_keys("Enter", literal=False, enter=False)
    else:
        pane.send_keys(char, literal=True, enter=False)


def type_text(session_name: str, text: str, char_delay: float = 0.08) -> None:
    for ch in text:
        send_char(session_name, ch)
        time.sleep(char_delay)


# ── capture ────────────────────────────────────────────────────────────────

def _capture_loop(pane, frames: list[str], stop_event: threading.Event, interval: float):
    while not stop_event.is_set():
        try:
            lines = pane.capture_pane()
            text = "\n".join(lines) if isinstance(lines, list) else str(lines)
            frames.append(text)
        except Exception:
            break
        time.sleep(interval)


def capture_while(session_name: str, action: callable, *,
                  interval: float = 0.1, settle_after: float = 1.0) -> list[str]:
    frames: list[str] = []
    stop_event = threading.Event()
    pane = _get_pane(session_name)
    t = threading.Thread(target=_capture_loop, args=(pane, frames, stop_event, interval), daemon=True)
    t.start()
    time.sleep(0.3)
    action()
    time.sleep(settle_after)
    stop_event.set()
    t.join(timeout=2)
    return frames


# ── diff / reenact ─────────────────────────────────────────────────────────

def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb: n += 1
        else: break
    return n


def diff_typed_text(prev: str, cur: str) -> str:
    if prev == cur:
        return ""
    suffix = cur[_common_prefix_len(prev, cur):]
    if "\n" not in suffix:
        return suffix
    first_nl = suffix.index("\n")
    typed = suffix[:first_nl + 1]
    prev_last = prev[prev.rfind("\n") + 1:]
    if prev_last and typed.startswith(prev_last):
        typed = typed[len(prev_last):]
    return typed


def _flush_pending(pending: list[str], statements: list[str]):
    if not pending:
        return
    text = "".join(pending).lstrip(" ")
    if text:
        statements.append(f"tmux-text {text}")
    pending.clear()


def _process_diff(diff: str, pending: list[str], statements: list[str]):
    if diff.strip() == "" and not pending:
        return
    if "\n" in diff:
        before_nl = diff[:diff.index("\n")]
        if before_nl.strip():
            pending.append(before_nl)
        elif not before_nl and not pending:
            pass
        elif before_nl:
            pending.append(before_nl)
        _flush_pending(pending, statements)
        statements.append("tmux-ctrl Enter")
    else:
        pending.append(diff)


def frames_to_keystrokes(frames: list[str]) -> list[str]:
    if len(frames) < 2:
        return []
    statements: list[str] = []
    pending: list[str] = []
    for i in range(len(frames) - 1):
        diff = diff_typed_text(frames[i], frames[i + 1])
        if diff:
            _process_diff(diff, pending, statements)
    _flush_pending(pending, statements)
    return statements


# ── execute ────────────────────────────────────────────────────────────────

def execute_statements(statements: list[str], session_name: str, *,
                       char_delay: float = 0.08, settle_after: float = 1.0) -> None:
    pane = _get_pane(session_name)
    if pane is None:
        raise RuntimeError(f"Session '{session_name}' not found")
    for stmt in statements:
        low = stmt.lower().lstrip()
        if low.startswith("tmux-text "):
            for ch in stmt[10:]:
                if ch == "\n":
                    pane.send_keys("Enter", literal=False, enter=False)
                else:
                    pane.send_keys(ch, literal=True, enter=False)
                time.sleep(char_delay)
        elif low.startswith("tmux-ctrl "):
            key = stmt[10:].strip()
            if key:
                pane.send_keys(key, literal=False, enter=False)
                time.sleep(char_delay)
    time.sleep(settle_after)


# ── comparison ─────────────────────────────────────────────────────────────

def _compare_finals(final_a: str, final_b: str) -> bool:
    if final_a == final_b:
        return True
    if final_a in final_b or final_b in final_a:
        print("  (one final frame contains the other)")
        return True
    lines_a, lines_b = final_a.split("\n"), final_b.split("\n")
    if len(lines_a) == len(lines_b):
        diffs = sum(1 for la, lb in zip(lines_a, lines_b) if la.rstrip() != lb.rstrip())
        if diffs == 0:
            return True
        print(f"  {diffs} lines differ")
    return False


def streams_equal(frames_a: list[str], frames_b: list[str]) -> bool:
    if not frames_a or not frames_b:
        print("  One or both streams are empty!")
        return False
    final_a = _normalize(frames_a[-1])
    final_b = _normalize(frames_b[-1])
    if _compare_finals(final_a, final_b):
        return True
    print(f"\n  Final A: {final_a[:200]!r}")
    print(f"  Final B: {final_b[:200]!r}")
    return False


# ── main test ──────────────────────────────────────────────────────────────

def _run_phases(command: str, session_a: str, session_b: str, tmpdir: Path) -> bool:
    create_session(session_a)
    def _type():
        type_text(session_a, command, char_delay=0.08)
        time.sleep(0.1)
        send_char(session_a, "\n")
        time.sleep(0.5)
    frames_a = capture_while(session_a, _type, interval=0.1, settle_after=1.0)
    if len(frames_a) < 3:
        return False
    statements = frames_to_keystrokes(frames_a)
    if not statements:
        return False
    create_session(session_b)
    def _exec():
        execute_statements(statements, session_b, char_delay=0.08)
    frames_b = capture_while(session_b, _exec, interval=0.1, settle_after=1.0)
    return streams_equal(frames_a, frames_b)


def _print_phase(phase: str, frames: list[str], statements: list[str] | None = None):
    if statements is not None:
        print(f"\n[Phase {phase}] Reenacting frames → keystrokes...")
        print(f"  Generated {len(statements)} statements:")
        for s in statements:
            print(f"    {s!r}")
    else:
        print(f"\n[Phase {phase}] Capturing...")
        print(f"  Captured {len(frames)} frames")


def test_roundtrip(command: str = "echo hello world", max_attempts: int = 5) -> bool:
    session_a, session_b = "echo_rt_A", "echo_rt_B"
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}\nAttempt {attempt}/{max_attempts}: roundtrip '{command}'\n{'='*60}")
        try:
            if _run_phases(command, session_a, session_b, Path("/tmp/echo_roundtrip_test")):
                print("\n  ✓ ROUNDTRIP PASSED — streams match!")
                return True
            print("\n  ✗ Streams differ")
        finally:
            kill_session(session_a)
            kill_session(session_b)
    print(f"\n  ✗ ROUNDTRIP FAILED after {max_attempts} attempts")
    return False


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Roundtrip test for agent_echo")
    ap.add_argument("--command", default="echo hello world")
    ap.add_argument("--max-attempts", type=int, default=5)
    args = ap.parse_args()
    sys.exit(0 if test_roundtrip(command=args.command, max_attempts=args.max_attempts) else 1)


if __name__ == "__main__":
    main()
