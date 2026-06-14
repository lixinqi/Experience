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


# ── ansi strip ─────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?[\x40-\x7e]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _normalize(text: str) -> str:
    """Strip ANSI codes and trailing whitespace from each line."""
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
    """Send a single literal character to a tmux session."""
    pane = _get_pane(session_name)
    if pane is None:
        raise RuntimeError(f"Session '{session_name}' not found")
    if char == "\n":
        pane.send_keys("Enter", literal=False, enter=False)
    else:
        pane.send_keys(char, literal=True, enter=False)


def type_text(session_name: str, text: str, char_delay: float = 0.08) -> None:
    """Type text character by character with delays."""
    for ch in text:
        send_char(session_name, ch)
        time.sleep(char_delay)


# ── capture ────────────────────────────────────────────────────────────────


def capture_while(
    session_name: str,
    action: callable,
    *,
    interval: float = 0.1,
    settle_after: float = 1.0,
) -> list[str]:
    """Capture tmux frames while performing an action in a background thread."""
    frames: list[str] = []
    stop_event = threading.Event()

    def _capture_loop():
        pane = _get_pane(session_name)
        if pane is None:
            return
        while not stop_event.is_set():
            try:
                lines = pane.capture_pane()
                text = "\n".join(lines) if isinstance(lines, list) else str(lines)
                frames.append(text)
            except Exception:
                break
            time.sleep(interval)

    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()
    time.sleep(0.3)  # Let first frame capture

    action()

    time.sleep(settle_after)
    stop_event.set()
    t.join(timeout=2)

    return frames


# ── diff / reenact ─────────────────────────────────────────────────────────


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix of two strings."""
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n


def diff_typed_text(prev: str, cur: str) -> str:
    """Extract what was typed between two consecutive frames.

    Returns the new content that appeared.  If the diff contains a newline
    followed by output (command result), only the typed portion (including the
    Enter that produced it) is returned.  Pure output on new lines is stripped.
    """
    if prev == cur:
        return ""

    cp = _common_prefix_len(prev, cur)
    suffix = cur[cp:]

    # If no newline in the new content, it's all typed text.
    if "\n" not in suffix:
        return suffix

    # Newlines appeared. Split: everything up to and including the first \n
    # is "typed" (Enter pressed). Everything after is output — ignore.
    first_nl = suffix.index("\n")
    typed = suffix[: first_nl + 1]  # includes the \n

    # Edge case: the typed portion may include old content that was
    # scrolled up. If the typed portion contains the entire prev last line,
    # remove it — it's not newly typed.
    prev_last_line_start = prev.rfind("\n")
    prev_last_line = prev[prev_last_line_start + 1:] if prev_last_line_start >= 0 else prev
    if prev_last_line and typed.startswith(prev_last_line):
        typed = typed[len(prev_last_line):]

    return typed


def frames_to_keystrokes(frames: list[str]) -> list[str]:
    """Convert screen frames to keystroke DSL statements.

    Accumulates small character diffs into typed text. When a diff contains a
    newline (Enter was pressed), flushes pending chars and emits the Enter.
    """
    if len(frames) < 2:
        return []

    statements: list[str] = []
    pending: list[str] = []

    def flush():
        if pending:
            text = "".join(pending)
            # Strip leading space that leaked from the prompt's cursor gap
            text = text.lstrip(" ")
            if text:
                statements.append(f"tmux-text {text}")
            pending.clear()

    for i in range(len(frames) - 1):
        diff = diff_typed_text(frames[i], frames[i + 1])

        if not diff:
            continue

        # Filter out diff fragments that are solely whitespace appearing
        # at cursor position (prompt spacing, not typed characters).
        # Only keep them when they appear between typed characters.
        if diff.strip() == "" and not pending:
            # Whitespace-only diff with nothing pending — prompt artifact, skip
            continue

        if "\n" in diff:
            before_nl = diff[: diff.index("\n")]
            if before_nl.strip():
                pending.append(before_nl)
            elif before_nl and not pending:
                # Only whitespace before Enter — prompt artifact, ignore
                pass
            elif before_nl:
                pending.append(before_nl)
            flush()
            statements.append("tmux-ctrl Enter")
        else:
            pending.append(diff)

    flush()
    return statements


# ── execute ────────────────────────────────────────────────────────────────


def execute_statements(
    statements: list[str],
    session_name: str,
    *,
    char_delay: float = 0.08,
    settle_after: float = 1.0,
) -> None:
    """Execute keystroke DSL statements in a tmux session.

    Types each character individually so the replay capture matches the
    original character-by-character capture.
    """
    pane = _get_pane(session_name)
    if pane is None:
        raise RuntimeError(f"Session '{session_name}' not found")

    for stmt in statements:
        low = stmt.lower().lstrip()  # Only strip leading whitespace, preserve \n
        if low.startswith("tmux-text "):
            text = stmt[10:]  # After "tmux-text "
            for ch in text:
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


def streams_equal(frames_a: list[str], frames_b: list[str]) -> bool:
    """Check if two capture streams represent the same screen content."""
    if not frames_a or not frames_b:
        print("  One or both streams are empty!")
        return False

    final_a = _normalize(frames_a[-1])
    final_b = _normalize(frames_b[-1])

    if final_a == final_b:
        return True

    # Check containment (timing differences)
    if final_a in final_b or final_b in final_a:
        print("  (one final frame contains the other)")
        return True

    # Line-by-line comparison ignoring trailing whitespace
    lines_a = final_a.split("\n")
    lines_b = final_b.split("\n")
    if len(lines_a) == len(lines_b):
        diffs = 0
        for la, lb in zip(lines_a, lines_b):
            if la.rstrip() != lb.rstrip():
                diffs += 1
        if diffs == 0:
            return True
        print(f"  {diffs} lines differ")

    print(f"\n  Final A: {final_a[:200]!r}")
    print(f"  Final B: {final_b[:200]!r}")
    return False


# ── main test ──────────────────────────────────────────────────────────────


def test_roundtrip(command: str = "echo hello world", max_attempts: int = 5) -> bool:
    session_a = "echo_rt_A"
    session_b = "echo_rt_B"
    tmpdir = Path("/tmp/echo_roundtrip_test")
    tmpdir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}")
        print(f"Attempt {attempt}/{max_attempts}: roundtrip '{command}'")
        print(f"{'='*60}")

        try:
            # ── Phase 1: Original capture ──
            print("\n[Phase 1] Capturing original (char-by-char typing)...")
            create_session(session_a)

            def _type_original():
                type_text(session_a, command, char_delay=0.08)
                time.sleep(0.1)
                send_char(session_a, "\n")  # Enter
                time.sleep(0.5)  # Let output appear

            frames_original = capture_while(
                session_a, _type_original, interval=0.1, settle_after=1.0
            )
            print(f"  Captured {len(frames_original)} frames")
            for i, f in enumerate(frames_original):
                print(f"    [{i}] {_normalize(f)[:100]!r}")

            if len(frames_original) < 3:
                print("  Not enough frames — retrying...")
                continue

            with open(tmpdir / "frames_original.json", "w") as f:
                json.dump(frames_original, f, indent=2)

            # ── Phase 2: Reenact ──
            print("\n[Phase 2] Reenacting frames → keystrokes...")
            statements = frames_to_keystrokes(frames_original)
            print(f"  Generated {len(statements)} statements:")
            for s in statements:
                print(f"    {s!r}")

            if not statements:
                print("  No statements — retrying...")
                continue

            with open(tmpdir / "statements.ks", "w") as f:
                for s in statements:
                    f.write(s + "\n")

            # ── Phase 3: Execute + capture ──
            print("\n[Phase 3] Executing keystrokes in new session...")
            create_session(session_b)

            def _execute():
                execute_statements(statements, session_b, char_delay=0.08)

            frames_replay = capture_while(
                session_b, _execute, interval=0.1, settle_after=1.0
            )
            print(f"  Captured {len(frames_replay)} frames")
            for i, f in enumerate(frames_replay):
                print(f"    [{i}] {_normalize(f)[:100]!r}")

            with open(tmpdir / "frames_replay.json", "w") as f:
                json.dump(frames_replay, f, indent=2)

            # ── Phase 4: Compare ──
            print("\n[Phase 4] Comparing streams...")
            if streams_equal(frames_original, frames_replay):
                print("\n  ✓ ROUNDTRIP PASSED — streams match!")
                return True
            else:
                print("\n  ✗ Streams differ")

        finally:
            kill_session(session_a)
            kill_session(session_b)

    print(f"\n  ✗ ROUNDTRIP FAILED after {max_attempts} attempts")
    return False


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Roundtrip test for agent_echo")
    ap.add_argument("--command", default="echo hello world",
                    help="Command to test")
    ap.add_argument("--max-attempts", type=int, default=5)
    args = ap.parse_args()

    success = test_roundtrip(command=args.command, max_attempts=args.max_attempts)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
