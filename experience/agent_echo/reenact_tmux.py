"""
reenact_tmux — Convert tmux capture stream to keystroke DSL.

Generated from reenact_tmux.viba:
    main := void
        <- ArgParse[$input.capture_stream TmuxCaptureStream.capture_stream]
        <- ArgParse[$input.frame_time_ranges TmuxCaptureStream.frame_time_ranges]
        <- ArgParse[$output TmuxActionSequence]
        <- Import[design.viba]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_capture_stream(path: Path) -> list[str]:
    """Load capture stream from JSON-lines file. Returns list of frame texts."""
    frames: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line)["text"])
    return frames


# ── diff logic ─────────────────────────────────────────────────────────────


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

    # Newlines appeared. Split: everything up to and including the first \\n
    # is "typed" (Enter pressed). Everything after is output — ignore.
    first_nl = suffix.index("\n")
    typed = suffix[: first_nl + 1]  # includes the \\n

    # Edge case: the typed portion may include old content that was
    # scrolled up.  Strip it.
    prev_last_line_start = prev.rfind("\n")
    prev_last_line = (
        prev[prev_last_line_start + 1 :] if prev_last_line_start >= 0 else prev
    )
    if prev_last_line and typed.startswith(prev_last_line):
        typed = typed[len(prev_last_line) :]

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
            text = text.lstrip(" ")  # strip prompt-cursor-gap space
            if text:
                statements.append(f"tmux-text {text}")
            pending.clear()

    for i in range(len(frames) - 1):
        diff = diff_typed_text(frames[i], frames[i + 1])

        if not diff:
            continue

        # Skip whitespace-only diffs when nothing is pending (prompt artifact)
        if diff.strip() == "" and not pending:
            continue

        if "\n" in diff:
            before_nl = diff[: diff.index("\n")]
            if before_nl.strip():
                pending.append(before_nl)
            elif before_nl and not pending:
                pass  # whitespace-only before Enter — prompt artifact
            elif before_nl:
                pending.append(before_nl)
            flush()
            statements.append("tmux-ctrl Enter")
        else:
            pending.append(diff)

    flush()
    return statements


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert tmux capture frames to keystroke DSL."
    )
    parser.add_argument(
        "--input-capture-stream",
        type=Path,
        required=True,
        help="Path to capture stream (JSON lines of frame text).",
    )
    parser.add_argument(
        "--input-frame-time-ranges",
        type=Path,
        required=True,
        help="Path to frame time ranges (JSON).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write keystroke DSL statements (one per line).",
    )
    args = parser.parse_args()

    frames = load_capture_stream(args.input_capture_stream)

    if not frames:
        print("No frames found in capture stream.", file=sys.stderr)
        sys.exit(1)

    statements = frames_to_keystrokes(frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for stmt in statements:
            f.write(stmt + "\n")

    print(f"Reenacted {len(frames)} frames → {len(statements)} keystroke statements")
    print(f"  Output: {args.output}")
    if statements:
        print("  Statements:")
        for s in statements:
            print(f"    {s}")


if __name__ == "__main__":
    main()
