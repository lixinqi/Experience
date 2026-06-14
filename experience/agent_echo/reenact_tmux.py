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
    """Load capture stream from JSON-lines file."""
    frames: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line)["text"])
    return frames


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n


def diff_typed_text(prev: str, cur: str) -> str:
    """Extract what was typed between two consecutive frames.

    Output on new lines (command results) is stripped — only typed chars and
    the Enter that produced them are returned.
    """
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


def _handle_diff(diff: str, pending: list[str], statements: list[str]):
    """Process a single frame diff: accumulate typed text or emit statements."""
    if not diff:
        return
    if diff.strip() == "" and not pending:
        return
    if "\n" in diff:
        before_nl = diff[:diff.index("\n")]
        if before_nl.strip():
            pending.append(before_nl)
        elif before_nl and not pending:
            pass
        elif before_nl:
            pending.append(before_nl)
        _flush_pending(pending, statements)
        statements.append("tmux-ctrl Enter")
    else:
        pending.append(diff)


def frames_to_keystrokes(frames: list[str]) -> list[str]:
    """Convert screen frames to keystroke DSL statements."""
    if len(frames) < 2:
        return []
    statements: list[str] = []
    pending: list[str] = []
    for i in range(len(frames) - 1):
        _handle_diff(diff_typed_text(frames[i], frames[i + 1]), pending, statements)
    _flush_pending(pending, statements)
    return statements


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert tmux capture frames to keystroke DSL.")
    p.add_argument("--input-capture-stream", type=Path, required=True)
    p.add_argument("--input-frame-time-ranges", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def main() -> None:
    args = _build_argparser().parse_args()
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
    for s in statements:
        print(f"    {s}")


if __name__ == "__main__":
    main()
