"""
capture_tmux — Capture tmux session frames at 0.5s intervals.

Generated from capture_tmux.viba:
    main := void
        <- ArgParse[$tmux_session_name str]
        <- ArgParse[$output.capture_stream TmuxCaptureStream.capture_stream]
        <- ArgParse[$output.frame_time_ranges TmuxCaptureStream.frame_time_ranges]
        <- Import[design.viba]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import libtmux

# Shared state for graceful shutdown
_shutdown_flag = False
_frames_buffer: list[str] = []
_time_ranges_buffer: list[tuple[float, float]] = []


def _on_shutdown(signum, frame):
    global _shutdown_flag
    _shutdown_flag = True


signal.signal(signal.SIGTERM, _on_shutdown)
signal.signal(signal.SIGINT, _on_shutdown)


def capture_frames(
    session_name: str,
    *,
    interval: float = 0.5,
    max_frames: int | None = None,
    settle_wait: float = 0.1,
) -> tuple[list[str], list[tuple[float, float]]]:
    """Capture tmux pane content at regular intervals.

    Handles SIGTERM/SIGINT gracefully — returns whatever frames were
    captured before the signal was received.

    Args:
        session_name: Tmux session name to capture.
        interval: Seconds between captures.
        max_frames: Stop after this many frames (None = capture until
                    session ends, signal received, or stdin EOF).
        settle_wait: Seconds to wait after connecting before first capture.

    Returns:
        (frames, time_ranges) where frames is list of screen text and
        time_ranges is list of (start_ts, end_ts) timestamps.
    """
    global _shutdown_flag, _frames_buffer, _time_ranges_buffer
    _shutdown_flag = False
    _frames_buffer = []
    _time_ranges_buffer = []

    server = libtmux.Server()
    session = None
    for s in server.sessions:
        if s.session_name == session_name:
            session = s
            break

    if session is None:
        raise RuntimeError(f"Tmux session '{session_name}' not found")

    pane = session.active_window.active_pane
    time.sleep(settle_wait)

    count = 0
    while not _shutdown_flag:
        start_ts = time.time()

        # Check session still exists
        try:
            _ = pane.capture_pane()
        except Exception:
            break

        server2 = libtmux.Server()
        cur_session = None
        for s in server2.sessions:
            if s.session_name == session_name:
                cur_session = s
                break
        if cur_session is None:
            break
        pane = cur_session.active_window.active_pane

        lines = pane.capture_pane()
        frame_text = "\n".join(lines) if isinstance(lines, list) else str(lines)
        _frames_buffer.append(frame_text)

        end_ts = time.time()
        _time_ranges_buffer.append((start_ts, end_ts))

        count += 1
        if max_frames is not None and count >= max_frames:
            break

        elapsed = end_ts - start_ts
        sleep_for = max(0, interval - elapsed)
        # Sleep in small chunks to respond to signals promptly
        while sleep_for > 0 and not _shutdown_flag:
            time.sleep(min(0.1, sleep_for))
            sleep_for -= 0.1

    return _frames_buffer, _time_ranges_buffer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture tmux session frames at 0.5s intervals."
    )
    parser.add_argument(
        "--tmux-session-name",
        type=str,
        required=True,
        help="Name of the tmux session to capture.",
    )
    parser.add_argument(
        "--output-capture-stream",
        type=Path,
        required=True,
        help="Path to write capture stream (JSON lines of frame text).",
    )
    parser.add_argument(
        "--output-frame-time-ranges",
        type=Path,
        required=True,
        help="Path to write frame time ranges (JSON).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between captures (default: 0.5).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to capture.",
    )
    args = parser.parse_args()

    frames, time_ranges = capture_frames(
        args.tmux_session_name,
        interval=args.interval,
        max_frames=args.max_frames,
    )

    # Write capture stream: one JSON object per line {index, text}
    args.output_capture_stream.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_capture_stream, "w") as f:
        for i, frame in enumerate(frames):
            f.write(json.dumps({"index": i, "text": frame}) + "\n")

    # Write time ranges
    args.output_frame_time_ranges.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_frame_time_ranges, "w") as f:
        json.dump(
            [
                {"index": i, "start": tr[0], "end": tr[1]}
                for i, tr in enumerate(time_ranges)
            ],
            f,
            indent=2,
        )

    print(f"Captured {len(frames)} frames from session '{args.tmux_session_name}'")
    print(f"  Capture stream: {args.output_capture_stream}")
    print(f"  Time ranges: {args.output_frame_time_ranges}")


if __name__ == "__main__":
    main()
