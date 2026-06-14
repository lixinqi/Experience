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
import time
from pathlib import Path

import libtmux

_shutdown_flag = False
_frames_buffer: list[str] = []
_time_ranges_buffer: list[tuple[float, float]] = []


def _on_shutdown(signum, frame):
    global _shutdown_flag
    _shutdown_flag = True


signal.signal(signal.SIGTERM, _on_shutdown)
signal.signal(signal.SIGINT, _on_shutdown)


def _get_session_pane(session_name: str):
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == session_name:
            return s.active_window.active_pane
    return None


def _sleep_interruptible(seconds: float):
    remaining = seconds
    while remaining > 0 and not _shutdown_flag:
        time.sleep(min(0.1, remaining))
        remaining -= 0.1


def _capture_one_frame(session_name: str, prev_pane) -> tuple[object, str]:
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == session_name:
            pane = s.active_window.active_pane
            lines = pane.capture_pane()
            text = "\n".join(lines) if isinstance(lines, list) else str(lines)
            return (pane, text)
    return (None, "")


def _capture_loop(session_name: str, pane, interval: float, max_frames: int | None):
    count = 0
    while not _shutdown_flag:
        start_ts = time.time()
        try:
            _ = pane.capture_pane()
        except Exception:
            break
        pane, text = _capture_one_frame(session_name, pane)
        if pane is None:
            break
        _frames_buffer.append(text)
        _time_ranges_buffer.append((start_ts, time.time()))
        count += 1
        if max_frames is not None and count >= max_frames:
            break
        _sleep_interruptible(max(0, interval - (time.time() - start_ts)))


def capture_frames(
    session_name: str,
    *,
    interval: float = 0.5,
    max_frames: int | None = None,
    settle_wait: float = 0.1,
) -> tuple[list[str], list[tuple[float, float]]]:
    """Capture tmux pane content at regular intervals.  Returns (frames, time_ranges)."""
    global _shutdown_flag, _frames_buffer, _time_ranges_buffer
    _shutdown_flag = False
    _frames_buffer = []
    _time_ranges_buffer = []
    pane = _get_session_pane(session_name)
    if pane is None:
        raise RuntimeError(f"Tmux session '{session_name}' not found")
    time.sleep(settle_wait)
    _capture_loop(session_name, pane, interval, max_frames)
    return _frames_buffer, _time_ranges_buffer


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Capture tmux session frames at 0.5s intervals.")
    p.add_argument("--tmux-session-name", type=str, required=True)
    p.add_argument("--output-capture-stream", type=Path, required=True)
    p.add_argument("--output-frame-time-ranges", type=Path, required=True)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=None)
    return p


def _write_output(stream_path: Path, ranges_path: Path,
                  frames: list[str], time_ranges: list[tuple[float, float]]):
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stream_path, "w") as f:
        for i, frame in enumerate(frames):
            f.write(json.dumps({"index": i, "text": frame}) + "\n")
    ranges_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ranges_path, "w") as f:
        json.dump([{"index": i, "start": tr[0], "end": tr[1]}
                    for i, tr in enumerate(time_ranges)], f, indent=2)


def main() -> None:
    args = _build_argparser().parse_args()
    frames, time_ranges = capture_frames(
        args.tmux_session_name, interval=args.interval, max_frames=args.max_frames)
    _write_output(args.output_capture_stream, args.output_frame_time_ranges,
                  frames, time_ranges)
    print(f"Captured {len(frames)} frames from session '{args.tmux_session_name}'")
    print(f"  Capture stream: {args.output_capture_stream}")
    print(f"  Time ranges: {args.output_frame_time_ranges}")


if __name__ == "__main__":
    main()
