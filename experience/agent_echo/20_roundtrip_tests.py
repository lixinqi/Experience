"""20 roundtrip tests with FutureTensor storage and validation."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import sympy

from experience.agent_echo.roundtrip_test import (
    create_session, kill_session, type_text, send_char,
    capture_while, frames_to_keystrokes, execute_statements, streams_equal,
)
from experience.agent_echo.validate_future_tensor_dir import validate_future_tensor_dir
from experience.future_tensor.future_tensor import FutureTensor

COMMANDS = [
    "date +%Y-%m-%d",
    "pwd",
    "whoami",
    "hostname",
    "uname -a",
    "echo $HOME",
    "echo $(( 3 + 4 ))",
    "printf 'hello world'",
    "ls /",
    "head -5 /etc/hosts",
    "cat /etc/shells",
    "dirname /usr/local/bin/python3",
    "echo hello | tr 'a-z' 'A-Z'",
    "echo one:two:three | cut -d: -f2",
    "ls -1 / | wc -l",
    "echo hello && echo world",
    "seq 1 5",
    "python3 --version",
    "test -d /tmp && echo yes || echo no",
    "echo hello world",
]


def _store_frames_as_future_tensor(frames: list[str], ft_dir: Path):
    """Store captured frames in a real FutureTensor on disk."""
    async def _noop_get(coords, trajactory):
        return ("", type("S", (), {"is_confidence": False})())

    ft_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    ft = FutureTensor(str(ft_dir), _noop_get, [sympy.Integer(n)])
    for i, frame in enumerate(frames):
        tmp = ft_dir / f"_tmp_{i}.txt"
        tmp.write_text(frame)
        ft.ft_reset_materialized_value([i], 1.0, str(tmp), symlink=False)
        tmp.unlink()

    logical_view = ft.ft_describe_logical_view()
    meta = {
        "ft_relative_to": ft.ft_relative_to,
        "ft_tensor_uid": ft.ft_tensor_uid,
        "ft_capacity_shape": ft.ft_capacity_shape,
        "logical_view": logical_view,
    }
    (ft_dir / "ft_meta.json").write_text(json.dumps(meta, indent=2))
    return ft


def _capture_and_reenact(session_name: str, cmd: str,
                          ft_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Type a command, capture frames, store as FutureTensor, reenact."""
    create_session(session_name)

    def _type():
        type_text(session_name, cmd, char_delay=0.08)
        time.sleep(0.1)
        send_char(session_name, "\n")
        time.sleep(0.5)

    frames = capture_while(session_name, _type, interval=0.1, settle_after=1.0)
    _store_frames_as_future_tensor(frames, ft_dir)
    ft_errors = validate_future_tensor_dir(ft_dir)
    return frames, frames_to_keystrokes(frames), ft_errors


def _execute_and_capture(session_name: str, statements: list[str],
                          ft_dir: Path) -> list[str]:
    """Replay keystrokes, capture frames, store as FutureTensor."""
    create_session(session_name)

    def _exec():
        execute_statements(statements, session_name, char_delay=0.08)

    frames = capture_while(session_name, _exec, interval=0.1, settle_after=1.0)
    _store_frames_as_future_tensor(frames, ft_dir)
    validate_future_tensor_dir(ft_dir)
    return frames


def _run_one(i: int, cmd: str) -> tuple[bool, str]:
    session_a, session_b = f"rt_A{i}", f"rt_B{i}"
    tmpdir = Path(f"/tmp/echo_rt_ft/{i:02d}")
    try:
        frames_a, statements, ft_errs_a = _capture_and_reenact(
            session_a, cmd, tmpdir / "capture_A")
        frames_b = _execute_and_capture(
            session_b, statements, tmpdir / "capture_B")
        ok = streams_equal(frames_a, frames_b)
        detail = ""
        if ft_errs_a:
            detail += f"FT-A errors: {ft_errs_a}; "
        if not ok:
            detail += f"statements: {statements}"
        return ok, detail
    except Exception as e:
        return False, str(e)
    finally:
        kill_session(session_a)
        kill_session(session_b)


def run_all() -> tuple[int, int]:
    passed, failed = 0, 0
    for i, cmd in enumerate(COMMANDS, 1):
        ok, detail = _run_one(i, cmd)
        if ok and not detail:
            passed += 1
            print(f"  [{i:2d}/20] ✓  {cmd!r}", flush=True)
        else:
            failed += 1
            print(f"  [{i:2d}/20] ✗  {cmd!r}  — {detail}", flush=True)
    return passed, failed


if __name__ == "__main__":
    print("20 roundtrip tests + FutureTensor validation\n")
    passed, failed = run_all()
    print(f"\n{'='*50}")
    print(f"Results: {passed}/20 passed, {failed}/20 failed")
    sys.exit(0 if failed == 0 else 1)
