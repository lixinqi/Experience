"""
50 distinct roundtrip tests — capture → reenact → execute → capture → compare.
"""
from __future__ import annotations

import sys
import time
from experience.agent_echo.roundtrip_test import (
    create_session, kill_session, type_text, send_char,
    capture_while, frames_to_keystrokes, execute_statements, streams_equal,
)

COMMANDS = [
    # 1-10: system info builtins
    "date +%Y-%m-%d",
    "pwd",
    "whoami",
    "hostname",
    "uname",
    "uname -a",
    "id",
    "getconf PAGE_SIZE",
    "which bash",
    "type bash",

    # 11-20: environment and expansion
    "echo $HOME",
    "echo $USER",
    "echo $SHELL",
    "echo ~",
    "echo ~/Documents",
    "echo $(hostname)",
    "echo $(( 3 + 4 ))",
    "echo $(( 7 * 9 ))",
    "printf 'hello world'",
    "printf 'one\ntwo\nthree'",

    # 21-30: file inspection
    "ls /",
    "ls -1 / | head -5",
    "ls /bin | head -5",
    "head -5 /etc/hosts",
    "cat /etc/shells",
    "wc -c /etc/hosts",
    "dirname /usr/local/bin/python3",
    "basename /usr/local/bin/python3",
    "readlink $(which python3)",
    "df -h .",

    # 31-40: text processing and pipes
    "echo hello world | wc -w",
    "echo hello | tr 'a-z' 'A-Z'",
    "echo abcdef | wc -c",
    "printf 'a\nb\nc' | wc -l",
    "echo the quick brown fox | tr ' ' '\n' | sort",
    "echo one:two:three | cut -d: -f2",
    "ls -1 / | wc -l",
    "echo hello && echo world",
    "echo first; echo second; echo third",
    "true && echo ok || echo fail",

    # 41-50: mixed real commands
    "python3 --version",
    "bash --version | head -1",
    "env | head -3",
    "echo hello world",
    "seq 1 5",
    "sleep 0.1 && echo done",
    "test -d /tmp && echo yes || echo no",
    "test -f /etc/hosts && echo exists",
    "cd / && pwd",
    "echo 0xDEAD 3.14159 key=val @user path/to/file",
]


def run_all() -> tuple[int, int]:
    passed = 0
    failed = 0

    for i, cmd in enumerate(COMMANDS, 1):
        session_a = f"rt_A{i}"
        session_b = f"rt_B{i}"
        try:
            create_session(session_a)

            def _type():
                type_text(session_a, cmd, char_delay=0.08)
                time.sleep(0.1)
                send_char(session_a, "\n")
                time.sleep(0.5)

            frames_a = capture_while(session_a, _type, interval=0.1, settle_after=1.0)
            statements = frames_to_keystrokes(frames_a)

            create_session(session_b)

            def _exec():
                execute_statements(statements, session_b, char_delay=0.08)

            frames_b = capture_while(session_b, _exec, interval=0.1, settle_after=1.0)

            if streams_equal(frames_a, frames_b):
                passed += 1
                print(f"  [{i:2d}/50] ✓  {cmd!r}", flush=True)
            else:
                failed += 1
                print(f"  [{i:2d}/50] ✗  {cmd!r}  — streams differ", flush=True)
                # Show the statements
                print(f"         statements: {statements!r}", flush=True)
        except Exception as e:
            failed += 1
            print(f"  [{i:2d}/50] ✗  {cmd!r}  — {e}", flush=True)
        finally:
            kill_session(session_a)
            kill_session(session_b)

    return passed, failed


if __name__ == "__main__":
    print("50 distinct roundtrip tests\n")
    passed, failed = run_all()
    print(f"\n{'='*50}")
    print(f"Results: {passed}/50 passed, {failed}/50 failed")
    sys.exit(0 if failed == 0 else 1)
