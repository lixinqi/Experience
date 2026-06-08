"""
Integration test: query_interactive_repl_coding_agent against interactive
Claude (no --print) running in a persistent tmux session.

The function sends a chat message to Claude via tmux send-keys.  Claude
interprets the message, uses its Write tool to write the keystroke DSL
command to the output file, and the function polls until content appears.
"""

import os
import subprocess
import sys
import time

from experience.react.query_interactive_repl_coding_agent import (
    query_interactive_repl_coding_agent,
)

SESSION_NAME = "test_interactive_agent"


def tmux_send(text: str):
    subprocess.run(
        ["tmux", "send-keys", "-t", SESSION_NAME, text], check=True,
    )


def tmux_enter():
    subprocess.run(
        ["tmux", "send-keys", "-t", SESSION_NAME, "Enter"], check=True,
    )


def setup_session():
    """Start bash, source env, launch interactive Claude (no --print)."""
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME], check=False,
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME, "bash"], check=True,
    )
    time.sleep(0.5)

    tmux_send("unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT"); tmux_enter()
    time.sleep(0.1)

    flash = os.path.expanduser("~/.llm-config/flash-env.sh")
    if os.path.isfile(flash):
        tmux_send("source " + flash); tmux_enter()
        time.sleep(0.1)

    tmux_send(
        "claude --add-dir /tmp --allow-dangerously-skip-permissions "
        "--permission-mode bypassPermissions"
    ); tmux_enter()
    time.sleep(4)


def teardown_session():
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME], check=False,
    )


# ── test cases ───────────────────────────────────────────────────────────

def _task_prompt(task: str, output_path: str) -> str:
    """Build a prompt that makes Claude write to the output file via Write tool."""
    return (
        f"Terminal:\n"
        f"bash-3.2$ \n\n"
        f"Task: {task}\n\n"
        f"Reply with keystroke DSL commands. "
        f"You can chain multiple commands with ; on one line.\n"
        f"tmux-text <text> — type literal text\n"
        f"tmux-ctrl <key>  — press a control key\n"
        f"Use tmux-text \";\" to type a literal semicolon.\n"
        f"Do not reply in chat. Use the Write tool to save your answer to "
        f"{output_path}."
    )


def test_echo_hello():
    """Task: echo hello → Claude writes keystroke to output file."""
    t0 = time.time()
    raw = query_interactive_repl_coding_agent(
        online_tmux_session_name=SESSION_NAME,
        get_query=lambda output_path: _task_prompt(
            "echo hello", output_path,
        ),
        interval_seconds=0.5,
    )
    elapsed = time.time() - t0
    print(f"  returned in {elapsed:.1f}s")
    print(f"  raw ({len(raw)} chars): {raw!r}")
    assert "tmux-text" in raw.lower(), f"unexpected: {raw!r}"
    print("  ✓")


def test_ls_command():
    """Task: list files → Claude writes tmux-text ls -la."""
    t0 = time.time()
    raw = query_interactive_repl_coding_agent(
        online_tmux_session_name=SESSION_NAME,
        get_query=lambda output_path: _task_prompt(
            "list files in current directory", output_path,
        ),
        interval_seconds=0.5,
    )
    elapsed = time.time() - t0
    print(f"  returned in {elapsed:.1f}s")
    print(f"  raw ({len(raw)} chars): {raw!r}")
    assert "tmux-text" in raw.lower(), f"unexpected: {raw!r}"
    print("  ✓")


def test_ctrl_key():
    """Task: interrupt → Claude writes tmux-ctrl C-c."""
    t0 = time.time()
    raw = query_interactive_repl_coding_agent(
        online_tmux_session_name=SESSION_NAME,
        get_query=lambda output_path: _task_prompt(
            "interrupt the current process", output_path,
        ),
        interval_seconds=0.5,
    )
    elapsed = time.time() - t0
    print(f"  returned in {elapsed:.1f}s")
    print(f"  raw ({len(raw)} chars): {raw!r}")
    assert "tmux-" in raw.lower(), f"unexpected: {raw!r}"
    print("  ✓")


# ── main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0

    def run(label, fn):
        global passed, failed
        print(f"  {label} ...", end=" ", flush=True)
        try:
            setup_session()   # kill old + create fresh at start
            fn()
            passed += 1
            # Leave session alive — next run kills it, or user inspects.
        except Exception as e:
            print(f"\n  ✗ {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("Running tests...\n")
    run("echo_hello", test_echo_hello)
    run("ls_command", test_ls_command)
    run("ctrl_key", test_ctrl_key)

    total = passed + failed
    print(f"\n  {passed}/{total} passed{' ✓' if failed == 0 else ''}")
    sys.exit(0 if failed == 0 else 1)
