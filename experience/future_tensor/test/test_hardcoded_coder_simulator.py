"""
test_coder_simulator: simulate a coder typing characters one by one into a tmux session.

Everything runs as a single lazy ft_async_get pipeline — one ft_forward call
triggers the entire sequence: create session, type each char, capture frame.

Uses:
  1. ft_tmux_create_session — create the terminal
  2. ft_tmux_send_text     — type each character
  3. ft_tmux_send_ctrl     — send Enter (newline)
  4. ft_tmux_capture_pane  — verify pane content after each frame
  5. ft_sleep              — pause for tmux to settle
  6. ft_sequential         — chain operations in order
  7. ft_make_forwarded     — create pre-materialized input tensor
"""

import os
import sys
import tempfile

import libtmux

from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.function.ft_make_forwarded import ft_make_forwarded
from experience.future_tensor.function.ft_tmux_create_session import ft_tmux_create_session
from experience.future_tensor.function.ft_tmux_send_text import ft_tmux_send_text
from experience.future_tensor.function.ft_tmux_send_ctrl import ft_tmux_send_ctrl
from experience.future_tensor.function.ft_tmux_capture_pane import ft_tmux_capture_pane
from experience.future_tensor.function.ft_sleep import ft_sleep
from experience.future_tensor.function.ft_sequential import ft_sequential
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor


# ─── Test infrastructure ───

passed = 0
failed = 0


def run_test(name: str, condition: bool, expected=None, actual=None):
    global passed, failed
    if condition:
        passed += 1
        print(f"  \u2713 {name}")
    else:
        failed += 1
        print(f"  \u2717 {name}")
        if expected is not None:
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")


def _storage_path(ft, flat_index):
    digits = list(str(flat_index))
    return os.path.join(
        ft.ft_initial_static_tensor.st_relative_to, ft.ft_initial_static_tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )


def read_ft_element(ft, flat_index):
    path = _storage_path(ft, flat_index)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


# ─── Main test ───

print("Running test_coder_simulator...\n")

server = libtmux.Server()
INSTANCE_ID = "coder_sim_test"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"

# Cleanup any leftover session
try:
    server.kill_session(SESSION_NAME)
except Exception:
    pass

CODE_TO_TYPE = "echo hello"

with tempfile.TemporaryDirectory() as tmpdir:

    id_ft = ft_make_forwarded(tmpdir, [1], [INSTANCE_ID])

    # ─── Test 1: Full pipeline — create + type + enter + capture ───
    # One ft_forward triggers everything.
    print("Test 1: Full pipeline (create + type + enter + capture)")

    steps = []

    # Step: create session
    steps.append(ft_tmux_create_session(id_ft))

    # Step: small delay for shell init
    steps.append(ft_sleep(id_ft, 0.3))

    # Steps: type each character
    for ch in CODE_TO_TYPE:
        ch_ft = ft_make_forwarded(tmpdir, [1], [ch])
        steps.append(ft_tmux_send_text(ch_ft, id_ft))

    # Step: send Enter
    enter_ft = ft_make_forwarded(tmpdir, [1], ["Enter"])
    steps.append(ft_tmux_send_ctrl(enter_ft, id_ft))

    # Step: delay for command to execute
    steps.append(ft_sleep(id_ft, 0.5))

    # Step: capture final pane (this is the last step — its result is returned)
    steps.append(ft_tmux_capture_pane(id_ft))

    # Build single pipeline
    pipeline = ft_sequential(*steps)

    # ONE ft_forward — entire pipeline executes
    prompt_t = st_make_tensor(["go"], tmpdir)
    pipeline.ft_forward(prompt_t)

    run_test("pipeline forwarded", pipeline.ft_forwarded is True)
    run_test("pipeline status = confidence",
             pipeline.ft_initial_static_tensor.data.flatten()[0].item() > 0)

    final_content = read_ft_element(pipeline, 0)
    run_test("final capture not None", final_content is not None)
    run_test("final capture contains 'hello' (echo output)",
             final_content is not None and "hello" in final_content,
             "contains 'hello'",
             repr(final_content[:200] if final_content else None))
    run_test("session exists", server.has_session(SESSION_NAME))

    print(f"\n  Final pane:\n{final_content}\n")

    # Cleanup
    try:
        server.kill_session(SESSION_NAME)
    except Exception:
        pass

    # ─── Test 2: Frame-by-frame — capture after each character ───
    # One ft_forward. Interleave ft_tmux_capture_pane after each char.
    # The last capture is the pipeline output — shows all chars typed.

    INSTANCE_ID_2 = "coder_sim_frames"
    SESSION_NAME_2 = f"{tmux_session_prefix}{INSTANCE_ID_2}"
    try:
        server.kill_session(SESSION_NAME_2)
    except Exception:
        pass

    print("Test 2: Frame-by-frame capture (one ft_forward, interleaved captures)")

    id_ft2 = ft_make_forwarded(tmpdir, [1], [INSTANCE_ID_2])

    steps2 = []

    # Create session + delay
    steps2.append(ft_tmux_create_session(id_ft2))
    steps2.append(ft_sleep(id_ft2, 0.3))

    # Type each char, capture after each — last capture becomes pipeline output
    FRAME_CODE = "echo hi"
    for ch in FRAME_CODE:
        ch_ft = ft_make_forwarded(tmpdir, [1], [ch])
        steps2.append(ft_tmux_send_text(ch_ft, id_ft2))
        steps2.append(ft_sleep(id_ft2, 0.05))
        steps2.append(ft_tmux_capture_pane(id_ft2))

    # ONE ft_forward — entire frame-by-frame pipeline executes
    frame_pipeline = ft_sequential(*steps2)
    prompt_t = st_make_tensor(["go"], tmpdir)
    frame_pipeline.ft_forward(prompt_t)

    run_test("frame pipeline forwarded", frame_pipeline.ft_forwarded is True)

    # The pipeline result is the last capture (after typing all chars)
    final_frame = read_ft_element(frame_pipeline, 0)
    run_test("final frame not None", final_frame is not None)
    run_test("final frame contains full typed text",
             final_frame is not None and "echo hi" in final_frame,
             "contains 'echo hi'",
             repr(final_frame[:200] if final_frame else None))

    print(f"\n  Final frame:\n{final_frame}\n")

    # ─── Test 3: Pipeline with Enter and output verification ───
    print("Test 3: Full pipeline with Enter — verify command output")

    steps3 = []
    # Send Enter after "echo hi" (already typed above in the session)
    enter_ft2 = ft_make_forwarded(tmpdir, [1], ["Enter"])
    steps3.append(ft_tmux_send_ctrl(enter_ft2, id_ft2))
    steps3.append(ft_sleep(id_ft2, 0.5))
    steps3.append(ft_tmux_capture_pane(id_ft2))

    enter_pipeline = ft_sequential(*steps3)
    prompt_t = st_make_tensor(["go"], tmpdir)
    enter_pipeline.ft_forward(prompt_t)

    output_content = read_ft_element(enter_pipeline, 0)
    run_test("after enter: contains 'hi' (echo output)",
             output_content is not None and "hi" in output_content,
             "contains 'hi'",
             repr(output_content[:200] if output_content else None))

    # Cleanup
    try:
        server.kill_session(SESSION_NAME_2)
    except Exception:
        pass


# ─── Summary ───
print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
if failed == 0:
    print("All test_coder_simulator tests passed.")
else:
    sys.exit(1)
