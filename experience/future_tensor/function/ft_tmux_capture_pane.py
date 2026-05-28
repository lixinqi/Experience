"""
FtTmuxCapturePane := torch.autograd.Function[
    $forward  Import[{future_tensor function tmux_capture_pane_forward.viba}],
    $backward Import[{future_tensor function tmux_capture_pane_backward.viba}]
]

ft_tmux_capture_pane = FtTmuxCapturePane.apply
"""

from typing import List

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.tmux_capture_pane_forward import tmux_capture_pane_forward
from experience.future_tensor.function.tmux_capture_pane_backward import tmux_capture_pane_backward


class FtTmuxCapturePane(torch.autograd.Function):
    """Autograd Function for capturing tmux pane content per FutureTensor element."""

    @staticmethod
    def forward(ctx, input_ft: FutureTensor):
        ctx.input_ft = input_ft
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_initial_static_tensor.st_relative_to
        return tmux_capture_pane_forward(input_ft)

    @staticmethod
    def backward(ctx, grad_output):
        return tmux_capture_pane_backward(ctx, grad_output)


def ft_tmux_capture_pane(input_ft: FutureTensor) -> FutureTensor:
    """Capture the tmux pane content for each element of ``input_ft``.

    Each element's symbolic content is used as the ``instance_id`` to look up
    the tmux session named ``f"{tmux_session_prefix}{instance_id}"``.
    The active pane's content is captured and stored as the output element.

    Args:
        input_ft: FutureTensor whose elements contain instance IDs.

    Returns:
        A FutureTensor with the same shape as ``input_ft``, where each element
        contains the captured pane text.
    """
    return FtTmuxCapturePane.apply(input_ft)


if __name__ == "__main__":
    import os
    import sys
    import tempfile

    import sympy

    from experience.future_tensor.status import Status
    from experience.future_tensor.function.tmux_session import tmux_session_prefix
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running tests for ft_tmux_capture_pane...\n")

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

    def make_forwarded_ft(shape, data_list, tmpdir):
        async def dummy_get(coords, trajactory):
            return ("unused", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(s) for s in shape])
        nested = _unflatten_data(data_list, shape)
        result_tensor = st_make_tensor(nested, tmpdir)
        assign_tensor(ft.ft_initial_static_tensor, result_tensor)
        ft.ft_forwarded = True
        return ft

    def _unflatten_data(flat_list, shape):
        if not shape:
            return flat_list[0] if flat_list else None
        if len(shape) == 1:
            return flat_list
        chunk_size = 1
        for s in shape[1:]:
            chunk_size *= s
        return [
            _unflatten_data(flat_list[i * chunk_size : (i + 1) * chunk_size], shape[1:])
            for i in range(shape[0])
        ]

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

    import libtmux
    server = libtmux.Server()

    # === Group 1: Forward shape and laziness ===
    print("Group 1: Forward shape and laziness")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3], ["cap0", "cap1", "cap2"], tmpdir)
        output = ft_tmux_capture_pane(ft)
        run_test("output shape matches input", output.ft_capacity_shape == [3])
        run_test("output is lazy", output.ft_forwarded is False)

    # === Group 2: Capture from real tmux session ===
    print("\nGroup 2: Capture from real tmux session")

    test_sessions = []

    with tempfile.TemporaryDirectory() as tmpdir:
        instance_id = "capture_test_01"
        session_name = f"{tmux_session_prefix}{instance_id}"

        # Create a tmux session and send some text
        try:
            server.kill_session(session_name)
        except Exception:
            pass
        sess = server.new_session(session_name=session_name, attach=False)
        test_sessions.append(session_name)
        pane = sess.active_window.active_pane
        pane.send_keys("echo hello_capture", enter=True)

        import time
        time.sleep(0.5)  # Wait for echo to complete

        ft = make_forwarded_ft([1], [instance_id], tmpdir)
        output = ft_tmux_capture_pane(ft)

        prompt_t = st_make_tensor(["p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("output forwarded", output.ft_forwarded is True)
        content = read_ft_element(output, 0)
        run_test("captured content is not None", content is not None)
        run_test("captured content contains sent text",
                 content is not None and "hello_capture" in content,
                 "contains 'hello_capture'", repr(content[:100] if content else None))
        run_test("confidence > 0",
                 output.ft_initial_static_tensor.data.flatten()[0].item() > 0)

    # === Group 3: Capture from non-existent session ===
    print("\nGroup 3: Non-existent session")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1], ["nonexistent_session_xyz"], tmpdir)
        output = ft_tmux_capture_pane(ft)

        prompt_t = st_make_tensor(["p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("nonexistent session: forwarded", output.ft_forwarded is True)
        run_test("nonexistent session: status < 0 (failed)",
                 output.ft_initial_static_tensor.data.flatten()[0].item() < 0)

    # === Group 4: Autograd connectivity ===
    print("\nGroup 4: Autograd connectivity")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["grad_a", "grad_b"], tmpdir)
        ft.requires_grad_(True)
        output = ft_tmux_capture_pane(ft)
        run_test("output has grad_fn", output.grad_fn is not None)
        run_test("output requires_grad", output.requires_grad is True)

    # === Group 5: Backward produces grad ===
    print("\nGroup 5: Backward produces grad")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["bw0", "bw1"], tmpdir)
        ft.requires_grad_(True)
        output = ft_tmux_capture_pane(ft)
        loss = output.sum()
        loss.backward()
        run_test("input has grad", ft.grad is not None)

    # === Group 6: 2nd derivative support ===
    print("\nGroup 6: 2nd derivative support")

    from experience.future_tensor.backward_dispatch import (
        need_reflection,
        dispatch_policy,
        TracePolicy,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["2nd_a", "2nd_b"], tmpdir)
        ft.requires_grad_(True)

        backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
        anchored = need_reflection(ft, backward_dispatch_start)
        output = ft_tmux_capture_pane(anchored)
        loss = output.sum()
        loss.backward(create_graph=True)

        run_test("2nd: grad exists", backward_dispatch_start.grad is not None)
        run_test("2nd: grad has grad_fn", backward_dispatch_start.grad.grad_fn is not None)

        records = []
        with dispatch_policy(TracePolicy(records)):
            backward_dispatch_start.grad.backward()

        run_test("2nd: TracePolicy collected records", len(records) >= 1)
        if records:
            run_test("2nd: record fn is tmux_capture_pane_backward",
                     records[0].fn is tmux_capture_pane_backward)
            run_test("2nd: record has grad_output",
                     "grad_output" in records[0].inputs)
            run_test("2nd: record has grad_input",
                     "grad_input" in records[0].inputs)
        else:
            for name in ["fn", "grad_output", "grad_input"]:
                run_test(f"2nd: record has {name}", False)

    # === Group 7: Multiple elements capture ===
    print("\nGroup 7: Multiple elements capture")

    with tempfile.TemporaryDirectory() as tmpdir:
        ids = ["multi_cap_a", "multi_cap_b"]
        for inst_id in ids:
            sname = f"{tmux_session_prefix}{inst_id}"
            try:
                server.kill_session(sname)
            except Exception:
                pass
            sess = server.new_session(session_name=sname, attach=False)
            test_sessions.append(sname)
            pane = sess.active_window.active_pane
            pane.send_keys(f"echo marker_{inst_id}", enter=True)

        time.sleep(0.5)

        ft = make_forwarded_ft([2], ids, tmpdir)
        output = ft_tmux_capture_pane(ft)

        prompt_t = st_make_tensor(["p", "p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("multi: forwarded", output.ft_forwarded is True)
        c0 = read_ft_element(output, 0)
        c1 = read_ft_element(output, 1)
        run_test("multi: elem 0 contains marker",
                 c0 is not None and "marker_multi_cap_a" in c0,
                 "contains 'marker_multi_cap_a'", repr(c0[:80] if c0 else None))
        run_test("multi: elem 1 contains marker",
                 c1 is not None and "marker_multi_cap_b" in c1,
                 "contains 'marker_multi_cap_b'", repr(c1[:80] if c1 else None))

    # Cleanup
    for name in test_sessions:
        try:
            server.kill_session(name)
        except Exception:
            pass

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All ft_tmux_capture_pane tests passed.")
    else:
        sys.exit(1)
