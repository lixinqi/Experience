"""
FtTmuxCreateSession := torch.autograd.Function[
    $forward  Import[{future_tensor function tmux_create_session_forward.viba}],
    $backward Import[{future_tensor function tmux_create_session_backward.viba}]
]

ft_tmux_create_session = FtTmuxCreateSession.apply
"""

from typing import List

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.tmux_create_session_forward import tmux_create_session_forward
from experience.future_tensor.function.tmux_create_session_backward import tmux_create_session_backward


class FtTmuxCreateSession(torch.autograd.Function):
    """Autograd Function for creating tmux sessions per FutureTensor element."""

    @staticmethod
    def forward(ctx, input_ft: FutureTensor):
        ctx.input_ft = input_ft
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_initial_static_tensor.st_relative_to
        return tmux_create_session_forward(input_ft)

    @staticmethod
    def backward(ctx, grad_output):
        return tmux_create_session_backward(ctx, grad_output)


def ft_tmux_create_session(input_ft: FutureTensor) -> FutureTensor:
    """Create a tmux session for each element of ``input_ft``.

    Each element's symbolic content is used as the ``instance_id``.
    The session name is ``f"{tmux_session_prefix}{instance_id}"``.

    Args:
        input_ft: FutureTensor whose elements contain instance IDs.

    Returns:
        A FutureTensor with the same shape as ``input_ft``.
    """
    return FtTmuxCreateSession.apply(input_ft)


if __name__ == "__main__":
    import os
    import sys
    import tempfile

    import sympy

    from experience.future_tensor.status import Status
    from experience.future_tensor.function.tmux_session import tmux_session_prefix
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running tests for ft_tmux_create_session...\n")

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

    # === Group 1: Forward shape and laziness ===
    print("Group 1: Forward shape and laziness")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3], ["id0", "id1", "id2"], tmpdir)
        output = ft_tmux_create_session(ft)
        run_test("output shape matches input", output.ft_capacity_shape == [3])
        run_test("output is lazy", output.ft_forwarded is False)

    # === Group 2: Session creation via ft_forward ===
    print("\nGroup 2: Session creation via ft_forward")

    import libtmux
    server = libtmux.Server()
    test_sessions = []

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["sess_a", "sess_b"], tmpdir)
        output = ft_tmux_create_session(ft)

        prompt_t = st_make_tensor(["p", "p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("output forwarded", output.ft_forwarded is True)

        expected_names = [
            f"{tmux_session_prefix}sess_a",
            f"{tmux_session_prefix}sess_b",
        ]
        for name in expected_names:
            run_test(f"session exists: {name}", server.has_session(name))
            test_sessions.append(name)

        # Idempotency: run again, should still succeed
        output2 = ft_tmux_create_session(ft)
        output2.ft_forward(prompt_t)
        for name in expected_names:
            run_test(f"idempotent: {name} still exists", server.has_session(name))

    # Cleanup test sessions
    for name in test_sessions:
        try:
            server.kill_session(name)
        except Exception:
            pass

    # === Group 3: Autograd connectivity ===
    print("\nGroup 3: Autograd connectivity")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["grad0", "grad1"], tmpdir)
        ft.requires_grad_(True)
        output = ft_tmux_create_session(ft)
        run_test("output has grad_fn", output.grad_fn is not None)
        run_test("output requires_grad", output.requires_grad is True)

    # === Group 4: Lazy async_get reads instance_id ===
    print("\nGroup 4: Lazy async_get")

    with tempfile.TemporaryDirectory() as tmpdir:
        async def lazy_get(coords, trajactory):
            return (f"lazy_{coords[0]}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, lazy_get, [sympy.Integer(2)])
        output = ft_tmux_create_session(ft)

        prompt_t = st_make_tensor(["p", "p"], tmpdir)
        output.ft_forward(prompt_t)

        expected_names = [
            f"{tmux_session_prefix}lazy_0",
            f"{tmux_session_prefix}lazy_1",
        ]
        for name in expected_names:
            run_test(f"lazy session exists: {name}", server.has_session(name))
            try:
                server.kill_session(name)
            except Exception:
                pass

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All ft_tmux_create_session tests passed.")
    else:
        sys.exit(1)
