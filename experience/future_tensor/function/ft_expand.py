"""
FtExpand := torch.autograd.Function[
    $forward Import[{future_tensor function expand_forward.viba}],
    $backward Import[{future_tensor function expand_backward.viba}]
]

ft_expand = FtExpand.apply
"""

from typing import List, Union

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.expand_forward import expand_forward
from experience.future_tensor.function.expand_2nd import ExpandGradFn
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher
from experience.future_tensor.function.expand_backward import expand_backward


class FtExpand(torch.autograd.Function):
    """Autograd Function for expanding (broadcasting) FutureTensors.

    Forward: expand_forward (broadcast size-1 dims to target size).
    Backward: expand_backward (reduce/sum along expanded dims).
    """

    @staticmethod
    def forward(ctx, input: FutureTensor, target_shape: List[Union[int, sympy.Basic]]):
        result = expand_forward(input, target_shape)

        ctx.input_shape = input.ft_capacity_shape
        ctx.expanded_shape = result.ft_capacity_shape

        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not grad_output.requires_grad:
            grad_output.requires_grad_(True)

        # 1st-derivative dispatch: skip GradFn if dispatcher handles it.
        dispatch = get_backward_dispatcher(expand_backward)
        if dispatch({"input_shape": ctx.input_shape, "expanded_shape": ctx.expanded_shape}):
            return grad_output, None

        # ExpandGradFn.forward handles FutureTensor attribute reconstruction
        # and calls expand_backward internally.
        grad_input = ExpandGradFn.apply(
            grad_output, ctx.input_shape, ctx.expanded_shape
        )

        # Return grads for (input, target_shape)
        return grad_input, None


def ft_expand(input: FutureTensor, target_shape: List[Union[int, sympy.Basic]]) -> FutureTensor:
    """Expand (broadcast) a FutureTensor with autograd support.

    Same behavior as torch.Tensor.expand() -- broadcasts dimensions of size 1
    to the corresponding size in target_shape.

    When target_shape contains sympy.Symbol, the output has capacity 0 on that
    axis (symbolic/dynamic dim).

    Args:
        input: Source FutureTensor.
        target_shape: Desired output shape. Use -1 to keep a dim unchanged.
            May contain sympy.Symbol for symbolic (dynamic) dimensions.

    Returns:
        A new FutureTensor with the expanded shape.
    """
    return FtExpand.apply(input, target_shape)


if __name__ == "__main__":
    import os
    import sys
    import tempfile

    import sympy

    from experience.future_tensor.status import Status
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
    from experience.future_tensor.function.expand_forward import _storage_path

    print("Running tests for ft_expand...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  \u2717 {name}")
            if expected is not None:
                print(f"    expected: {expected}, actual: {actual}")

    def read_ft_element(ft, flat_index):
        path = _storage_path(ft, flat_index)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    def make_forwarded_ft(shape, data_list, tmpdir):
        async def dummy_get(coords, trajactory):
            return ("unused", Status.confidence(1.0))
        ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(s) for s in shape])
        nested = _unflatten_data(data_list, shape)
        result_tensor = st_make_tensor(nested, tmpdir)
        assign_tensor(ft.ft_static_tensor, result_tensor)
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
            _unflatten_data(flat_list[i * chunk_size:(i + 1) * chunk_size], shape[1:])
            for i in range(shape[0])
        ]

    # === Group 1: Forward shape validation ===
    print("Group 1: Forward shape validation")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1, 3], ["a", "b", "c"], tmpdir)
        output = ft_expand(ft, [4, 3])
        run_test("expand [1,3]->[4,3] shape", output.ft_capacity_shape == [4, 3])
        run_test("expand forwarded (input was forwarded)", output.ft_forwarded is True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1, 3], ["a", "b", "c"], tmpdir)
        try:
            ft_expand(ft, [4, 5])
            run_test("non-1 dim mismatch raises", False)
        except ValueError:
            run_test("non-1 dim mismatch raises", True)

    # === Group 2: Autograd connectivity ===
    print("\nGroup 2: Autograd connectivity")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1, 3], ["a", "b", "c"], tmpdir)
        ft.requires_grad_(True)
        output = ft_expand(ft, [4, 3])
        run_test("output has grad_fn", output.grad_fn is not None)
        run_test("output requires_grad", output.requires_grad is True)

    # === Group 3: Backward produces grad ===
    print("\nGroup 3: Backward produces grad")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1, 3], ["a", "b", "c"], tmpdir)
        ft.requires_grad_(True)
        output = ft_expand(ft, [4, 3])
        loss = output.sum()
        loss.backward()
        run_test("input has grad", ft.grad is not None)
        run_test("grad shape matches input", list(ft.grad.shape) == [])  # scalar

    # === Group 4: Multiple inputs backward ===
    print("\nGroup 4: Multiple expand shapes")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3, 1], ["x", "y", "z"], tmpdir)
        ft.requires_grad_(True)
        output = ft_expand(ft, [3, 5])
        run_test("expand [3,1]->[3,5] shape", output.ft_capacity_shape == [3, 5])
        loss = output.sum()
        loss.backward()
        run_test("grad exists for [3,1]->[3,5]", ft.grad is not None)

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["p", "q"], tmpdir)
        ft.requires_grad_(True)
        output = ft_expand(ft, [3, 2])
        run_test("expand [2]->[3,2] shape", output.ft_capacity_shape == [3, 2])
        loss = output.sum()
        loss.backward()
        run_test("grad exists for [2]->[3,2]", ft.grad is not None)

    # === Group 5: 2nd derivative support ===
    print("\nGroup 5: 2nd derivative support")

    from experience.future_tensor.backward_dispatch import (
        need_reflection,
        dispatch_policy,
        TracePolicy,
    )
    from experience.future_tensor.function.expand_backward import expand_backward

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1, 3], ["a", "b", "c"], tmpdir)
        ft.requires_grad_(True)

        backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
        anchored = need_reflection(ft, backward_dispatch_start)
        output = ft_expand(anchored, [4, 3])
        loss = output.sum()
        loss.backward(create_graph=True)

        run_test("2nd: grad exists", backward_dispatch_start.grad is not None)
        run_test("2nd: grad has grad_fn", backward_dispatch_start.grad.grad_fn is not None)

        records = []
        with dispatch_policy(TracePolicy(records)):
            backward_dispatch_start.grad.backward()

        run_test("2nd: TracePolicy collected records", len(records) >= 1)
        if records:
            run_test("2nd: record fn is expand_backward",
                     records[0].fn is expand_backward)
            run_test("2nd: record has grad_output",
                     "grad_output" in records[0].inputs)
            run_test("2nd: record has input_shape",
                     "input_shape" in records[0].inputs)
            run_test("2nd: record has expanded_shape",
                     "expanded_shape" in records[0].inputs)
            run_test("2nd: record has grad_input",
                     "grad_input" in records[0].inputs)
        else:
            for name in ["fn", "grad_output", "input_shape", "expanded_shape", "grad_input"]:
                run_test(f"2nd: record has {name}", False)

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All ft_expand tests passed.")
    else:
        sys.exit(1)
