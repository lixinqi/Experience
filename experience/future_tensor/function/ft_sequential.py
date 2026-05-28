"""
FtSequential := torch.autograd.Function[
    $forward Import[{future_tensor function sequential_forward.viba}],
    $backward Import[{future_tensor function sequential_backward.viba}]
]

ft_sequential = FtSequential.apply
"""

from typing import List, Tuple

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.sequential_forward import sequential_forward
from experience.future_tensor.function.sequential_backward import sequential_backward
from experience.future_tensor.function.sequential_2nd import SequentialGradFn
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class FtSequential(torch.autograd.Function):
    """Autograd Function for sequential evaluation of FutureTensors.

    Forward: evaluate inputs in sequence, return the last input's result.
    Backward: route grad_output to all inputs.
    """

    @staticmethod
    def forward(ctx, *inputs):
        if not inputs:
            raise ValueError("ft_sequential: inputs must not be empty")

        result = sequential_forward(inputs)

        ctx.inputs = inputs
        ctx.num_inputs = len(inputs)

        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not grad_output.requires_grad:
            grad_output.requires_grad_(True)

        # 1st-derivative dispatch: skip GradFn if dispatcher handles it.
        dispatch = get_backward_dispatcher(sequential_backward)
        if dispatch({"num_inputs": ctx.num_inputs}):
            return tuple(grad_output for _ in range(ctx.num_inputs))

        # SequentialGradFn.forward handles FutureTensor attribute reconstruction
        # and calls sequential_backward internally.
        first_input = ctx.inputs[0]
        grad_for_all = SequentialGradFn.apply(grad_output, ctx.num_inputs, first_input)

        # Return grads for all inputs
        return tuple(grad_for_all for _ in range(ctx.num_inputs))


def ft_sequential(*inputs: FutureTensor) -> FutureTensor:
    """Sequential evaluation of FutureTensors.

    Evaluates each input's ``ft_async_get`` in sequence at pull time.
    If any input returns a non-confidence status, that result is returned early.
    Otherwise the last input's result is returned.

    Args:
        *inputs: One or more FutureTensors. All must have the same shape.

    Returns:
        A FutureTensor with the same shape as the inputs.
    """
    return FtSequential.apply(*inputs)


if __name__ == "__main__":
    import os
    import sys
    import tempfile

    import sympy
    import torch

    from experience.future_tensor.status import Status
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running tests for ft_sequential...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ {name}")
            if expected is not None:
                print(f"    expected: {expected}, actual: {actual}")

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
            _unflatten_data(flat_list[i * chunk_size:(i + 1) * chunk_size], shape[1:])
            for i in range(shape[0])
        ]

    # === Group 1: Forward validation (tests 1-5) ===

    print("Group 1: Forward validation")

    try:
        ft_sequential()
        run_test("empty inputs raises ValueError", False)
    except ValueError:
        run_test("empty inputs raises ValueError", True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ft_a = make_forwarded_ft([2], ["a0", "a1"], tmpdir)
        ft_b = make_forwarded_ft([3], ["b0", "b1", "b2"], tmpdir)
        try:
            ft_sequential(ft_a, ft_b)
            run_test("mismatched shapes raise ValueError", False)
        except ValueError:
            run_test("mismatched shapes raise ValueError", True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ft_a = make_forwarded_ft([2], ["a0", "a1"], tmpdir)
        output = ft_sequential(ft_a)
        run_test("single input returns same shape", output.ft_capacity_shape == [2])
        run_test("single input is lazy", output.ft_forwarded is False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ft_a = make_forwarded_ft([3], ["x0", "x1", "x2"], tmpdir)
        ft_b = make_forwarded_ft([3], ["y0", "y1", "y2"], tmpdir)
        output = ft_sequential(ft_a, ft_b)
        run_test("two inputs return same shape", output.ft_capacity_shape == [3])
        run_test("two inputs is lazy", output.ft_forwarded is False)

    # === Group 2: Sequential async_get (tests 6-15) ===

    print("\nGroup 2: Sequential async_get")

    with tempfile.TemporaryDirectory() as tmpdir:
        call_order = []

        async def get_a(coords, prompt):
            call_order.append("a")
            return (f"a{coords}", Status.confidence(0.9))

        async def get_b(coords, prompt):
            call_order.append("b")
            return (f"b{coords}", Status.confidence(0.8))

        async def get_c(coords, prompt):
            call_order.append("c")
            return (f"c{coords}", Status.confidence(0.7))

        ft_a = FutureTensor(tmpdir, get_a, [sympy.Integer(2)])
        ft_b = FutureTensor(tmpdir, get_b, [sympy.Integer(2)])
        ft_c = FutureTensor(tmpdir, get_c, [sympy.Integer(2)])

        output = ft_sequential(ft_a, ft_b, ft_c)
        run_test("lazy before ft_forward", output.ft_forwarded is False)

        prompt_t = st_make_tensor(["p", "p"], tmpdir)
        output.ft_forward(prompt_t)
        run_test("forwarded after ft_forward", output.ft_forwarded is True)
        run_test("all getters called in order",
                 call_order == ["a", "b", "c", "a", "b", "c"])
        run_test("element 0 is from last input", read_ft_element(output, 0) == "c[0]")
        run_test("element 1 is from last input", read_ft_element(output, 1) == "c[1]")

    # === Group 3: Early return on error (tests 16-20) ===

    print("\nGroup 3: Early return on error")

    with tempfile.TemporaryDirectory() as tmpdir:
        call_order = []

        async def ok_get(coords, trajactory):
            call_order.append("ok")
            return (f"ok{coords}", Status.confidence(1.0))

        async def fail_get(coords, trajactory):
            call_order.append("fail")
            return ("", Status.self_confidence_but_failed(0.5))

        async def never_called_get(coords, trajactory):
            call_order.append("never")
            return ("never", Status.confidence(1.0))

        ft_ok = FutureTensor(tmpdir, ok_get, [sympy.Integer(2)])
        ft_fail = FutureTensor(tmpdir, fail_get, [sympy.Integer(2)])
        ft_never = FutureTensor(tmpdir, never_called_get, [sympy.Integer(2)])

        output = ft_sequential(ft_ok, ft_fail, ft_never)
        prompt_t = st_make_tensor(["p", "p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("early return: ok called", "ok" in call_order)
        run_test("early return: fail called", "fail" in call_order)
        run_test("early return: never NOT called", "never" not in call_order)
        # The error status should propagate
        run_test("error status propagated",
                 output.ft_initial_static_tensor.data[0].item() < 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        call_order = []

        async def overflow_get(coords, trajactory):
            call_order.append("overflow")
            return ("", Status.kContextOverflow)

        ft_overflow = FutureTensor(tmpdir, overflow_get, [sympy.Integer(1)])
        ft_after = FutureTensor(tmpdir, lambda c, p: (call_order.append("after") or ("after", Status.confidence(1.0))), [sympy.Integer(1)])

        output = ft_sequential(ft_overflow, ft_after)
        prompt_t = st_make_tensor(["p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("overflow early return: overflow called", "overflow" in call_order)
        run_test("overflow early return: after NOT called", "after" not in call_order)
        run_test("overflow status propagated",
                 output.ft_initial_static_tensor.data[0].item() == -3.0)

    # === Group 4: Autograd connectivity (tests 21-25) ===

    print("\nGroup 4: Autograd connectivity")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft_a = make_forwarded_ft([2], ["a0", "a1"], tmpdir)
        ft_b = make_forwarded_ft([2], ["b0", "b1"], tmpdir)
        ft_a.requires_grad_(True)
        ft_b.requires_grad_(True)

        output = ft_sequential(ft_a, ft_b)
        run_test("output has grad_fn", output.grad_fn is not None)
        run_test("output requires_grad", output.requires_grad is True)

    # === Group 5: Backward routing (tests 26-35) ===

    print("\nGroup 5: Backward routing")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft_a = make_forwarded_ft([2], ["a0", "a1"], tmpdir)
        ft_b = make_forwarded_ft([2], ["b0", "b1"], tmpdir)
        ft_a.requires_grad_(True)
        ft_b.requires_grad_(True)

        output = ft_sequential(ft_a, ft_b)
        loss = output.sum()
        loss.backward()

        run_test("input_a has grad", ft_a.grad is not None)
        run_test("input_b has grad", ft_b.grad is not None)

    with tempfile.TemporaryDirectory() as tmpdir:
        ft_a = make_forwarded_ft([3], ["a0", "a1", "a2"], tmpdir)
        ft_b = make_forwarded_ft([3], ["b0", "b1", "b2"], tmpdir)
        ft_c = make_forwarded_ft([3], ["c0", "c1", "c2"], tmpdir)
        ft_a.requires_grad_(True)
        ft_b.requires_grad_(True)
        ft_c.requires_grad_(True)

        output = ft_sequential(ft_a, ft_b, ft_c)
        loss = output.sum()
        loss.backward()

        run_test("three inputs: a has grad", ft_a.grad is not None)
        run_test("three inputs: b has grad", ft_b.grad is not None)
        run_test("three inputs: c has grad", ft_c.grad is not None)

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All ft_sequential tests passed.")
    else:
        sys.exit(1)
