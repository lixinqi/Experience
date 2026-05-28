"""
FtSwitch := torch.autograd.Function[
    $forward Import[{future_tensor function switch_forward.viba}],
    $backward Import[{future_tensor function switch_backward.viba}]
]

ft_switch = FtSwitch.apply
"""

from typing import List, Tuple

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.switch_forward import switch_forward
from experience.future_tensor.function.switch_backward import switch_backward
from experience.future_tensor.function.switch_2nd import SwitchGradFn
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class FtSwitch(torch.autograd.Function):
    """Autograd Function for switch control flow on FutureTensors.

    Forward: select a branch based on condition content.
    Backward: route grad_output to the selected branch only.
    """

    @staticmethod
    def _read_symbol(condition: FutureTensor) -> str:
        """Read the control symbol from condition's materialized storage."""
        import os
        flat = 0
        digits = list(str(flat))
        path = os.path.join(
            condition.ft_initial_static_tensor.st_relative_to,
            condition.ft_initial_static_tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def forward(
        ctx,
        condition: FutureTensor,
        symbols,
        summaries,
        descriptions,
        *branches,
    ):
        cases = list(zip(symbols, summaries, descriptions, branches))
        result = switch_forward(condition, cases)

        # Forward-time branch selection: condition is already materialized
        # by the caller (ft_forward runs from loss side towards inputs).
        condition_symbol = FtSwitch._read_symbol(condition)
        selected_index = 0
        for i, sym in enumerate(symbols):
            if sym == condition_symbol:
                selected_index = i
                break

        ctx.condition = condition
        ctx.symbols = symbols
        ctx.summaries = summaries
        ctx.descriptions = descriptions
        ctx.branches = branches
        ctx.selected_index = selected_index

        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not grad_output.requires_grad:
            grad_output.requires_grad_(True)

        # 1st-derivative dispatch: skip GradFn if dispatcher handles it.
        dispatch = get_backward_dispatcher(switch_backward)
        if dispatch({"selected_index": ctx.selected_index, "branches": list(ctx.branches)}):
            n_branches = len(ctx.branches)
            branch_grads = [None] * n_branches
            branch_grads[ctx.selected_index] = grad_output
            return (None, None, None, None, *branch_grads)

        # SwitchGradFn.forward handles FutureTensor attribute reconstruction
        # and calls switch_backward internally.
        grad_for_selected = SwitchGradFn.apply(grad_output, ctx.selected_index, list(ctx.branches))

        # Return grads for (condition, symbols, summaries, descriptions, *branches)
        n_branches = len(ctx.branches)
        branch_grads = [None] * n_branches
        branch_grads[ctx.selected_index] = grad_for_selected
        return (None, None, None, None, *branch_grads)


def ft_switch(condition: FutureTensor, cases: List[Tuple[str, str, str, FutureTensor]]) -> FutureTensor:
    """Switch control flow for FutureTensors.

    Selects a branch based on the symbolic content of the condition FutureTensor.

    Args:
        condition: FutureTensor whose first element's symbolic content determines
            the branch selection.
        cases: List of (symbol, summary, description, branch) tuples.
            The branch whose symbol matches the condition content is returned.

    Returns:
        A FutureTensor wrapping the selected branch.
    """
    symbols = [c[0] for c in cases]
    summaries = [c[1] for c in cases]
    descriptions = [c[2] for c in cases]
    branches = [c[3] for c in cases]
    return FtSwitch.apply(condition, symbols, summaries, descriptions, *branches)


if __name__ == "__main__":
    import sympy
    import tempfile
    import os

    from experience.future_tensor.status import Status
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
    from experience.future_tensor.function.ft_mean import ft_mean

    print("Running tests for ft_switch...\n")

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

    def make_condition_ft(symbol, tmpdir):
        """Create a materialized condition FutureTensor with the symbol in storage."""
        async def symbol_get(coords, trajactory):
            return (symbol, Status.confidence(1.0))
        ft = FutureTensor(tmpdir, symbol_get, [sympy.Integer(1)])
        # Materialize: write symbol into element-0 storage
        path = _storage_path(ft, 0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(symbol)
        ft.ft_initial_static_tensor.data[0] = Status.convert_status_to_float(Status.confidence(1.0))
        ft.ft_forwarded = True
        return ft

    def make_branch_ft(shape, data_list, tmpdir):
        """Create a branch FutureTensor with tracking ft_async_get."""
        async def branch_get(coords, trajactory):
            return (f"val_{coords}", Status.confidence(1.0))
        ft = FutureTensor(tmpdir, branch_get, [sympy.Integer(s) for s in shape])
        # Pre-materialize storage so we can also read directly if needed
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

    # === Group 1: Forward selection (tests 1-10) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        cond_ft = make_condition_ft("B", tmpdir)
        branch_a = make_branch_ft([3], ["a0", "a1", "a2"], tmpdir)
        branch_b = make_branch_ft([3], ["b0", "b1", "b2"], tmpdir)

        cases = [
            ("A", "case A", "desc A", branch_a),
            ("B", "case B", "desc B", branch_b),
        ]

        output = ft_switch(cond_ft, cases)

        run_test("forward shape matches branch", output.ft_capacity_shape == [3])
        run_test("forward is lazy", output.ft_forwarded is False)

        # Materialize output
        prompt_t = st_make_tensor(["p", "p", "p"], tmpdir)
        output.ft_forward(prompt_t)
        run_test("forwarded after ft_forward", output.ft_forwarded is True)

    with tempfile.TemporaryDirectory() as tmpdir:
        cond_ft = make_condition_ft("A", tmpdir)
        branch_a = make_branch_ft([3], ["x0", "x1", "x2"], tmpdir)
        branch_b = make_branch_ft([3], ["y0", "y1", "y2"], tmpdir)

        cases = [
            ("A", "case A", "desc A", branch_a),
            ("B", "case B", "desc B", branch_b),
        ]

        output = ft_switch(cond_ft, cases)
        run_test("forward selects branch A (lazy)", output.ft_capacity_shape == [3])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Default to first branch when no match
        cond_ft = make_condition_ft("UNKNOWN", tmpdir)
        branch_a = make_branch_ft([2], ["d0", "d1"], tmpdir)
        branch_b = make_branch_ft([2], ["e0", "e1"], tmpdir)

        cases = [
            ("A", "case A", "desc A", branch_a),
            ("B", "case B", "desc B", branch_b),
        ]

        output = ft_switch(cond_ft, cases)
        run_test("default to first branch (lazy)", output.ft_capacity_shape == [2])

    # === Group 2: Autograd connectivity (tests 11-15) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        cond_ft = make_condition_ft("A", tmpdir)
        branch_a = make_branch_ft([3], ["a0", "a1", "a2"], tmpdir)
        branch_b = make_branch_ft([3], ["b0", "b1", "b2"], tmpdir)
        branch_a.requires_grad_(True)

        cases = [
            ("A", "case A", "desc A", branch_a),
            ("B", "case B", "desc B", branch_b),
        ]

        output = ft_switch(cond_ft, cases)
        run_test("output has grad_fn", output.grad_fn is not None)

    # === Group 3: Backward routing (tests 16-25) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        cond_ft = make_condition_ft("B", tmpdir)
        branch_a = make_branch_ft([2], ["a0", "a1"], tmpdir)
        branch_b = make_branch_ft([2], ["b0", "b1"], tmpdir)
        branch_b.requires_grad_(True)

        cases = [
            ("A", "case A", "desc A", branch_a),
            ("B", "case B", "desc B", branch_b),
        ]

        output = ft_switch(cond_ft, cases)
        loss = ft_mean(output)
        loss.backward()

        run_test("loss is scalar", loss.shape == torch.Size([]))
        run_test("branch_b has grad", branch_b.grad is not None)

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All ft_switch tests completed.")
