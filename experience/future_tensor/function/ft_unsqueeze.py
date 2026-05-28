"""
FtUnsequeeze := torch.autograd.Function[
    $forward Import[{future_tensor function unsqueeze_forward.viba}],
    $backward Import[{future_tensor function slice_forward.viba}]
]

ft_unsqueeze = FtUnsequeeze.apply

# the same behavior with torch unsqueeze
"""

from typing import List

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor, _tensor_to_future
from experience.future_tensor.function.unsqueeze_forward import unsqueeze_forward
from experience.future_tensor.function.slice_forward import slice_forward
from experience.future_tensor.function.unsqueeze_2nd import UnsqueezeGradFn
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class FtUnsqueeze(torch.autograd.Function):
    """Autograd Function for unsqueezing FutureTensors.

    Forward: unsqueeze_forward (insert dim of size 1)
    Backward: slice_forward (squeeze = slice with int index on the inserted dim)
    """

    @staticmethod
    def forward(ctx, input: FutureTensor, dim: int):
        ctx.dim = dim
        ctx.input_ft = input
        result = unsqueeze_forward(input, dim)
        ctx.output_shape = result.ft_capacity_shape
        return result

    @staticmethod
    def backward(ctx, grad_output: FutureTensor):
        # grad_output from the autograd engine may be a plain torch.Tensor.
        if not hasattr(grad_output, "ft_initial_static_tensor"):
            # Reconstruct FutureTensor attributes on the EXISTING tensor so
            # that any grad_fn chain is preserved rather than severed.
            shape = getattr(ctx, "output_shape", list(grad_output.shape))
            relative_to = ctx.input_ft.ft_initial_static_tensor.st_relative_to
            async def dummy_get(coords, trajactory):
                return ("", Status.confidence(0.0))
            ref_ft = FutureTensor(relative_to, dummy_get, [sympy.Integer(s) for s in shape])
            if grad_output.numel() == 1:
                if shape:
                    ref_ft.ft_initial_static_tensor.data.flatten().fill_(grad_output.item())
                else:
                    ref_ft.ft_initial_static_tensor.data.fill_(grad_output.item())
            else:
                ref_ft.ft_initial_static_tensor.data.copy_(grad_output.data.view(ref_ft.ft_initial_static_tensor.shape))
            ref_ft.ft_forwarded = True

            # Monkey-patch attributes onto the existing grad_output tensor
            grad_output.ft_initial_static_tensor = ref_ft.ft_initial_static_tensor
            grad_output.ft_capacity_shape = ref_ft.ft_capacity_shape
            grad_output.ft_async_get = ref_ft.ft_async_get
            grad_output.ft_forwarded = ref_ft.ft_forwarded
            grad_output.ft_shape_schema = ref_ft.ft_shape_schema
            grad_output.ft_incremental_concated_tensors = ref_ft.ft_incremental_concated_tensors

        # Backward of unsqueeze is squeeze (slice with int index at dim)
        dim = ctx.dim
        ndim_output = len(grad_output.ft_capacity_shape)
        if dim < 0:
            dim = ndim_output + dim
        slices = [slice(None)] * ndim_output
        slices[dim] = 0  # int index collapses the dim

        # 1st-derivative dispatch: skip GradFn if dispatcher handles it.
        dispatch = get_backward_dispatcher(unsqueeze_forward)
        if dispatch({}):
            return grad_output, None

        if not grad_output.requires_grad:
            grad_output.requires_grad_(True)
        return UnsqueezeGradFn.apply(grad_output, dim, slices), None


def ft_unsqueeze(input: FutureTensor, dim: int) -> FutureTensor:
    """Unsqueeze a FutureTensor with autograd support.

    Same behavior as torch.unsqueeze — inserts a dimension of size 1 at `dim`.

    Args:
        input: Source FutureTensor.
        dim: Position at which to insert the new dimension.

    Returns:
        A new FutureTensor with the unsqueezed shape.
    """
    return FtUnsqueeze.apply(input, dim)


if __name__ == "__main__":
    import sympy
    from experience.future_tensor.status import Status
    import tempfile
    import os

    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running 100 tests for ft_unsqueeze...\n")

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

    # === Group 1: 1D ft_unsqueeze (tests 1-20) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"e{i}" for i in range(6)]
        ft = make_forwarded_ft([6], data, tmpdir)

        r = ft_unsqueeze(ft, 0)
        run_test("1", r.ft_capacity_shape == [1, 6])
        run_test("2", read_ft_element(r, 0) == "e0")
        run_test("3", read_ft_element(r, 5) == "e5")

        r = ft_unsqueeze(ft, 1)
        run_test("4", r.ft_capacity_shape == [6, 1])
        run_test("5", read_ft_element(r, 0) == "e0")
        run_test("6", read_ft_element(r, 5) == "e5")

        r = ft_unsqueeze(ft, -1)
        run_test("7", r.ft_capacity_shape == [6, 1])
        run_test("8", read_ft_element(r, 0) == "e0")

        r = ft_unsqueeze(ft, -2)
        run_test("9", r.ft_capacity_shape == [1, 6])
        run_test("10", read_ft_element(r, 0) == "e0")

    with tempfile.TemporaryDirectory() as tmpdir:
        data = ["single"]
        ft = make_forwarded_ft([1], data, tmpdir)

        r = ft_unsqueeze(ft, 0)
        run_test("11", r.ft_capacity_shape == [1, 1])
        run_test("12", read_ft_element(r, 0) == "single")

        r = ft_unsqueeze(ft, 1)
        run_test("13", r.ft_capacity_shape == [1, 1])
        run_test("14", read_ft_element(r, 0) == "single")

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"x{i}" for i in range(10)]
        ft = make_forwarded_ft([10], data, tmpdir)

        r = ft_unsqueeze(ft, 0)
        run_test("15", r.ft_capacity_shape == [1, 10])
        run_test("16", read_ft_element(r, 0) == "x0")
        run_test("17", read_ft_element(r, 9) == "x9")
        run_test("18", r.ft_forwarded is True)

        r = ft_unsqueeze(ft, 1)
        run_test("19", r.ft_capacity_shape == [10, 1])
        run_test("20", read_ft_element(r, 9) == "x9")

    # === Group 2: 2D ft_unsqueeze (tests 21-45) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"r{i}c{j}" for i in range(3) for j in range(4)]
        ft = make_forwarded_ft([3, 4], data, tmpdir)

        r = ft_unsqueeze(ft, 0)
        run_test("21", r.ft_capacity_shape == [1, 3, 4])
        run_test("22", read_ft_element(r, 0) == "r0c0")
        run_test("23", read_ft_element(r, 11) == "r2c3")

        r = ft_unsqueeze(ft, 1)
        run_test("24", r.ft_capacity_shape == [3, 1, 4])
        run_test("25", read_ft_element(r, 0) == "r0c0")
        run_test("26", read_ft_element(r, 4) == "r1c0")
        run_test("27", read_ft_element(r, 11) == "r2c3")

        r = ft_unsqueeze(ft, 2)
        run_test("28", r.ft_capacity_shape == [3, 4, 1])
        run_test("29", read_ft_element(r, 0) == "r0c0")
        run_test("30", read_ft_element(r, 1) == "r0c1")
        run_test("31", read_ft_element(r, 11) == "r2c3")

        r = ft_unsqueeze(ft, -1)
        run_test("32", r.ft_capacity_shape == [3, 4, 1])
        run_test("33", read_ft_element(r, 0) == "r0c0")

        r = ft_unsqueeze(ft, -2)
        run_test("34", r.ft_capacity_shape == [3, 1, 4])
        run_test("35", read_ft_element(r, 0) == "r0c0")

        r = ft_unsqueeze(ft, -3)
        run_test("36", r.ft_capacity_shape == [1, 3, 4])
        run_test("37", read_ft_element(r, 0) == "r0c0")

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"a{i}b{j}" for i in range(5) for j in range(2)]
        ft = make_forwarded_ft([5, 2], data, tmpdir)

        r = ft_unsqueeze(ft, 0)
        run_test("38", r.ft_capacity_shape == [1, 5, 2])
        run_test("39", read_ft_element(r, 0) == "a0b0")
        run_test("40", read_ft_element(r, 9) == "a4b1")

        r = ft_unsqueeze(ft, 1)
        run_test("41", r.ft_capacity_shape == [5, 1, 2])
        run_test("42", read_ft_element(r, 0) == "a0b0")
        run_test("43", read_ft_element(r, 2) == "a1b0")

        r = ft_unsqueeze(ft, 2)
        run_test("44", r.ft_capacity_shape == [5, 2, 1])
        run_test("45", read_ft_element(r, 0) == "a0b0")

    # === Group 3: 3D ft_unsqueeze (tests 46-60) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"v{i}" for i in range(24)]
        ft = make_forwarded_ft([2, 3, 4], data, tmpdir)

        r = ft_unsqueeze(ft, 0)
        run_test("46", r.ft_capacity_shape == [1, 2, 3, 4])
        run_test("47", read_ft_element(r, 0) == "v0")
        run_test("48", read_ft_element(r, 23) == "v23")

        r = ft_unsqueeze(ft, 1)
        run_test("49", r.ft_capacity_shape == [2, 1, 3, 4])
        run_test("50", read_ft_element(r, 0) == "v0")
        run_test("51", read_ft_element(r, 12) == "v12")

        r = ft_unsqueeze(ft, 2)
        run_test("52", r.ft_capacity_shape == [2, 3, 1, 4])
        run_test("53", read_ft_element(r, 0) == "v0")
        run_test("54", read_ft_element(r, 4) == "v4")

        r = ft_unsqueeze(ft, 3)
        run_test("55", r.ft_capacity_shape == [2, 3, 4, 1])
        run_test("56", read_ft_element(r, 0) == "v0")
        run_test("57", read_ft_element(r, 23) == "v23")

        r = ft_unsqueeze(ft, -1)
        run_test("58", r.ft_capacity_shape == [2, 3, 4, 1])
        r = ft_unsqueeze(ft, -4)
        run_test("59", r.ft_capacity_shape == [1, 2, 3, 4])
        run_test("60", read_ft_element(r, 0) == "v0")

    # === Group 4: Lazy ft_unsqueeze (tests 61-75) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        received = []

        async def tracking_get(coords, trajactory):
            received.append(coords)
            return (f"val_{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, tracking_get, [sympy.Integer(s) for s in [3, 4]])
        r = ft_unsqueeze(ft, 0)
        run_test("61", r.ft_capacity_shape == [1, 3, 4])
        run_test("62", r.ft_forwarded is False)

        prompt = st_make_tensor([[["p"] * 4] * 3], tmpdir)
        r.ft_forward(prompt)
        run_test("63", r.ft_forwarded is True)
        run_test("64", received[0] == [0, 0])
        run_test("65", received[1] == [0, 1])
        run_test("66", received[4] == [1, 0])
        run_test("67", read_ft_element(r, 0) == "val_[0, 0]")
        run_test("68", read_ft_element(r, 11) == "val_[2, 3]")

    with tempfile.TemporaryDirectory() as tmpdir:
        received2 = []

        async def tracking_get2(coords, prompt):
            received2.append(coords)
            return (f"r{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, tracking_get2, [sympy.Integer(s) for s in [5]])
        r = ft_unsqueeze(ft, 1)
        run_test("69", r.ft_capacity_shape == [5, 1])

        prompt = st_make_tensor([["p"]] * 5, tmpdir)
        r.ft_forward(prompt)
        run_test("70", received2[0] == [0])
        run_test("71", received2[4] == [4])
        run_test("72", read_ft_element(r, 0) == "r[0]")
        run_test("73", read_ft_element(r, 4) == "r[4]")

    with tempfile.TemporaryDirectory() as tmpdir:
        async def mid_get(coords, trajactory):
            return (f"m{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, mid_get, [sympy.Integer(s) for s in [2, 3]])
        r = ft_unsqueeze(ft, 1)
        run_test("74", r.ft_capacity_shape == [2, 1, 3])
        prompt = st_make_tensor([[["p"] * 3]] * 2, tmpdir)
        r.ft_forward(prompt)
        run_test("75", read_ft_element(r, 0) == "m[0, 0]")

    # === Group 5: Chained ft_unsqueeze (tests 76-85) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"d{i}" for i in range(4)]
        ft = make_forwarded_ft([4], data, tmpdir)

        r = ft_unsqueeze(ft, 0)
        r = ft_unsqueeze(r, 0)
        run_test("76", r.ft_capacity_shape == [1, 1, 4])
        run_test("77", read_ft_element(r, 0) == "d0")
        run_test("78", read_ft_element(r, 3) == "d3")

        r = ft_unsqueeze(ft, 0)
        r = ft_unsqueeze(r, 2)
        run_test("79", r.ft_capacity_shape == [1, 4, 1])
        run_test("80", read_ft_element(r, 0) == "d0")
        run_test("81", read_ft_element(r, 3) == "d3")

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"r{i}c{j}" for i in range(2) for j in range(3)]
        ft = make_forwarded_ft([2, 3], data, tmpdir)

        r = ft_unsqueeze(ft, 0)
        r = ft_unsqueeze(r, 2)
        r = ft_unsqueeze(r, 4)
        run_test("82", r.ft_capacity_shape == [1, 2, 1, 3, 1])
        run_test("83", read_ft_element(r, 0) == "r0c0")
        run_test("84", read_ft_element(r, 5) == "r1c2")

        r = ft_unsqueeze(ft, -1)
        run_test("85", r.ft_capacity_shape == [2, 3, 1])

    # === Group 6: Unsqueeze + slice combo (tests 86-100) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.ft_slice import ft_slice as slice_op

        data = [f"v{i}" for i in range(8)]
        ft = make_forwarded_ft([8], data, tmpdir)

        r = ft_unsqueeze(ft, 0)
        r2 = slice_op(r, [0, slice(2, 6)])
        run_test("86", r2.ft_capacity_shape == [4])
        run_test("87", read_ft_element(r2, 0) == "v2")
        run_test("88", read_ft_element(r2, 3) == "v5")

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.ft_slice import ft_slice as slice_op

        data = [f"r{i}c{j}" for i in range(4) for j in range(5)]
        ft = make_forwarded_ft([4, 5], data, tmpdir)

        # Slice then unsqueeze
        r = slice_op(ft, [slice(1, 3), slice(None)])
        r2 = ft_unsqueeze(r, 0)
        run_test("89", r2.ft_capacity_shape == [1, 2, 5])
        run_test("90", read_ft_element(r2, 0) == "r1c0")
        run_test("91", read_ft_element(r2, 4) == "r1c4")
        run_test("92", read_ft_element(r2, 5) == "r2c0")
        run_test("93", read_ft_element(r2, 9) == "r2c4")

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.ft_slice import ft_slice as slice_op

        data = [f"item_{i}" for i in range(12)]
        ft = make_forwarded_ft([3, 4], data, tmpdir)

        # Unsqueeze at dim 1, then slice at dim 0
        r = ft_unsqueeze(ft, 1)  # [3, 1, 4]
        r2 = slice_op(r, [slice(0, 2), slice(None), slice(None)])  # [2, 1, 4]
        run_test("94", r2.ft_capacity_shape == [2, 1, 4])
        run_test("95", read_ft_element(r2, 0) == "item_0")
        run_test("96", read_ft_element(r2, 3) == "item_3")
        run_test("97", read_ft_element(r2, 4) == "item_4")
        run_test("98", read_ft_element(r2, 7) == "item_7")

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.ft_slice import ft_slice as slice_op

        # Unsqueeze -> squeeze via int slice (round-trip)
        data = [f"q{i}" for i in range(5)]
        ft = make_forwarded_ft([5], data, tmpdir)

        r = ft_unsqueeze(ft, 0)  # [1, 5]
        r2 = slice_op(r, [0, slice(None)])  # squeeze dim 0 -> [5]
        run_test("99", r2.ft_capacity_shape == [5])
        run_test("100", read_ft_element(r2, 0) == "q0")

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All ft_unsqueeze tests completed.")
