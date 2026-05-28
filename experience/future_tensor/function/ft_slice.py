"""
FtSlice := torch.autograd.Function[
    $forward Import[{future_tensor function slice_forward.viba}],
    $backward Import[{future_tensor function slice_backward.viba}]
]

ft_slice = FtSlice.apply

# the same behavior with torch slice
"""

from typing import List, Union

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor, _tensor_to_future
from experience.future_tensor.function.slice_forward import slice_forward
from experience.future_tensor.function.slice_backward import slice_backward
from experience.future_tensor.function.slice_2nd import SliceGradFn
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class FtSlice(torch.autograd.Function):
    """Autograd Function for slicing FutureTensors.

    Forward: slice_forward (same as torch slice)
    Backward: slice_backward (scatter grad back to original positions)
    """

    @staticmethod
    def forward(ctx, input: FutureTensor, slices: List[Union[int, slice]]):
        ctx.original_shape = input.ft_capacity_shape
        ctx.slices = slices
        ctx.input_ft = input
        result = slice_forward(input, slices)
        ctx.output_shape = result.ft_capacity_shape
        return result

    @staticmethod
    def backward(ctx, grad_output: FutureTensor):
        # grad_output from the autograd engine may be a plain torch.Tensor
        # (it strips FutureTensor monkey-patched attributes).
        if not hasattr(grad_output, "ft_initial_static_tensor"):
            # Reconstruct FutureTensor attributes on the EXISTING tensor so
            # that any grad_fn chain (e.g. RecurrentGradFnBackward) is
            # preserved rather than severed by creating a new tensor object.
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

        # 1st-derivative dispatch: skip GradFn if dispatcher handles it.
        dispatch = get_backward_dispatcher(slice_backward)
        if dispatch({}):
            return grad_output, None

        if not grad_output.requires_grad:
            grad_output.requires_grad_(True)
        return SliceGradFn.apply(grad_output, ctx.original_shape, ctx.slices), None


def ft_slice(input: FutureTensor, slices: List[Union[int, slice]]) -> FutureTensor:
    """Slice a FutureTensor with autograd support.

    Same behavior as torch slice — supports int indexing (dim collapse),
    slice(start, stop, step), and negative indices.

    Args:
        input: Source FutureTensor.
        slices: List of int or slice objects, one per dimension.

    Returns:
        A new FutureTensor with the sliced shape.
    """
    return FtSlice.apply(input, slices)


if __name__ == "__main__":
    import sympy
    from experience.future_tensor.status import Status
    import tempfile
    import os

    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running 100 tests for ft_slice...\n")

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

    # === Group 1: 1D ft_slice (tests 1-25) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"v{i}" for i in range(10)]
        ft = make_forwarded_ft([10], data, tmpdir)

        r = ft_slice(ft, [slice(0, 5)])
        run_test("1", r.ft_capacity_shape == [5])
        run_test("2", read_ft_element(r, 0) == "v0")
        run_test("3", read_ft_element(r, 4) == "v4")

        r = ft_slice(ft, [slice(5, 10)])
        run_test("4", r.ft_capacity_shape == [5])
        run_test("5", read_ft_element(r, 0) == "v5")
        run_test("6", read_ft_element(r, 4) == "v9")

        r = ft_slice(ft, [slice(0, 10, 2)])
        run_test("7", r.ft_capacity_shape == [5])
        run_test("8", read_ft_element(r, 0) == "v0")
        run_test("9", read_ft_element(r, 1) == "v2")
        run_test("10", read_ft_element(r, 4) == "v8")

        r = ft_slice(ft, [slice(1, 8, 3)])
        run_test("11", r.ft_capacity_shape == [3])
        run_test("12", read_ft_element(r, 0) == "v1")
        run_test("13", read_ft_element(r, 1) == "v4")
        run_test("14", read_ft_element(r, 2) == "v7")

        r = ft_slice(ft, [0])
        run_test("15", r.ft_capacity_shape == [])

        r = ft_slice(ft, [9])
        run_test("16", r.ft_capacity_shape == [])

        r = ft_slice(ft, [-1])
        run_test("17", r.ft_capacity_shape == [])

        r = ft_slice(ft, [slice(-5, None)])
        run_test("18", r.ft_capacity_shape == [5])
        run_test("19", read_ft_element(r, 0) == "v5")
        run_test("20", read_ft_element(r, 4) == "v9")

        r = ft_slice(ft, [slice(None, 3)])
        run_test("21", r.ft_capacity_shape == [3])
        run_test("22", read_ft_element(r, 0) == "v0")
        run_test("23", read_ft_element(r, 2) == "v2")

        r = ft_slice(ft, [slice(None)])
        run_test("24", r.ft_capacity_shape == [10])
        run_test("25", read_ft_element(r, 9) == "v9")

    # === Group 2: 2D ft_slice (tests 26-50) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"r{i}c{j}" for i in range(5) for j in range(4)]
        ft = make_forwarded_ft([5, 4], data, tmpdir)

        r = ft_slice(ft, [slice(1, 3), slice(None)])
        run_test("26", r.ft_capacity_shape == [2, 4])
        run_test("27", read_ft_element(r, 0) == "r1c0")
        run_test("28", read_ft_element(r, 7) == "r2c3")

        r = ft_slice(ft, [slice(None), slice(1, 3)])
        run_test("29", r.ft_capacity_shape == [5, 2])
        run_test("30", read_ft_element(r, 0) == "r0c1")
        run_test("31", read_ft_element(r, 1) == "r0c2")
        run_test("32", read_ft_element(r, 2) == "r1c1")

        r = ft_slice(ft, [2, slice(None)])
        run_test("33", r.ft_capacity_shape == [4])
        run_test("34", read_ft_element(r, 0) == "r2c0")
        run_test("35", read_ft_element(r, 3) == "r2c3")

        r = ft_slice(ft, [slice(None), 0])
        run_test("36", r.ft_capacity_shape == [5])
        run_test("37", read_ft_element(r, 0) == "r0c0")
        run_test("38", read_ft_element(r, 4) == "r4c0")

        r = ft_slice(ft, [1, 2])
        run_test("39", r.ft_capacity_shape == [])

        r = ft_slice(ft, [slice(0, 4, 2), slice(0, 4, 2)])
        run_test("40", r.ft_capacity_shape == [2, 2])
        run_test("41", read_ft_element(r, 0) == "r0c0")
        run_test("42", read_ft_element(r, 1) == "r0c2")
        run_test("43", read_ft_element(r, 2) == "r2c0")
        run_test("44", read_ft_element(r, 3) == "r2c2")

        r = ft_slice(ft, [slice(-2, None), slice(-2, None)])
        run_test("45", r.ft_capacity_shape == [2, 2])
        run_test("46", read_ft_element(r, 0) == "r3c2")
        run_test("47", read_ft_element(r, 3) == "r4c3")

        r = ft_slice(ft, [slice(0, 1), slice(0, 1)])
        run_test("48", r.ft_capacity_shape == [1, 1])
        run_test("49", read_ft_element(r, 0) == "r0c0")

        r = ft_slice(ft, [4, 3])
        run_test("50", r.ft_capacity_shape == [])

    # === Group 3: 3D ft_slice (tests 51-65) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"d{d}r{r}c{c}" for d in range(2) for r in range(3) for c in range(4)]
        ft = make_forwarded_ft([2, 3, 4], data, tmpdir)

        r = ft_slice(ft, [0, slice(None), slice(None)])
        run_test("51", r.ft_capacity_shape == [3, 4])
        run_test("52", read_ft_element(r, 0) == "d0r0c0")
        run_test("53", read_ft_element(r, 11) == "d0r2c3")

        r = ft_slice(ft, [slice(None), 1, slice(None)])
        run_test("54", r.ft_capacity_shape == [2, 4])
        run_test("55", read_ft_element(r, 0) == "d0r1c0")
        run_test("56", read_ft_element(r, 4) == "d1r1c0")

        r = ft_slice(ft, [slice(None), slice(None), 2])
        run_test("57", r.ft_capacity_shape == [2, 3])
        run_test("58", read_ft_element(r, 0) == "d0r0c2")
        run_test("59", read_ft_element(r, 3) == "d1r0c2")

        r = ft_slice(ft, [1, 2, 3])
        run_test("60", r.ft_capacity_shape == [])

        r = ft_slice(ft, [slice(0, 2), slice(0, 2), slice(0, 2)])
        run_test("61", r.ft_capacity_shape == [2, 2, 2])
        run_test("62", read_ft_element(r, 0) == "d0r0c0")
        run_test("63", read_ft_element(r, 1) == "d0r0c1")
        run_test("64", read_ft_element(r, 4) == "d1r0c0")
        run_test("65", read_ft_element(r, 7) == "d1r1c1")

    # === Group 4: Lazy ft_slice (tests 66-80) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        received = []

        async def tracking_get(coords, trajactory):
            received.append(coords)
            return (f"out_{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, tracking_get, [sympy.Integer(s) for s in [6, 4]])
        r = ft_slice(ft, [slice(2, 4), slice(1, 3)])
        run_test("66", r.ft_capacity_shape == [2, 2])
        run_test("67", r.ft_forwarded is False)

        prompt_t = st_make_tensor([["p", "p"], ["p", "p"]], tmpdir)
        r.ft_forward(prompt_t)
        run_test("68", r.ft_forwarded is True)
        run_test("69", received[0] == [2, 1])
        run_test("70", received[1] == [2, 2])
        run_test("71", received[2] == [3, 1])
        run_test("72", received[3] == [3, 2])
        run_test("73", read_ft_element(r, 0) == "out_[2, 1]")
        run_test("74", read_ft_element(r, 3) == "out_[3, 2]")

    with tempfile.TemporaryDirectory() as tmpdir:
        async def int_get(coords, trajactory):
            return (f"g{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, int_get, [sympy.Integer(s) for s in [5, 5]])
        r = ft_slice(ft, [3, slice(None)])
        run_test("75", r.ft_capacity_shape == [5])
        prompt_t = st_make_tensor(["p"] * 5, tmpdir)
        r.ft_forward(prompt_t)
        run_test("76", read_ft_element(r, 0) == "g[3, 0]")
        run_test("77", read_ft_element(r, 4) == "g[3, 4]")

    with tempfile.TemporaryDirectory() as tmpdir:
        async def step_get(coords, trajactory):
            return (str(coords), Status.confidence(1.0))

        ft = FutureTensor(tmpdir, step_get, [sympy.Integer(s) for s in [12]])
        r = ft_slice(ft, [slice(0, 12, 4)])
        run_test("78", r.ft_capacity_shape == [3])
        prompt_t = st_make_tensor(["p"] * 3, tmpdir)
        r.ft_forward(prompt_t)
        run_test("79", read_ft_element(r, 0) == "[0]")
        run_test("80", read_ft_element(r, 2) == "[8]")

    # === Group 5: Chained ft_slice (tests 81-90) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"w{i}" for i in range(20)]
        ft = make_forwarded_ft([20], data, tmpdir)

        r = ft_slice(ft, [slice(5, 15)])
        r2 = ft_slice(r, [slice(2, 7)])
        run_test("81", r2.ft_capacity_shape == [5])
        run_test("82", read_ft_element(r2, 0) == "w7")
        run_test("83", read_ft_element(r2, 4) == "w11")

        r = ft_slice(ft, [slice(0, 20, 2)])
        r2 = ft_slice(r, [slice(1, 5)])
        run_test("84", r2.ft_capacity_shape == [4])
        run_test("85", read_ft_element(r2, 0) == "w2")
        run_test("86", read_ft_element(r2, 3) == "w8")

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"r{i}c{j}" for i in range(6) for j in range(6)]
        ft = make_forwarded_ft([6, 6], data, tmpdir)

        r = ft_slice(ft, [slice(1, 5), slice(None)])
        r2 = ft_slice(r, [slice(None), slice(2, 5)])
        run_test("87", r2.ft_capacity_shape == [4, 3])
        run_test("88", read_ft_element(r2, 0) == "r1c2")
        run_test("89", read_ft_element(r2, 2) == "r1c4")
        run_test("90", read_ft_element(r2, 9) == "r4c2")

    # === Group 6: Edge cases (tests 91-100) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # Single element tensor
        ft = make_forwarded_ft([1], ["solo"], tmpdir)
        r = ft_slice(ft, [slice(None)])
        run_test("91", r.ft_capacity_shape == [1])
        run_test("92", read_ft_element(r, 0) == "solo")

        r = ft_slice(ft, [0])
        run_test("93", r.ft_capacity_shape == [])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty result
        ft = make_forwarded_ft([5], [f"x{i}" for i in range(5)], tmpdir)
        r = ft_slice(ft, [slice(2, 2)])
        run_test("94", r.ft_capacity_shape == [0])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Reverse slice
        data = [f"r{i}" for i in range(6)]
        ft = make_forwarded_ft([6], data, tmpdir)
        r = ft_slice(ft, [slice(5, 1, -1)])
        run_test("95", r.ft_capacity_shape == [4])
        run_test("96", read_ft_element(r, 0) == "r5")
        run_test("97", read_ft_element(r, 3) == "r2")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Large tensor
        n = 100
        data = [f"i{i}" for i in range(n)]
        ft = make_forwarded_ft([n], data, tmpdir)
        r = ft_slice(ft, [slice(25, 75)])
        run_test("98", r.ft_capacity_shape == [50])
        run_test("99", read_ft_element(r, 0) == "i25")
        run_test("100", read_ft_element(r, 49) == "i74")

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All ft_slice tests completed.")
