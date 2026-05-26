"""
unsqueeze_forward :=
    FutureTensor
    <- FutureTensor
    <- ...
    # inline

# the same behavior with torch unsqueeze
"""

import os
from typing import List

import torch

import sympy

from experience.future_tensor.future_tensor import FutureTensor


def unsqueeze_forward(input: FutureTensor, dim: int) -> FutureTensor:
    """Unsqueeze a FutureTensor — insert a dimension of size 1.

    Same semantics as torch.unsqueeze. The new dimension is inserted at `dim`.

    Args:
        input: Source FutureTensor.
        dim: Position at which to insert the new dimension.
            Supports negative indexing.
    """
    input_shape = input.ft_capacity_shape
    input_schema = input.ft_shape_schema
    ndim = len(input_shape)

    # Normalize negative dim (allow dim in range [-(ndim+1), ndim])
    if dim < 0:
        dim = ndim + 1 + dim
    assert 0 <= dim <= ndim, f"dim {dim} out of range for {ndim}D tensor"

    # New shape: insert 1 at dim
    output_shape = input_shape[:dim] + [1] + input_shape[dim:]
    output_schema = list(input_schema[:dim]) + [sympy.Integer(1)] + list(input_schema[dim:])

    # Coordinate mapping: remove the unsqueezed dim to get input coords
    def map_coords(out_coords: List[int]) -> List[int]:
        # out_coords[dim] is always 0 (since that dim has size 1)
        return out_coords[:dim] + out_coords[dim + 1:]

    async def unsqueezed_async_get(coordinates: List[int], trajactory: str) -> str:
        original_coords = map_coords(coordinates)
        return await input.ft_async_get(original_coords, trajactory)

    result = FutureTensor(input.ft_static_tensor.st_relative_to, unsqueezed_async_get, list(output_schema))
    result.ft_capacity_shape = list(output_shape)

    # If input is already forwarded, copy storage directly
    if input.ft_forwarded:
        _copy_unsqueezed_storage(input, result, output_shape, dim)
        result.ft_forwarded = True

    return result


def _copy_unsqueezed_storage(input, output, output_shape, dim):
    """Copy all element storage from input to output (same data, different shape)."""
    import shutil
    import itertools

    input_shape = input.ft_capacity_shape

    for in_coords in itertools.product(*[range(s) for s in input_shape]) if input_shape else [()]:
        in_coords = list(in_coords) if input_shape else []

        # Map to output coords (insert 0 at dim)
        out_coords = in_coords[:dim] + [0] + in_coords[dim:]

        in_flat = _coords_to_flat(in_coords, input_shape) if input_shape else 0
        out_flat = _coords_to_flat(out_coords, output_shape)

        src_path = _storage_path(input, in_flat)
        dst_path = _storage_path(output, out_flat)
        if os.path.isfile(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            # Copy coefficient (confidence) from input
            output.ft_static_tensor.data.flatten()[out_flat] = input.ft_static_tensor.data.flatten()[in_flat]


def _coords_to_flat(coordinates: List[int], shape: List[int]) -> int:
    flat = 0
    stride = 1
    for i in reversed(range(len(shape))):
        flat += coordinates[i] * stride
        stride *= shape[i]
    return flat


def _storage_path(ft: FutureTensor, flat_index: int) -> str:
    digits = list(str(flat_index))
    return os.path.join(
        ft.ft_static_tensor.st_relative_to, ft.ft_static_tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )


if __name__ == "__main__":
    import sympy
    import torch

    import tempfile
    import itertools
    import asyncio

    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
    from experience.future_tensor.status import Status

    print("Running 100 tests for unsqueeze_forward...\n")

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

    # === Group 1: 1D unsqueeze at dim=0 (tests 1-15) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"e{i}" for i in range(5)]
        ft = make_forwarded_ft([5], data, tmpdir)

        r = unsqueeze_forward(ft, 0)
        run_test("1D dim=0 shape", r.ft_capacity_shape == [1, 5])
        run_test("1D dim=0 forwarded", r.ft_forwarded is True)
        run_test("1D dim=0 [0,0]", read_ft_element(r, 0) == "e0")
        run_test("1D dim=0 [0,1]", read_ft_element(r, 1) == "e1")
        run_test("1D dim=0 [0,4]", read_ft_element(r, 4) == "e4")

        r = unsqueeze_forward(ft, 1)
        run_test("1D dim=1 shape", r.ft_capacity_shape == [5, 1])
        run_test("1D dim=1 forwarded", r.ft_forwarded is True)
        run_test("1D dim=1 [0,0]", read_ft_element(r, 0) == "e0")
        run_test("1D dim=1 [1,0]", read_ft_element(r, 1) == "e1")
        run_test("1D dim=1 [4,0]", read_ft_element(r, 4) == "e4")

        r = unsqueeze_forward(ft, -1)
        run_test("1D dim=-1 shape", r.ft_capacity_shape == [5, 1])
        run_test("1D dim=-1 [0,0]", read_ft_element(r, 0) == "e0")
        run_test("1D dim=-1 [4,0]", read_ft_element(r, 4) == "e4")

        r = unsqueeze_forward(ft, -2)
        run_test("1D dim=-2 shape", r.ft_capacity_shape == [1, 5])
        run_test("1D dim=-2 [0,0]", read_ft_element(r, 0) == "e0")

    # === Group 2: 2D unsqueeze (tests 16-40) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"r{i}c{j}" for i in range(3) for j in range(4)]
        ft = make_forwarded_ft([3, 4], data, tmpdir)

        # dim=0: [3,4] -> [1,3,4]
        r = unsqueeze_forward(ft, 0)
        run_test("2D dim=0 shape", r.ft_capacity_shape == [1, 3, 4])
        run_test("2D dim=0 [0,0,0]", read_ft_element(r, 0) == "r0c0")
        run_test("2D dim=0 [0,1,2]", read_ft_element(r, 6) == "r1c2")
        run_test("2D dim=0 [0,2,3]", read_ft_element(r, 11) == "r2c3")

        # dim=1: [3,4] -> [3,1,4]
        r = unsqueeze_forward(ft, 1)
        run_test("2D dim=1 shape", r.ft_capacity_shape == [3, 1, 4])
        run_test("2D dim=1 [0,0,0]", read_ft_element(r, 0) == "r0c0")
        run_test("2D dim=1 [0,0,3]", read_ft_element(r, 3) == "r0c3")
        run_test("2D dim=1 [1,0,0]", read_ft_element(r, 4) == "r1c0")
        run_test("2D dim=1 [2,0,3]", read_ft_element(r, 11) == "r2c3")

        # dim=2: [3,4] -> [3,4,1]
        r = unsqueeze_forward(ft, 2)
        run_test("2D dim=2 shape", r.ft_capacity_shape == [3, 4, 1])
        run_test("2D dim=2 [0,0,0]", read_ft_element(r, 0) == "r0c0")
        run_test("2D dim=2 [0,1,0]", read_ft_element(r, 1) == "r0c1")
        run_test("2D dim=2 [1,0,0]", read_ft_element(r, 4) == "r1c0")
        run_test("2D dim=2 [2,3,0]", read_ft_element(r, 11) == "r2c3")

        # Negative dims
        r = unsqueeze_forward(ft, -1)
        run_test("2D dim=-1 shape", r.ft_capacity_shape == [3, 4, 1])
        run_test("2D dim=-1 [0,0,0]", read_ft_element(r, 0) == "r0c0")

        r = unsqueeze_forward(ft, -2)
        run_test("2D dim=-2 shape", r.ft_capacity_shape == [3, 1, 4])
        run_test("2D dim=-2 [0,0,0]", read_ft_element(r, 0) == "r0c0")

        r = unsqueeze_forward(ft, -3)
        run_test("2D dim=-3 shape", r.ft_capacity_shape == [1, 3, 4])
        run_test("2D dim=-3 [0,0,0]", read_ft_element(r, 0) == "r0c0")

    # === Group 3: 3D unsqueeze (tests 41-60) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"v{i}" for i in range(24)]
        ft = make_forwarded_ft([2, 3, 4], data, tmpdir)

        r = unsqueeze_forward(ft, 0)
        run_test("3D dim=0 shape", r.ft_capacity_shape == [1, 2, 3, 4])
        run_test("3D dim=0 [0,0,0,0]", read_ft_element(r, 0) == "v0")
        run_test("3D dim=0 [0,1,2,3]", read_ft_element(r, 23) == "v23")

        r = unsqueeze_forward(ft, 1)
        run_test("3D dim=1 shape", r.ft_capacity_shape == [2, 1, 3, 4])
        run_test("3D dim=1 [0,0,0,0]", read_ft_element(r, 0) == "v0")
        run_test("3D dim=1 [1,0,0,0]", read_ft_element(r, 12) == "v12")

        r = unsqueeze_forward(ft, 2)
        run_test("3D dim=2 shape", r.ft_capacity_shape == [2, 3, 1, 4])
        run_test("3D dim=2 [0,0,0,0]", read_ft_element(r, 0) == "v0")
        run_test("3D dim=2 [0,1,0,0]", read_ft_element(r, 4) == "v4")
        run_test("3D dim=2 [1,0,0,0]", read_ft_element(r, 12) == "v12")

        r = unsqueeze_forward(ft, 3)
        run_test("3D dim=3 shape", r.ft_capacity_shape == [2, 3, 4, 1])
        run_test("3D dim=3 [0,0,0,0]", read_ft_element(r, 0) == "v0")
        run_test("3D dim=3 [0,0,1,0]", read_ft_element(r, 1) == "v1")
        run_test("3D dim=3 [0,1,0,0]", read_ft_element(r, 4) == "v4")
        run_test("3D dim=3 [1,0,0,0]", read_ft_element(r, 12) == "v12")
        run_test("3D dim=3 [1,2,3,0]", read_ft_element(r, 23) == "v23")

        # Double unsqueeze
        r = unsqueeze_forward(ft, 0)
        r2 = unsqueeze_forward(r, 0)
        run_test("double unsqueeze shape", r2.ft_capacity_shape == [1, 1, 2, 3, 4])
        run_test("double unsqueeze [0,0,0,0,0]", read_ft_element(r2, 0) == "v0")
        run_test("double unsqueeze [0,0,1,2,3]", read_ft_element(r2, 23) == "v23")

    # === Group 4: Lazy unsqueeze (not forwarded) (tests 61-80) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        received = []

        async def tracking_get(coords, trajactory):
            received.append(coords)
            return (f"val_{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, tracking_get, [sympy.Integer(4), sympy.Integer(3)])
        r = unsqueeze_forward(ft, 0)
        run_test("lazy unsqueeze shape", r.ft_capacity_shape == [1, 4, 3])
        run_test("lazy unsqueeze not forwarded", r.ft_forwarded is False)

        # Forward the unsqueezed tensor
        prompt_data = ["p"] * 12
        prompt_t = st_make_tensor([prompt_data[:4], prompt_data[4:8], prompt_data[8:]], tmpdir)
        # Need shape [1,4,3]
        prompt_t2 = st_make_tensor([["p"] * 3] * 4, tmpdir)
        # Actually need [1,4,3] prompts
        prompt_flat = st_make_tensor([[["p"] * 3] * 4], tmpdir)
        r.ft_forward(prompt_flat)

        run_test("lazy forwarded after ft_forward", r.ft_forwarded is True)
        # Coordinates should strip dim 0
        run_test("lazy coord [0,0,0] -> [0,0]", received[0] == [0, 0])
        run_test("lazy coord [0,0,1] -> [0,1]", received[1] == [0, 1])
        run_test("lazy coord [0,1,0] -> [1,0]", received[3] == [1, 0])
        run_test("lazy coord [0,3,2] -> [3,2]", received[11] == [3, 2])
        run_test("lazy result [0]", read_ft_element(r, 0) == "val_[0, 0]")
        run_test("lazy result [11]", read_ft_element(r, 11) == "val_[3, 2]")

    with tempfile.TemporaryDirectory() as tmpdir:
        received2 = []

        async def tracking_get2(coords, prompt):
            received2.append(coords)
            return (f"r{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, tracking_get2, [sympy.Integer(5)])
        r = unsqueeze_forward(ft, 1)
        run_test("lazy dim=1 shape", r.ft_capacity_shape == [5, 1])

        prompt_flat = st_make_tensor([["p"]] * 5, tmpdir)
        r.ft_forward(prompt_flat)
        run_test("lazy dim=1 coord [0,0] -> [0]", received2[0] == [0])
        run_test("lazy dim=1 coord [4,0] -> [4]", received2[4] == [4])
        run_test("lazy dim=1 result [0]", read_ft_element(r, 0) == "r[0]")
        run_test("lazy dim=1 result [4]", read_ft_element(r, 4) == "r[4]")

    with tempfile.TemporaryDirectory() as tmpdir:
        async def simple_get(coords, trajactory):
            return (f"x{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, simple_get, [sympy.Integer(2), sympy.Integer(3)])
        r = unsqueeze_forward(ft, 1)
        run_test("lazy mid-dim shape", r.ft_capacity_shape == [2, 1, 3])

        prompt_flat = st_make_tensor([[["p"] * 3]] * 2, tmpdir)
        r.ft_forward(prompt_flat)
        run_test("lazy mid-dim [0,0,0]", read_ft_element(r, 0) == "x[0, 0]")
        run_test("lazy mid-dim [0,0,2]", read_ft_element(r, 2) == "x[0, 2]")
        run_test("lazy mid-dim [1,0,0]", read_ft_element(r, 3) == "x[1, 0]")
        run_test("lazy mid-dim [1,0,2]", read_ft_element(r, 5) == "x[1, 2]")

    # === Group 5: Chained unsqueeze (tests 81-90) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"d{i}" for i in range(6)]
        ft = make_forwarded_ft([2, 3], data, tmpdir)

        r = unsqueeze_forward(ft, 0)
        r = unsqueeze_forward(r, 2)
        run_test("chain [2,3]->dim0->dim2 shape", r.ft_capacity_shape == [1, 2, 1, 3])
        run_test("chain [0,0,0,0]", read_ft_element(r, 0) == "d0")
        run_test("chain [0,0,0,2]", read_ft_element(r, 2) == "d2")
        run_test("chain [0,1,0,0]", read_ft_element(r, 3) == "d3")
        run_test("chain [0,1,0,2]", read_ft_element(r, 5) == "d5")

    with tempfile.TemporaryDirectory() as tmpdir:
        data = ["a", "b", "c"]
        ft = make_forwarded_ft([3], data, tmpdir)

        r = unsqueeze_forward(ft, 0)
        r = unsqueeze_forward(r, 0)
        r = unsqueeze_forward(r, 0)
        run_test("triple unsqueeze shape", r.ft_capacity_shape == [1, 1, 1, 3])
        run_test("triple [0,0,0,0]", read_ft_element(r, 0) == "a")
        run_test("triple [0,0,0,1]", read_ft_element(r, 1) == "b")
        run_test("triple [0,0,0,2]", read_ft_element(r, 2) == "c")

    # === Group 6: Unsqueeze + slice interaction (tests 91-100) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.slice_forward import slice_forward

        data = [f"item_{i}" for i in range(8)]
        ft = make_forwarded_ft([8], data, tmpdir)

        # Unsqueeze then slice
        r = unsqueeze_forward(ft, 0)  # [1, 8]
        run_test("unsqueeze+slice: after unsqueeze", r.ft_capacity_shape == [1, 8])
        r2 = slice_forward(r, [0, slice(2, 6)])  # int + slice -> [4]
        run_test("unsqueeze+slice: after slice shape", r2.ft_capacity_shape == [4])
        run_test("unsqueeze+slice: [0]", read_ft_element(r2, 0) == "item_2")
        run_test("unsqueeze+slice: [3]", read_ft_element(r2, 3) == "item_5")

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.slice_forward import slice_forward

        data = [f"r{i}c{j}" for i in range(4) for j in range(3)]
        ft = make_forwarded_ft([4, 3], data, tmpdir)

        # Slice then unsqueeze
        r = slice_forward(ft, [slice(1, 3), slice(None)])  # [2, 3]
        run_test("slice+unsqueeze: after slice", r.ft_capacity_shape == [2, 3])
        r2 = unsqueeze_forward(r, 1)  # [2, 1, 3]
        run_test("slice+unsqueeze: after unsqueeze shape", r2.ft_capacity_shape == [2, 1, 3])
        run_test("slice+unsqueeze: [0,0,0] = r1c0", read_ft_element(r2, 0) == "r1c0")
        run_test("slice+unsqueeze: [0,0,2] = r1c2", read_ft_element(r2, 2) == "r1c2")
        run_test("slice+unsqueeze: [1,0,0] = r2c0", read_ft_element(r2, 3) == "r2c0")
        run_test("slice+unsqueeze: [1,0,2] = r2c2", read_ft_element(r2, 5) == "r2c2")

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All unsqueeze_forward tests completed.")
