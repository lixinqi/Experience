"""
slice_forward :=
    FutureTensor
    <- FutureTensor
    <- list[slice]
    # inline

# the same behavior with torch slice
"""

import itertools
import os
from typing import List, Union

import torch

import sympy

from experience.future_tensor.future_tensor import FutureTensor


def _resolve_slice(s: Union[int, slice], dim_size: int):
    """Resolve a slice or int index into a list of concrete indices.

    Returns (indices, collapsed) where collapsed=True means this dim is removed.
    """
    if isinstance(s, int):
        idx = s if s >= 0 else s + dim_size
        return [idx], True
    start, stop, step = s.indices(dim_size)
    return list(range(start, stop, step)), False


def slice_forward(input: FutureTensor, slices: List[Union[int, slice]]) -> FutureTensor:
    """Slice a FutureTensor — same semantics as torch tensor slicing.

    Returns a new FutureTensor whose ft_async_get maps sliced coordinates
    back to the original tensor's coordinate space.

    Args:
        input: Source FutureTensor.
        slices: List of int or slice objects, one per dimension.
    """
    input_shape = input.ft_capacity_shape

    # Pad slices with slice(None) if fewer than ndim
    full_slices = list(slices)
    while len(full_slices) < len(input_shape):
        full_slices.append(slice(None))

    # Resolve each slice into (indices, collapsed)
    per_dim = []
    for d, s in enumerate(full_slices):
        indices, collapsed = _resolve_slice(s, input_shape[d])
        per_dim.append((indices, collapsed))

    # Compute output shape (drop collapsed dims)
    output_shape = [len(indices) for indices, collapsed in per_dim if not collapsed]

    # Compute output schema: preserve symbolic info from input for non-collapsed dims
    input_schema = input.ft_shape_schema
    # Pad schema with sympy.Integer(1) if input has fewer dims (shouldn't happen for slice)
    padded_schema = list(input_schema)
    while len(padded_schema) < len(full_slices):
        padded_schema.append(sympy.Integer(1))

    output_schema = []
    for d, (indices, collapsed) in enumerate(per_dim):
        if not collapsed:
            # If slicing takes all elements of a symbolic dim, preserve the symbol
            # Otherwise, the result is concrete
            if len(indices) == input_shape[d] and d < len(input_schema):
                output_schema.append(padded_schema[d])
            else:
                output_schema.append(sympy.Integer(len(indices)))

    # Build coordinate mapping: output_coords -> input_coords
    def map_coords(out_coords: List[int]) -> List[int]:
        in_coords = []
        out_dim = 0
        for indices, collapsed in per_dim:
            if collapsed:
                in_coords.append(indices[0])
            else:
                in_coords.append(indices[out_coords[out_dim]])
                out_dim += 1
        return in_coords

    # Create new ft_async_get that delegates to input's ft_async_get
    # with remapped coordinates
    async def sliced_async_get(coordinates: List[int], trajactory: str) -> str:
        original_coords = map_coords(coordinates)
        return await input.ft_async_get(original_coords, trajactory)

    result = FutureTensor(input.ft_static_tensor.st_relative_to, sliced_async_get, list(output_schema))
    result.ft_capacity_shape = list(output_shape)

    # If input is already forwarded, slice the storage directly
    if input.ft_forwarded:
        _copy_sliced_storage(input, result, per_dim, output_shape)
        result.ft_forwarded = True

    return result


def _copy_sliced_storage(
    input: FutureTensor, output: FutureTensor, per_dim, output_shape
):
    """Copy element storage from input to output based on slice mapping."""
    import shutil

    input_shape = input.ft_capacity_shape
    out_ranges = [range(s) for s in output_shape] if output_shape else [()]

    for out_coords in itertools.product(*out_ranges) if output_shape else [()]:
        out_coords = list(out_coords) if output_shape else []

        # Map to input coords
        in_coords = []
        out_dim = 0
        for indices, collapsed in per_dim:
            if collapsed:
                in_coords.append(indices[0])
            else:
                in_coords.append(indices[out_coords[out_dim]])
                out_dim += 1

        # Compute flat indices
        in_flat = _coords_to_flat(in_coords, input_shape)
        out_flat = _coords_to_flat(out_coords, output_shape) if output_shape else 0

        # Copy file
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
    import asyncio

    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
    from experience.future_tensor.status import Status

    print("Running 100 tests for slice_forward...\n")

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
        """Create a FutureTensor that's already materialized with given data."""
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

    # === Group 1: Basic 1D slicing (tests 1-20) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"elem_{i}" for i in range(10)]
        ft = make_forwarded_ft([10], data, tmpdir)

        # Test 1-5: slice(start, stop)
        r = slice_forward(ft, [slice(0, 5)])
        run_test("1D slice(0,5) shape", r.ft_capacity_shape == [5])
        run_test("1D slice(0,5) elem0", read_ft_element(r, 0) == "elem_0")
        run_test("1D slice(0,5) elem4", read_ft_element(r, 4) == "elem_4")
        run_test("1D slice(0,5) forwarded", r.ft_forwarded is True)
        run_test("1D slice(0,5) numel", r.ft_static_tensor.numel() == 5)

        # Test 6-10: slice(start, stop, step)
        r = slice_forward(ft, [slice(1, 9, 2)])
        run_test("1D slice(1,9,2) shape", r.ft_capacity_shape == [4])
        run_test("1D slice(1,9,2) elem0", read_ft_element(r, 0) == "elem_1")
        run_test("1D slice(1,9,2) elem1", read_ft_element(r, 1) == "elem_3")
        run_test("1D slice(1,9,2) elem2", read_ft_element(r, 2) == "elem_5")
        run_test("1D slice(1,9,2) elem3", read_ft_element(r, 3) == "elem_7")

        # Test 11-15: int index (collapse)
        r = slice_forward(ft, [3])
        run_test("1D int[3] shape", r.ft_capacity_shape == [])
        run_test("1D int[3] forwarded", r.ft_forwarded is True)

        # slice(None) = full
        r = slice_forward(ft, [slice(None)])
        run_test("1D slice(None) shape", r.ft_capacity_shape == [10])
        run_test("1D slice(None) elem0", read_ft_element(r, 0) == "elem_0")
        run_test("1D slice(None) elem9", read_ft_element(r, 9) == "elem_9")

        # Test 16-20: negative indexing
        r = slice_forward(ft, [slice(-3, None)])
        run_test("1D slice(-3,None) shape", r.ft_capacity_shape == [3])
        run_test("1D slice(-3,None) elem0", read_ft_element(r, 0) == "elem_7")
        run_test("1D slice(-3,None) elem2", read_ft_element(r, 2) == "elem_9")
        r = slice_forward(ft, [-1])
        run_test("1D int[-1] shape", r.ft_capacity_shape == [])
        r = slice_forward(ft, [-2])
        run_test("1D int[-2] shape", r.ft_capacity_shape == [])

    # === Group 2: 2D slicing (tests 21-45) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"r{i}c{j}" for i in range(4) for j in range(5)]
        ft = make_forwarded_ft([4, 5], data, tmpdir)

        # Test 21-25: slice first dim
        r = slice_forward(ft, [slice(0, 2), slice(None)])
        run_test("2D rows[0:2] shape", r.ft_capacity_shape == [2, 5])
        run_test("2D rows[0:2] [0,0]", read_ft_element(r, 0) == "r0c0")
        run_test("2D rows[0:2] [0,4]", read_ft_element(r, 4) == "r0c4")
        run_test("2D rows[0:2] [1,0]", read_ft_element(r, 5) == "r1c0")
        run_test("2D rows[0:2] [1,4]", read_ft_element(r, 9) == "r1c4")

        # Test 26-30: slice second dim
        r = slice_forward(ft, [slice(None), slice(1, 4)])
        run_test("2D cols[1:4] shape", r.ft_capacity_shape == [4, 3])
        run_test("2D cols[1:4] [0,0]", read_ft_element(r, 0) == "r0c1")
        run_test("2D cols[1:4] [0,2]", read_ft_element(r, 2) == "r0c3")
        run_test("2D cols[1:4] [3,0]", read_ft_element(r, 9) == "r3c1")
        run_test("2D cols[1:4] [3,2]", read_ft_element(r, 11) == "r3c3")

        # Test 31-35: int index on first dim (row select)
        r = slice_forward(ft, [2, slice(None)])
        run_test("2D row[2] shape", r.ft_capacity_shape == [5])
        run_test("2D row[2] elem0", read_ft_element(r, 0) == "r2c0")
        run_test("2D row[2] elem4", read_ft_element(r, 4) == "r2c4")
        r = slice_forward(ft, [0, slice(None)])
        run_test("2D row[0] elem0", read_ft_element(r, 0) == "r0c0")
        run_test("2D row[0] elem4", read_ft_element(r, 4) == "r0c4")

        # Test 36-40: int index on second dim (col select)
        r = slice_forward(ft, [slice(None), 3])
        run_test("2D col[3] shape", r.ft_capacity_shape == [4])
        run_test("2D col[3] elem0", read_ft_element(r, 0) == "r0c3")
        run_test("2D col[3] elem1", read_ft_element(r, 1) == "r1c3")
        run_test("2D col[3] elem2", read_ft_element(r, 2) == "r2c3")
        run_test("2D col[3] elem3", read_ft_element(r, 3) == "r3c3")

        # Test 41-45: both dims int (scalar)
        r = slice_forward(ft, [1, 2])
        run_test("2D [1,2] shape", r.ft_capacity_shape == [])
        r = slice_forward(ft, [3, 4])
        run_test("2D [3,4] shape", r.ft_capacity_shape == [])
        r = slice_forward(ft, [0, 0])
        run_test("2D [0,0] shape", r.ft_capacity_shape == [])
        r = slice_forward(ft, [slice(1, 3), slice(2, 4)])
        run_test("2D [1:3,2:4] shape", r.ft_capacity_shape == [2, 2])
        run_test("2D [1:3,2:4] [0,0]", read_ft_element(r, 0) == "r1c2")

    # === Group 3: 3D slicing (tests 46-60) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"d{d}r{r}c{c}" for d in range(3) for r in range(4) for c in range(2)]
        ft = make_forwarded_ft([3, 4, 2], data, tmpdir)

        r = slice_forward(ft, [0, slice(None), slice(None)])
        run_test("3D dim0=0 shape", r.ft_capacity_shape == [4, 2])
        run_test("3D dim0=0 [0,0]", read_ft_element(r, 0) == "d0r0c0")
        run_test("3D dim0=0 [3,1]", read_ft_element(r, 7) == "d0r3c1")

        r = slice_forward(ft, [slice(None), 2, slice(None)])
        run_test("3D dim1=2 shape", r.ft_capacity_shape == [3, 2])
        run_test("3D dim1=2 [0,0]", read_ft_element(r, 0) == "d0r2c0")
        run_test("3D dim1=2 [2,1]", read_ft_element(r, 5) == "d2r2c1")

        r = slice_forward(ft, [slice(None), slice(None), 1])
        run_test("3D dim2=1 shape", r.ft_capacity_shape == [3, 4])
        run_test("3D dim2=1 [0,0]", read_ft_element(r, 0) == "d0r0c1")
        run_test("3D dim2=1 [2,3]", read_ft_element(r, 11) == "d2r3c1")

        r = slice_forward(ft, [1, 2, 0])
        run_test("3D scalar [1,2,0] shape", r.ft_capacity_shape == [])

        r = slice_forward(ft, [slice(0, 2), slice(1, 3), slice(None)])
        run_test("3D [0:2,1:3,:] shape", r.ft_capacity_shape == [2, 2, 2])
        run_test("3D [0:2,1:3,:] [0,0,0]", read_ft_element(r, 0) == "d0r1c0")
        run_test("3D [0:2,1:3,:] [0,0,1]", read_ft_element(r, 1) == "d0r1c1")
        run_test("3D [0:2,1:3,:] [1,1,0]", read_ft_element(r, 6) == "d1r2c0")
        run_test("3D [0:2,1:3,:] [1,1,1]", read_ft_element(r, 7) == "d1r2c1")

    # === Group 4: Lazy slicing (not forwarded) with ft_async_get delegation (tests 61-75) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        received = []

        async def tracking_get(coords, trajactory):
            received.append(coords)
            return (f"val_{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, tracking_get, [sympy.Integer(6)])
        # NOT forwarded — slice should create lazy result
        r = slice_forward(ft, [slice(2, 5)])
        run_test("lazy slice shape", r.ft_capacity_shape == [3])
        run_test("lazy slice not forwarded", r.ft_forwarded is False)

        # Forward the sliced tensor
        prompt_t = st_make_tensor(["p0", "p1", "p2"], tmpdir)
        r.ft_forward(prompt_t)
        run_test("lazy slice forwarded after ft_forward", r.ft_forwarded is True)
        run_test("lazy mapped coord [0] -> [2]", received[0] == [2])
        run_test("lazy mapped coord [1] -> [3]", received[1] == [3])
        run_test("lazy mapped coord [2] -> [4]", received[2] == [4])
        run_test("lazy result elem0", read_ft_element(r, 0) == "val_[2]")
        run_test("lazy result elem2", read_ft_element(r, 2) == "val_[4]")

    with tempfile.TemporaryDirectory() as tmpdir:
        received2 = []

        async def tracking_get2(coords, prompt):
            received2.append(coords)
            return (f"v{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, tracking_get2, [sympy.Integer(4), sympy.Integer(3)])
        r = slice_forward(ft, [1, slice(None)])
        run_test("lazy 2D row select shape", r.ft_capacity_shape == [3])

        prompt_t = st_make_tensor(["a", "b", "c"], tmpdir)
        r.ft_forward(prompt_t)
        run_test("lazy 2D mapped [0] -> [1,0]", received2[0] == [1, 0])
        run_test("lazy 2D mapped [1] -> [1,1]", received2[1] == [1, 1])
        run_test("lazy 2D mapped [2] -> [1,2]", received2[2] == [1, 2])

    with tempfile.TemporaryDirectory() as tmpdir:
        async def step_get(coords, trajactory):
            return (str(coords), Status.confidence(1.0))

        ft = FutureTensor(tmpdir, step_get, [sympy.Integer(10)])
        r = slice_forward(ft, [slice(0, 10, 3)])
        run_test("lazy step=3 shape", r.ft_capacity_shape == [4])
        prompt_t = st_make_tensor(["p"] * 4, tmpdir)
        r.ft_forward(prompt_t)
        run_test("lazy step elem0 -> [0]", read_ft_element(r, 0) == "[0]")
        run_test("lazy step elem1 -> [3]", read_ft_element(r, 1) == "[3]")
        run_test("lazy step elem2 -> [6]", read_ft_element(r, 2) == "[6]")
        run_test("lazy step elem3 -> [9]", read_ft_element(r, 3) == "[9]")

    # === Group 5: Edge cases (tests 76-90) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # Single element
        ft = make_forwarded_ft([1], ["only"], tmpdir)
        r = slice_forward(ft, [slice(None)])
        run_test("single elem full slice shape", r.ft_capacity_shape == [1])
        run_test("single elem full slice val", read_ft_element(r, 0) == "only")

        r = slice_forward(ft, [0])
        run_test("single elem int index shape", r.ft_capacity_shape == [])

        # Empty slice
        ft = make_forwarded_ft([5], [f"x{i}" for i in range(5)], tmpdir)
        r = slice_forward(ft, [slice(3, 3)])
        run_test("empty slice shape", r.ft_capacity_shape == [0])

        # Reverse slice
        r = slice_forward(ft, [slice(4, 1, -1)])
        run_test("reverse slice shape", r.ft_capacity_shape == [3])
        run_test("reverse [0]", read_ft_element(r, 0) == "x4")
        run_test("reverse [1]", read_ft_element(r, 1) == "x3")
        run_test("reverse [2]", read_ft_element(r, 2) == "x2")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Large tensor
        n = 100
        data = [f"item_{i}" for i in range(n)]
        ft = make_forwarded_ft([n], data, tmpdir)
        r = slice_forward(ft, [slice(50, 60)])
        run_test("large tensor slice shape", r.ft_capacity_shape == [10])
        run_test("large tensor [0]", read_ft_element(r, 0) == "item_50")
        run_test("large tensor [9]", read_ft_element(r, 9) == "item_59")

        r = slice_forward(ft, [slice(0, 100, 10)])
        run_test("large step=10 shape", r.ft_capacity_shape == [10])
        run_test("large step=10 [0]", read_ft_element(r, 0) == "item_0")
        run_test("large step=10 [5]", read_ft_element(r, 5) == "item_50")
        run_test("large step=10 [9]", read_ft_element(r, 9) == "item_90")

    # === Group 6: Chained slices (tests 91-100) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"v{i}" for i in range(20)]
        ft = make_forwarded_ft([20], data, tmpdir)

        # Slice then slice again
        r1 = slice_forward(ft, [slice(5, 15)])
        run_test("chain: first slice shape", r1.ft_capacity_shape == [10])
        r2 = slice_forward(r1, [slice(2, 7)])
        run_test("chain: second slice shape", r2.ft_capacity_shape == [5])
        run_test("chain: [0] = v7", read_ft_element(r2, 0) == "v7")
        run_test("chain: [4] = v11", read_ft_element(r2, 4) == "v11")

    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"r{i}c{j}" for i in range(6) for j in range(6)]
        ft = make_forwarded_ft([6, 6], data, tmpdir)

        r1 = slice_forward(ft, [slice(1, 5), slice(None)])
        run_test("chain 2D: first shape", r1.ft_capacity_shape == [4, 6])
        r2 = slice_forward(r1, [slice(None), slice(2, 4)])
        run_test("chain 2D: second shape", r2.ft_capacity_shape == [4, 2])
        run_test("chain 2D: [0,0] = r1c2", read_ft_element(r2, 0) == "r1c2")
        run_test("chain 2D: [0,1] = r1c3", read_ft_element(r2, 1) == "r1c3")
        run_test("chain 2D: [3,0] = r4c2", read_ft_element(r2, 6) == "r4c2")
        run_test("chain 2D: [3,1] = r4c3", read_ft_element(r2, 7) == "r4c3")

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All slice_forward tests completed.")
