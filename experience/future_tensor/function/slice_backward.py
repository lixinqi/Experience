"""
slice_backward :=
    FutureTensor
    <- FutureTensor
    <- list[slice]
    # inline

# the same behavior with torch slice
# Backward of slice is unsqueeze (pad zeros in dropped positions).
"""

import itertools
import os
from typing import List, Union

import torch

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def slice_backward(
    grad_output: FutureTensor,
    original_shape: List[int],
    slices: List[Union[int, slice]],
) -> FutureTensor:
    """Backward pass of slice: scatter grad_output back into original shape.

    Creates a FutureTensor of original_shape where sliced positions receive
    grad_output values and non-sliced positions get None/empty content.

    Args:
        grad_output: The gradient FutureTensor from upstream (shape = sliced shape).
        original_shape: The shape of the original (pre-slice) tensor.
        slices: The slice spec used in the forward pass.
    """
    from experience.future_tensor.function.slice_forward import _resolve_slice

    # Resolve slices
    full_slices = list(slices)
    while len(full_slices) < len(original_shape):
        full_slices.append(slice(None))

    per_dim = []
    for d, s in enumerate(full_slices):
        indices, collapsed = _resolve_slice(s, original_shape[d])
        per_dim.append((indices, collapsed))

    # Build reverse mapping: for each position in original_shape,
    # determine if it maps to a position in grad_output
    def reverse_map(in_coords: List[int]):
        """Map input coords -> output coords, or None if not in slice."""
        out_coords = []
        for d, (indices, collapsed) in enumerate(per_dim):
            c = in_coords[d]
            if c not in indices:
                return None
            if not collapsed:
                out_coords.append(indices.index(c))
        return out_coords

    async def scatter_async_get(coordinates: List[int], trajactory: str):
        out_coords = reverse_map(coordinates)
        if out_coords is None:
            return ("", Status.confidence(0.0))  # Zero/empty for positions not in the slice
        return await grad_output.ft_async_get(out_coords, trajactory)

    result = FutureTensor(grad_output.ft_initial_static_tensor.st_relative_to, scatter_async_get, [sympy.Integer(s) for s in original_shape])

    # If grad_output is forwarded, materialize immediately
    if grad_output.ft_forwarded:
        _scatter_storage(grad_output, result, original_shape, per_dim)
        result.ft_forwarded = True

    return result


def _scatter_storage(grad_output, result, original_shape, per_dim):
    """Scatter grad_output storage back to result at original positions."""
    import shutil

    out_shape = grad_output.ft_capacity_shape

    for out_coords in itertools.product(*[range(s) for s in out_shape]) if out_shape else [()]:
        out_coords = list(out_coords) if out_shape else []

        # Map output coords back to input coords
        in_coords = []
        out_dim = 0
        for indices, collapsed in per_dim:
            if collapsed:
                in_coords.append(indices[0])
            else:
                in_coords.append(indices[out_coords[out_dim]])
                out_dim += 1

        out_flat = _coords_to_flat(out_coords, out_shape) if out_shape else 0
        in_flat = _coords_to_flat(in_coords, original_shape)

        src_path = _storage_path(grad_output, out_flat)
        dst_path = _storage_path(result, in_flat)
        if os.path.isfile(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            # Copy coefficient (confidence) from grad_output
            result.ft_initial_static_tensor.data.flatten()[in_flat] = grad_output.ft_initial_static_tensor.data.flatten()[out_flat]


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
        ft.ft_initial_static_tensor.st_relative_to, ft.ft_initial_static_tensor.st_tensor_uid,
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

    print("Running 100 tests for slice_backward...\n")

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

    # === Group 1: Basic 1D scatter (tests 1-20) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # Grad output from slicing [2:5] of a [10] tensor
        grad = make_forwarded_ft([3], ["g0", "g1", "g2"], tmpdir)
        r = slice_backward(grad, [10], [slice(2, 5)])
        run_test("1D scatter shape", r.ft_capacity_shape == [10])
        run_test("1D scatter forwarded", r.ft_forwarded is True)
        run_test("1D scatter [0] empty", read_ft_element(r, 0) is None)
        run_test("1D scatter [1] empty", read_ft_element(r, 1) is None)
        run_test("1D scatter [2] = g0", read_ft_element(r, 2) == "g0")
        run_test("1D scatter [3] = g1", read_ft_element(r, 3) == "g1")
        run_test("1D scatter [4] = g2", read_ft_element(r, 4) == "g2")
        run_test("1D scatter [5] empty", read_ft_element(r, 5) is None)
        run_test("1D scatter [9] empty", read_ft_element(r, 9) is None)

        # Coeff check
        run_test("1D coeff [2] = 1", r.ft_initial_static_tensor.data.flatten()[2].item() == 1.0)
        run_test("1D coeff [0] = 0", r.ft_initial_static_tensor.data.flatten()[0].item() == 0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step slice backward
        grad = make_forwarded_ft([4], ["s0", "s1", "s2", "s3"], tmpdir)
        r = slice_backward(grad, [10], [slice(1, 9, 2)])
        run_test("1D step scatter shape", r.ft_capacity_shape == [10])
        run_test("1D step [0] empty", read_ft_element(r, 0) is None)
        run_test("1D step [1] = s0", read_ft_element(r, 1) == "s0")
        run_test("1D step [2] empty", read_ft_element(r, 2) is None)
        run_test("1D step [3] = s1", read_ft_element(r, 3) == "s1")
        run_test("1D step [5] = s2", read_ft_element(r, 5) == "s2")
        run_test("1D step [7] = s3", read_ft_element(r, 7) == "s3")
        run_test("1D step [9] empty", read_ft_element(r, 9) is None)
        run_test("1D step coeff [1]", r.ft_initial_static_tensor.data.flatten()[1].item() == 1.0)

    # === Group 2: 2D scatter (tests 21-45) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # Forward sliced rows [1:3] from [4,5] -> grad shape [2,5]
        grad_data = [f"g{i}{j}" for i in range(2) for j in range(5)]
        grad = make_forwarded_ft([2, 5], grad_data, tmpdir)
        r = slice_backward(grad, [4, 5], [slice(1, 3), slice(None)])
        run_test("2D row scatter shape", r.ft_capacity_shape == [4, 5])
        run_test("2D row [0,0] empty", read_ft_element(r, 0) is None)  # row 0
        run_test("2D row [1,0] = g00", read_ft_element(r, 5) == "g00")
        run_test("2D row [1,4] = g04", read_ft_element(r, 9) == "g04")
        run_test("2D row [2,0] = g10", read_ft_element(r, 10) == "g10")
        run_test("2D row [2,4] = g14", read_ft_element(r, 14) == "g14")
        run_test("2D row [3,0] empty", read_ft_element(r, 15) is None)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Forward sliced cols [2:4] from [4,5] -> grad shape [4,2]
        grad_data = [f"g{i}{j}" for i in range(4) for j in range(2)]
        grad = make_forwarded_ft([4, 2], grad_data, tmpdir)
        r = slice_backward(grad, [4, 5], [slice(None), slice(2, 4)])
        run_test("2D col scatter shape", r.ft_capacity_shape == [4, 5])
        run_test("2D col [0,0] empty", read_ft_element(r, 0) is None)
        run_test("2D col [0,1] empty", read_ft_element(r, 1) is None)
        run_test("2D col [0,2] = g00", read_ft_element(r, 2) == "g00")
        run_test("2D col [0,3] = g01", read_ft_element(r, 3) == "g01")
        run_test("2D col [0,4] empty", read_ft_element(r, 4) is None)
        run_test("2D col [1,2] = g10", read_ft_element(r, 7) == "g10")
        run_test("2D col [3,3] = g31", read_ft_element(r, 18) == "g31")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Int index on dim0: slice [2, :] from [4,3] -> grad shape [3]
        grad = make_forwarded_ft([3], ["ga", "gb", "gc"], tmpdir)
        r = slice_backward(grad, [4, 3], [2, slice(None)])
        run_test("2D int dim0 scatter shape", r.ft_capacity_shape == [4, 3])
        run_test("2D int [0,0] empty", read_ft_element(r, 0) is None)
        run_test("2D int [2,0] = ga", read_ft_element(r, 6) == "ga")
        run_test("2D int [2,1] = gb", read_ft_element(r, 7) == "gb")
        run_test("2D int [2,2] = gc", read_ft_element(r, 8) == "gc")
        run_test("2D int [3,0] empty", read_ft_element(r, 9) is None)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Int index on dim1: slice [:,1] from [3,4] -> grad shape [3]
        grad = make_forwarded_ft([3], ["x", "y", "z"], tmpdir)
        r = slice_backward(grad, [3, 4], [slice(None), 1])
        run_test("2D int dim1 scatter shape", r.ft_capacity_shape == [3, 4])
        run_test("2D int dim1 [0,0] empty", read_ft_element(r, 0) is None)
        run_test("2D int dim1 [0,1] = x", read_ft_element(r, 1) == "x")
        run_test("2D int dim1 [1,1] = y", read_ft_element(r, 5) == "y")
        run_test("2D int dim1 [2,1] = z", read_ft_element(r, 9) == "z")
        run_test("2D int dim1 [2,3] empty", read_ft_element(r, 11) is None)

    # === Group 3: 3D scatter (tests 46-60) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # Slice [0,:,:] from [3,4,2] -> grad shape [4,2]
        grad_data = [f"g{i}{j}" for i in range(4) for j in range(2)]
        grad = make_forwarded_ft([4, 2], grad_data, tmpdir)
        r = slice_backward(grad, [3, 4, 2], [0, slice(None), slice(None)])
        run_test("3D dim0=0 scatter shape", r.ft_capacity_shape == [3, 4, 2])
        run_test("3D [0,0,0] = g00", read_ft_element(r, 0) == "g00")
        run_test("3D [0,3,1] = g31", read_ft_element(r, 7) == "g31")
        run_test("3D [1,0,0] empty", read_ft_element(r, 8) is None)
        run_test("3D [2,3,1] empty", read_ft_element(r, 23) is None)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Slice [:,:,0] from [2,3,4] -> grad shape [2,3]
        grad_data = [f"g{i}{j}" for i in range(2) for j in range(3)]
        grad = make_forwarded_ft([2, 3], grad_data, tmpdir)
        r = slice_backward(grad, [2, 3, 4], [slice(None), slice(None), 0])
        run_test("3D dim2=0 scatter shape", r.ft_capacity_shape == [2, 3, 4])
        run_test("3D [0,0,0] = g00", read_ft_element(r, 0) == "g00")
        run_test("3D [0,0,1] empty", read_ft_element(r, 1) is None)
        run_test("3D [0,1,0] = g01", read_ft_element(r, 4) == "g01")
        run_test("3D [1,2,0] = g12", read_ft_element(r, 20) == "g12")
        run_test("3D [1,2,3] empty", read_ft_element(r, 23) is None)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Slice [1,2,:] from [3,4,5] -> scalar extraction -> grad shape [5]
        grad = make_forwarded_ft([5], [f"e{i}" for i in range(5)], tmpdir)
        r = slice_backward(grad, [3, 4, 5], [1, 2, slice(None)])
        run_test("3D two-int scatter shape", r.ft_capacity_shape == [3, 4, 5])
        # Position [1,2,*] = flat indices: 1*20 + 2*5 + k = 30+k
        run_test("3D [1,2,0] = e0", read_ft_element(r, 30) == "e0")
        run_test("3D [1,2,4] = e4", read_ft_element(r, 34) == "e4")
        run_test("3D [0,0,0] empty", read_ft_element(r, 0) is None)
        run_test("3D [2,3,4] empty", read_ft_element(r, 59) is None)

    # === Group 4: Lazy backward (tests 61-75) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        received = []

        async def lazy_grad_get(coords, trajactory):
            received.append(coords)
            return (f"lazy_{coords}", Status.confidence(1.0))

        grad = FutureTensor(tmpdir, lazy_grad_get, [sympy.Integer(3)])
        # Don't forward grad — backward should be lazy too
        r = slice_backward(grad, [8], [slice(2, 5)])
        run_test("lazy backward shape", r.ft_capacity_shape == [8])
        run_test("lazy backward not forwarded", r.ft_forwarded is False)

        # Forward the result
        prompt_t = st_make_tensor(["p"] * 8, tmpdir)
        r.ft_forward(prompt_t)
        run_test("lazy backward forwarded after ft_forward", r.ft_forwarded is True)
        # Position [2] in original maps to [0] in grad
        run_test("lazy backward elem2", read_ft_element(r, 2) == "lazy_[0]")
        run_test("lazy backward elem3", read_ft_element(r, 3) == "lazy_[1]")
        run_test("lazy backward elem4", read_ft_element(r, 4) == "lazy_[2]")
        # Non-sliced positions get empty string
        run_test("lazy backward elem0 empty", read_ft_element(r, 0) == "")
        run_test("lazy backward elem7 empty", read_ft_element(r, 7) == "")

    with tempfile.TemporaryDirectory() as tmpdir:
        async def lazy_2d_get(coords, trajactory):
            return (f"g{coords}", Status.confidence(1.0))

        grad = FutureTensor(tmpdir, lazy_2d_get, [sympy.Integer(2), sympy.Integer(2)])
        r = slice_backward(grad, [4, 4], [slice(1, 3), slice(0, 2)])
        prompt_t = st_make_tensor(["p"] * 16, tmpdir)
        r.ft_forward(prompt_t)
        # [1,0] -> grad[0,0]
        run_test("lazy 2D [1,0] = g[0,0]", read_ft_element(r, 4) == "g[0, 0]")
        # [2,1] -> grad[1,1]
        run_test("lazy 2D [2,1] = g[1,1]", read_ft_element(r, 9) == "g[1, 1]")
        # [0,0] -> not in slice
        run_test("lazy 2D [0,0] empty", read_ft_element(r, 0) == "")
        # [3,3] -> not in slice
        run_test("lazy 2D [3,3] empty", read_ft_element(r, 15) == "")

    # === Group 5: Negative indexing and edge cases (tests 76-90) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        grad = make_forwarded_ft([3], ["a", "b", "c"], tmpdir)
        r = slice_backward(grad, [5], [slice(-3, None)])
        run_test("neg slice scatter shape", r.ft_capacity_shape == [5])
        run_test("neg [0] empty", read_ft_element(r, 0) is None)
        run_test("neg [1] empty", read_ft_element(r, 1) is None)
        run_test("neg [2] = a", read_ft_element(r, 2) == "a")
        run_test("neg [3] = b", read_ft_element(r, 3) == "b")
        run_test("neg [4] = c", read_ft_element(r, 4) == "c")

    with tempfile.TemporaryDirectory() as tmpdir:
        grad = make_forwarded_ft([2], ["m", "n"], tmpdir)
        r = slice_backward(grad, [6], [slice(4, 0, -2)])
        run_test("neg step scatter shape", r.ft_capacity_shape == [6])
        # slice(4, 0, -2) on size 6 -> indices [4, 2]
        run_test("neg step [4] = m", read_ft_element(r, 4) == "m")
        run_test("neg step [2] = n", read_ft_element(r, 2) == "n")
        run_test("neg step [0] empty", read_ft_element(r, 0) is None)
        run_test("neg step [3] empty", read_ft_element(r, 3) is None)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Scalar backward: int index on all dims
        grad = make_forwarded_ft([], ["scalar_grad"], tmpdir)
        r = slice_backward(grad, [3, 4], [1, 2])
        run_test("scalar backward shape", r.ft_capacity_shape == [3, 4])
        # Position [1,2] = flat 1*4+2 = 6
        run_test("scalar [1,2] = scalar_grad", read_ft_element(r, 6) == "scalar_grad")
        run_test("scalar [0,0] empty", read_ft_element(r, 0) is None)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Full slice backward (identity)
        grad = make_forwarded_ft([5], [f"f{i}" for i in range(5)], tmpdir)
        r = slice_backward(grad, [5], [slice(None)])
        run_test("identity backward shape", r.ft_capacity_shape == [5])
        for i in range(5):
            run_test(f"identity [{i}] = f{i}", read_ft_element(r, i) == f"f{i}")

    # === Group 6: Round-trip (forward + backward) consistency (tests 91-100) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.slice_forward import slice_forward

        data = [f"orig_{i}" for i in range(10)]
        ft = make_forwarded_ft([10], data, tmpdir)

        # Forward slice
        slices_spec = [slice(3, 7)]
        fwd = slice_forward(ft, slices_spec)
        run_test("roundtrip fwd shape", fwd.ft_capacity_shape == [4])
        run_test("roundtrip fwd [0]", read_ft_element(fwd, 0) == "orig_3")

        # Backward scatter
        grad_data = ["grad_0", "grad_1", "grad_2", "grad_3"]
        grad_ft = make_forwarded_ft([4], grad_data, tmpdir)
        bwd = slice_backward(grad_ft, [10], slices_spec)
        run_test("roundtrip bwd shape", bwd.ft_capacity_shape == [10])
        run_test("roundtrip bwd [3] = grad_0", read_ft_element(bwd, 3) == "grad_0")
        run_test("roundtrip bwd [6] = grad_3", read_ft_element(bwd, 6) == "grad_3")
        run_test("roundtrip bwd [0] empty", read_ft_element(bwd, 0) is None)
        run_test("roundtrip bwd [7] empty", read_ft_element(bwd, 7) is None)

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.slice_forward import slice_forward

        data = [f"r{i}c{j}" for i in range(5) for j in range(5)]
        ft = make_forwarded_ft([5, 5], data, tmpdir)
        slices_spec = [slice(1, 4), slice(2, 5)]
        fwd = slice_forward(ft, slices_spec)
        run_test("roundtrip 2D fwd shape", fwd.ft_capacity_shape == [3, 3])

        grad_data = [f"g{i}{j}" for i in range(3) for j in range(3)]
        grad_ft = make_forwarded_ft([3, 3], grad_data, tmpdir)
        bwd = slice_backward(grad_ft, [5, 5], slices_spec)
        run_test("roundtrip 2D bwd shape", bwd.ft_capacity_shape == [5, 5])
        # [1,2] = flat 7, should be g00
        run_test("roundtrip 2D [1,2] = g00", read_ft_element(bwd, 7) == "g00")
        # [3,4] = flat 19, should be g22
        run_test("roundtrip 2D [3,4] = g22", read_ft_element(bwd, 19) == "g22")

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All slice_backward tests completed.")
