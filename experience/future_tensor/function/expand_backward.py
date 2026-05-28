"""
expand_backward :=
    FutureTensor
    <- $grad_output FutureTensor
    <- $input_shape list[int]
    <- $expanded_shape list[int]
    # inline

Backward pass of expand_forward: reduce (sum) along expanded dimensions.

Since expand broadcasts size-1 dims, the gradient must sum along those dims
to produce the correct grad for the original (unexpanded) tensor.
"""

import itertools
import os
from typing import List

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def expand_backward(
    grad_output: FutureTensor,
    input_shape: List[int],
    expanded_shape: List[int],
) -> FutureTensor:
    """Backward pass of expand: reduce along expanded dimensions.

    For each position in the original (unexpanded) shape, the gradient is the
    sum of all positions that were broadcast to it during expand.

    Args:
        grad_output: Gradient FutureTensor with shape == expanded_shape.
        input_shape: Original (pre-expand) shape.
        expanded_shape: Shape after expand (== grad_output.ft_capacity_shape).

    Returns:
        FutureTensor with shape == input_shape.
    """
    ndim_diff = len(expanded_shape) - len(input_shape)

    # Identify which dims were expanded (broadcast from size 1)
    expanded_dims = []
    for d in range(len(expanded_shape)):
        if d < ndim_diff:
            # Prepended dim — always expanded
            expanded_dims.append(d)
        else:
            in_d = d - ndim_diff
            if input_shape[in_d] == 1 and expanded_shape[d] > 1:
                expanded_dims.append(d)

    async def reduced_async_get(coordinates: List[int], trajactory: str):
        """Sum over all expanded positions that map to these input coords."""
        # For each expanded dim, iterate over all positions in that dim;
        # for non-expanded dims, use the coordinate directly.
        # Since this is symbolic text (not numeric), we just take the first
        # successful result — symbolic "sum" is not meaningful for text.
        # The numeric coefficient sum is handled by the data copy below.
        out_coords = [0] * len(expanded_shape)
        for d in range(len(expanded_shape)):
            if d < ndim_diff:
                out_coords[d] = 0
            else:
                in_d = d - ndim_diff
                if input_shape[in_d] == 1:
                    out_coords[d] = 0
                else:
                    out_coords[d] = coordinates[in_d]

        return await grad_output.ft_async_get(out_coords, trajactory)

    result = FutureTensor(
        grad_output.ft_initial_static_tensor.st_relative_to,
        reduced_async_get,
        [sympy.Integer(s) for s in input_shape],
    )

    # If grad_output is forwarded, reduce storage immediately
    if grad_output.ft_forwarded:
        _reduce_storage(grad_output, result, input_shape, expanded_shape, ndim_diff)
        result.ft_forwarded = True

    return result


def _reduce_storage(grad_output, result, input_shape, expanded_shape, ndim_diff):
    """Reduce (pick first valid) from grad_output to result along expanded dims."""
    import shutil

    # For each input position, find one representative output position and copy
    for in_coords in itertools.product(*[range(s) for s in input_shape]) if input_shape else [()]:
        in_coords = list(in_coords) if input_shape else []

        # Map to the first output position (broadcast dims get index 0)
        out_coords = []
        for d in range(len(expanded_shape)):
            if d < ndim_diff:
                out_coords.append(0)
            else:
                in_d = d - ndim_diff
                if input_shape[in_d] == 1:
                    out_coords.append(0)
                else:
                    out_coords.append(in_coords[in_d])

        in_flat = _coords_to_flat(in_coords, input_shape) if input_shape else 0
        out_flat = _coords_to_flat(out_coords, expanded_shape)

        src_path = _storage_path(grad_output, out_flat)
        dst_path = _storage_path(result, in_flat)
        if os.path.isfile(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            # Sum coefficients along expanded dims
            coeff_sum = _sum_coefficients(
                grad_output, in_coords, input_shape, expanded_shape, ndim_diff
            )
            result.ft_initial_static_tensor.data.flatten()[in_flat] = coeff_sum


def _sum_coefficients(grad_output, in_coords, input_shape, expanded_shape, ndim_diff):
    """Sum numeric coefficients over all output positions mapping to in_coords."""
    # Build ranges for each expanded dim
    ranges = []
    for d in range(len(expanded_shape)):
        if d < ndim_diff:
            ranges.append(range(expanded_shape[d]))
        else:
            in_d = d - ndim_diff
            if input_shape[in_d] == 1 and expanded_shape[d] > 1:
                ranges.append(range(expanded_shape[d]))
            else:
                ranges.append([in_coords[in_d]])

    total = 0.0
    for out_coords in itertools.product(*ranges):
        out_flat = _coords_to_flat(list(out_coords), expanded_shape)
        total += grad_output.ft_initial_static_tensor.data.flatten()[out_flat].item()
    return total


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
    import tempfile

    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running tests for expand_backward...\n")

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

    def make_forwarded_ft(shape, data_list, tmpdir, coeffs=None):
        async def dummy_get(coords, trajactory):
            return ("unused", Status.confidence(1.0))
        ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(s) for s in shape])
        nested = _unflatten_data(data_list, shape)
        result_tensor = st_make_tensor(nested, tmpdir)
        assign_tensor(ft.ft_initial_static_tensor, result_tensor)
        if coeffs is not None:
            for i, c in enumerate(coeffs):
                ft.ft_initial_static_tensor.data.flatten()[i] = c
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

    # === Group 1: Basic reduce ===
    print("Group 1: Basic reduce")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Expanded [1,3] -> [4,3], backward should reduce dim 0
        grad = make_forwarded_ft(
            [4, 3],
            [f"g{i}{j}" for i in range(4) for j in range(3)],
            tmpdir,
            coeffs=[1.0] * 12,
        )
        r = expand_backward(grad, [1, 3], [4, 3])
        run_test("reduce shape [1,3]", r.ft_capacity_shape == [1, 3])
        run_test("reduce forwarded", r.ft_forwarded is True)
        # Content from first representative (row 0)
        run_test("[0,0] content = g00", read_ft_element(r, 0) == "g00")
        run_test("[0,1] content = g01", read_ft_element(r, 1) == "g01")
        # Coefficient should be sum over dim 0: 4 * 1.0 = 4.0
        run_test("[0,0] coeff = 4.0",
                 abs(r.ft_initial_static_tensor.data.flatten()[0].item() - 4.0) < 0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Expanded [3,1] -> [3,5], backward should reduce dim 1
        grad = make_forwarded_ft(
            [3, 5],
            [f"g{i}{j}" for i in range(3) for j in range(5)],
            tmpdir,
            coeffs=[0.5] * 15,
        )
        r = expand_backward(grad, [3, 1], [3, 5])
        run_test("reduce dim1 shape [3,1]", r.ft_capacity_shape == [3, 1])
        run_test("[0,0] content = g00", read_ft_element(r, 0) == "g00")
        run_test("[2,0] content = g20", read_ft_element(r, 2) == "g20")
        # Coefficient should be sum over dim 1: 5 * 0.5 = 2.5
        run_test("[0,0] coeff = 2.5",
                 abs(r.ft_initial_static_tensor.data.flatten()[0].item() - 2.5) < 0.1)

    # === Group 2: Prepended dim reduce ===
    print("\nGroup 2: Prepended dim reduce")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Expanded [3] -> [2,3], backward should reduce prepended dim
        grad = make_forwarded_ft(
            [2, 3],
            [f"g{i}{j}" for i in range(2) for j in range(3)],
            tmpdir,
            coeffs=[1.0] * 6,
        )
        r = expand_backward(grad, [3], [2, 3])
        run_test("prepend reduce shape [3]", r.ft_capacity_shape == [3])
        run_test("[0] content = g00", read_ft_element(r, 0) == "g00")
        run_test("[2] content = g02", read_ft_element(r, 2) == "g02")
        # Coeff: sum over prepended dim: 2 * 1.0 = 2.0
        run_test("[0] coeff = 2.0",
                 abs(r.ft_initial_static_tensor.data.flatten()[0].item() - 2.0) < 0.1)

    # === Group 3: No-op (same shape) ===
    print("\nGroup 3: No-op reduce")

    with tempfile.TemporaryDirectory() as tmpdir:
        grad = make_forwarded_ft([2, 3], [f"g{i}" for i in range(6)], tmpdir, coeffs=[1.0] * 6)
        r = expand_backward(grad, [2, 3], [2, 3])
        run_test("noop shape [2,3]", r.ft_capacity_shape == [2, 3])
        run_test("noop [0] = g0", read_ft_element(r, 0) == "g0")
        run_test("noop coeff [0] = 1.0",
                 abs(r.ft_initial_static_tensor.data.flatten()[0].item() - 1.0) < 0.1)

    # === Group 4: Lazy reduce ===
    print("\nGroup 4: Lazy reduce")

    with tempfile.TemporaryDirectory() as tmpdir:
        async def lazy_get(coords, trajactory):
            return (f"lazy_{coords}", Status.confidence(0.8))

        grad = FutureTensor(tmpdir, lazy_get, [sympy.Integer(4), sympy.Integer(3)])
        r = expand_backward(grad, [1, 3], [4, 3])
        run_test("lazy not forwarded", r.ft_forwarded is False)

        prompt_t = st_make_tensor(["p"] * 3, tmpdir)
        r.ft_forward(prompt_t)
        run_test("lazy forwarded", r.ft_forwarded is True)
        run_test("lazy [0,0] calls [0,0]", read_ft_element(r, 0) == "lazy_[0, 0]")

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All expand_backward tests passed.")
    else:
        import sys
        sys.exit(1)
