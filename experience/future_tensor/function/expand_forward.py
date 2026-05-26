"""
expand_forward :=
    FutureTensor
    <- FutureTensor
    <- list[int | sympy.Symbol]
    # inline

# the same behavior with torch expand
# When target_shape contains sympy.Symbol, the output has capacity 0 on that
# axis (symbolic/dynamic dim). ft_async_get still maps coords back to input.
"""

import itertools
import os
from typing import List, Union

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor


def _is_symbolic(dim) -> bool:
    """Check if a dimension is symbolic (sympy expression, not concrete int)."""
    return isinstance(dim, sympy.Basic) and not isinstance(dim, sympy.Integer)


def expand_forward(input: FutureTensor, target_shape: List[Union[int, sympy.Basic]]) -> FutureTensor:
    """Expand a FutureTensor -- broadcast dims of size 1 to target size.

    Same semantics as torch.Tensor.expand(). Dimensions of size 1 in the input
    are broadcast to the corresponding size in target_shape. Passing -1 for a
    dimension means keeping the current size.

    When a target dim is a sympy.Symbol, the output has capacity 0 on that axis
    (symbolic/dynamic dim that grows via ft_incremental_concated_tensors).

    Args:
        input: Source FutureTensor.
        target_shape: Desired output shape. Use -1 to keep a dim unchanged.
            May contain sympy.Symbol for symbolic (dynamic) dimensions.

    Returns:
        A new FutureTensor with the expanded shape.
    """
    input_shape = input.ft_capacity_shape

    # Allow target_shape to have more dims than input (prepend size-1 dims)
    ndim_diff = len(target_shape) - len(input_shape)
    if ndim_diff < 0:
        raise ValueError(
            f"expand: target_shape has fewer dims ({len(target_shape)}) "
            f"than input ({len(input_shape)})"
        )
    padded_input_shape = [1] * ndim_diff + input_shape

    # Resolve -1 and validate; build output_shape (capacity) and output_schema
    output_shape = []  # concrete capacity (0 for symbolic dims)
    output_schema = []  # preserves sympy.Symbol for symbolic dims
    for d, (t, i) in enumerate(zip(target_shape, padded_input_shape)):
        if _is_symbolic(t):
            # Symbolic dim: capacity is 0, schema preserves the symbol
            output_shape.append(0)
            output_schema.append(t)
        elif t == -1:
            output_shape.append(i)
            output_schema.append(sympy.Integer(i))
        elif i == 1:
            t_int = int(t) if isinstance(t, sympy.Integer) else t
            output_shape.append(t_int)
            output_schema.append(sympy.Integer(t_int))
        elif i == t or (isinstance(t, sympy.Integer) and int(t) == i):
            t_int = int(t) if isinstance(t, sympy.Integer) else t
            output_shape.append(t_int)
            output_schema.append(sympy.Integer(t_int))
        else:
            raise ValueError(
                f"expand: cannot expand dim {d} from size {i} to {t} "
                f"(only size-1 dims can be expanded)"
            )

    # Build coordinate mapping: output_coords -> input_coords
    def map_coords(out_coords: List[int]) -> List[int]:
        in_coords = []
        for d in range(ndim_diff, len(output_shape)):
            in_d = d - ndim_diff
            if input_shape[in_d] == 1:
                in_coords.append(0)
            else:
                in_coords.append(out_coords[d])
        return in_coords

    async def expanded_async_get(coordinates: List[int], trajactory: str):
        original_coords = map_coords(coordinates)
        if input.ft_forwarded:
            coeff, filepath = input.ft_get_materialized_value(original_coords)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    from experience.future_tensor.status import Status
                    return (f.read(), Status.confidence(coeff))
            from experience.future_tensor.status import Status
            return ("", Status.confidence(0.0))
        return await input.ft_async_get(original_coords, trajactory)

    result = FutureTensor(
        input.ft_static_tensor.st_relative_to,
        expanded_async_get,
        list(output_schema),
    )
    result.ft_capacity_shape = list(output_shape)

    # If input is already forwarded AND no symbolic dims, copy storage directly
    has_symbolic = any(_is_symbolic(t) for t in target_shape)
    if input.ft_forwarded and not has_symbolic:
        _copy_expanded_storage(input, result, output_shape, padded_input_shape, ndim_diff)
        result.ft_forwarded = True
    # Symbolic dims: don't mark forwarded — no concrete storage to back
    # ft_get_materialized_value. Downstream ops use ft_async_get which works.

    return result


def _copy_expanded_storage(input, output, output_shape, padded_input_shape, ndim_diff):
    """Copy element storage from input to output, broadcasting size-1 dims."""
    import shutil

    input_shape = input.ft_capacity_shape

    for out_coords in itertools.product(*[range(s) for s in output_shape]) if output_shape else [()]:
        out_coords = list(out_coords) if output_shape else []

        # Map to input coords
        in_coords = []
        for d in range(ndim_diff, len(output_shape)):
            in_d = d - ndim_diff
            if input_shape[in_d] == 1:
                in_coords.append(0)
            else:
                in_coords.append(out_coords[d])

        in_flat = _coords_to_flat(in_coords, input_shape) if input_shape else 0
        out_flat = _coords_to_flat(out_coords, output_shape)

        src_path = _storage_path(input, in_flat)
        dst_path = _storage_path(output, out_flat)
        if os.path.isfile(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            output.ft_static_tensor.data.flatten()[out_flat] = (
                input.ft_static_tensor.data.flatten()[in_flat]
            )


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
    import tempfile

    from experience.future_tensor.status import Status
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running tests for expand_forward...\n")

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

    # === Group 1: Basic expand (1D -> broadcast) ===
    print("Group 1: Basic expand")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1, 3], ["a", "b", "c"], tmpdir)
        r = expand_forward(ft, [4, 3])
        run_test("expand [1,3]->[4,3] shape", r.ft_capacity_shape == [4, 3])
        run_test("expand forwarded", r.ft_forwarded is True)
        run_test("[0,0]=a", read_ft_element(r, 0) == "a")
        run_test("[0,2]=c", read_ft_element(r, 2) == "c")
        run_test("[3,0]=a (broadcast)", read_ft_element(r, 9) == "a")
        run_test("[3,2]=c (broadcast)", read_ft_element(r, 11) == "c")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3, 1], ["x", "y", "z"], tmpdir)
        r = expand_forward(ft, [3, 5])
        run_test("expand [3,1]->[3,5] shape", r.ft_capacity_shape == [3, 5])
        run_test("[0,0]=x", read_ft_element(r, 0) == "x")
        run_test("[0,4]=x (broadcast)", read_ft_element(r, 4) == "x")
        run_test("[2,0]=z", read_ft_element(r, 10) == "z")
        run_test("[2,4]=z (broadcast)", read_ft_element(r, 14) == "z")

    # === Group 2: -1 keeps size ===
    print("\nGroup 2: -1 keeps size")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1, 4], ["a", "b", "c", "d"], tmpdir)
        r = expand_forward(ft, [3, -1])
        run_test("expand [1,4]->[3,-1] shape", r.ft_capacity_shape == [3, 4])
        run_test("[2,3]=d (broadcast)", read_ft_element(r, 11) == "d")

    # === Group 3: Prepend dims ===
    print("\nGroup 3: Prepend dims")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3], ["p", "q", "r"], tmpdir)
        r = expand_forward(ft, [2, 3])
        run_test("expand [3]->[2,3] shape", r.ft_capacity_shape == [2, 3])
        run_test("[0,0]=p", read_ft_element(r, 0) == "p")
        run_test("[1,2]=r (broadcast)", read_ft_element(r, 5) == "r")

    # === Group 4: No-op expand (same shape) ===
    print("\nGroup 4: No-op expand")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2, 3], [f"e{i}" for i in range(6)], tmpdir)
        r = expand_forward(ft, [2, 3])
        run_test("expand [2,3]->[2,3] shape", r.ft_capacity_shape == [2, 3])
        run_test("[0,0]=e0", read_ft_element(r, 0) == "e0")
        run_test("[1,2]=e5", read_ft_element(r, 5) == "e5")

    # === Group 5: Lazy expand ===
    print("\nGroup 5: Lazy expand")

    with tempfile.TemporaryDirectory() as tmpdir:
        calls = []

        async def tracking_get(coords, trajactory):
            calls.append(coords)
            return (f"val_{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, tracking_get, [sympy.Integer(1), sympy.Integer(2)])
        r = expand_forward(ft, [3, 2])
        run_test("lazy expand not forwarded", r.ft_forwarded is False)

        prompt_t = st_make_tensor([["p"] * 2] * 3, tmpdir)
        r.ft_forward(prompt_t)
        run_test("lazy expand forwarded", r.ft_forwarded is True)
        # All rows should delegate to row 0 of input
        run_test("lazy [0,0] calls input [0,0]", read_ft_element(r, 0) == "val_[0, 0]")
        run_test("lazy [2,1] calls input [0,1]", read_ft_element(r, 5) == "val_[0, 1]")

    # === Group 6: Error cases ===
    print("\nGroup 6: Error cases")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3, 2], ["a"] * 6, tmpdir)
        try:
            expand_forward(ft, [3])
            run_test("fewer dims raises ValueError", False)
        except ValueError:
            run_test("fewer dims raises ValueError", True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3, 2], ["a"] * 6, tmpdir)
        try:
            expand_forward(ft, [4, 2])
            run_test("non-1 dim expand raises ValueError", False)
        except ValueError:
            run_test("non-1 dim expand raises ValueError", True)

    # === Group 7: Symbolic dim expand ===
    print("\nGroup 7: Symbolic dim expand")

    import asyncio

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1], ["my_instance"], tmpdir)
        n = sympy.Symbol("n")
        r = expand_forward(ft, [sympy.Integer(1), n])
        run_test("symbolic [1]->[1,n] capacity [1,0]", r.ft_capacity_shape == [1, 0])
        run_test("symbolic schema preserved", r.ft_shape_schema == [sympy.Integer(1), n])
        run_test("symbolic expand not forwarded (no concrete storage)", r.ft_forwarded is False)

    with tempfile.TemporaryDirectory() as tmpdir:
        async def source_get(coords, trajactory):
            return (f"src_{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, source_get, [sympy.Integer(1)])
        ft.ft_forwarded = False
        n = sympy.Symbol("n")
        r = expand_forward(ft, [sympy.Integer(1), n])
        run_test("symbolic lazy capacity [1,0]", r.ft_capacity_shape == [1, 0])
        run_test("symbolic lazy not forwarded", r.ft_forwarded is False)

        # ft_async_get maps [0, i] -> [0] for any i
        result = asyncio.run(r.ft_async_get([0, 5], "prompt"))
        run_test("symbolic [0,5] maps to [0]", result[0] == "src_[0]")
        result2 = asyncio.run(r.ft_async_get([0, 99], "prompt"))
        run_test("symbolic [0,99] maps to [0]", result2[0] == "src_[0]")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3], ["x", "y", "z"], tmpdir)
        n = sympy.Symbol("n")
        r = expand_forward(ft, [n, 3])
        run_test("symbolic [3]->[n,3] capacity [0,3]", r.ft_capacity_shape == [0, 3])
        run_test("symbolic [3]->[n,3] schema", r.ft_shape_schema == [n, sympy.Integer(3)])
        run_test("symbolic prepend not forwarded (symbolic dim)", r.ft_forwarded is False)

    with tempfile.TemporaryDirectory() as tmpdir:
        async def multi_get(coords, trajactory):
            return (f"val_{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, multi_get, [sympy.Integer(1), sympy.Integer(1)])
        n = sympy.Symbol("n")
        m = sympy.Symbol("m")
        r = expand_forward(ft, [n, m])
        run_test("symbolic [1,1]->[n,m] capacity [0,0]", r.ft_capacity_shape == [0, 0])
        run_test("symbolic [1,1]->[n,m] schema", r.ft_shape_schema == [n, m])
        # All coords map to [0,0]
        result = asyncio.run(r.ft_async_get([7, 3], "prompt"))
        run_test("symbolic [7,3] maps to [0,0]", result[0] == "val_[0, 0]")

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All expand_forward tests passed.")
    else:
        import sys
        sys.exit(1)
