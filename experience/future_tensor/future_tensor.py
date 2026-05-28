"""
Status = Import[status].Status

FutureTensor :=
    torch.Tensor[(), bfloat16, value=1]
    * $ft_initial_static_tensor SymbolicTensor[...]
    * $ft_incremental_concated_tensors list[($tensor SymbolicTensor[...], $concat_axis int)]
    * $ft_shape_schema list[sympy.Symbol]
    * $ft_capacity_shape list[int]
    * $ft_forwarded bool
    * $ft_forward (void <- $prompt FutureTensor)
    * $ft_async_get (Awaitable[($output str, $status Status)] <- $coordinates list[int] <- $prompt str)
    * $ft_get_materialized_value (($coefficient float, $element_file_path str) <- $coordinates list[int])
    * $ft_reset_materialized_value (void <- $coordinates list[int] <- $coefficient float <- $filepath str <- $symlink bool)
"""

import asyncio
import itertools
import os
import shutil
from typing import Callable, Awaitable, List, Tuple

import sympy
import torch

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
from experience.future_tensor.status import Status


def _read_element(tensor: torch.Tensor, flat_index: int) -> str:
    """Read the symbolic content at a flat index."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    path = os.path.realpath(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _coords_to_flat(coordinates: List[int], shape: List[int]) -> int:
    """Convert multi-dimensional coordinates to flat index."""
    flat = 0
    stride = 1
    for i in reversed(range(len(shape))):
        flat += coordinates[i] * stride
        stride *= shape[i]
    return flat


def _resolve_coords_to_tensor(ft, coordinates):
    """Map logical coordinates to (tensor, local_coords, local_shape).

    Resolves across ft_initial_static_tensor + ft_incremental_concated_tensors.
    Assumes chunks are appended in flat-index order (sequential iteration).
    """
    static_shape = list(ft.ft_initial_static_tensor.shape)
    ndim = len(static_shape)

    if ndim == 0:
        # Scalar tensor — always resolved from static tensor
        return ft.ft_initial_static_tensor, list(coordinates), static_shape

    if ndim > 0 and all(coordinates[i] < static_shape[i] for i in range(ndim)):
        return ft.ft_initial_static_tensor, list(coordinates), static_shape

    logical_flat = _coords_to_flat(coordinates, ft.ft_capacity_shape)
    num_static = 1
    for s in static_shape:
        num_static *= s
    chunk_idx = logical_flat - num_static

    if chunk_idx < 0 or chunk_idx >= len(ft.ft_incremental_concated_tensors):
        raise IndexError(f"Coordinate {coordinates} not materialized")
    chunk, _ = ft.ft_incremental_concated_tensors[chunk_idx]
    chunk_shape = list(chunk.shape)
    return chunk, [0] * len(chunk_shape), chunk_shape


def _storage_path_for_tensor(
    tensor: torch.Tensor, coordinates: List[int], shape: List[int]
) -> str:
    """Get storage file path for a tensor at given coordinates."""
    flat_index = _coords_to_flat(coordinates, shape)
    digits = list(str(flat_index))
    return os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )


def _grow_capacity_for(ft, coordinates: List[int], result: Tuple[str, "Status"]):
    """Append result to incremental storage, grow ft_capacity_shape."""
    shape = ft.ft_capacity_shape
    new_shape = [max(shape[i], coordinates[i] + 1) if i < len(coordinates) else shape[i]
                 for i in range(len(shape))]
    if new_shape == shape:
        return
    axis = next((i for i in range(len(shape))
                 if i < len(coordinates) and coordinates[i] + 1 > shape[i]), 0)
    chunk = make_tensor([result[0]], ft.ft_initial_static_tensor.st_relative_to)
    ft.ft_incremental_concated_tensors.append((chunk, axis))
    ft.ft_capacity_shape = new_shape


def _resolve_concrete_shape(ft_shape_schema) -> List[int]:
    """Resolve sympy schema to concrete shape (Symbol → 0)."""
    concrete: List[int] = []
    for dim in ft_shape_schema:
        if isinstance(dim, (int, sympy.Integer)):
            concrete.append(int(dim))
        elif isinstance(dim, sympy.Symbol):
            concrete.append(0)
        else:
            concrete.append(int(dim))
    return concrete


async def _cached_async_get_impl(ft, raw_async_get, symbolic_axes, coordinates, trajactory):
    """Write-through cached ft_async_get. Only caches confidence results."""
    key = tuple(coordinates)
    if key in ft._ft_cache:
        return ft._ft_cache[key]
    shape = ft.ft_capacity_shape
    if shape and len(coordinates) == len(shape) and all(
            coordinates[i] < shape[i] for i in range(len(shape))):
        try:
            tensor, local_coords, local_shape = _resolve_coords_to_tensor(ft, coordinates)
            flat_idx = _coords_to_flat(local_coords, local_shape)
            coeff = tensor.data.flatten()[flat_idx].item()
            if coeff > 0:
                fpath = _storage_path_for_tensor(tensor, local_coords, local_shape)
                if os.path.isfile(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        result = (f.read(), Status.confidence(coeff))
                    ft._ft_cache[key] = result
                    return result
        except IndexError:
            pass
    result = await raw_async_get(coordinates, trajactory)
    if result[1].is_confidence:
        ft._ft_cache[key] = result
        if symbolic_axes:
            _grow_capacity_for(ft, coordinates, result)
    return result


def _ft_forward_impl(ft, prompt_tensor: torch.Tensor) -> None:
    """Materialize FutureTensor by calling ft_async_get for each pending element."""
    if ft.ft_forwarded:
        return
    shape = ft.ft_capacity_shape
    all_coordinates = [list(c) for c in itertools.product(*[range(s) for s in shape])]
    pending: List[List[int]] = []
    for c in all_coordinates:
        try:
            t, lc, ls = _resolve_coords_to_tensor(ft, c)
            if t.data.flatten()[_coords_to_flat(lc, ls)].item() <= 0:
                pending.append(c)
        except IndexError:
            pending.append(c)
    if not pending:
        ft.ft_forwarded = True
        return
    prompts = [_read_element(prompt_tensor, _coords_to_flat(c, shape)) for c in pending]

    async def _gather():
        return await asyncio.gather(*[ft.ft_async_get(c, p) for c, p in zip(pending, prompts)])

    results = asyncio.run(_gather())
    for coords, (content, status) in zip(pending, results):
        tensor, local_coords, local_shape = _resolve_coords_to_tensor(ft, coords)
        flat_idx = _coords_to_flat(local_coords, local_shape)
        tensor.data.flatten()[flat_idx] = Status.convert_status_to_float(status)
        dst = _storage_path_for_tensor(tensor, local_coords, local_shape)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)
    ft.ft_forwarded = True
    for c in all_coordinates:
        try:
            t, lc, ls = _resolve_coords_to_tensor(ft, c)
            if t.data.flatten()[_coords_to_flat(lc, ls)].item() <= 0:
                ft.ft_forwarded = False
                break
        except IndexError:
            ft.ft_forwarded = False
            break


def _ft_get_materialized_value_impl(ft, coordinates: List[int]) -> Tuple[float, str]:
    """Look up element coefficient and file path by coordinates."""
    tensor, local_coords, local_shape = _resolve_coords_to_tensor(ft, coordinates)
    flat_idx = _coords_to_flat(local_coords, local_shape)
    coefficient = tensor.data.flatten()[flat_idx].item()
    return (coefficient, _storage_path_for_tensor(tensor, local_coords, local_shape))


def _ft_reset_materialized_value_impl(
    ft, coordinates: List[int], coefficient: float, filepath: str, symlink: bool = False,
) -> None:
    """Overwrite element at coordinates with given coefficient and filepath."""
    tensor, local_coords, local_shape = _resolve_coords_to_tensor(ft, coordinates)
    flat_idx = _coords_to_flat(local_coords, local_shape)
    tensor.data.flatten()[flat_idx] = coefficient
    dst_path = _storage_path_for_tensor(tensor, local_coords, local_shape)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if symlink:
        if os.path.exists(dst_path) or os.path.islink(dst_path):
            os.remove(dst_path)
        os.symlink(os.path.realpath(filepath), dst_path)
    else:
        shutil.copy2(filepath, dst_path)


def _unflatten(flat_list: List[str], shape: List[int]):
    """Rebuild nested list structure from flat list given shape."""
    if not shape:
        return flat_list[0] if flat_list else None
    if len(shape) == 1:
        return flat_list
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    return [
        _unflatten(flat_list[i * chunk_size:(i + 1) * chunk_size], shape[1:])
        for i in range(shape[0])
    ]


def _assign_symbolic_only(lvalue: torch.Tensor, rvalue: torch.Tensor) -> None:
    """Copy symbolic storage from rvalue to lvalue without overwriting data coefficients."""
    assert lvalue.shape == rvalue.shape
    shape = list(lvalue.shape)
    for coords in itertools.product(*[range(s) for s in shape]):
        coords = list(coords)
        lvalue_path = _storage_path_for_tensor(lvalue, coords, shape)
        rvalue_path = _storage_path_for_tensor(rvalue, coords, shape)
        os.makedirs(os.path.dirname(lvalue_path), exist_ok=True)
        if os.path.isfile(rvalue_path):
            shutil.copy2(rvalue_path, lvalue_path)


def FutureTensor(
    relative_to: str,
    ft_async_get: Callable[[List[int], str], Awaitable[Tuple[str, "Status"]]],
    ft_shape_schema: List["sympy.Symbol"],
) -> torch.Tensor:
    """Create a FutureTensor: a scalar bfloat16 torch.Tensor monkey-patched with ft_* attributes.

    FutureTensor is an interface — a plain torch.Tensor of shape (), dtype bfloat16, value
    always 1. It is monkey-patched with ft_* attributes. Not a subclass of SymbolicTensor.

    Args:
        relative_to: Storage root directory.
        ft_async_get: Async callable (coordinates, prompt) -> (str, Status).
        ft_shape_schema: Declared logical shape schema (list of sympy.Symbol or sympy.Integer).

    Returns:
        A scalar torch.Tensor (shape=(), dtype=bfloat16, value=1) with ft_* attributes.
    """
    from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor

    concrete_shape = _resolve_concrete_shape(ft_shape_schema)
    symbolic_axes = [i for i, d in enumerate(ft_shape_schema) if isinstance(d, sympy.Symbol)]
    ft = torch.ones((), dtype=torch.bfloat16)
    ft.ft_initial_static_tensor = make_none_tensor(concrete_shape, relative_to)
    ft.ft_incremental_concated_tensors = []
    ft.ft_shape_schema = list(ft_shape_schema)
    ft.ft_capacity_shape = list(concrete_shape)
    ft.ft_forwarded = False
    ft._ft_cache = {}
    if ft_async_get is not None:
        raw = ft_async_get
        ft.ft_async_get = lambda c, t: _cached_async_get_impl(ft, raw, symbolic_axes, c, t)
    else:
        ft.ft_async_get = None
    ft.ft_forward = lambda p: _ft_forward_impl(ft, p)
    ft.ft_get_materialized_value = lambda c: _ft_get_materialized_value_impl(ft, c)
    ft.ft_reset_materialized_value = lambda c, coeff, fp, symlink=False: _ft_reset_materialized_value_impl(ft, c, coeff, fp, symlink)
    return ft


def _tensor_to_future(tensor: torch.Tensor, reference_ft) -> torch.Tensor:
    """Convert a plain torch.Tensor into a FutureTensor using reference for metadata.

    The autograd engine passes plain torch.Tensor as grad_output to backward().
    This helper wraps it back into a FutureTensor so that backward functions
    (slice_backward, unsqueeze_forward, etc.) receive the type they expect.

    Args:
        tensor: Plain torch.Tensor (usually grad_output from autograd engine).
        reference_ft: A FutureTensor to borrow relative_to and metadata from.

    Returns:
        A FutureTensor with the same shape and data coefficients as ``tensor``.
    """
    from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor

    shape = list(tensor.shape)
    relative_to = reference_ft.ft_initial_static_tensor.st_relative_to

    async def dummy_get(coords, trajactory):
        return ("", Status.confidence(0.0))

    ft = FutureTensor(relative_to, dummy_get, [sympy.Integer(s) for s in shape])
    ft.ft_initial_static_tensor.data.copy_(tensor.data)
    ft.ft_forwarded = True
    return ft


if __name__ == "__main__":
    import tempfile

    print("Running tests for future_tensor...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_storage(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to,
            tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        with open(path) as f:
            return f.read()

    # ── Test 1: Basic structure ──────────────────────────────────────────

    print("Test 1: FutureTensor is a scalar bfloat16 torch.Tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def echo_get(coordinates, trajactory):
            return (f"output({coordinates}, {trajactory})", Status.confidence(0.9))

        ft = FutureTensor(tmpdir, echo_get, [sympy.Integer(3)])
        run_test("FT is torch.Tensor", isinstance(ft, torch.Tensor))
        run_test("FT shape is ()", list(ft.shape) == [])
        run_test("FT dtype is bfloat16", ft.dtype == torch.bfloat16)
        run_test("FT value is 1", ft.item() == 1.0)
        run_test("ft_capacity_shape is [3]", ft.ft_capacity_shape == [3])
        run_test("ft_initial_static_tensor shape is [3]", list(ft.ft_initial_static_tensor.shape) == [3])
        run_test("Not forwarded initially", ft.ft_forwarded is False)

        prompt_t = make_tensor(["prompt_0", "prompt_1", "prompt_2"], tmpdir)
        ft.ft_forward(prompt_t)

        run_test("Forwarded after ft_forward", ft.ft_forwarded is True)
        run_test("Element 0", read_storage(ft.ft_initial_static_tensor, 0) == "output([0], prompt_0)")
        run_test("Element 1", read_storage(ft.ft_initial_static_tensor, 1) == "output([1], prompt_1)")
        run_test("Element 2", read_storage(ft.ft_initial_static_tensor, 2) == "output([2], prompt_2)")
        run_test("FT value still 1 after forward", ft.item() == 1.0)
        run_test("Status in ft_initial_static_tensor coeff[0]",
                 abs(ft.ft_initial_static_tensor.data.flatten()[0].item() - 0.9) < 0.01)

    # ── Test 2: ft_get_materialized_value / ft_reset_materialized_value ──

    print("Test 2: ft_get_materialized_value / ft_reset_materialized_value")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def simple_get(coordinates, trajactory):
            return (f"content_{coordinates[0]}", Status.confidence(0.8))

        ft = FutureTensor(tmpdir, simple_get, [sympy.Integer(2)])
        prompt_t = make_tensor(["a", "b"], tmpdir)
        ft.ft_forward(prompt_t)

        coeff, path = ft.ft_get_materialized_value([0])
        run_test("ft_get_materialized_value coeff", abs(coeff - 0.8) < 0.01)
        run_test("ft_get_materialized_value path exists", os.path.isfile(path))

        # Reset element 0
        new_content_path = os.path.join(tmpdir, "new_content.txt")
        with open(new_content_path, "w") as f:
            f.write("new_content")
        ft.ft_reset_materialized_value([0], 0.5, new_content_path, symlink=False)
        coeff2, path2 = ft.ft_get_materialized_value([0])
        run_test("ft_reset_materialized_value coeff", abs(coeff2 - 0.5) < 0.01)
        with open(path2) as f:
            run_test("ft_reset_materialized_value content", f.read() == "new_content")

    # ── Test 3: Idempotent forward ─────────────────────────────────────

    print("Test 3: Idempotent forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        counter = [0]

        async def counting_get(coordinates, trajactory):
            counter[0] += 1
            return ("x", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, counting_get, [sympy.Integer(2)])
        prompt_t = make_tensor(["a", "b"], tmpdir)

        ft.ft_forward(prompt_t)
        first_count = counter[0]
        run_test("Called 2 times on first forward", first_count == 2, 2, first_count)

        ft.ft_forward(prompt_t)
        run_test("No additional calls on second forward", counter[0] == first_count)

    # ── Test 4: ft_incremental_concated_tensors ──────────────────────────

    print("Test 4: ft_incremental_concated_tensors empty by default")
    with tempfile.TemporaryDirectory() as tmpdir:
        ft = FutureTensor(tmpdir, None, [sympy.Integer(3)])
        run_test("ft_incremental_concated_tensors is []", ft.ft_incremental_concated_tensors == [])

    # ── Test 5: FtMean ───────────────────────────────────────────────────

    from experience.future_tensor.function.ft_mean import ft_mean

    print("Test 5: FtMean")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def dummy_get(coords, trajactory):
            return ("unused", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(4)])
        ft.ft_initial_static_tensor.data[0] = 1.0
        ft.ft_initial_static_tensor.data[1] = 2.0
        ft.ft_initial_static_tensor.data[2] = 3.0
        ft.ft_initial_static_tensor.data[3] = 4.0
        ft.ft_forwarded = True

        m = ft_mean(ft)
        run_test("FtMean returns scalar", m.shape == torch.Size([]))
        run_test("FtMean value correct", abs(m.item() - 2.5) < 0.01)

    print("\nAll tests completed.")
