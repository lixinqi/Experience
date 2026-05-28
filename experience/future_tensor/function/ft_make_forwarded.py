"""
ft_make_forwarded :=
    FutureTensor
    <- $relative_to str
    <- $shape list[int]
    <- $data list[str]
    # inline

Create a pre-forwarded (materialized) FutureTensor from a flat list of string data.
This is a leaf node — no autograd backward is needed.
"""

from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor


def ft_make_forwarded(
    relative_to: str,
    shape: List[int],
    data: List[str],
) -> FutureTensor:
    """Create a pre-forwarded FutureTensor populated with string data.

    The returned tensor is already materialized (ft_forwarded=True) and can be
    used directly as input to other FutureTensor ops without calling ft_forward.

    Args:
        relative_to: Base directory for tensor storage.
        shape: Shape of the tensor (e.g. [3] or [2, 4]).
        data: Flat list of string values to store. Length must equal product of shape.

    Returns:
        A FutureTensor with data written to disk, marked as forwarded.
    """
    async def dummy_get(coords, trajactory):
        return ("unused", Status.confidence(1.0))

    ft = FutureTensor(relative_to, dummy_get, [sympy.Integer(s) for s in shape])
    nested = _unflatten_data(data, shape)
    result_tensor = st_make_tensor(nested, relative_to)
    assign_tensor(ft.ft_initial_static_tensor, result_tensor)
    ft.ft_forwarded = True
    return ft


def _unflatten_data(flat_list: List[str], shape: List[int]):
    """Reshape a flat list into nested lists matching shape."""
    if not shape:
        return flat_list[0] if flat_list else None
    if len(shape) == 1:
        return flat_list
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    return [
        _unflatten_data(flat_list[i * chunk_size : (i + 1) * chunk_size], shape[1:])
        for i in range(shape[0])
    ]
