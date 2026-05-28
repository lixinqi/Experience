"""ft_unary: apply a pure function elementwise over a FutureTensor's outputs.

This is the unary map operator for FutureTensor pipelines:
    FutureTensor -> (coords, prompt, output, status) -> (new_output, new_status)
"""

import sympy
from typing import Callable, List, Tuple

import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def ft_unary(
    input_ft: torch.Tensor,
    fn: Callable[[List[int], str, str, Status], Tuple[str, Status]],
) -> torch.Tensor:
    """Apply a pure function elementwise over a FutureTensor's outputs.

    Args:
        input_ft: The input FutureTensor.
        fn: Pure callback with signature
            (coordinates, prompt, output, status) -> (new_output, new_status).

    Returns:
        A new FutureTensor of the same shape whose ft_async_get wraps `fn`
        around the input's ft_async_get.
    """

    async def wrapped(coords: List[int], prompt: str) -> Tuple[str, Status]:
        output, status = await input_ft.ft_async_get(coords, prompt)
        return fn(coords, prompt, output, status)

    return FutureTensor(
        input_ft.ft_initial_static_tensor.st_relative_to,
        wrapped,
        [sympy.Integer(s) for s in input_ft.ft_capacity_shape],
    )
