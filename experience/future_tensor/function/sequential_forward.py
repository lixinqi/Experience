"""
sequential_forward :=
    FutureTensor
    <- $inputs tuple[FutureTensor, ...]
    # inline

Pure lazy sequential: evaluation of each input is deferred to the returned
ft_async_get.  No input is evaluated inside sequential_forward itself.
"""

from typing import List, Tuple

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def sequential_forward(
    inputs: Tuple[FutureTensor, ...],
) -> FutureTensor:
    """Lazy forward pass of sequential.

    Returns a **lazy** FutureTensor whose ``ft_async_get`` will, at pull time:
      1. Call each input's ``ft_async_get`` in sequence.
      2. If any returns a non-confidence status, early return that result.
      3. Otherwise return the last input's result.

    Args:
        inputs: Non-empty tuple of FutureTensors. All must have the same shape.

    Returns:
        A lazy FutureTensor with the same shape as the inputs.
    """
    if not inputs:
        raise ValueError("sequential_forward: inputs must not be empty")

    expected_shape = inputs[0].ft_capacity_shape
    for i, inp in enumerate(inputs):
        if inp.ft_capacity_shape != expected_shape:
            raise ValueError(
                f"sequential_forward: input {i} has shape {inp.ft_capacity_shape}, "
                f"expected {expected_shape}"
            )

    async def sequential_async_get(coordinates: List[int], trajactory: str):
        last_result = None
        for inp in inputs:
            result = await inp.ft_async_get(coordinates, trajactory)
            status = result[1]
            if not status.is_confidence:
                return result
            last_result = result
        return last_result

    result = FutureTensor(
        inputs[0].ft_initial_static_tensor.st_relative_to,
        sequential_async_get,
        inputs[0].ft_shape_schema,
    )
    result.ft_capacity_shape = list(expected_shape)
    # ft_forwarded stays False — purely lazy

    return result
