"""
ft_first_line: Filter FutureTensor that extracts the first non-empty line.

A human types one line at a time. LLM expert output may be multi-line
garbage — this filter ensures only the first meaningful line passes through.

No autograd — pure data transformation, transparent to backward.
"""

from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def ft_first_line(input_ft: FutureTensor) -> FutureTensor:
    """Extract the first non-empty line from each element of input_ft.

    Args:
        input_ft: FutureTensor whose elements may contain multi-line text.

    Returns:
        A lazy FutureTensor with the same shape. Each element contains only
        the first non-empty stripped line of the corresponding input element,
        or empty string if no non-empty line exists.
    """
    shape = input_ft.ft_capacity_shape
    relative_to = input_ft.ft_initial_static_tensor.st_relative_to

    async def first_line_get(coordinates: List[int], trajactory: str):
        if input_ft.ft_forwarded:
            _coeff, filepath = input_ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text, _status = await input_ft.ft_async_get(coordinates, trajactory)

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        first = lines[0] if lines else ""
        if first:
            return (first, Status.confidence(1.0))
        return ("", Status.self_confidence_but_failed(0.1))

    result = FutureTensor(
        relative_to,
        first_line_get,
        [sympy.Integer(s) for s in shape],
    )
    result.ft_capacity_shape = list(shape)
    return result
