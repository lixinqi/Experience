"""
read_file_forward :=
    FutureTensor
    <- $input FutureTensor
    # inline

Read file contents from the local filesystem.
Each element's symbolic content provides the file path to read.
The file content is stored as the output element's content.
"""

import os
from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def read_file_forward(input_ft: FutureTensor) -> FutureTensor:
    """Forward: create a lazy FutureTensor whose ft_async_get reads a file."""
    shape = input_ft.ft_capacity_shape
    schema = input_ft.ft_shape_schema
    relative_to = input_ft.ft_initial_static_tensor.st_relative_to

    async def read_file_async_get(coordinates: List[int], trajactory: str):
        if input_ft.ft_forwarded:
            _coeff, filepath = input_ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                file_path = f.read().strip()
        else:
            content, _status = await input_ft.ft_async_get(coordinates, trajactory)
            file_path = content.strip()

        if not file_path:
            return ("", Status.self_confidence_but_failed(0.0))

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return (text, Status.confidence(1.0))
        except Exception:
            return ("", Status.self_confidence_but_failed(0.0))

    result = FutureTensor(
        relative_to,
        read_file_async_get,
        list(schema),
    )
    result.ft_capacity_shape = list(shape)
    return result
