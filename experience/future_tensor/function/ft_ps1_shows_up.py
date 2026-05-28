"""
ft_ps1_shows_up: Check whether a PS1 prompt regexp matches in captured pane text.

Pure runtime op (no autograd). Returns "true" if PS1 regexp matches anywhere in
the capture text, "false" otherwise. Used to conditionally skip sleep when the
terminal is already idle.
"""

import re
from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def ft_ps1_shows_up(ps1_pattern_ft: FutureTensor, capture_ft: FutureTensor) -> FutureTensor:
    """Check whether PS1 regexp matches in captured terminal output.

    Args:
        ps1_pattern_ft: FutureTensor whose elements contain PS1 regexps.
                        Broadcastable to capture_ft shape.
        capture_ft: FutureTensor whose elements contain captured pane text.

    Returns:
        A lazy FutureTensor with the same shape as capture_ft. Each element
        is "true" if the PS1 regexp matches in the capture text, "false" otherwise.
    """
    shape = capture_ft.ft_capacity_shape
    relative_to = capture_ft.ft_initial_static_tensor.st_relative_to

    async def ps1_check_get(coordinates: List[int], trajactory: str):
        # Read capture text
        if capture_ft.ft_forwarded:
            _coeff, filepath = capture_ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                capture_text = f.read()
        else:
            capture_text, _status = await capture_ft.ft_async_get(coordinates, trajactory)

        # Read PS1 pattern — same shape or broadcastable
        ps1_shape = ps1_pattern_ft.ft_capacity_shape
        if ps1_shape == shape:
            ps1_coords = coordinates
        elif ps1_shape == [1] or ps1_shape == []:
            ps1_coords = [0] if ps1_shape == [1] else []
        else:
            # Broadcast: truncate coords to ps1 shape dims
            ps1_coords = coordinates[:len(ps1_shape)]

        if ps1_pattern_ft.ft_forwarded:
            _coeff, filepath = ps1_pattern_ft.ft_get_materialized_value(ps1_coords)
            with open(filepath, "r", encoding="utf-8") as f:
                pattern = f.read().strip()
        else:
            pattern, _status = await ps1_pattern_ft.ft_async_get(ps1_coords, trajactory)
            pattern = pattern.strip()

        # Match PS1 regexp against capture text
        if pattern and re.search(pattern, capture_text, re.MULTILINE):
            return ("true", Status.confidence(1.0))
        return ("false", Status.confidence(1.0))

    result = FutureTensor(
        relative_to,
        ps1_check_get,
        [sympy.Integer(s) for s in shape],
    )
    result.ft_capacity_shape = list(shape)
    return result
