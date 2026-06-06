"""
ft_speculative_keystroke: Speculative execution with foveal validation.

Simple: validate current_foveal at current_fixation.
  - Pass → output keystroke.
  - Fail → output empty (forces engine re-query).
"""

import os
from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.react.fixation import extract_foveal, foveal_matches
from experience.react.react_types import KeystrokeNode


async def _read_ft(ft, coordinates, trajectory):
    """Read text from a FutureTensor."""
    if ft.ft_forwarded:
        _coeff, filepath = ft.ft_get_materialized_value(coordinates)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        return ""
    text, _status = await ft.ft_async_get(coordinates, trajectory)
    return text


def _foveal_ok(node: dict, screen: str, fixation_key: str, foveal_key: str) -> bool:
    """Check if expected foveal matches actual screen content at fixation."""
    expected = node[foveal_key]
    if not expected:
        return True
    actual = extract_foveal(screen, tuple(node[fixation_key]))
    return foveal_matches(actual, expected)


def ft_speculative_keystroke(
    engine_input: FutureTensor,
    screen_capture: FutureTensor,
) -> FutureTensor:
    """Speculative keystroke with foveal validation.

    Walks KeystrokeNode tree. Outputs keystroke if validation passes, empty if not.
    """
    shape = engine_input.ft_capacity_shape
    schema = engine_input.ft_shape_schema
    relative_to = engine_input.ft_initial_static_tensor.st_relative_to
    async def speculative_get(coordinates: List[int], trajectory: str):
        screen = await _read_ft(screen_capture, coordinates, trajectory)

        # Always do a fresh engine pull. Each react_loop iteration captures
        # a new screen, so the engine must decide based on current state.
        # The tree-walking optimization is removed — it prevented the
        # autonomous loop from re-querying the LLM on each iteration.
        text = await _read_ft(engine_input, coordinates, trajectory)
        tree = KeystrokeNode.deserialize(text)
        if not tree:
            return ("", Status.self_confidence_but_failed(0.5))
        foveal_pass = _foveal_ok(tree, screen, "current_fixation", "current_foveal")
        confidence = 1.0 if foveal_pass else 0.5
        return (tree["keystrokes"], Status.confidence(confidence))

    result = FutureTensor(
        relative_to, speculative_get,
        list(schema),
    )
    result.ft_capacity_shape = list(shape)
    return result
