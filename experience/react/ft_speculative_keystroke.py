"""
ft_speculative_keystroke: Speculative execution with foveal validation.

Simple: validate current_foveal at current_fixation.
  - Pass → output keystroke.
  - Fail → output empty (forces engine re-query).
"""

import os
from typing import Dict, List

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
    state: Dict[tuple, dict] = {}

    async def speculative_get(coordinates: List[int], trajectory: str):
        key = tuple(coordinates[:-1]) if len(coordinates) > 1 else ()
        iter_idx = coordinates[-1] if coordinates else 0
        screen = await _read_ft(screen_capture, coordinates, trajectory)
        cur = state.get(key)

        # Done — tree exhausted
        if cur and cur.get("done"):
            return ("", Status.confidence(1.0))

        # Fresh pull or first iteration
        if cur is None or iter_idx == 0:
            text = await _read_ft(engine_input, coordinates, trajectory)
            tree = KeystrokeNode.deserialize(text)
            if not tree:
                return ("", Status.self_confidence_but_failed(0.5))
            foveal_pass = _foveal_ok(tree, screen, "current_fixation", "current_foveal")
            state[key] = {"node": tree}
            confidence = 1.0 if foveal_pass else 0.5
            return (tree["keystrokes"], Status.confidence(confidence))

        # Advance: validate post-send prediction from last node, then advance to child
        node = cur["node"]
        pred_ok = _foveal_ok(node, screen, "predicted_next_fixation", "predicted_next_foveal")

        children = node.get("children", [])
        if not children or not children[0]:
            state[key] = {"node": node, "done": True}
            return ("", Status.confidence(1.0))

        child = children[0]
        child_ok = _foveal_ok(child, screen, "current_fixation", "current_foveal")
        state[key] = {"node": child}
        confidence = 1.0 if (pred_ok and child_ok) else 0.5
        return (child["keystrokes"], Status.confidence(confidence))

    result = FutureTensor(
        relative_to, speculative_get,
        list(schema),
    )
    result.ft_capacity_shape = list(shape)
    return result
