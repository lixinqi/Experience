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

    Walks a KeystrokeNode tree across iterations.  First iteration pulls
    a fresh tree from the engine.  Subsequent iterations advance to child
    nodes.  When the tree is exhausted the engine is re-queried.
    """
    shape = engine_input.ft_capacity_shape
    schema = engine_input.ft_shape_schema
    relative_to = engine_input.ft_initial_static_tensor.st_relative_to
    state: dict = {}  # prefix -> {"node": KeystrokeNode dict}

    async def speculative_get(coordinates: List[int], trajectory: str):
        key = tuple(coordinates[:-1]) if len(coordinates) > 1 else ()
        iter_idx = coordinates[-1] if coordinates else 0
        screen = await _read_ft(screen_capture, coordinates, trajectory)
        cur = state.get(key)

        # Fresh tree from engine: no cached tree, or first iteration.
        if cur is None or iter_idx == 0:
            if cur is None and iter_idx > 0:
                # Tree was exhausted in a prior iteration — start fresh.
                pass
            text = await _read_ft(engine_input, coordinates, trajectory)
            tree = KeystrokeNode.deserialize(text)
            if not tree:
                return ("", Status.self_confidence_but_failed(0.5))
            ok = _foveal_ok(tree, screen, "current_fixation", "current_foveal")
            # If tree has children, keep it for walking.  If single-node
            # (no children), still return it but the next iter_idx==0 will refresh.
            state[key] = {"node": tree, "fresh": True}
            return (tree["keystrokes"], Status.confidence(1.0 if ok else 0.5))

        # Walk to next child.
        node = cur["node"]
        children = node.get("children", [])
        if not children or not children[0]:
            # Tree exhausted — re-query engine next time.
            del state[key]
            return ("", Status.confidence(1.0))

        child = children[0]
        child_ok = _foveal_ok(child, screen, "current_fixation", "current_foveal")
        state[key] = {"node": child}
        return (child["keystrokes"], Status.confidence(1.0 if child_ok else 0.5))

    result = FutureTensor(
        relative_to, speculative_get,
        list(schema),
    )
    result.ft_capacity_shape = list(shape)
    return result
