"""
LsEngine: hardcoded engine that types "ls" + Enter.

ADT:
    LsEngine :=
        FutureTensor[
          Awaitable[$output KeystrokeNode]
            <- $coordinates list[int]
            <- $prompt str
        ]
        <- $task str
        <- $mail FutureTensor
        <- $fixation FutureTensor
        <- $capture FutureTensor
"""

import tempfile
from typing import List, Tuple

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.react.react_types import KeystrokeNode
from experience.react.fixation import extract_foveal


def _parse_fixation(fixation_text: str) -> Tuple[int, int]:
    """Parse "row,col" string into tuple."""
    try:
        parts = fixation_text.strip().split(",")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (0, 0)


def _build_plan(fixation: Tuple[int, int], foveal: str) -> KeystrokeNode:
    """Build hardcoded KeystrokeNode chain: T ls -> C Enter."""
    row, col = fixation
    node_type = KeystrokeNode(
        current_fixation=(row, col),
        current_foveal=foveal,
        keystrokes="T ls",
        predicted_next_fixation=(row, col + 2),
        predicted_next_foveal="ls",
    )
    node_enter = KeystrokeNode(
        current_fixation=(row, col + 2),
        current_foveal="ls",
        keystrokes="C Enter",
        predicted_next_fixation=(row + 1, 0),
        predicted_next_foveal="$",
    )
    node_type.children.append(node_enter)
    return node_type


async def _read_ft(ft: FutureTensor, coordinates: List[int]) -> str:
    """Read text from a FutureTensor at coordinates (handles forwarded or lazy)."""
    if ft.ft_forwarded:
        try:
            _, filepath = ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except (IndexError, FileNotFoundError, OSError):
            pass
    text, _ = await ft.ft_async_get(coordinates, "")
    return text


def engine_step(
    capture: FutureTensor,
    fixation: FutureTensor,
    mail: FutureTensor,
    task: str,
) -> FutureTensor:
    shape = list(capture.ft_capacity_shape)
    schema = list(capture.ft_shape_schema)

    async def _engine_get(coordinates, trajectory):
        capture_text = await _read_ft(capture, coordinates)
        fixation_text = await _read_ft(fixation, coordinates)
        fix_point = _parse_fixation(fixation_text)
        foveal = extract_foveal(capture_text, fix_point)
        plan = _build_plan(fix_point, foveal)
        return (plan.serialize(), Status.confidence(1.0))

    ft = FutureTensor(tempfile.mkdtemp(prefix="ls_engine_"), _engine_get, schema)
    ft.ft_capacity_shape = shape
    ft.requires_grad_(True)
    return ft