"""
EchoHelloWorldEngine: simplest possible engine implementing engine_step contract.

Hardcodes the plan: type "echo hello world" + Enter. No LLM, no experience.

Signature (per engine_step.viba):
    engine_step($capture FutureTensor, $fixation FutureTensor, $mail FutureTensor,
                $coordinates list[int], $task str) -> EngineOutput
"""

from typing import List, Tuple

from experience.future_tensor.future_tensor import FutureTensor
from experience.react.types import KeystrokeNode, EngineOutput, ft_read
from experience.react.fixation import extract_foveal


def _parse_fixation(fixation_text: str) -> Tuple[int, int]:
    """Parse "row,col" string into tuple."""
    try:
        parts = fixation_text.strip().split(",")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (0, 0)


def _build_plan(fixation: Tuple[int, int], foveal: str) -> KeystrokeNode:
    """Build hardcoded KeystrokeNode chain: T echo hello world → C Enter."""
    row, col = fixation
    text = "echo hello world"
    node_type = KeystrokeNode(
        current_fixation=(row, col),
        current_foveal=foveal,
        keystrokes="T echo hello world",
        predicted_next_fixation=(row, col + len(text)),
        predicted_next_foveal="orld",
    )
    node_enter = KeystrokeNode(
        current_fixation=(row, col + len(text)),
        current_foveal="orld",
        keystrokes="C Enter",
        predicted_next_fixation=(row + 1, 0),
        predicted_next_foveal="$",
    )
    node_type.children.append(node_enter)
    return node_type


async def engine_step(
    capture: FutureTensor,
    fixation: FutureTensor,
    mail: FutureTensor,
    coordinates: List[int],
    task: str,
) -> EngineOutput:
    """Engine step per engine_step.viba contract."""
    capture_text = ft_read(capture, coordinates)
    fix_point = _parse_fixation(ft_read(fixation, coordinates))
    foveal = extract_foveal(capture_text, fix_point)
    return EngineOutput(plan=_build_plan(fix_point, foveal))
