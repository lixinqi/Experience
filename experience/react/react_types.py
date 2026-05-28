"""Shared types for the react package."""

import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Awaitable

from experience.future_tensor.future_tensor import FutureTensor


@dataclass
class KeystrokeNode:
    """Recursive tree of keystrokes with predicted fixation points."""
    current_fixation: Tuple[int, int] = (0, 0)
    current_foveal: str = ""
    keystrokes: str = ""
    predicted_next_fixation: Tuple[int, int] = (0, 0)
    predicted_next_foveal: str = ""
    children: List["KeystrokeNode"] = field(default_factory=list)
    comments: str = ""

    def serialize(self) -> str:
        """Serialize this KeystrokeNode tree to JSON string."""
        def _to_dict(n):
            if n is None:
                return None
            d = {
                "current_fixation": list(n.current_fixation),
                "current_foveal": n.current_foveal,
                "keystrokes": n.keystrokes,
                "predicted_next_fixation": list(n.predicted_next_fixation),
                "predicted_next_foveal": n.predicted_next_foveal,
                "children": [_to_dict(c) for c in n.children],
            }
            if n.comments:
                d["comments"] = n.comments
            return d
        return json.dumps(_to_dict(self))

    @staticmethod
    def deserialize(text: str) -> Optional[dict]:
        """Deserialize JSON string to KeystrokeNode dict."""
        if not text or not text.strip():
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None


@dataclass
class ReactConfig:
    """Configuration for the ReAct loop."""
    max_iterations: int = 20
    settle_delay: float = 0.3
    mismatch_tolerance: int = 3
    fixation_start: Tuple[int, int] = (0, 0)
    step_budget: int = 8


# engine_step signature per engine_step.viba:
#   $output KeystrokeNode
#   <- $capture FutureTensor
#   <- $fixation FutureTensor
#   <- $mail FutureTensor
#   <- $coordinates list[int]
#   <- $task str
#
# CodingAgentEngine.__call__ returns a FutureTensor (future op):
#   FutureTensor[Awaitable[$output KeystrokeNode] <- $coordinates <- $prompt]
#   <- $task * $mail * $fixation * $capture
EngineStepFn = Callable[
    [FutureTensor, FutureTensor, FutureTensor, str],
    FutureTensor,
]

ValidatorFn = Callable[[str, int], bool]


def ft_read(ft: FutureTensor, coordinates: List[int]) -> str:
    """Read materialized string content from a FutureTensor at coordinates."""
    _, filepath = ft.ft_get_materialized_value(coordinates)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return ""
