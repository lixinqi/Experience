"""
ft_validate_ctrl: Filter FutureTensor that validates control sequences.

Only passes through recognized tmux control key names (Enter, Escape, C-c,
C-d, etc.). Garbage text that isn't a valid control sequence is replaced
with empty string.

No autograd — pure data filter, transparent to backward.
"""

from typing import List, Set

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status

# Recognized tmux key names (literal=False sends these as special keys)
VALID_CTRL_KEYS: Set[str] = {
    "Enter", "Escape", "Tab", "Space", "BSpace",
    "Up", "Down", "Left", "Right",
    "Home", "End", "PageUp", "PageDown", "Insert", "Delete",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
}
# Also allow C-<key> and M-<key> patterns
_CTRL_PREFIXES = ("C-", "M-")


def _is_valid_ctrl(text: str) -> bool:
    """Check if text is a recognized tmux control key name."""
    if text in VALID_CTRL_KEYS:
        return True
    if any(text.startswith(p) and len(text) <= 4 for p in _CTRL_PREFIXES):
        return True
    return False


def ft_validate_ctrl(input_ft: FutureTensor) -> FutureTensor:
    """Filter: only pass through valid tmux control key names.

    Extracts the first non-empty line from input, then checks if it's a
    recognized control key. Invalid sequences are replaced with empty string.

    Args:
        input_ft: FutureTensor whose elements should contain control key names.

    Returns:
        A lazy FutureTensor with the same shape. Each element contains the
        validated control key name, or empty string if invalid.
    """
    shape = input_ft.ft_capacity_shape
    relative_to = input_ft.ft_initial_static_tensor.st_relative_to

    async def validate_get(coordinates: List[int], trajactory: str):
        if input_ft.ft_forwarded:
            _coeff, filepath = input_ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text, _status = await input_ft.ft_async_get(coordinates, trajactory)

        # Extract first non-empty line
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        key = lines[0] if lines else ""

        if key and _is_valid_ctrl(key):
            return (key, Status.confidence(1.0))
        return ("", Status.self_confidence_but_failed(0.1))

    result = FutureTensor(
        relative_to,
        validate_get,
        [sympy.Integer(s) for s in shape],
    )
    result.ft_capacity_shape = list(shape)
    return result
