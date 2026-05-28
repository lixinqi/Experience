"""
ft_tmux_speculative_complete: Speculative multi-action dispatch from a single LLM call.

Expert generates multiple actions (separated by sep) in one LLM request.
This op consumes them one at a time, checking screen_capture to confirm
the previous action was applied before returning the next one.

State (bunch text + position) persists across recurrent iterations,
captured in autograd context for backward.

No autograd — pure runtime filter, transparent to backward.
Like ft_first_line but stateful: returns action[0] on first call,
then action[1] when screen confirms action[0] was typed, etc.
When all actions consumed or screen doesn't match, returns empty
(which forces upstream ft_expert to re-generate).
"""

from typing import Dict, List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def _parse_bunch(text: str, sep: str = "\n\n") -> List[str]:
    """Split multi-action text into individual actions.

    If lines carry "T " / "C " prefixes, split on single newline
    (each prefixed line is one action). Otherwise split on sep.
    """
    if not text or not text.strip():
        return []
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if any(l.startswith("T ") or l.startswith("C ") for l in lines):
        return lines
    parts = [p.strip() for p in text.split(sep) if p.strip()]
    return parts


def _screen_confirms_action(captured_text: str, action: str) -> bool:
    """Check if the terminal screen confirms the given action was applied.

    Supports "T " / "C " prefix protocol.
    """
    if not captured_text or not action:
        return False

    action_clean = action.strip()
    if action_clean.startswith("T "):
        action_clean = action_clean[2:]
    elif action_clean.startswith("C "):
        action_clean = action_clean[2:]

    lines = [l for l in captured_text.split("\n") if l.strip()]
    if not lines:
        return False

    ctrl_keys = {"Enter", "Escape", "Tab", "C-c", "C-d"}
    if action_clean in ctrl_keys:
        last = lines[-1]
        return "λ" in last or "$ " in last or last.rstrip().endswith("$")

    return action_clean in lines[-1]


async def _read_ft(ft, coordinates, trajactory):
    """Read text from a FutureTensor."""
    if ft.ft_forwarded:
        _coeff, filepath = ft.ft_get_materialized_value(coordinates)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    text, _status = await ft.ft_async_get(coordinates, trajactory)
    return text


def ft_tmux_speculative_complete(
    bunch_input: FutureTensor,
    screen_capture: FutureTensor,
    sep: str = "\n\n",
) -> FutureTensor:
    """Speculative multi-action dispatch from a single LLM call.

    On each recurrent iteration:
      1. No cached state or first iter → pull fresh bunch, return first action.
      2. Screen confirms previous action → advance, return next action.
      3. Bunch exhausted or screen doesn't confirm → re-pull, return first action.
    """
    shape = bunch_input.ft_capacity_shape
    relative_to = bunch_input.ft_initial_static_tensor.st_relative_to

    _state: Dict[tuple, dict] = {}

    async def _pull_fresh(prefix_key, coordinates, trajactory):
        """Pull bunch from upstream, cache it, return first action or None."""
        bunch_text = await _read_ft(bunch_input, coordinates, trajactory)
        actions = _parse_bunch(bunch_text, sep)
        if not actions:
            _state.pop(prefix_key, None)
            return None
        _state[prefix_key] = {"bunch": actions, "position": 0, "last_action": actions[0]}
        return actions[0]

    async def speculative_get(coordinates: List[int], trajactory: str):
        prefix_key = tuple(coordinates[:-1]) if len(coordinates) > 1 else ()
        iter_idx = coordinates[-1] if coordinates else 0
        state = _state.get(prefix_key)

        screen_text = await _read_ft(screen_capture, coordinates, trajactory)

        # No state or first iteration → fresh pull
        if state is None or iter_idx == 0:
            action = await _pull_fresh(prefix_key, coordinates, trajactory)
            if action is None:
                return ("", Status.self_confidence_but_failed(0.1))
            return (action, Status.confidence(1.0))

        # Have cached bunch — check confirmation
        bunch = state["bunch"]
        pos = state["position"]

        if _screen_confirms_action(screen_text, state["last_action"]):
            next_pos = pos + 1
            if next_pos < len(bunch):
                next_action = bunch[next_pos]
                _state[prefix_key] = {"bunch": bunch, "position": next_pos, "last_action": next_action}
                return (next_action, Status.confidence(1.0))
            # Bunch exhausted — re-pull
            action = await _pull_fresh(prefix_key, coordinates, trajactory)
            if action is None:
                return ("", Status.self_confidence_but_failed(0.1))
            return (action, Status.confidence(1.0))

        # Screen doesn't confirm — re-pull
        action = await _pull_fresh(prefix_key, coordinates, trajactory)
        if action is None:
            return ("", Status.self_confidence_but_failed(0.1))
        return (action, Status.confidence(1.0))

    result = FutureTensor(
        relative_to,
        speculative_get,
        [sympy.Integer(s) for s in shape],
    )
    result.ft_capacity_shape = list(shape)
    result._speculative_state = _state
    return result
