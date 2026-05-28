"""
ft_terminal_idle_gate: Gate a FutureTensor on terminal-idle state.

Passes text through only when the terminal's command line is empty (idle).
If there is already content on the command line, returns empty string —
a human doesn't re-type when text is already there.

Prefix-aware: only gates "T " prefixed inputs (text needs idle terminal).
"C " prefixed inputs (ctrl keys) pass through always — ctrl keys like
Enter are valid regardless of terminal state.

No autograd — pure runtime gate, transparent to backward.
"""

from typing import List

import libtmux
import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.function.tmux_send_text_forward import _broadcast_coords


def _terminal_has_command(pane) -> bool:
    """Check if the terminal's last prompt line already has a command typed.

    Detects prompt markers (λ, $) and checks whether there is content
    after the prompt + path token.
    """
    captured = pane.capture_pane()
    if not captured:
        return False
    non_empty = [l for l in captured if l.strip()]
    if not non_empty:
        return False
    last = non_empty[-1]

    if "λ" in last:
        after = last.split("λ", 1)[1].strip()
        tokens = after.split()
        # Skip the path token (starts with / or ~)
        pi = next((i for i, t in enumerate(tokens) if t.startswith(("/", "~"))), -1)
        cmd_tokens = tokens[pi + 1:] if pi >= 0 else tokens
        return len(cmd_tokens) > 0

    if "$ " in last:
        after = last.split("$ ", 1)[1].strip()
        return len(after) > 0

    return False


def ft_terminal_idle_gate(
    input_ft: FutureTensor,
    session_name_ft: FutureTensor,
) -> FutureTensor:
    """Gate text through only when the terminal command line is idle.

    If the terminal already has content on the command line, returns empty
    string with confidence 1.0 (skip silently). Otherwise passes through
    the input text unchanged.

    Args:
        input_ft: FutureTensor whose elements contain text to gate.
        session_name_ft: FutureTensor whose elements contain instance IDs
            for tmux session lookup. Broadcastable to input_ft shape.

    Returns:
        A lazy FutureTensor with the same shape as input_ft.
    """
    shape = input_ft.ft_capacity_shape
    relative_to = input_ft.ft_initial_static_tensor.st_relative_to
    session_shape = session_name_ft.ft_capacity_shape

    async def gate_get(coordinates: List[int], trajactory: str):
        # Read text from input
        if input_ft.ft_forwarded:
            _coeff, filepath = input_ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text, _status = await input_ft.ft_async_get(coordinates, trajactory)

        if not text.strip():
            return ("", Status.confidence(1.0))

        # "C " prefixed inputs (ctrl keys) pass through always —
        # ctrl keys like Enter are valid regardless of terminal state
        if text.startswith("C "):
            return (text, Status.confidence(1.0))

        # Read instance_id from session_name_ft (broadcast coordinates)
        session_coords = _broadcast_coords(coordinates, session_shape, shape)
        if session_name_ft.ft_forwarded:
            _coeff, filepath = session_name_ft.ft_get_materialized_value(session_coords)
            with open(filepath, "r", encoding="utf-8") as f:
                instance_id = f.read().strip()
        else:
            content, _status = await session_name_ft.ft_async_get(session_coords, trajactory)
            instance_id = content.strip()

        session_name = f"{tmux_session_prefix}{instance_id}"

        # Check terminal state (only for "T " prefixed or unprefixed text)
        try:
            server = libtmux.Server()
            session = None
            for s in server.sessions:
                if s.session_name == session_name:
                    session = s
                    break
            if session is None:
                return ("", Status.self_confidence_but_failed(0.0))
            pane = session.active_window.active_pane
            if _terminal_has_command(pane):
                return ("", Status.confidence(1.0))
        except Exception:
            pass

        # Terminal is idle — pass through
        return (text, Status.confidence(1.0))

    result = FutureTensor(
        relative_to,
        gate_get,
        [sympy.Integer(s) for s in shape],
    )
    result.ft_capacity_shape = list(shape)
    return result
