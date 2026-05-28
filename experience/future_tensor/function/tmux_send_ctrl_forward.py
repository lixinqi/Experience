"""
tmux_send_ctrl_forward :=
    FutureTensor
    <- $input FutureTensor
    <- $session_name FutureTensor  # broadcastable to $input
    # inline

Send control characters to a tmux pane via libtmux with literal=False, enter=False.
$input elements contain the control sequence to send.
$session_name elements contain the instance_id for session lookup.
$session_name shape is broadcastable to $input shape (size-1 dims broadcast).
Output shape = $input shape.
"""

from typing import List

import libtmux
import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.function.tmux_send_text_forward import _broadcast_coords


def tmux_send_ctrl_forward(
    input_ft: FutureTensor,
    session_name_ft: FutureTensor,
) -> FutureTensor:
    """Forward: create a lazy FutureTensor whose ft_async_get sends ctrl to tmux."""
    shape = input_ft.ft_capacity_shape
    schema = input_ft.ft_shape_schema
    relative_to = input_ft.ft_initial_static_tensor.st_relative_to
    session_shape = session_name_ft.ft_capacity_shape

    async def send_async_get(coordinates: List[int], trajactory: str):
        # Read ctrl sequence from input_ft
        if input_ft.ft_forwarded:
            _coeff, filepath = input_ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                ctrl = f.read()
        else:
            ctrl, _status = await input_ft.ft_async_get(coordinates, trajactory)

        # Guard: never send empty/whitespace ctrl sequence to the terminal
        if not ctrl.strip():
            return ("", Status.confidence(1.0))

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
            pane.send_keys(ctrl, literal=False, enter=False)
            return ("", Status.confidence(1.0))
        except Exception:
            return ("", Status.self_confidence_but_failed(0.0))

    result = FutureTensor(
        relative_to,
        send_async_get,
        list(schema),
    )
    result.ft_capacity_shape = list(shape)
    return result
