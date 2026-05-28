"""
tmux_create_session_forward :=
    FutureTensor
    <- $input FutureTensor
    # inline

Create one tmux session per element using libtmux.
Session name: f"{tmux_session_prefix}{instance_id}" where instance_id
is read from the input FutureTensor's symbolic content.
"""

from typing import List

import libtmux
import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_session import tmux_session_prefix


def tmux_create_session_forward(input_ft: FutureTensor) -> FutureTensor:
    """Forward: create a lazy FutureTensor whose ft_async_get creates tmux sessions."""
    shape = input_ft.ft_capacity_shape
    schema = input_ft.ft_shape_schema
    relative_to = input_ft.ft_initial_static_tensor.st_relative_to

    async def tmux_async_get(coordinates: List[int], trajactory: str):
        if input_ft.ft_forwarded:
            _coeff, filepath = input_ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                instance_id = f.read().strip()
        else:
            content, _status = await input_ft.ft_async_get(coordinates, trajactory)
            instance_id = content.strip()

        session_name = f"{tmux_session_prefix}{instance_id}"

        try:
            server = libtmux.Server()
            if server.has_session(session_name):
                session = server.sessions.get(session_name=session_name)
                pane = session.active_window.active_pane
                pane.send_keys("clear", enter=True)
                return ("", Status.confidence(1.0))
            server.new_session(session_name=session_name, attach=False)
            return ("", Status.confidence(1.0))
        except Exception:
            return ("", Status.self_confidence_but_failed(0.0))

    result = FutureTensor(
        relative_to,
        tmux_async_get,
        list(schema),
    )
    result.ft_capacity_shape = list(shape)
    return result
