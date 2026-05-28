"""
FtTmuxSendCtrl := torch.autograd.Function[
    $forward  Import[{future_tensor function tmux_send_ctrl_forward.viba}],
    $backward Import[{future_tensor function tmux_send_ctrl_backward.viba}]
]

ft_tmux_send_ctrl = FtTmuxSendCtrl.apply
"""

import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.tmux_send_ctrl_forward import tmux_send_ctrl_forward
from experience.future_tensor.function.tmux_send_ctrl_backward import tmux_send_ctrl_backward


class FtTmuxSendCtrl(torch.autograd.Function):
    """Autograd Function for sending control characters to tmux sessions."""

    @staticmethod
    def forward(ctx, input_ft: FutureTensor, session_name_ft: FutureTensor):
        ctx.input_ft = input_ft
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_initial_static_tensor.st_relative_to
        return tmux_send_ctrl_forward(input_ft, session_name_ft)

    @staticmethod
    def backward(ctx, grad_output):
        return tmux_send_ctrl_backward(ctx, grad_output), None


def ft_tmux_send_ctrl(
    input_ft: FutureTensor,
    session_name_ft: FutureTensor,
) -> FutureTensor:
    """Send control characters to tmux sessions.

    Each element of ``input_ft`` contains the control sequence to send.
    Each element of ``session_name_ft`` contains the instance_id for session lookup.
    ``session_name_ft`` is broadcastable to ``input_ft`` (size-1 dims broadcast).

    Args:
        input_ft: FutureTensor whose elements contain control sequences.
        session_name_ft: FutureTensor whose elements contain instance IDs.
            Broadcastable to input_ft shape.

    Returns:
        A FutureTensor with the same shape as ``input_ft``.
    """
    return FtTmuxSendCtrl.apply(input_ft, session_name_ft)
