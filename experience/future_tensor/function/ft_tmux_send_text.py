"""
FtTmuxSendText := torch.autograd.Function[
    $forward  Import[{future_tensor function tmux_send_text_forward.viba}],
    $backward Import[{future_tensor function tmux_send_text_backward.viba}]
]

ft_tmux_send_text = FtTmuxSendText.apply
"""

import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.tmux_send_text_forward import tmux_send_text_forward
from experience.future_tensor.function.tmux_send_text_backward import tmux_send_text_backward


class FtTmuxSendText(torch.autograd.Function):
    """Autograd Function for sending text to tmux sessions."""

    @staticmethod
    def forward(ctx, input_ft: FutureTensor, session_name_ft: FutureTensor):
        ctx.input_ft = input_ft
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_initial_static_tensor.st_relative_to
        return tmux_send_text_forward(input_ft, session_name_ft)

    @staticmethod
    def backward(ctx, grad_output):
        return tmux_send_text_backward(ctx, grad_output), None


def ft_tmux_send_text(
    input_ft: FutureTensor,
    session_name_ft: FutureTensor,
) -> FutureTensor:
    """Send text to tmux sessions.

    Each element of ``input_ft`` contains the text to send.
    Each element of ``session_name_ft`` contains the instance_id for session lookup.
    ``session_name_ft`` is broadcastable to ``input_ft`` (size-1 dims broadcast).

    Args:
        input_ft: FutureTensor whose elements contain text to send.
        session_name_ft: FutureTensor whose elements contain instance IDs.
            Broadcastable to input_ft shape.

    Returns:
        A FutureTensor with the same shape as ``input_ft``.
    """
    return FtTmuxSendText.apply(input_ft, session_name_ft)
