"""
FtTmuxSendKeys := torch.autograd.Function[
    $forward  Import[{future_tensor function tmux_send_keys_forward.viba}],
    $backward Import[{future_tensor function tmux_send_keys_backward.viba}]
]

ft_tmux_send_keys = FtTmuxSendKeys.apply

Merged op: handles both text ("T " prefix) and ctrl ("C " prefix) in one op.
Prefix protocol:
    "T echo hello"  → pane.send_keys("echo hello", literal=True, enter=False)
    "C Enter"       → pane.send_keys("Enter", literal=False, enter=False)
    (no prefix)     → fallback: literal=True, enter=False
"""

import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.tmux_send_keys_forward import tmux_send_keys_forward
from experience.future_tensor.function.tmux_send_keys_backward import tmux_send_keys_backward


class FtTmuxSendKeys(torch.autograd.Function):
    """Autograd Function for sending keys (text or ctrl) to tmux sessions."""

    @staticmethod
    def forward(ctx, input_ft: FutureTensor, session_name_ft: FutureTensor):
        ctx.input_ft = input_ft
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_initial_static_tensor.st_relative_to
        return tmux_send_keys_forward(input_ft, session_name_ft)

    @staticmethod
    def backward(ctx, grad_output):
        return tmux_send_keys_backward(ctx, grad_output), None


def ft_tmux_send_keys(
    input_ft: FutureTensor,
    session_name_ft: FutureTensor,
) -> FutureTensor:
    """Send keys to tmux sessions (unified text + ctrl).

    Each element of ``input_ft`` contains a prefixed command:
        "T <text>"  — send literal text (literal=True)
        "C <key>"   — send control/key name (literal=False)
        (no prefix) — fallback to literal text

    Each element of ``session_name_ft`` contains the instance_id for session lookup.
    ``session_name_ft`` is broadcastable to ``input_ft`` (size-1 dims broadcast).

    Args:
        input_ft: FutureTensor whose elements contain prefixed key commands.
        session_name_ft: FutureTensor whose elements contain instance IDs.
            Broadcastable to input_ft shape.

    Returns:
        A FutureTensor with the same shape as ``input_ft``.
    """
    return FtTmuxSendKeys.apply(input_ft, session_name_ft)
