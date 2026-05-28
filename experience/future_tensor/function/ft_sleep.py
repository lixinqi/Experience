"""
FtSleep := torch.autograd.Function[
    $forward  Import[{future_tensor function sleep_forward.viba}],
    $backward Import[{future_tensor function sleep_backward.viba}]
]

ft_sleep = FtSleep.apply
"""

import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.sleep_forward import sleep_forward
from experience.future_tensor.function.sleep_backward import sleep_backward


class FtSleep(torch.autograd.Function):
    """Autograd Function for async sleep within a FutureTensor pipeline."""

    @staticmethod
    def forward(ctx, input_ft: FutureTensor, seconds: float):
        ctx.input_ft = input_ft
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_initial_static_tensor.st_relative_to
        return sleep_forward(input_ft, seconds)

    @staticmethod
    def backward(ctx, grad_output):
        return sleep_backward(ctx, grad_output), None


def ft_sleep(input_ft: FutureTensor, seconds: float) -> FutureTensor:
    """Pause the pipeline for ``seconds`` before continuing.

    Useful for giving tmux sessions time to initialize or commands to complete.

    Args:
        input_ft: FutureTensor providing shape and relative_to context.
        seconds: Duration to sleep in seconds.

    Returns:
        A lazy FutureTensor with the same shape as ``input_ft``.
    """
    return FtSleep.apply(input_ft, seconds)
