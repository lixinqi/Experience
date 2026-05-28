"""
FtMean := torch.autograd.Function[
    $forward  (scalar <- $input FutureTensor)
    $backward (FutureTensor <- $grad_output scalar)
]

ft_mean = FtMean.apply

# Compute mean of ft_initial_static_tensor coefficients with autograd support.
"""

import torch

from experience.future_tensor.future_tensor import _tensor_to_future


class FtMean(torch.autograd.Function):
    """Mean of a FutureTensor's ft_initial_static_tensor coefficients.

    Forward returns the scalar mean of ``input_ft.ft_initial_static_tensor.data``.
    Backward returns a FutureTensor with the same shape as ``input_ft`` where
    each coefficient is ``grad_output / numel``.

    This creates an autograd edge so that ``loss.backward()`` properly
    propagates through the FutureTensor op chain.
    """

    @staticmethod
    def forward(ctx, input_ft):
        ctx.input_ft = input_ft
        return input_ft.ft_initial_static_tensor.data.mean()

    @staticmethod
    def backward(ctx, grad_output):
        input_ft = ctx.input_ft
        shape = input_ft.ft_capacity_shape
        numel = max(input_ft.ft_initial_static_tensor.data.numel(), 1)
        val = (grad_output / numel).detach().to(torch.bfloat16)
        return _tensor_to_future(torch.full(shape, val.item(), dtype=torch.bfloat16), input_ft)


def ft_mean(input_ft):
    """Compute the mean of a FutureTensor's coefficients with autograd support."""
    return FtMean.apply(input_ft)
