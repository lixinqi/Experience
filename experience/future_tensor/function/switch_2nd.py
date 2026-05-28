"""
SwitchGradFn: autograd.Function wrapping switch_backward.

  forward  = switch_backward  (1st derivative: route grad to selected branch)
  backward = 2nd-derivative dispatch via the active Policy

FtSwitch.backward() calls SwitchGradFn.apply(...) instead of switch_backward(...)
directly so that backward_dispatch_start.grad.backward() naturally triggers
SwitchGradFn.backward() (the 2nd-derivative dispatch).
"""

import sympy
import torch
from typing import List

from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class SwitchGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS switch_backward (the 1st derivative)."""

    @staticmethod
    def forward(ctx, grad_output, selected_index: int, branches: List):
        from experience.future_tensor.function.switch_backward import switch_backward
        from experience.future_tensor.future_tensor import FutureTensor
        from experience.future_tensor.status import Status

        # Reconstruct FutureTensor attributes if stripped by autograd
        if not hasattr(grad_output, "ft_initial_static_tensor"):
            selected_branch = branches[selected_index]
            shape = selected_branch.ft_capacity_shape
            relative_to = selected_branch.ft_initial_static_tensor.st_relative_to

            async def dummy_get(coords, trajactory):
                return ("", Status.confidence(0.0))

            ref_ft = FutureTensor(relative_to, dummy_get, [sympy.Integer(s) for s in shape])
            if grad_output.numel() == 1:
                if shape:
                    ref_ft.ft_initial_static_tensor.data.flatten().fill_(grad_output.item())
                else:
                    ref_ft.ft_initial_static_tensor.data.fill_(grad_output.item())
            else:
                ref_ft.ft_initial_static_tensor.data.copy_(grad_output.data.view(ref_ft.ft_initial_static_tensor.shape))
            ref_ft.ft_forwarded = True

            # Monkey-patch attributes onto the existing grad_output tensor
            grad_output.ft_initial_static_tensor = ref_ft.ft_initial_static_tensor
            grad_output.ft_capacity_shape = ref_ft.ft_capacity_shape
            grad_output.ft_async_get = ref_ft.ft_async_get
            grad_output.ft_forwarded = ref_ft.ft_forwarded
            grad_output.ft_shape_schema = ref_ft.ft_shape_schema
            grad_output.ft_incremental_concated_tensors = ref_ft.ft_incremental_concated_tensors

        ctx.save_for_backward(grad_output)
        ctx.selected_index = selected_index
        ctx.branches = branches
        ctx._switch_backward_fn = switch_backward
        ctx._grad_input = grad_output

        # Run actual backward
        grad_input = switch_backward(grad_output, selected_index, branches)

        # Force creation of SwitchGradFnBackward by returning a new tensor.
        # Inside autograd.Function forward, ``+ 0`` is not tracked by autograd;
        # the returned tensor simply gets SwitchGradFnBackward as its grad_fn.
        return grad_input + 0

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_backward_dispatcher(ctx._switch_backward_fn)
        dispatch({
            "grad_output":    grad_output,
            "selected_index": ctx.selected_index,
            "branches":       ctx.branches,
            "grad_input":     ctx._grad_input,
        })

        # Gradient for (grad_output, selected_index, branches)
        return grad_grad_input, None, None
