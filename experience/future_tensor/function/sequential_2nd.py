"""
SequentialGradFn: autograd.Function wrapping sequential_backward.

  forward  = sequential_backward  (1st derivative)
  backward = 2nd-derivative dispatch via the active Policy

FtSequential.backward() calls SequentialGradFn.apply(...) instead of
sequential_backward(...) directly so that
backward_dispatch_start.grad.backward() naturally triggers
SequentialGradFn.backward() (the 2nd-derivative dispatch).
"""

import sympy
import torch

from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class SequentialGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS sequential_backward (the 1st derivative)."""

    @staticmethod
    def forward(ctx, grad_output, num_inputs: int, first_input):
        from experience.future_tensor.function.sequential_backward import sequential_backward
        from experience.future_tensor.future_tensor import FutureTensor
        from experience.future_tensor.status import Status

        # Reconstruct FutureTensor attributes if stripped by autograd
        if not hasattr(grad_output, "ft_initial_static_tensor"):
            shape = first_input.ft_capacity_shape
            relative_to = first_input.ft_initial_static_tensor.st_relative_to

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
        ctx.num_inputs = num_inputs
        ctx._sequential_backward_fn = sequential_backward
        ctx._grad_input = grad_output

        # Run actual backward
        grad_input = sequential_backward(grad_output, num_inputs)

        # Force creation of SequentialGradFnBackward by returning a new tensor.
        # Inside autograd.Function forward, ``+ 0`` is not tracked by autograd;
        # the returned tensor simply gets SequentialGradFnBackward as its grad_fn.
        return grad_input + 0

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_backward_dispatcher(ctx._sequential_backward_fn)
        dispatch({
            "grad_output": grad_output,
            "num_inputs":  ctx.num_inputs,
            "grad_input":  ctx._grad_input,
        })

        # Gradient for (grad_output, num_inputs, first_input)
        return grad_grad_input, None, None
