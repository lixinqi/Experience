"""
retrieve_experience_backward :=
    FutureTensor
    <- $ctx AutogradContext
    <- $grad_output FutureTensor
    # inline

Backward for ft_retrieve_experience.
Experience retrieval selects entries — gradient passes through to input unchanged.
Reconstructs FutureTensor attributes on grad_output (stripped by autograd),
enables requires_grad for 2nd-derivative graph recording, and calls
RetrieveExperienceGradFn.apply.
"""

from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.retrieve_experience_2nd import RetrieveExperienceGradFn
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


def retrieve_experience_backward(ctx, grad_output) -> FutureTensor:
    """Backward for ft_retrieve_experience: reconstruct attrs + GradFn.

    Retrieval is a selection op — the gradient passes through to input.
    The selected_indexes_map is recorded for dispatch but does not produce
    a gradient for experience (experience gradients come from ft_expert's
    backward, not from retrieval).
    """
    if not hasattr(grad_output, "ft_initial_static_tensor"):
        shape: List[int] = ctx.shape
        relative_to: str = ctx.relative_to

        async def dummy_get(coords, trajactory):
            return ("", Status.confidence(0.0))

        ref_ft = FutureTensor(
            relative_to, dummy_get, [sympy.Integer(s) for s in shape],
        )
        if grad_output.numel() == 1:
            if shape:
                ref_ft.ft_initial_static_tensor.data.flatten().fill_(grad_output.item())
            else:
                ref_ft.ft_initial_static_tensor.data.fill_(grad_output.item())
        else:
            ref_ft.ft_initial_static_tensor.data.copy_(
                grad_output.data.view(ref_ft.ft_initial_static_tensor.shape)
            )
        ref_ft.ft_forwarded = True

        grad_output.ft_initial_static_tensor = ref_ft.ft_initial_static_tensor
        grad_output.ft_capacity_shape = ref_ft.ft_capacity_shape
        grad_output.ft_async_get = ref_ft.ft_async_get
        grad_output.ft_forwarded = ref_ft.ft_forwarded
        grad_output.ft_shape_schema = ref_ft.ft_shape_schema
        grad_output.ft_incremental_concated_tensors = (
            ref_ft.ft_incremental_concated_tensors
        )

    if not grad_output.requires_grad:
        grad_output.requires_grad_(True)

    dispatch = get_backward_dispatcher(retrieve_experience_backward)
    if dispatch({
        "input": ctx.input_ft.ft_initial_static_tensor,
        "output": ctx.output_ft.ft_initial_static_tensor,
        "experience": ctx.experience,
        "indexes_map": ctx.indexes_map,
    }):
        return grad_output

    return RetrieveExperienceGradFn.apply(
        grad_output,
        ctx.input_ft.ft_initial_static_tensor,
        ctx.output_ft.ft_initial_static_tensor,
        ctx.experience,
        ctx.indexes_map,
    )
