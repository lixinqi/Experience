"""
build_expert_context_backward :=
    FutureTensor
    <- $ctx AutogradContext
    <- $grad_output FutureTensor
    # inline

Backward for ft_build_expert_context.
Context construction is deterministic string concatenation — gradient passes
through to the input (and experience_text) unchanged.
Reconstructs FutureTensor attributes on grad_output (stripped by autograd),
enables requires_grad for 2nd-derivative graph recording, and calls
BuildExpertContextGradFn.apply.
"""

from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.build_expert_context_2nd import BuildExpertContextGradFn
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


def build_expert_context_backward(ctx, grad_output) -> FutureTensor:
    """Backward for ft_build_expert_context: reconstruct attrs + GradFn.

    Context construction is deterministic concatenation of input + experience_text
    + task_prompt. The gradient passes through to input unchanged.
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

    dispatch = get_backward_dispatcher(build_expert_context_backward)
    if dispatch({
        "input": ctx.input_ft.ft_initial_static_tensor,
        "experience_text": ctx.experience_text_ft.ft_initial_static_tensor,
        "output": ctx.output_ft.ft_initial_static_tensor,
        "task_prompt": ctx.task_prompt_st,
    }):
        return grad_output

    return BuildExpertContextGradFn.apply(
        grad_output,
        ctx.input_ft.ft_initial_static_tensor,
        ctx.experience_text_ft.ft_initial_static_tensor,
        ctx.output_ft.ft_initial_static_tensor,
        ctx.task_prompt_st,
    )
