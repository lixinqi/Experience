"""
FtBuildExpertContext := torch.autograd.Function[
    $forward  Import[{future_tensor function build_expert_context_forward.viba}],
    $backward Import[{future_tensor function build_expert_context_backward.viba}]
]

ft_build_expert_context = FtBuildExpertContext.apply
"""

import torch
from typing import Callable, Dict, Optional

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.build_expert_context_forward import (
    build_expert_context_forward,
)
from experience.future_tensor.function.build_expert_context_backward import (
    build_expert_context_backward,
)
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function import symbolic_grad_registry
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class FtBuildExpertContext(torch.autograd.Function):
    """Autograd Function for building expert context in a FutureTensor pipeline.

    Forward: FutureTensor -- lazy, async prompt construction (input + experience_text + task).
    Backward: Pass-through -- context construction is deterministic concatenation.
    """

    @staticmethod
    def forward(
        ctx,
        input_ft: FutureTensor,
        experience_text_ft: FutureTensor,
        task_prompt_st: torch.Tensor,
        output_prompt: Optional[Callable[..., str]] = None,
    ) -> FutureTensor:
        output = build_expert_context_forward(
            input_ft, experience_text_ft, task_prompt_st, output_prompt,
        )

        # Save for backward
        ctx.input_ft = input_ft
        ctx.experience_text_ft = experience_text_ft
        ctx.output_ft = output
        ctx.task_prompt_st = task_prompt_st

        # Save st_* attrs for task_prompt (save_for_backward strips them)
        ctx.task_prompt_st_attrs = {}
        for attr in ("st_relative_to", "st_tensor_uid"):
            if hasattr(task_prompt_st, attr):
                ctx.task_prompt_st_attrs[attr] = getattr(task_prompt_st, attr)

        # Save shape/relative_to for attr reconstruction in backward
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_initial_static_tensor.st_relative_to

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # If grad_output lacks st_* attrs, wrap as TODO symbolic tensor
        output_st = ctx.output_ft.ft_initial_static_tensor
        symbolic_grad = symbolic_grad_registry.pop(output_st.st_tensor_uid)
        if symbolic_grad is not None:
            grad_output = symbolic_grad
        elif not hasattr(grad_output, "st_relative_to"):
            symbolic_grad_output = todo_tensor_like(output_st)
            symbolic_grad_output.data.copy_(grad_output.data)
            grad_output = symbolic_grad_output

        # Restore task_prompt st_* attrs
        task_prompt_st = ctx.task_prompt_st
        for attr, val in ctx.task_prompt_st_attrs.items():
            setattr(task_prompt_st, attr, val)

        # 1st-derivative dispatch
        dispatch = get_backward_dispatcher(build_expert_context_backward)
        if dispatch({
            "input": ctx.input_ft.ft_initial_static_tensor,
            "experience_text": ctx.experience_text_ft.ft_initial_static_tensor,
            "output": output_st,
            "task_prompt": task_prompt_st,
        }):
            # Return grads for (input_ft, experience_text_ft, task_prompt_st, output_prompt)
            return grad_output, grad_output, grad_output, None

        grad_input = build_expert_context_backward(ctx, grad_output)

        # Register symbolic grad for upstream
        input_st = ctx.input_ft.ft_initial_static_tensor
        if grad_input is not None and hasattr(input_st, "st_tensor_uid"):
            symbolic_grad_registry.register(input_st.st_tensor_uid, grad_input)

        # Grad also flows to experience_text (pass-through)
        exp_text_st = ctx.experience_text_ft.ft_initial_static_tensor
        if grad_input is not None and hasattr(exp_text_st, "st_tensor_uid"):
            symbolic_grad_registry.register(exp_text_st.st_tensor_uid, grad_input)

        # Grad also flows to task_prompt (pass-through)
        if grad_input is not None and hasattr(task_prompt_st, "st_tensor_uid"):
            symbolic_grad_registry.register(task_prompt_st.st_tensor_uid, grad_input)

        # Return grads for (input_ft, experience_text_ft, task_prompt_st, output_prompt)
        return grad_input, grad_input, grad_input, None


def ft_build_expert_context(
    input_ft: FutureTensor,
    experience_text_ft: FutureTensor,
    task_prompt_st: torch.Tensor,
    output_prompt: Optional[Callable[..., str]] = None,
) -> FutureTensor:
    """Build the full LLM prompt with autograd support.

    Combines input content, retrieved experience text, and task_prompt into
    a full prompt for downstream LLM call.

    Backward: gradient passes through to input, experience_text, and task_prompt
    (context construction is deterministic concatenation).

    Args:
        input_ft: FutureTensor containing input text.
        experience_text_ft: FutureTensor containing formatted experience text.
        task_prompt_st: 0D trainable symbolic tensor containing task description.
        output_prompt: Custom prompt builder: (task_prompt, exp_text, input_text, prompt) -> str.

    Returns:
        FutureTensor whose elements are the full LLM prompt strings.
    """
    return FtBuildExpertContext.apply(
        input_ft, experience_text_ft, task_prompt_st, output_prompt,
    )
