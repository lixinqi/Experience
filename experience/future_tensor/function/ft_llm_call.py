"""
FtLlmCall := torch.autograd.Function[
    $forward  Import[{future_tensor function llm_call_forward.viba}],
    $backward Import[{future_tensor function llm_call_backward.viba}]
]

ft_llm_call = FtLlmCall.apply

LLM call wrapped as a FutureTensor autograd Function.
Takes prompt text, sends to LLM, returns response.
Supports 1st and 2nd derivatives via symbolic gradients.
"""

import torch
from typing import Dict, Optional

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.llm_call_forward import llm_call_forward
from experience.future_tensor.function.llm_call_backward import llm_call_backward
from experience.future_tensor.function.llm_call_2nd import LlmCallGradFn
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function import symbolic_grad_registry
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class FtLlmCall(torch.autograd.Function):
    """Autograd Function for LLM call in a FutureTensor pipeline.

    Forward: FutureTensor -- lazy, async LLM call (prompt → response).
    Backward: Symbolic gradient -- LLM reflects on how to improve the prompt.
    """

    @staticmethod
    def forward(
        ctx,
        prompt_ft: FutureTensor,
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
    ) -> FutureTensor:
        output = llm_call_forward(prompt_ft, llm_method, llm_env)

        # Save for backward
        ctx.prompt_ft = prompt_ft
        ctx.output_ft = output
        ctx.llm_method = llm_method
        ctx.llm_env = llm_env

        # Save shape/relative_to for attr reconstruction in backward
        ctx.shape = prompt_ft.ft_capacity_shape
        ctx.relative_to = prompt_ft.ft_initial_static_tensor.st_relative_to

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # After forward + ft_forward, FutureTensors have materialized .ft_initial_static_tensor
        output_st = ctx.output_ft.ft_initial_static_tensor
        input_st = ctx.prompt_ft.ft_initial_static_tensor

        # Check if input has content
        has_content = input_st.data.sum().item() > 0
        if has_content:
            input_st.requires_grad_(True)
        else:
            input_st.requires_grad_(False)

        # If grad_output lacks st_* attrs, wrap as TODO symbolic tensor
        symbolic_grad = symbolic_grad_registry.pop(output_st.st_tensor_uid)
        if symbolic_grad is not None:
            grad_output = symbolic_grad
        elif not hasattr(grad_output, "st_relative_to"):
            symbolic_grad_output = todo_tensor_like(output_st)
            symbolic_grad_output.data.copy_(grad_output.data)
            grad_output = symbolic_grad_output

        # 1st-derivative dispatch: skip GradFn if dispatcher handles it.
        dispatch = get_backward_dispatcher(llm_call_backward)
        if dispatch({
            "input": input_st, "output": output_st,
            "llm_method": ctx.llm_method, "llm_env": ctx.llm_env,
        }):
            return None, None, None

        # Call GradFn for 2nd-derivative support
        grad_input = LlmCallGradFn.apply(
            grad_output,
            input_st,
            output_st,
            ctx.llm_method,
            ctx.llm_env,
        )

        # Register symbolic grad for upstream
        if grad_input is not None:
            symbolic_grad_registry.register(input_st.st_tensor_uid, grad_input)

        # Return grads for (prompt_ft, llm_method, llm_env)
        return grad_input, None, None


def ft_llm_call(
    prompt_ft: FutureTensor,
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> FutureTensor:
    """LLM call with autograd support.

    Async forward: each element's prompt text is sent to an LLM, and the
    LLM response becomes the output.

    Backward: LLM reflects on how to improve the prompt given the output
    gradient (symbolic gradient via text diffs).

    Args:
        prompt_ft: FutureTensor with prompt text per element.
        llm_method: LLM method name (default "raw_llm_api").
        llm_env: Optional environment variables for LLM config.

    Returns:
        FutureTensor output containing LLM responses.
    """
    return FtLlmCall.apply(prompt_ft, llm_method, llm_env)
