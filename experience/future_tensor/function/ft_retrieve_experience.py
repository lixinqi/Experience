"""
FtRetrieveExperience := torch.autograd.Function[
    $forward  Import[{future_tensor function retrieve_experience_forward.viba}],
    $backward Import[{future_tensor function retrieve_experience_backward.viba}]
]

ft_retrieve_experience = FtRetrieveExperience.apply
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.retrieve_experience_forward import (
    retrieve_experience_forward,
)
from experience.future_tensor.function.retrieve_experience_backward import (
    retrieve_experience_backward,
)
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function import symbolic_grad_registry
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class FtRetrieveExperience(torch.autograd.Function):
    """Autograd Function for experience retrieval in a FutureTensor pipeline.

    Forward: FutureTensor -- lazy, async experience retrieval (query + select + format).
    Backward: Pass-through -- retrieval is a selection op, gradient flows to input.
    """

    @staticmethod
    def forward(
        ctx,
        input_ft: FutureTensor,
        experience: torch.Tensor,
        topk: int = 16,
        retrieval_method: Optional[Callable[[str, str], float]] = None,
        skip_query_gen: bool = False,
        query_prompt: Optional[Callable[..., str]] = None,
        task_prompt: str = "",
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
    ) -> FutureTensor:
        output, indexes_map = retrieve_experience_forward(
            input_ft, experience, topk, retrieval_method,
            skip_query_gen, query_prompt, task_prompt, llm_method, llm_env,
        )

        # Save for backward
        ctx.input_ft = input_ft
        ctx.output_ft = output
        ctx.experience = experience
        ctx.indexes_map = indexes_map

        # Save shape/relative_to for attr reconstruction in backward
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_initial_static_tensor.st_relative_to

        # Save st_* attrs for experience
        ctx.experience_st_attrs = {}
        for attr in ("st_relative_to", "st_tensor_uid"):
            if hasattr(experience, attr):
                ctx.experience_st_attrs[attr] = getattr(experience, attr)

        # Also stash indexes_map on the output FutureTensor for downstream access
        output._selected_indexes_map = indexes_map

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Restore experience st_* attrs
        for attr, val in ctx.experience_st_attrs.items():
            setattr(ctx.experience, attr, val)

        # If grad_output lacks st_* attrs, wrap as TODO symbolic tensor
        output_st = ctx.output_ft.ft_initial_static_tensor
        symbolic_grad = symbolic_grad_registry.pop(output_st.st_tensor_uid)
        if symbolic_grad is not None:
            grad_output = symbolic_grad
        elif not hasattr(grad_output, "st_relative_to"):
            symbolic_grad_output = todo_tensor_like(output_st)
            symbolic_grad_output.data.copy_(grad_output.data)
            grad_output = symbolic_grad_output

        # 1st-derivative dispatch
        dispatch = get_backward_dispatcher(retrieve_experience_backward)
        if dispatch({
            "input": ctx.input_ft.ft_initial_static_tensor,
            "output": output_st,
            "experience": ctx.experience,
            "indexes_map": ctx.indexes_map,
        }):
            # Return grads for (input_ft, experience, topk, retrieval_method,
            #                   skip_query_gen, query_prompt, task_prompt, llm_method, llm_env)
            return grad_output, None, None, None, None, None, None, None, None

        grad_input = retrieve_experience_backward(ctx, grad_output)

        # Register symbolic grad for upstream
        input_st = ctx.input_ft.ft_initial_static_tensor
        if grad_input is not None and hasattr(input_st, "st_tensor_uid"):
            symbolic_grad_registry.register(input_st.st_tensor_uid, grad_input)

        # Return grads for (input_ft, experience, topk, retrieval_method,
        #                   skip_query_gen, query_prompt, task_prompt, llm_method, llm_env)
        return grad_input, None, None, None, None, None, None, None, None


def ft_retrieve_experience(
    input_ft: FutureTensor,
    experience: torch.Tensor,
    topk: int = 16,
    retrieval_method: Optional[Callable[[str, str], float]] = None,
    skip_query_gen: bool = False,
    query_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> FutureTensor:
    """Retrieve and format experience entries with autograd support.

    For each element:
      1. Read input content from upstream.
      2. Generate query (or use input directly if skip_query_gen).
      3. select_qkv_indexes to find topk experience entries.
      4. Format entries as text.

    Backward: gradient passes through to input (retrieval is a selection op).

    Args:
        input_ft: FutureTensor with text content.
        experience: ExperienceTensor (last dim=3: query, key, value).
        topk: Number of top experience entries.
        retrieval_method: Custom similarity function.
        skip_query_gen: If True, use input directly as query.
        query_prompt: Custom query prompt builder.
        task_prompt: High-level task description.
        llm_method: LLM method for query generation.
        llm_env: Optional environment variables.

    Returns:
        FutureTensor of formatted experience text.
        Access ._selected_indexes_map on the result for index tracking.
    """
    return FtRetrieveExperience.apply(
        input_ft, experience, topk, retrieval_method,
        skip_query_gen, query_prompt, task_prompt, llm_method, llm_env,
    )
