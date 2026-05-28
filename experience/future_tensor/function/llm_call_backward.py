"""
llm_call_backward :=
    FutureTensor
    <- $ctx AutogradContext
    <- $grad_output FutureTensor
    # inline

Backward for ft_llm_call.
Reconstructs FutureTensor attributes on grad_output (stripped by autograd),
enables requires_grad for 2nd-derivative graph recording, and calls
LlmCallGradFn.apply.

Also contains llm_call_backward_compute: the actual gradient computation
that asks an LLM to write a "better version of the prompt" given grad_output.
"""

import os
import shutil
import tempfile
import itertools
from typing import Dict, List, Optional, Tuple, Union

import torch
import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.llm_call_2nd import LlmCallGradFn
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.tensor_util.dump_view import dump_view
from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
from experience.llm_client.agent_task import AgentTask
from experience.llm_client.task_handler import TaskHandler
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


def _scalar_slice_indices(size: torch.Size) -> List[List[int]]:
    """Generate all coordinate tuples for iterating over each scalar element."""
    ranges = [range(s) for s in size]
    return [list(coord) for coord in itertools.product(*ranges)]


def _read_storage(tensor: torch.Tensor, flat_index: int) -> Optional[str]:
    """Read the text content at a flat index, resolving symlinks."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    real_path = os.path.realpath(path)
    if not os.path.isfile(real_path):
        return None
    with open(real_path, encoding="utf-8") as f:
        return f.read()


def default_prompt_for_llm_call_grad_input(
    workspace_dir: str,
    const_grad_output_view: str,
    const_input_view: str,
    const_output_view: str,
    mutable_grad_input_dir: str,
) -> str:
    """Default prompt for computing input gradient in llm_call backward pass."""
    return (
        "You are a symbolic gradient calculator for backward pass.\n\n"
        "During forward pass, the prompt was sent to an LLM which produced the output.\n"
        "Now given the output gradient (how output should change), compute a better\n"
        "version of the prompt that would produce improved output.\n\n"
        "Context (read-only):\n"
        f"- Output gradient (text diff): \"{const_grad_output_view}\"\n"
        f"- Original prompt: \"{const_input_view}\"\n"
        f"- Original output: \"{const_output_view}\"\n\n"
        "Compute and write:\n"
        f"1. Prompt gradient in \"{mutable_grad_input_dir}\":\n"
        "   How should the prompt text change to improve the output?\n"
        f"2. File \"{mutable_grad_input_dir}/<xxx>/data\" must be a better version "
        f"of \"{const_input_view}/<xxx>/data\"\n\n"
        "Replace all TODO with improved prompt content.\n"
    )


def llm_call_backward_compute(
    grad_output: torch.Tensor,
    input_st: torch.Tensor,
    output_st: torch.Tensor,
    llm_method: str,
    llm_env: Optional[Dict[str, str]],
) -> Optional[torch.Tensor]:
    """Compute grad_input for ft_llm_call backward.

    For each element: LLM is asked "given how the output should change
    (grad_output), the original prompt, and the output, write a better
    version of the prompt".

    Returns grad_input as a symbolic tensor (unified diffs), or None if
    input doesn't require grad.
    """
    if not input_st.requires_grad:
        return None

    grad_input = todo_tensor_like(input_st)

    # Numeric channel: copy coefficients
    grad_input.data.copy_(grad_output.data)

    # Symbolic channel: per-element LLM reflection
    coords_list = _scalar_slice_indices(input_st.size())

    all_tasks: List[AgentTask] = []
    element_contexts: List[Tuple] = []

    for coords in coords_list:
        int_slices = [c for c in coords]

        scalar_grad_output = slice_view(grad_output, int_slices)
        scalar_input = slice_view(input_st, int_slices)
        scalar_output = slice_view(output_st, int_slices)
        scalar_grad_input_view = slice_view(grad_input, int_slices)
        scalar_grad_input_value = slice_tensor(grad_input, int_slices)

        workspace_dir = tempfile.mkdtemp()
        grad_output_view_dir = os.path.join(workspace_dir, "const_grad_output_view")
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        output_view_dir = os.path.join(workspace_dir, "const_output_view")
        grad_input_dir = os.path.join(workspace_dir, "mutable_grad_input_dir")

        dump_view(scalar_grad_output, grad_output_view_dir, "txt")
        dump_view(scalar_input, input_view_dir, "txt")
        dump_view(scalar_output, output_view_dir, "txt")
        dump_view(scalar_grad_input_value, grad_input_dir, "txt")

        prompt = default_prompt_for_llm_call_grad_input(
            workspace_dir, grad_output_view_dir, input_view_dir,
            output_view_dir, grad_input_dir,
        )
        agent_task = AgentTask(
            workspace_dir=workspace_dir,
            output_relative_dir=["mutable_grad_input_dir"],
            prompt=prompt,
        )
        all_tasks.append(agent_task)
        element_contexts.append((
            workspace_dir, scalar_grad_input_view,
            scalar_input, scalar_grad_input_value,
        ))

    # Run all LLM tasks in batch
    if all_tasks:
        TaskHandler()(all_tasks, llm_method, llm_env=llm_env)

    # Copy back results
    from experience.symbolic_tensor.function.st_moe_forward import _copy_back_to_storage_view

    for workspace_dir, scalar_grad_input_view, scalar_input, scalar_grad_input_value in element_contexts:
        output_dir = os.path.join(workspace_dir, "mutable_grad_input_dir")
        _copy_back_to_storage_view(output_dir, scalar_grad_input_view)
        shutil.rmtree(workspace_dir, ignore_errors=True)

    # Compute diff: grad_input = diff(original_input, improved_input)
    grad_input_diff = get_diff_tensor(input_st, grad_input)
    return grad_input_diff


def llm_call_backward(ctx, grad_output) -> Optional[torch.Tensor]:
    """Backward for ft_llm_call: reconstruct attrs + GradFn.

    Reconstructs FutureTensor attributes on grad_output (stripped by autograd),
    enables requires_grad, and calls LlmCallGradFn.apply for 2nd-derivative
    support.
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

    dispatch = get_backward_dispatcher(llm_call_backward)
    if dispatch({
        "input": ctx.prompt_ft.ft_initial_static_tensor,
        "output": ctx.output_ft.ft_initial_static_tensor,
        "llm_method": ctx.llm_method,
        "llm_env": ctx.llm_env,
    }):
        return grad_output

    return LlmCallGradFn.apply(
        grad_output,
        ctx.prompt_ft.ft_initial_static_tensor,
        ctx.output_ft.ft_initial_static_tensor,
        ctx.llm_method,
        ctx.llm_env,
    )
