"""
build_expert_context_forward :=
    FutureTensor
    <- $input FutureTensor
    <- $experience_text FutureTensor
    <- $task_prompt SymbolicTensor[0D]  # trainable
    <- $output_prompt Callable | None
    # inline

Build the full LLM prompt from input + retrieved experience text + task_prompt.
Decomposed from ft_expert_forward steps 8-10 (prompt construction).
"""

from typing import Callable, List, Optional, Tuple

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.symbolic_tensor.function.st_moe_forward import _read_file_content
from experience.symbolic_tensor.tensor_util.st_setitem import st_setitem


def _default_prompt(task_prompt: str, experience_text: str, input_text: str, prompt: str) -> str:
    """Default inline prompt: trajectory first, then task + experience + input."""
    return (
        f"{prompt}\n\n"
        f"{task_prompt}\n\n"
        f"Examples of correct input->output:\n{experience_text}\n\n"
        f"Now for this input:\n{input_text}\n\n"
        "Reply with ONLY the output. One line. No explanation. No prefix."
    )


def _read_task_prompt(task_prompt_st: torch.Tensor) -> str:
    """Read text content from a 0D symbolic tensor."""
    content = _read_file_content(task_prompt_st, 0)
    return content or ""


def build_expert_context_forward(
    input: FutureTensor,
    experience_text: FutureTensor,
    task_prompt_st: torch.Tensor,
    output_prompt: Optional[Callable[..., str]] = None,
) -> FutureTensor:
    """Forward: create a lazy FutureTensor whose ft_async_get builds the LLM prompt.

    For each element:
      1. Pull experience text from upstream.
      2. Read input content.
      3. Read task_prompt from symbolic tensor.
      4. Build full prompt via output_prompt callable (or default).

    Args:
        input: FutureTensor containing input text per element.
        experience_text: FutureTensor containing formatted experience text per element.
        task_prompt_st: 0D symbolic tensor containing task description (trainable).
        output_prompt: Custom prompt builder: (task_prompt, exp_text, input_text, prompt) -> str.

    Returns:
        FutureTensor whose elements are the full LLM prompt strings.
    """
    input_shape = input.ft_capacity_shape
    input_schema = input.ft_shape_schema
    relative_to = input.ft_initial_static_tensor.st_relative_to

    async def _async_get(
        coordinates: List[int], trajactory: str,
    ) -> Tuple[str, Status]:
        # If already computed, read from disk
        if output.ft_initial_static_tensor.data[tuple(coordinates)].item() > 0:
            flat_idx = sum(
                c * s for c, s in zip(coordinates, output.ft_initial_static_tensor.stride())
            )
            content = _read_file_content(output.ft_initial_static_tensor, flat_idx)
            if content is not None:
                return (content, Status.confidence(1.0))

        if isinstance(trajactory, dict):
            actual_trajactory = trajactory.get("trajactory", "")
        else:
            actual_trajactory = trajactory

        # Pull experience text from upstream
        exp_text, exp_status = await experience_text.ft_async_get(
            coordinates, actual_trajactory,
        )

        if not exp_status.is_confidence or not exp_text:
            return ("", exp_status)

        # Read input content (already materialized by retrieve_experience_forward)
        if input.ft_initial_static_tensor.data[tuple(coordinates)].item() > 0:
            flat_idx = sum(
                c * s for c, s in zip(coordinates, input.ft_initial_static_tensor.stride())
            )
            input_content = _read_file_content(input.ft_initial_static_tensor, flat_idx) or ""
        else:
            input_content = ""

        # Read task_prompt from symbolic tensor
        task_prompt = _read_task_prompt(task_prompt_st)

        # Build prompt
        if output_prompt is not None:
            full_prompt = output_prompt(task_prompt, exp_text, input_content, actual_trajactory)
        else:
            full_prompt = _default_prompt(task_prompt, exp_text, input_content, actual_trajactory)

        # Write-through
        st_setitem(
            output.ft_initial_static_tensor, coordinates, full_prompt,
            coefficient=1.0,
        )

        return (full_prompt, Status.confidence(1.0))

    output = FutureTensor(
        relative_to,
        _async_get,
        list(input_schema),
    )

    return output
