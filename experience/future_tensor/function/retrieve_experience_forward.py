"""
retrieve_experience_forward :=
    FutureTensor
    <- $input FutureTensor
    <- $experience SymbolicTensor[..., 3]
    <- $topk int
    <- $retrieval_method Callable | None
    <- $skip_query_gen bool
    <- $query_prompt Callable | None
    <- $task_prompt str
    <- $llm_method str
    <- $llm_env dict | None
    # inline

Retrieve and format experience entries for each element of input.
Decomposed from ft_expert_forward steps 2-7.
"""

import threading
from typing import Callable, Dict, List, Optional, Tuple

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.function.get_query_tensor import get_query_tensor
from experience.symbolic_tensor.function.select_qkv_indexes import select_qkv_indexes, multi_similarity
from experience.symbolic_tensor.function.st_moe_forward import (
    _read_file_content,
    _replace_last_tensor_with_full_slice,
)
from experience.symbolic_tensor.tensor_util.st_setitem import st_setitem


def _read_experience_entries(
    experience_sliced_view: torch.Tensor,
) -> List[Tuple[str, str, str]]:
    """Read QKV entries from a sliced experience view tensor.

    Returns list of (query, key, value) tuples.
    """
    entries = []
    shape = list(experience_sliced_view.shape)
    if len(shape) < 2:
        return entries
    num_entries = shape[0]
    for i in range(num_entries):
        q = _read_file_content(experience_sliced_view, i * 3) or ""
        k = _read_file_content(experience_sliced_view, i * 3 + 1) or ""
        v = _read_file_content(experience_sliced_view, i * 3 + 2) or ""
        entries.append((q, k, v))
    return entries


def retrieve_experience_forward(
    input: FutureTensor,
    experience: torch.Tensor,
    topk: int = 16,
    retrieval_method: Optional[Callable[[str, str], float]] = None,
    skip_query_gen: bool = False,
    query_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> Tuple[FutureTensor, Dict[tuple, List[torch.Tensor]]]:
    """Forward: create a lazy FutureTensor whose ft_async_get retrieves experience.

    For each element:
      1. Read input content from upstream FutureTensor.
      2. Generate query (or use input directly if skip_query_gen).
      3. select_qkv_indexes to find topk experience entries.
      4. Read and format entries as text.

    Returns:
        (experience_text_ft, selected_indexes_map)
    """
    input_shape = input.ft_capacity_shape
    input_schema = input.ft_shape_schema
    relative_to = input.ft_initial_static_tensor.st_relative_to

    _selected_indexes_map: Dict[tuple, List[torch.Tensor]] = {}
    _qkv_lock = threading.Lock()

    async def _async_get(
        coordinates: List[int], trajactory: str,
    ) -> Tuple[str, Status]:
        # If already computed at this coordinate, read from disk
        if output.ft_initial_static_tensor.data[tuple(coordinates)].item() > 0:
            flat_idx = sum(
                c * s for c, s in zip(coordinates, output.ft_initial_static_tensor.stride())
            )
            content = _read_file_content(output.ft_initial_static_tensor, flat_idx)
            if content is not None:
                return (content, Status.confidence(1.0))

        # Unwrap prompt
        if isinstance(trajactory, dict):
            actual_trajactory = trajactory.get("trajactory", "")
        else:
            actual_trajactory = trajactory

        # Materialize input at this coordinate if needed
        if (
            not input.ft_forwarded
            and input.ft_initial_static_tensor.data[tuple(coordinates)].item() == 0
        ):
            input_content_result, input_status = await input.ft_async_get(
                coordinates, actual_trajactory,
            )
            if input_content_result:
                st_setitem(
                    input.ft_initial_static_tensor, coordinates, input_content_result,
                    coefficient=Status.convert_status_to_float(input_status),
                )

        # Read input content
        if input.ft_initial_static_tensor.data[tuple(coordinates)].item() > 0:
            flat_idx = sum(
                c * s for c, s in zip(coordinates, input.ft_initial_static_tensor.stride())
            )
            input_content = _read_file_content(input.ft_initial_static_tensor, flat_idx)
        else:
            input_content = ""

        # Run retrieval in thread executor (contains sync LLM calls for query gen)
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync_retrieve():
            if skip_query_gen:
                batch_query = input_content or ""
                effective_retrieval = retrieval_method or multi_similarity
            else:
                scalar_input = make_tensor(
                    [input_content] if input_content else ["TODO"],
                    relative_to,
                )
                input_query = get_query_tensor(
                    scalar_input, query_prompt=query_prompt,
                    task_prompt=task_prompt, llm_method=llm_method, llm_env=llm_env,
                )
                batch_query = _read_file_content(input_query, 0)
                effective_retrieval = retrieval_method

            if batch_query is None:
                _selected_indexes_map[tuple(coordinates)] = []
                return ("", Status.self_confidence_but_failed(0.1))

            # Select topk experience entries
            with _qkv_lock:
                select_experience_query_indexes, cold_start = select_qkv_indexes(
                    experience, batch_query, topk,
                    retrieval_method=effective_retrieval,
                )
            _selected_indexes_map[tuple(coordinates)] = select_experience_query_indexes

            if cold_start:
                return ("", Status.self_confidence_but_failed(0.1))

            # Expand last index to get full q/k/v
            select_experience_indexes = _replace_last_tensor_with_full_slice(
                select_experience_query_indexes, experience.size()[-1],
            )

            # Slice and read experience entries
            experience_sliced_view = slice_view(experience, select_experience_indexes)
            entries = _read_experience_entries(experience_sliced_view)

            if not entries:
                return ("", Status.self_confidence_but_failed(0.1))

            # Format entries as text (INPUT→OUTPUT for LLM clarity)
            formatted_parts = []
            for q, k, v in entries:
                if v:
                    label = q.replace("\n", " ") if q else k.replace("\n", " ")
                    formatted_parts.append(f"  INPUT: {label}\n  OUTPUT: {v}")
            experience_text = "\n\n".join(formatted_parts) if formatted_parts else ""

            if not experience_text:
                return ("", Status.self_confidence_but_failed(0.1))

            return (experience_text, Status.confidence(1.0))

        result_content, result_status = await loop.run_in_executor(
            None, _sync_retrieve,
        )

        # Write-through
        if result_content:
            st_setitem(
                output.ft_initial_static_tensor, coordinates, result_content,
                coefficient=Status.convert_status_to_float(result_status),
            )

        return (result_content, result_status)

    output = FutureTensor(
        relative_to,
        _async_get,
        list(input_schema),
    )

    return output, _selected_indexes_map
