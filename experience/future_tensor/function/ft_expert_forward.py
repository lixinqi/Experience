"""
ft_expert_forward: Async FutureTensor version of st_moe_forward.

Parameter mapping:
    st_moe.input      <=> ft_expert.input (direct, FutureTensor, requires_grad)
    st_moe.context    <=> ft_expert.prompt (from upstream ft_async_get, requires_grad=False)
    st_moe.experience <=> ft_expert.experience (direct)

The output is a FutureTensor whose ft_async_get:
  1. Takes (coordinates, prompt) from upstream
  2. If input coefficient > 0, reads input[coordinates] content as st_moe.input
  3. Uses prompt as st_moe.context (requires_grad=False)
  4. Runs expert: query, select topk, workspace, LLM
  5. Returns (output_content, Status)
"""

import os
import shutil
import tempfile
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

import sympy

from experience.future_tensor.future_tensor import FutureTensor, _read_element, _coords_to_flat
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.tensor_util.dump_view import dump_view
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.get_query_tensor import get_query_tensor
from experience.symbolic_tensor.function.select_qkv_indexes import select_qkv_indexes, multi_similarity
from experience.symbolic_tensor.function.st_moe_forward import (
    default_prompt_for_output,
    _copy_back_to_storage_view,
    _read_file_content,
    _replace_last_tensor_with_full_slice,
)
from experience.llm_client.agent_task import AgentTask
from experience.llm_client.task_handler import TaskHandler
from experience.symbolic_tensor.tensor_util.st_setitem import st_setitem


def ft_expert_forward(
    input: FutureTensor,
    experience: torch.Tensor,
    output_prompt: Optional[Callable[..., str]] = None,
    query_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    topk: int = 16,
    retrieval_method: Optional[Callable[[str, str], float]] = None,
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
    skip_query_gen: bool = False,
) -> Tuple[FutureTensor, FutureTensor, Any]:
    """Async FutureTensor version of st_moe_forward.

    Parameter mapping:
        st_moe.input      <=> ft_expert.input (direct, FutureTensor, requires_grad)
        st_moe.context    <=> ft_expert.prompt (from upstream ft_async_get, requires_grad=False)
        st_moe.experience <=> ft_expert.experience (direct)

    For each element, the ft_async_get callback:
      1. Receives a prompt string (= st_moe.context)
      2. If input coefficient > 0, reads input[coordinates] content (= st_moe.input)
      3. Generates query keywords, selects topk experience entries
      4. Builds workspace (with context view), runs LLM
      5. Returns (output_content, Status)

    Args:
        input: FutureTensor (= st_moe.input). May be a none tensor (zero coefficients)
            when first in chain.
        experience: ExperienceTensor (last dim=3: query, key, value).
        output_prompt: Custom forward prompt builder. None uses default.
        query_prompt: Custom query prompt builder. None uses default.
        task_prompt: High-level task description.
        topk: Number of top experience entries per element.
        retrieval_method: Custom similarity function. None uses default.
        llm_method: "coding_agent" or "raw_llm_api".
        llm_env: Optional environment variables for LLM.

    Returns:
        (output, prompt_tensor, selected_experience_qkv_indexes_map):
            output: FutureTensor of same shape as input.
            prompt_tensor: FutureTensor storing prompts (for backward as context).
            selected_experience_qkv_indexes_map: Dict mapping coordinates to indexes.
    """
    input_shape = input.ft_capacity_shape
    input_schema = input.ft_shape_schema

    # Create prompt_tensor: same shape as input, stores prompts for backward (= st_moe.context)
    prompt_tensor = FutureTensor(
        input.ft_static_tensor.st_relative_to,
        ft_async_get=None,  # not used — written directly via st_setitem
        ft_shape_schema=list(input_schema),
    )

    # Mutable container for selected indexes (filled during ft_async_get)
    _selected_indexes_map: Dict[tuple, List[torch.Tensor]] = {}

    # Lock for thread-safe access to select_qkv_indexes (creates cached view dir)
    _qkv_lock = threading.Lock()

    def _sync_expert_for_element(
        coordinates: List[int], trajactory: str,
    ) -> Tuple[str, Status]:
        """Synchronous per-element expert: query, select, workspace, LLM, read output.

        Runs in a thread executor so that internal asyncio.run() calls
        (inside TaskHandler / get_query_tensor) don't conflict with the
        outer event loop driving FutureTensor.ft_forward.
        """
        # 1. Store the trajactory into prompt_tensor for backward (= st_moe.context)
        if isinstance(trajactory, dict):
            trajactory = trajactory.get("trajactory", "")
        st_setitem(prompt_tensor.ft_static_tensor, coordinates, trajactory)

        # 2. Build scalar_input from input[coordinates] (= st_moe.input)
        #    Coefficient-based check: input is always a FutureTensor, but may be
        #    a "none tensor" with zero coefficients when first in chain.
        if input.ft_static_tensor.data[tuple(coordinates)].item() > 0:
            flat_idx = sum(c * s for c, s in zip(coordinates, input.ft_static_tensor.stride()))
            input_content = _read_file_content(input.ft_static_tensor, flat_idx)
        else:
            input_content = ""

        if skip_query_gen:
            # Use input_content directly as query (no LLM keyword generation)
            batch_input_query_file_content = input_content or ""
            effective_retrieval = retrieval_method or multi_similarity
            scalar_input = make_tensor(
                [input_content] if input_content else ["TODO"],
                input.ft_static_tensor.st_relative_to,
            )
            scalar_context = make_tensor([trajactory], input.ft_static_tensor.st_relative_to)
        else:
            scalar_input = make_tensor(
                [input_content] if input_content else ["TODO"],
                input.ft_static_tensor.st_relative_to,
            )

            # 3. Build scalar_context from trajactory (= st_moe.context, requires_grad=False)
            scalar_context = make_tensor([trajactory], input.ft_static_tensor.st_relative_to)

            # 4. Get query from scalar_input
            input_query = get_query_tensor(
                scalar_input, query_prompt=query_prompt,
                task_prompt=task_prompt, llm_method=llm_method, llm_env=llm_env,
            )

            # 5. Read query file content
            batch_input_query_file_content = _read_file_content(input_query, 0)
            effective_retrieval = retrieval_method

        if batch_input_query_file_content is None:
            _selected_indexes_map[tuple(coordinates)] = []
            return ("", Status.self_confidence_but_failed(0.1))

        # 6. Select topk experience entries (lock: creates cached qkv_data_view dir)
        with _qkv_lock:
            select_experience_query_indexes, cold_start = select_qkv_indexes(
                experience, batch_input_query_file_content, topk,
                retrieval_method=effective_retrieval,
            )
        _selected_indexes_map[tuple(coordinates)] = select_experience_query_indexes

        # Cold-start: no meaningful experience — skip LLM entirely
        if cold_start:
            return ("", Status.self_confidence_but_failed(0.1))

        # Replace last index tensor with full slice to keep q/k/v
        select_experience_indexes = _replace_last_tensor_with_full_slice(
            select_experience_query_indexes, experience.size()[-1],
        )

        # 7. Slice experience to get relevant entries
        experience_sliced_view = slice_view(experience, select_experience_indexes)

        # 8. Create scalar TODO output
        scalar_output = todo_tensor_like(scalar_input)
        scalar_output_view = slice_view(scalar_output, [0])
        scalar_output_value = slice_tensor(scalar_output, [0])

        # 9. Create workspace and dump views
        workspace_dir = tempfile.mkdtemp()
        exp_view_dir = os.path.join(workspace_dir, "const_experiance_view")
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        context_view_dir = os.path.join(workspace_dir, "const_context_view")
        output_dir = os.path.join(workspace_dir, "mutable_output_dir")

        scalar_input_view = slice_view(scalar_input, [0])
        scalar_context_view = slice_view(scalar_context, [0])
        dump_view(experience_sliced_view, exp_view_dir, "txt")
        dump_view(scalar_input_view, input_view_dir, "txt")
        # context: dump alongside input (prompt as context)
        dump_view(scalar_context_view, context_view_dir, "txt")
        dump_view(scalar_output_value, output_dir, "txt")

        # 10. Build prompt and run LLM
        prompt_str = (output_prompt or default_prompt_for_output)(
            task_prompt, workspace_dir, exp_view_dir, input_view_dir, output_dir,
        )
        agent_task = AgentTask(
            workspace_dir=workspace_dir,
            output_relative_dir="mutable_output_dir",
            prompt=prompt_str,
        )
        TaskHandler()([agent_task], llm_method, llm_env=llm_env)

        # 11. Copy back results
        _copy_back_to_storage_view(output_dir, scalar_output_view)

        # 12. Read output content
        output_content = _read_file_content(scalar_output, 0)

        # Cleanup
        shutil.rmtree(workspace_dir, ignore_errors=True)

        if output_content is None or output_content.strip() == "TODO":
            return ("", Status.self_confidence_but_failed(0.5))
        return (output_content, Status.confidence(1.0))

    async def ft_expert_forward_async_get(
        coordinates: List[int], trajactory: str
    ) -> Tuple[str, Status]:
        # If output already computed at this coordinate, return from disk
        if output.ft_static_tensor.data[tuple(coordinates)].item() > 0:
            flat_idx = sum(c * s for c, s in zip(coordinates, output.ft_static_tensor.stride()))
            content = _read_file_content(output.ft_static_tensor, flat_idx)
            if content is not None:
                return (content, Status.confidence(1.0))

        # Ensure input is materialized at this coordinate (lazy pull + write-through)
        if isinstance(trajactory, dict):
            actual_trajactory = trajactory.get("trajactory", "")
        else:
            actual_trajactory = trajactory
        if not input.ft_forwarded and input.ft_static_tensor.data[tuple(coordinates)].item() == 0:
            input_content, input_status = await input.ft_async_get(coordinates, actual_trajactory)
            if input_content:
                st_setitem(input.ft_static_tensor, coordinates, input_content,
                           coefficient=Status.convert_status_to_float(input_status))

        import asyncio
        loop = asyncio.get_event_loop()
        result_content, result_status = await loop.run_in_executor(
            None, _sync_expert_for_element, coordinates, trajactory,
        )

        # Write-through: persist output so downstream ops can read from this tensor
        if result_content:
            st_setitem(output.ft_static_tensor, coordinates, result_content,
                       coefficient=Status.convert_status_to_float(result_status))
        return (result_content, result_status)

    output = FutureTensor(
        input.ft_static_tensor.st_relative_to,
        ft_expert_forward_async_get,
        ft_shape_schema=list(input_schema),
    )

    return output, prompt_tensor, _selected_indexes_map


def build_nested_indexes_list(
    indexes_map: Dict[tuple, List[torch.Tensor]],
    shape: List[int],
) -> Any:
    """Convert flat indexes_map to nested list matching shape.

    Called after ft_forward has populated the map.
    """
    import itertools
    if not shape:
        return indexes_map.get((), [])

    coords_list = [
        list(coords)
        for coords in itertools.product(*[range(s) for s in shape])
    ]

    flat_results = [indexes_map.get(tuple(c), []) for c in coords_list]

    def _unflatten(flat: list, shp: List[int]) -> Any:
        if not shp:
            return flat[0]
        if len(shp) == 1:
            return flat
        chunk = 1
        for s in shp[1:]:
            chunk *= s
        return [_unflatten(flat[i * chunk:(i + 1) * chunk], shp[1:]) for i in range(shp[0])]

    return _unflatten(flat_results, shape)


if __name__ == "__main__":
    import sympy
    import torch

    import subprocess

    # Source anthropic env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    print("Running 20 tests for ft_expert_forward...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
            print(f"  \u2713 {name}")
        else:
            failed += 1
            print(f"  \u2717 {name}")
            if expected is not None:
                print(f"    expected: {expected}, actual: {actual}")

    def read_storage(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to, tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # ── Tests 1-3: Function structure ──
    print("Tests 1-3: Function structure")
    run_test("ft_expert_forward is callable", callable(ft_expert_forward))
    run_test("build_nested_indexes_list is callable", callable(build_nested_indexes_list))
    run_test("default_prompt_for_output is callable", callable(default_prompt_for_output))

    # ── Tests 4-7: Forward returns correct types and shapes ──
    print("Tests 4-7: Forward returns correct types and shapes")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def simple_get(coords, trajactory):
            return (f"result_{coords}", Status.confidence(0.9))

        ft_input = FutureTensor(tmpdir, simple_get, [sympy.Integer(2)])

        experience_data = [
            ["greeting\nhello", "Hello in English", "Bonjour en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        output, prompt_tensor, indexes_map = ft_expert_forward(
            ft_input, experience_tensor, topk=2,
        )

        run_test("output is FutureTensor", isinstance(output, torch.Tensor))
        run_test("prompt_tensor is FutureTensor", isinstance(prompt_tensor, torch.Tensor))
        run_test("output shape matches input", output.ft_capacity_shape == [2])
        run_test("indexes_map is dict", isinstance(indexes_map, dict))

    # ── Tests 5a-5c: Forward with none tensor (zero coefficients) ──
    print("Tests 5a-5c: Forward with none tensor (zero coefficients)")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello", "Hello", "Bonjour"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        # Create a "none tensor" FutureTensor: ft_async_get returns empty with low coeff
        async def none_get(coords, trajactory):
            return ("", Status.self_confidence_but_failed(0.0))

        ft_none_input = FutureTensor(tmpdir, none_get, [sympy.Integer(1)])

        output, prompt_tensor, indexes_map = ft_expert_forward(
            ft_none_input, experience_tensor, topk=1,
        )

        run_test("output shape is [1] with none tensor", output.ft_capacity_shape == [1])
        run_test("prompt_tensor shape is [1]", prompt_tensor.shape == [1])
        run_test("indexes_map is dict (none tensor)", isinstance(indexes_map, dict))

    # ── Tests 8-11: Forward + ft_forward materializes (real LLM) ──
    print("Tests 8-11: Forward + ft_forward (real LLM)")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        # ft_input returns the input content that becomes st_moe.input
        async def moe_get(coords, trajactory):
            return ("Hello world in English", Status.confidence(0.9))

        ft_input = FutureTensor(tmpdir, moe_get, [sympy.Integer(1)])

        output, prompt_tensor, indexes_map = ft_expert_forward(
            ft_input, experience_tensor, topk=2,
            task_prompt="Translate English to French.",
        )

        run_test("output not forwarded yet", output.ft_forwarded is False)

        # First materialize the input, then the output
        input_prompts = make_tensor(["materialize input"], tmpdir)
        ft_input.ft_forward(input_prompts)

        output_prompts = make_tensor(["Translate this"], tmpdir)
        output.ft_forward(output_prompts)

        run_test("output forwarded", output.ft_forwarded is True)
        content_0 = read_storage(output.ft_static_tensor, 0)
        run_test("output[0] has content",
                 content_0 is not None and content_0.strip() != "TODO",
                 "not TODO", content_0)
        run_test("output coeff > 0", output.ft_static_tensor.data[0].item() > 0)

    # ── Tests 12-14: Prompt tensor stores prompts (= context) ──
    print("Tests 12-14: Prompt tensor stores prompts (context)")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello", "Hello", "Bonjour"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        async def prompt_capture_get(coords, trajactory):
            return (f"captured:{trajactory[:10]}", Status.confidence(0.8))

        ft_input = FutureTensor(tmpdir, prompt_capture_get, [sympy.Integer(2)])
        output, prompt_tensor, indexes_map = ft_expert_forward(
            ft_input, experience_tensor, topk=1,
        )

        # Materialize input first
        input_prompts = make_tensor(["input0", "input1"], tmpdir)
        ft_input.ft_forward(input_prompts)

        output_prompts = make_tensor(["context_A", "context_B"], tmpdir)
        output.ft_forward(output_prompts)

        # prompt_tensor should have the prompts (= context) stored via st_setitem
        pt0 = read_storage(prompt_tensor.ft_static_tensor, 0)
        pt1 = read_storage(prompt_tensor.ft_static_tensor, 1)
        run_test("prompt_tensor[0] stored",
                 pt0 is not None,
                 "not None", pt0)
        run_test("prompt_tensor[1] stored",
                 pt1 is not None,
                 "not None", pt1)
        run_test("prompt_tensor[0] has content",
                 pt0 is not None and len(pt0) > 0)

    # ── Tests 15-17: Selected indexes map populated ──
    print("Tests 15-17: Selected indexes map")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello", "Hello", "Bonjour"],
            ["farewell\ngoodbye", "Goodbye", "Au revoir"],
            ["thanks\nmerci", "Thank you", "Merci"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        async def idx_get(coords, trajactory):
            return (f"result_{coords}", Status.confidence(0.9))

        ft_input = FutureTensor(tmpdir, idx_get, [sympy.Integer(2)])
        output, prompt_tensor, indexes_map = ft_expert_forward(
            ft_input, experience_tensor, topk=2,
        )

        # Materialize input first
        input_prompts = make_tensor(["Hello world", "Goodbye world"], tmpdir)
        ft_input.ft_forward(input_prompts)

        output_prompts = make_tensor(["ctx A", "ctx B"], tmpdir)
        output.ft_forward(output_prompts)

        run_test("indexes_map has entries after ft_forward",
                 len(indexes_map) > 0,
                 ">0", len(indexes_map))

        # build_nested_indexes_list
        nested = build_nested_indexes_list(indexes_map, [2])
        run_test("nested indexes is list of 2",
                 isinstance(nested, list) and len(nested) == 2)
        run_test("each entry is list of tensors",
                 all(isinstance(entry, list) for entry in nested))

    # ── Tests 18-20: Multi-element forward ──
    print("Tests 18-20: Multi-element forward (real LLM)")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello", "Hello in English", "Bonjour en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        async def multi_get(coords, trajactory):
            return (f"element_{coords[0]}: {trajactory[:20]}", Status.confidence(0.85))

        ft_input = FutureTensor(tmpdir, multi_get, [sympy.Integer(3)])
        output, prompt_tensor, indexes_map = ft_expert_forward(
            ft_input, experience_tensor, topk=2,
            task_prompt="Translate English to French.",
        )

        # Materialize input first
        input_prompts = make_tensor(
            ["Hello world", "Goodbye friend", "Thank you"],
            tmpdir,
        )
        ft_input.ft_forward(input_prompts)

        output_prompts = make_tensor(
            ["ctx_hello", "ctx_goodbye", "ctx_thanks"],
            tmpdir,
        )
        output.ft_forward(output_prompts)

        run_test("3-element output forwarded", output.ft_forwarded is True)
        all_have_content = all(
            read_storage(output.ft_static_tensor, i) is not None
            for i in range(3)
        )
        run_test("all 3 elements have content", all_have_content)
        run_test("indexes_map has 3 entries",
                 len(indexes_map) == 3, 3, len(indexes_map))

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All ft_expert_forward tests completed.")
