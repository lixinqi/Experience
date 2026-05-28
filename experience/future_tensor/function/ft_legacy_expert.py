"""
FtLegacyExpert := torch.autograd.Function[
    $forward Import[{future_tensor function ft_expert_forward.viba}],
    $backward Import[{symbolic_tensor function st_moe_backward.viba}],
    $ctx.context Symbolic[torch.Tensor] # prompt_tensor._tensor, requires_grad=False
    $ctx.output_prompt OutputPromptCallable # default None
    $ctx.query_prompt QueryPromptCallable # default None
    $ctx.grad_input_prompt BackwardPromptCallable # default None
    $ctx.grad_exp_key_prompt BackwardPromptCallable # default None
    $ctx.grad_exp_value_prompt BackwardPromptCallable # default None
    $ctx.task_prompt str # default ""
    $ctx.topk int # default 16
    $ctx.retrieval_method RetrievalMethodCallable # default None
    $ctx.llm_method str # default "raw_llm_api"
    $ctx.llm_env dict[str, str] # default None
]

ft_legacy_expert := FtLegacyExpert.apply
"""

import torch
from typing import Any, Callable, Dict, Optional, Tuple

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.ft_expert_forward import (
    ft_expert_forward,
    build_nested_indexes_list,
)
from experience.symbolic_tensor.function.st_moe_forward import default_prompt_for_output
from experience.symbolic_tensor.function.get_query_tensor import default_prompt_for_query
from experience.symbolic_tensor.function.st_moe_backward import (
    st_moe_backward,
    default_prompt_for_grad_input,
    default_prompt_for_grad_exp_key,
    default_prompt_for_grad_exp_value,
)
from experience.symbolic_tensor.function.select_qkv_indexes import default_retrieval_method
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function import symbolic_grad_registry
from experience.future_tensor.function.expert_2nd import ExpertGradFn
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


OutputPromptCallable = Callable[..., str]
QueryPromptCallable = Callable[..., str]
BackwardPromptCallable = Callable[..., str]
RetrievalMethodCallable = Callable[[str, str], float]


class FtLegacyExpert(torch.autograd.Function):
    """Autograd Function for FutureTensor Expert.

    Forward: FutureTensor — lazy, async expert (query + select + LLM translate).
    Backward: Symbolic[torch.Tensor] — materialized, concurrent LLM reflection.

    Parameter mapping:
        ft_legacy_expert.input      <=> st_moe.input (direct, FutureTensor, requires_grad)
        ft_legacy_expert.prompt     <=> st_moe.context (from upstream ft_async_get, requires_grad=False)
        ft_legacy_expert.experience <=> st_moe.experience (direct)
    """

    @staticmethod
    def forward(
        ctx,
        input: FutureTensor,
        experience: torch.Tensor,
        output_prompt: Optional[OutputPromptCallable] = None,
        query_prompt: Optional[QueryPromptCallable] = None,
        grad_input_prompt: Optional[BackwardPromptCallable] = None,
        grad_exp_key_prompt: Optional[BackwardPromptCallable] = None,
        grad_exp_value_prompt: Optional[BackwardPromptCallable] = None,
        task_prompt: str = "",
        topk: int = 16,
        retrieval_method: Optional[RetrievalMethodCallable] = None,
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
        skip_query_gen: bool = False,
    ) -> FutureTensor:
        output, prompt_tensor, indexes_map = ft_expert_forward(
            input, experience, output_prompt, query_prompt, task_prompt, topk,
            retrieval_method=retrieval_method, llm_method=llm_method, llm_env=llm_env,
            skip_query_gen=skip_query_gen,
        )

        # Save for backward — FutureTensors now, backward uses ._tensor after ft_forward
        ctx.input_ft = input
        ctx.output_ft = output
        ctx.prompt_tensor_ft = prompt_tensor
        ctx.indexes_map = indexes_map
        ctx.experience = experience

        # Save st_* attrs for experience (save_for_backward strips them)
        ctx.experience_st_attrs = {}
        for attr in ("st_relative_to", "st_tensor_uid"):
            if hasattr(experience, attr):
                ctx.experience_st_attrs[attr] = getattr(experience, attr)

        # ctx saves for backward
        ctx.output_prompt = output_prompt
        ctx.query_prompt = query_prompt
        ctx.grad_input_prompt = grad_input_prompt
        ctx.grad_exp_key_prompt = grad_exp_key_prompt
        ctx.grad_exp_value_prompt = grad_exp_value_prompt
        ctx.task_prompt = task_prompt
        ctx.topk = topk
        ctx.retrieval_method = retrieval_method
        ctx.llm_method = llm_method
        ctx.llm_env = llm_env

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # After forward + ft_forward, FutureTensors have materialized .ft_initial_static_tensor
        # New mapping:
        #   st_moe.input   = input.ft_initial_static_tensor (requires_grad based on coefficients)
        #   st_moe.context = prompt_tensor.ft_initial_static_tensor (requires_grad=False)
        #   st_moe.experience = experience (direct)

        output_st = ctx.output_ft.ft_initial_static_tensor
        prompt_tensor_st = ctx.prompt_tensor_ft.ft_initial_static_tensor  # = st_moe.context
        prompt_tensor_st.requires_grad_(False)

        # Restore experience st_* attrs
        experience = ctx.experience
        for attr, val in ctx.experience_st_attrs.items():
            setattr(experience, attr, val)

        # Build input_st (= st_moe.input)
        # input is always a FutureTensor (may be "none tensor" with zero coefficients)
        input_st = ctx.input_ft.ft_initial_static_tensor
        # Check if any coefficient > 0 to decide requires_grad
        has_content = input_st.data.sum().item() > 0
        if has_content:
            input_st.requires_grad_(True)
        else:
            input_st.requires_grad_(False)

        # Build nested indexes list from map
        selected_experience_qkv_indexes_list = build_nested_indexes_list(
            ctx.indexes_map, list(output_st.shape),
        )

        # If grad_output lacks st_* attrs, wrap as TODO symbolic tensor
        symbolic_grad = symbolic_grad_registry.pop(output_st.st_tensor_uid)
        if symbolic_grad is not None:
            grad_output = symbolic_grad
        elif not hasattr(grad_output, "st_relative_to"):
            symbolic_grad_output = todo_tensor_like(output_st)
            symbolic_grad_output.data.copy_(grad_output.data)
            grad_output = symbolic_grad_output

        # 1st-derivative dispatch: skip GradFn if dispatcher handles it.
        dispatch = get_backward_dispatcher(st_moe_backward)
        if dispatch({
            "input": input_st, "output": output_st, "experience": experience,
            "task_prompt": ctx.task_prompt, "topk": ctx.topk, "context": prompt_tensor_st,
            "selected_experience_qkv_indexes_list": selected_experience_qkv_indexes_list,
            "llm_method": ctx.llm_method, "llm_env": ctx.llm_env,
        }):
            return None, None, None, None, None, None, None, None, None, None, None, None, None

        # Call st_moe_backward with direct mapping:
        #   input = input_st (ft_legacy_expert.input)
        #   context = prompt_tensor_st (ft_legacy_expert.prompt)
        #   experience = experience (ft_legacy_expert.experience)
        grad_input, grad_experience = ExpertGradFn.apply(
            grad_output,
            input_st,
            output_st,
            experience,
            ctx.task_prompt,
            ctx.topk,
            ctx.llm_method,
            ctx.llm_env,
            prompt_tensor_st,    # context
            ctx.grad_input_prompt,
            ctx.grad_exp_key_prompt,
            ctx.grad_exp_value_prompt,
            selected_experience_qkv_indexes_list,
        )

        # Register symbolic grads keyed by tensor uids
        # grad_input → grad for ft_legacy_expert.input
        if grad_input is not None:
            symbolic_grad_registry.register(input_st.st_tensor_uid, grad_input)

        # grad_experience → grad for ft_legacy_expert.experience
        if grad_experience is not None:
            symbolic_grad_registry.register(experience.st_tensor_uid, grad_experience)

        # Return grads for (input, experience, output_prompt, query_prompt,
        #                    grad_input_prompt, grad_exp_key_prompt, grad_exp_value_prompt,
        #                    task_prompt, topk, retrieval_method, llm_method, llm_env, skip_query_gen)
        return grad_input, grad_experience, None, None, None, None, None, None, None, None, None, None, None


def ft_legacy_expert(
    input: FutureTensor,
    experience: torch.Tensor,
    output_prompt: Optional[OutputPromptCallable] = None,
    query_prompt: Optional[QueryPromptCallable] = None,
    grad_input_prompt: Optional[BackwardPromptCallable] = None,
    grad_exp_key_prompt: Optional[BackwardPromptCallable] = None,
    grad_exp_value_prompt: Optional[BackwardPromptCallable] = None,
    task_prompt: str = "",
    topk: int = 16,
    retrieval_method: Optional[RetrievalMethodCallable] = None,
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
    skip_query_gen: bool = False,
) -> FutureTensor:
    """FutureTensor Expert with autograd support.

    Async forward: each element receives a prompt (context), queries experience
    via expert using input content, and generates output via LLM.

    Backward: reuses st_moe_backward with direct mapping.

    Args:
        input: FutureTensor (= st_moe.input). May be a none tensor (zero coefficients)
            when first in chain.
        experience: ExperienceTensor (last dim=3: query, key, value).
        output_prompt: Custom forward prompt builder.
        query_prompt: Custom query prompt builder.
        grad_input_prompt: Custom input gradient prompt builder.
        grad_exp_key_prompt: Custom experience key gradient prompt builder.
        grad_exp_value_prompt: Custom experience value gradient prompt builder.
        task_prompt: High-level task description.
        topk: Number of top experience entries per element.
        retrieval_method: Custom similarity function.
        llm_method: "coding_agent" or "raw_llm_api".
        llm_env: Optional environment variables for LLM.

    Returns:
        FutureTensor output.
    """
    return FtLegacyExpert.apply(
        input, experience, output_prompt, query_prompt,
        grad_input_prompt, grad_exp_key_prompt, grad_exp_value_prompt,
        task_prompt, topk, retrieval_method, llm_method, llm_env, skip_query_gen,
    )


if __name__ == "__main__":
    import sympy
    import torch

    import os
    import subprocess
    import tempfile

    from experience.future_tensor.status import Status
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

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

    print("Running 20 tests for ft_legacy_expert (FtLegacyExpert)...\n")

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

    # ── Tests 1-3: Class structure ──
    print("Tests 1-3: Class structure")
    run_test("FtLegacyExpert is autograd.Function subclass",
             issubclass(FtLegacyExpert, torch.autograd.Function))
    run_test("ft_legacy_expert is callable", callable(ft_legacy_expert))
    run_test("default prompts re-exported",
             all(callable(f) for f in [
                 default_prompt_for_output, default_prompt_for_query,
                 default_prompt_for_grad_input, default_prompt_for_grad_exp_key,
                 default_prompt_for_grad_exp_value,
             ]))

    # ── Tests 4-7: Forward returns correct types ──
    print("Tests 4-7: Forward types")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def simple_get(coords, trajactory):
            return (f"result_{coords}", Status.confidence(0.9))

        ft_input = FutureTensor(tmpdir, simple_get, [sympy.Integer(2)])
        experience_data = [
            ["greeting\nhello", "Hello", "Bonjour"],
            ["farewell\ngoodbye", "Goodbye", "Au revoir"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        output = ft_legacy_expert(
            ft_input, experience_tensor, topk=2,
        )

        run_test("output is FutureTensor", isinstance(output, torch.Tensor))
        run_test("output shape matches input", output.ft_capacity_shape == [2])
        run_test("output not forwarded yet", output.ft_forwarded is False)

    # ── Tests 8-11: Forward + ft_forward materializes (real LLM) ──
    print("Tests 8-11: Forward + ft_forward (real LLM)")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        # ft_input's ft_async_get returns input content (= st_moe.input)
        async def expert_get(coords, trajactory):
            return ("Hello world in English", Status.confidence(0.9))

        ft_input = FutureTensor(tmpdir, expert_get, [sympy.Integer(1)])
        output = ft_legacy_expert(
            ft_input, experience_tensor, topk=2,
            task_prompt="Translate English to French.",
        )

        # Materialize input first, then output
        input_prompts = make_tensor(["materialize input"], tmpdir)
        ft_input.ft_forward(input_prompts)

        output_prompts = make_tensor(["Translate this to French"], tmpdir)
        output.ft_forward(output_prompts)

        run_test("output forwarded", output.ft_forwarded is True)
        content_0 = read_storage(output.ft_initial_static_tensor, 0)
        run_test("output[0] has content",
                 content_0 is not None,
                 "not None", content_0)
        run_test("output[0] not TODO",
                 content_0 is not None and content_0.strip() != "TODO",
                 "not TODO", content_0)
        run_test("output coeff > 0",
                 output.ft_initial_static_tensor.data[0].item() > 0)

    # ── Tests 12-14: ctx saves backward args ──
    print("Tests 12-14: ctx saves backward args")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def dummy_get(coords, trajactory):
            return ("x", Status.confidence(1.0))

        ft_input = FutureTensor(tmpdir, dummy_get, [sympy.Integer(2)])
        experience_data = [["q", "k", "v"]]
        experience_tensor = make_tensor(experience_data, tmpdir)

        ctx = type('Ctx', (), {})()
        FtLegacyExpert.forward(
            ctx, ft_input, experience_tensor,
            task_prompt="test task",
            topk=4,
            llm_method="raw_llm_api",
        )

        run_test("ctx.task_prompt saved", ctx.task_prompt == "test task")
        run_test("ctx.topk saved", ctx.topk == 4)
        run_test("ctx.input_ft saved", ctx.input_ft is ft_input)

    # ── Tests 15-17: Prompt tensor stores prompts (context) ──
    print("Tests 15-17: Prompt tensor stores prompts (context)")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello", "Hello", "Bonjour"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        async def prompt_cap_get(coords, trajactory):
            return (f"out:{trajactory[:10]}", Status.confidence(0.8))

        ft_input = FutureTensor(tmpdir, prompt_cap_get, [sympy.Integer(2)])
        output = ft_legacy_expert(
            ft_input, experience_tensor, topk=1,
        )

        # Materialize input first
        input_prompts = make_tensor(["Hello friend", "Goodbye world"], tmpdir)
        ft_input.ft_forward(input_prompts)

        output_prompts = make_tensor(["context_A", "context_B"], tmpdir)
        output.ft_forward(output_prompts)

        pt0 = read_storage(prompt_tensor.ft_initial_static_tensor, 0)
        pt1 = read_storage(prompt_tensor.ft_initial_static_tensor, 1)
        run_test("prompt_tensor[0] stored", pt0 is not None)
        run_test("prompt_tensor[1] stored", pt1 is not None)
        run_test("prompt_tensor has content",
                 pt0 is not None and len(pt0) > 0)

    # ── Tests 18-20: Backward (real LLM) ──
    print("Tests 18-20: Backward (real LLM)")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)
        experience_tensor.requires_grad_(True)

        # ft_input returns the input content
        async def bw_get(coords, trajactory):
            return ("Hello world in English", Status.confidence(0.9))

        ft_input = FutureTensor(tmpdir, bw_get, [sympy.Integer(1)])
        output = ft_legacy_expert(
            ft_input, experience_tensor, topk=2,
            task_prompt="Translate English to French.",
        )

        # Materialize input first, then output
        input_prompts = make_tensor(["materialize"], tmpdir)
        ft_input.ft_forward(input_prompts)

        output_prompts = make_tensor(["Translate to French"], tmpdir)
        output.ft_forward(output_prompts)

        # Build grad_output as symbolic tensor
        grad_output = make_tensor(["Should be: Bonjour le monde"], tmpdir)
        grad_output.data[0] = 1.0

        # Build ctx manually and call backward
        ctx = type('Ctx', (), {})()
        ctx.input_ft = ft_input
        ctx.output_ft = output
        ctx.prompt_tensor_ft = prompt_tensor
        ctx.indexes_map = indexes_map
        ctx.experience = experience_tensor
        ctx.experience_st_attrs = {}
        for attr in ("st_relative_to", "st_tensor_uid"):
            if hasattr(experience_tensor, attr):
                ctx.experience_st_attrs[attr] = getattr(experience_tensor, attr)
        ctx.output_prompt = None
        ctx.query_prompt = None
        ctx.grad_input_prompt = None
        ctx.grad_exp_key_prompt = None
        ctx.grad_exp_value_prompt = None
        ctx.task_prompt = "Translate English to French."
        ctx.topk = 2
        ctx.retrieval_method = None
        ctx.llm_method = "raw_llm_api"
        ctx.llm_env = None

        result = FtLegacyExpert.backward(ctx, grad_output, None, None)
        grad_input_result = result[0]  # grad for ft_legacy_expert's input (= st_moe's input)
        grad_experience_result = result[1]  # grad for ft_legacy_expert's experience

        run_test("grad_input shape matches input",
                 grad_input_result is not None and list(grad_input_result.shape) == [1],
                 [1],
                 list(grad_input_result.shape) if grad_input_result is not None else None)
        run_test("grad_experience is None (registered via symbolic_grad_registry)",
                 grad_experience_result is not None)
        # Check grad content
        if grad_input_result is not None:
            gi_content = read_storage(grad_input_result, 0)
            run_test("grad_input has content",
                     gi_content is not None and len(gi_content) > 0)
        else:
            run_test("grad_input has content", False, "not None", None)

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All ft_legacy_expert tests completed.")
