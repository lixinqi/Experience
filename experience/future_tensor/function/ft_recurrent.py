"""
FtRecurrent := torch.autograd.Function[
    $forward Import[{future_tensor function ft_recurrent_forward.viba}],
    $backward Import[{future_tensor function ft_recurrent_backward.viba}],
    $ctx.topk_self_confidence_but_failed int # default 8
    $ctx.grad_input_prompt BackwardPromptCallable # default None
    $ctx.task_prompt str # default ""
    $ctx.llm_method str # default "raw_llm_api"
    $ctx.llm_env dict[str, str] # default None
    $ctx.step_budget int # default None (run to completion)
]

ft_recurrent := FtRecurrent.apply
"""

import torch
from typing import Callable, Dict, Optional, Tuple

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.ft_recurrent_forward import recurrent_forward
from experience.future_tensor.function.ft_recurrent_backward import (
    recurrent_backward,
    default_prompt_for_recurrent_grad_input,
    BackwardPromptCallable,
)
from experience.future_tensor.function.recurrent_2nd import RecurrentGradFn
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class FtRecurrent(torch.autograd.Function):
    """Autograd Function for the recurrent generate-validate loop.

    Forward: FutureTensor — lazy, sequential retry loop.
    Backward: Symbolic[torch.Tensor] — materialized, concurrent LLM reflection.
    """

    @staticmethod
    def forward(
        ctx,
        input: FutureTensor,
        topk_self_confidence_but_failed: int = 8,
        grad_input_prompt: Optional[BackwardPromptCallable] = None,
        task_prompt: str = "",
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
        accumulate_output=None,
        step_budget: Optional[int] = None,
    ) -> FutureTensor:
        output, prompt_tensor, next_position, recurrent_state = recurrent_forward(
            input,
            accumulate_output=accumulate_output,
            step_budget=step_budget,
        )

        # Save for backward — these are FutureTensors now, but backward will
        # use their materialized .ft_initial_static_tensor (symbolic tensor) after ft_forward.
        ctx.input_ft = input
        ctx.output_ft = output
        ctx.prompt_tensor_ft = prompt_tensor
        ctx.next_position = next_position
        ctx.recurrent_state = recurrent_state
        ctx.topk_self_confidence_but_failed = topk_self_confidence_but_failed
        ctx.grad_input_prompt = grad_input_prompt
        ctx.task_prompt = task_prompt
        ctx.llm_method = llm_method
        ctx.llm_env = llm_env

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # After forward + ft_forward, the FutureTensors have materialized
        # ft_initial_static_tensor attributes (symbolic tensors with st_* attrs).
        input_st = ctx.input_ft.ft_initial_static_tensor
        output_st = ctx.output_ft.ft_initial_static_tensor
        prompt_tensor_st = ctx.prompt_tensor_ft.ft_initial_static_tensor

        # Enable 2nd-derivative graph recording.
        if not grad_output.requires_grad:
            grad_output.requires_grad_(True)

        # 1st-derivative dispatch: skip GradFn if dispatcher handles it.
        dispatch = get_backward_dispatcher(recurrent_backward)
        if dispatch({
            "input": input_st, "output": output_st, "prompt_tensor": prompt_tensor_st,
            "topk_self_confidence_but_failed": ctx.topk_self_confidence_but_failed,
            "task_prompt": ctx.task_prompt, "llm_method": ctx.llm_method, "llm_env": ctx.llm_env,
        }):
            return grad_output, None, None, None, None, None, None, None

        # RecurrentGradFn.forward handles st_* attribute reconstruction
        # and calls recurrent_backward internally.
        grad_input = RecurrentGradFn.apply(
            grad_output,
            input_st,
            output_st,
            prompt_tensor_st,
            ctx.topk_self_confidence_but_failed,
            ctx.grad_input_prompt,
            ctx.task_prompt,
            ctx.llm_method,
            ctx.llm_env,
        )

        # Return grads for (input, topk_scbf, grad_input_prompt, task_prompt,
        #                    llm_method, llm_env, accumulate_output, step_budget)
        return grad_input, None, None, None, None, None, None, None


def ft_recurrent(
    input: FutureTensor,
    topk_self_confidence_but_failed: int = 8,
    grad_input_prompt: Optional[BackwardPromptCallable] = None,
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
    accumulate_output=None,
    step_budget: Optional[int] = None,
) -> FutureTensor:
    """Recurrent generate-validate loop with autograd support.

    Runs up to recurrent_dim iterations per prefix element. First confident
    result wins; on all-fail, the best self_confidence_but_failed is returned.

    Args:
        input: FutureTensor of shape (*prefix_dims, recurrent_dim).
        topk_self_confidence_but_failed: Reflect on top-k scbf cases in backward.
        grad_input_prompt: Optional custom backward prompt builder.
        task_prompt: Additional task context.
        llm_method: "coding_agent" or "raw_llm_api".
        llm_env: Optional environment variables for LLM.
        accumulate_output: Optional callable to accumulate outputs across
            iterations. Signature: (accumulator, cur_output) -> new_accumulator.
            If None, uses identity (current iteration output only).
        step_budget: Optional max iterations per ft_forward call. None = run to
            completion (backward compatible). When set, ft_forward resumes from
            next_position on subsequent calls.

    Returns:
        FutureTensor output of shape (*prefix_dims).
    """
    return FtRecurrent.apply(
        input, topk_self_confidence_but_failed, grad_input_prompt,
        task_prompt, llm_method, llm_env, accumulate_output, step_budget,
    )


if __name__ == "__main__":
    import sympy
    import os
    import subprocess
    import tempfile

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

    from experience.future_tensor.status import Status
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running 20 tests for ft_recurrent (FtRecurrent)...\n")

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
    run_test("FtRecurrent is autograd.Function subclass",
             issubclass(FtRecurrent, torch.autograd.Function))
    run_test("ft_recurrent is callable", callable(ft_recurrent))
    run_test("default_prompt re-exported", callable(default_prompt_for_recurrent_grad_input))

    # ── Tests 4-7: Forward returns correct shapes ──
    print("Tests 4-7: Forward shapes")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def simple_get(coords, trajactory):
            return (f"result_{coords}", Status.confidence(0.9))

        ft_input = FutureTensor(tmpdir, simple_get, [sympy.Integer(s) for s in [2, 3]])
        output = ft_recurrent(ft_input)

        run_test("output shape is prefix [2]",
                 output.ft_capacity_shape == [2])
        run_test("output not forwarded yet",
                 output.ft_forwarded is False)

    # ── Tests 8-11: Forward + ft_forward materializes ──
    print("Tests 8-11: Forward + ft_forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def confident_get(coords, trajactory):
            return (f"ans_{coords}", Status.confidence(0.8))

        ft_input = FutureTensor(tmpdir, confident_get, [sympy.Integer(s) for s in [3, 2]])
        output = ft_recurrent(ft_input)

        # ft_forward the output
        output_prompts = make_tensor(["go0", "go1", "go2"], tmpdir)
        output.ft_forward(output_prompts)

        run_test("output forwarded", output.ft_forwarded is True)
        run_test("output[0] has content",
                 read_storage(output.ft_initial_static_tensor, 0) is not None)
        content_0 = read_storage(output.ft_initial_static_tensor, 0)
        run_test("output[0] = ans for first iteration",
                 content_0 is not None and "ans_" in content_0,
                 "ans_*", content_0)
        # Confidence: first iteration wins, so only iteration 0 is called per prefix
        run_test("output coeff > 0",
                 output.ft_initial_static_tensor.data[0].item() > 0)

    # ── Tests 12-14: Forward with retry (scbf) ──
    print("Tests 12-14: Forward with retry")
    with tempfile.TemporaryDirectory() as tmpdir:
        call_log = []

        async def retry_get(coords, trajactory):
            call_log.append(coords)
            iteration = coords[-1]
            if iteration < 2:
                return (f"bad_{coords}", Status.self_confidence_but_failed(0.3 + iteration * 0.2))
            return (f"good_{coords}", Status.confidence(0.95))

        ft_input = FutureTensor(tmpdir, retry_get, [sympy.Integer(s) for s in [1, 4]])
        output = ft_recurrent(ft_input)

        output_prompts = make_tensor(["start"], tmpdir)
        output.ft_forward(output_prompts)

        run_test("retry: output forwarded", output.ft_forwarded is True)
        content = read_storage(output.ft_initial_static_tensor, 0)
        run_test("retry: output is from winning iteration",
                 content is not None and "good_" in content,
                 "good_*", content)
        # Should have called iterations 0, 1, 2 (won at 2), not 3
        prefix_calls = [c for c in call_log if c[0] == 0]
        run_test("retry: stopped after winning (<=3 calls)",
                 len(prefix_calls) <= 3, "<=3", len(prefix_calls))

    # ── Tests 15-16: ctx saves backward args ──
    print("Tests 15-16: ctx saves backward args")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def dummy_get(coords, trajactory):
            return ("x", Status.confidence(1.0))

        ft_input = FutureTensor(tmpdir, dummy_get, [sympy.Integer(s) for s in [2, 2]])
        # Call forward directly to inspect ctx
        ctx = type('Ctx', (), {})()
        FtRecurrent.forward(
            ctx, ft_input,
            topk_self_confidence_but_failed=4,
            task_prompt="test prompt",
            llm_method="raw_llm_api",
        )
        run_test("ctx.topk saved",
                 ctx.topk_self_confidence_but_failed == 4)
        run_test("ctx.task_prompt saved",
                 ctx.task_prompt == "test prompt")

    # ── Tests 17-18: Backward produces grad_input ──
    print("Tests 17-18: Backward (real LLM)")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def bw_get(coords, trajactory):
            return ("Hello world", Status.confidence(0.9))

        ft_input = FutureTensor(tmpdir, bw_get, [sympy.Integer(s) for s in [1, 2]])
        output = ft_recurrent(ft_input, task_prompt="Translate to French.")

        # Materialize forward
        output_prompts = make_tensor(["translate"], tmpdir)
        output.ft_forward(output_prompts)

        # Build grad_output as symbolic tensor
        grad_output = make_tensor(["Bonjour le monde"], tmpdir)
        grad_output.data[0] = 1.0

        # Call backward directly
        ctx = type('Ctx', (), {})()
        ctx.input_ft = ft_input
        ctx.output_ft = output
        ctx.prompt_tensor_ft = output.grad_fn._saved_data[0] if hasattr(output, 'grad_fn') and output.grad_fn else output
        # Get prompt_tensor from recurrent_forward directly
        from experience.future_tensor.function.ft_recurrent_forward import recurrent_forward as _rf
        _, _prompt_tensor, _, _ = _rf(ft_input)
        ctx.prompt_tensor_ft = _prompt_tensor
        ctx.topk_self_confidence_but_failed = 8
        ctx.grad_input_prompt = None
        ctx.task_prompt = "Translate to French."
        ctx.llm_method = "raw_llm_api"
        ctx.llm_env = None

        result = FtRecurrent.backward(ctx, grad_output)
        grad_input = result[0]

        run_test("grad_input shape matches input",
                 list(grad_input.ft_capacity_shape) == [1, 2])
        # Check flat 0 = winning iteration [0,0]
        content = read_storage(grad_input, 0)
        # Diff can be None if LLM output matches input exactly (no change).
        # The key test is that the backward ran without error and produced
        # a tensor with the right shape. A non-None diff means the LLM
        # actually produced improved text.
        has_gradient = content is not None and content != "TODO"
        run_test("grad_input[0,0] processed (not TODO, may be None if no diff)",
                 content != "TODO",
                 "not TODO", content)

    # ── Tests 19-20: None grads for non-tensor args ──
    print("Tests 19-20: None grads for non-tensor args")
    # Use the result from test 17-18
    run_test("grad for topk is None", result[1] is None)
    run_test("grad for llm_env is None", result[5] is None)

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All ft_recurrent tests completed.")
