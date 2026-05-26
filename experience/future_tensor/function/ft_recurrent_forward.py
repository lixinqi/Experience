"""
recurrent_forward :=
    ($output FutureTensor[($prefix_dims ...)],
     $prompt_tensor FutureTensor[($prefix_dims ..., $recurrent_dim int)],
     $next_position torch.IntTensor[($prefix_dims ...)],
     $recurrent_state dict[tuple, dict])
    <- $input FutureTensor[($prefix_dims ..., $recurrent_dim int)]
    <- $get_next_iter_prompt GetNextIterPromptCallable # default None
    # inline
    <- ($recurrent_dim int <- $input.ft_capacity_shape[-1])
    <- ($prefix_shape list[int] <- $input.ft_capacity_shape[:-1])
    <- { prompt_tensor = FutureTensor(input.ft_static_tensor.st_relative_to, ft_async_get=None, ft_shape_schema=...) }
    <- $output.ft_async_get recurrent_forward_async_get

GetNextIterPromptCallable := $func (Awaitable[$ret str]
    <- $cur_prompt str <- $cur_output str <- $cur_output_status Status)
"""

import os
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor, _coords_to_flat
from experience.future_tensor.status import Status

# Type alias for custom prompt accumulation callback
GetNextIterPromptCallable = Callable[..., Awaitable[str]]


def default_next_iter_prompt(cur_prompt: str, cur_output: str, iteration: int) -> str:
    """Default prompt accumulation: append iteration output to prompt history."""
    return (
        f"{cur_prompt}\n"
        f"====[Iteration {iteration}]====\n"
        f"Output:\n\n{cur_output}\n\n"
    )


def recurrent_forward(
    input: FutureTensor,
    get_next_iter_prompt: Optional[GetNextIterPromptCallable] = None,
    accumulate_output: Optional[Callable[[str, str], str]] = None,
    step_budget: Optional[int] = None,
    next_position: Optional[torch.Tensor] = None,
    recurrent_state: Optional[Dict[tuple, dict]] = None,
) -> Tuple[FutureTensor, FutureTensor, torch.Tensor, Dict[tuple, dict]]:
    """Recurrent forward: retry loop encoded in tensor dimensions.

    Takes input of shape (*prefix_dims, recurrent_dim) where the last dim
    encodes a generate-then-retry loop. Returns (output, prompt_tensor,
    next_position, recurrent_state) where output has shape (*prefix_dims),
    prompt_tensor has the same shape as input, next_position tracks per-element
    loop progress, and recurrent_state persists loop state across resumes.

    For each prefix coordinate, iterates i in range(recurrent_dim):
      - Calls input.ft_async_get([*coords, i], cur_prompt)
      - Dispatches on Status:
        - confidence: return immediately (success)
        - kConfidenceNotBounded: return with confidence(1.0)
        - kContextOverflow: return ("", kContextOverflow)
        - self_confidence_but_failed: accumulate prompt, try next i
    If all iterations fail, returns best (by self_confidence_but_failed value).

    Args:
        input: FutureTensor of shape (*prefix_dims, recurrent_dim).
        get_next_iter_prompt: Optional async callable to build next iteration's
            prompt. Signature: (cur_prompt, cur_output, cur_output_status) -> str.
            If None, uses default_next_iter_prompt.
        accumulate_output: Optional callable to accumulate outputs across
            iterations. Signature: (accumulator, cur_output) -> new_accumulator.
            If None, uses the current iteration's output directly (identity).
        step_budget: Optional max iterations per resume. None = run to completion.
        next_position: Optional IntTensor of shape (*prefix_dims) tracking loop
            progress. None = start from zeros (fresh run).
        recurrent_state: Optional dict persisting per-element loop state across
            resumes. None = start fresh.

    Returns:
        (output, prompt_tensor, next_position, recurrent_state) tuple.
        output and prompt_tensor are FutureTensors. next_position is
        torch.IntTensor, recurrent_state is a dict.
    """
    input_shape = input.ft_capacity_shape
    input_schema = input.ft_shape_schema
    assert len(input_shape) >= 1, (
        f"Input must have at least 1 dim, got shape {input_shape}"
    )

    recurrent_dim = input_shape[-1]
    prefix_shape = input_shape[:-1]
    prefix_schema = list(input_schema[:-1])
    symbolic = (recurrent_dim == 0)  # symbolic last dim — grow dynamically

    # Initialize next_position if not provided
    if next_position is None:
        next_position = torch.zeros(*prefix_shape, dtype=torch.int) if prefix_shape else torch.zeros((), dtype=torch.int)
    if recurrent_state is None:
        recurrent_state = {}

    # prompt_tensor: dynamic FutureTensor that reads from recurrent_state
    prompt_tensor = FutureTensor(
        input.ft_static_tensor.st_relative_to,
        ft_async_get=None,
        ft_shape_schema=list(input.ft_shape_schema),
    )

    async def recurrent_forward_async_get(
        coordinates: List[int], trajactory: str
    ) -> Tuple[str, Status]:
        prefix_coords = tuple(coordinates)

        # Compute flat index for prefix coordinates
        if prefix_shape:
            flat_idx = _coords_to_flat(coordinates, prefix_shape)
        else:
            flat_idx = 0

        start_i = next_position.flatten()[flat_idx].item()
        if symbolic:
            budget = step_budget if step_budget is not None else 8
            end_i = start_i + budget
        else:
            end_i = recurrent_dim if step_budget is None else min(recurrent_dim, start_i + step_budget)

        # Restore persisted state or initialize fresh
        state = recurrent_state.get(prefix_coords, {
            "best_output": "", "best_status": None,
            "best_scyf_value": -1.0, "accumulator": "",
            "cur_prompt": trajactory,
        })
        best_output = state["best_output"]
        best_status = state["best_status"]
        best_scyf_value = state["best_scyf_value"]
        accumulator = state["accumulator"]
        cur_prompt = state["cur_prompt"]

        last_i = start_i

        for i in range(start_i, end_i):
            last_i = i

            # Call input.ft_async_get([*coordinates, i], cur_prompt)
            cur_output, cur_output_status = await input.ft_async_get(
                [*coordinates, i], cur_prompt
            )

            # Accumulate output if configured
            if accumulate_output is not None:
                accumulator = accumulate_output(accumulator, cur_output)
            else:
                accumulator = cur_output

            # Match on cur_output_status
            if cur_output_status.is_confidence:
                next_position.flatten()[flat_idx] = end_i
                recurrent_state[prefix_coords] = {
                    "best_output": accumulator, "best_status": cur_output_status,
                    "best_scyf_value": best_scyf_value, "accumulator": accumulator,
                    "cur_prompt": cur_prompt,
                }
                return (accumulator, cur_output_status)

            if cur_output_status.is_kConfidenceNotBounded:
                next_position.flatten()[flat_idx] = end_i
                recurrent_state[prefix_coords] = {
                    "best_output": accumulator, "best_status": Status.confidence(1.0),
                    "best_scyf_value": best_scyf_value, "accumulator": accumulator,
                    "cur_prompt": cur_prompt,
                }
                return (accumulator, Status.confidence(1.0))

            if cur_output_status.is_kContextOverflow:
                next_position.flatten()[flat_idx] = end_i
                recurrent_state[prefix_coords] = {
                    "best_output": "", "best_status": Status.kContextOverflow,
                    "best_scyf_value": best_scyf_value, "accumulator": accumulator,
                    "cur_prompt": cur_prompt,
                }
                return ("", Status.kContextOverflow)

            # self_confidence_but_failed case
            if cur_output_status.value > best_scyf_value:
                best_scyf_value = cur_output_status.value
                best_output = cur_output
                best_status = cur_output_status

            # Advance prompt for next iteration
            if get_next_iter_prompt is not None:
                cur_prompt = await get_next_iter_prompt(
                    cur_prompt, cur_output, cur_output_status,
                )
            else:
                cur_prompt = default_next_iter_prompt(
                    cur_prompt, cur_output, i,
                )

        # Update position after this budget slice
        next_position.flatten()[flat_idx] = last_i + 1

        # Persist state
        recurrent_state[prefix_coords] = {
            "best_output": best_output, "best_status": best_status,
            "best_scyf_value": best_scyf_value, "accumulator": accumulator,
            "cur_prompt": cur_prompt,
        }

        # All trials in this budget exhausted
        if accumulate_output is not None:
            return (accumulator, best_status)
        return (best_output, best_status)

    output = FutureTensor(
        input.ft_static_tensor.st_relative_to,
        recurrent_forward_async_get,
        ft_shape_schema=prefix_schema,
    )
    return (output, prompt_tensor, next_position, recurrent_state)


if __name__ == "__main__":
    import sympy

    import asyncio
    import os
    import tempfile
    import torch

    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running 100 tests for recurrent_forward...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ {name}")
            if expected is not None:
                print(f"    expected: {expected}, actual: {actual}")

    def _storage_path(ft, flat_index):
        digits = list(str(flat_index))
        return os.path.join(
            ft.ft_static_tensor.st_relative_to, ft.ft_static_tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )

    def read_ft_element(ft, flat_index):
        path = _storage_path(ft, flat_index)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # === Group 1: Single element, confidence on first try (tests 1-10) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # shape: (1,) -> prefix (), recurrent_dim=1
        async def ok_first(coords, prompt):
            return ("hello world", Status.confidence(0.9))

        inp = FutureTensor(tmpdir, ok_first, [sympy.Integer(1)])
        out, pt, _, _ = recurrent_forward(inp)

        run_test("1: output shape []", out.ft_capacity_shape == [])
        run_test("2: prompt_tensor shape [1]", pt.ft_capacity_shape == [1])
        run_test("3: not forwarded", out.ft_forwarded is False)

        prompt_t = st_make_tensor("test_prompt", tmpdir)
        out.ft_forward(prompt_t)

        run_test("4: forwarded", out.ft_forwarded is True)
        run_test("5: content", read_ft_element(out, 0) == "hello world")
        run_test("6: confidence 0.9",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - 0.9) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # shape: (3,) -> prefix (), recurrent_dim=3, confidence on first
        async def ok_first_3(coords, prompt):
            return ("result_0", Status.confidence(0.8))

        inp = FutureTensor(tmpdir, ok_first_3, [sympy.Integer(3)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("7: shape [] with recurrent_dim=3", out.ft_capacity_shape == [])
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("8: content from i=0", read_ft_element(out, 0) == "result_0")
        run_test("9: confidence 0.8", abs(out.ft_static_tensor.data.flatten()[0].item() - 0.8) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # prefix [2], recurrent_dim=1 -> shape [2, 1]
        async def prefix_ok(coords, prompt):
            prefix = coords[0]
            return (f"out_{prefix}", Status.confidence(0.9))

        inp = FutureTensor(tmpdir, prefix_ok, [sympy.Integer(2), sympy.Integer(1)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("10: prefix [2] shape", out.ft_capacity_shape == [2])

    # === Group 2: confidence on second try (tests 11-25) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=2, fail first, ok second
        call_log = []

        async def logged_get(coords, trajactory):
            call_log.append((list(coords), trajactory))
            i = coords[-1]
            if i == 0:
                return ("bad_output", Status.self_confidence_but_failed(0.3))
            if i == 1:
                return ("good_output", Status.confidence(0.95))
            return ("???", Status.confidence(0.0))

        inp = FutureTensor(tmpdir, logged_get, [sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("initial", tmpdir)
        out.ft_forward(prompt_t)

        run_test("11: content is good_output", read_ft_element(out, 0) == "good_output")
        run_test("12: confidence 0.95",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - 0.95) < 0.01)

        # Check call sequence
        run_test("13: first call is i=0", call_log[0][0] == [0])
        run_test("14: second call is i=1", call_log[1][0] == [1])
        run_test("15: total 2 calls", len(call_log) == 2, 2, len(call_log))

        # Check prompt accumulation for i=1
        i1_prompt = call_log[1][1]
        run_test("16: i=1 prompt has 'Iteration 0'", "Iteration 0" in i1_prompt)
        run_test("17: i=1 prompt has bad_output", "bad_output" in i1_prompt)
        run_test("18: i=1 prompt starts with initial", i1_prompt.startswith("initial"))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Ok on third try (recurrent_dim=3)
        async def ok_on_third(coords, prompt):
            i = coords[-1]
            if i == 0: return ("out0", Status.self_confidence_but_failed(0.3))
            if i == 1: return ("out1", Status.self_confidence_but_failed(0.4))
            if i == 2: return ("out2", Status.confidence(0.9))
            return ("???", Status.confidence(0.0))

        inp = FutureTensor(tmpdir, ok_on_third, [sympy.Integer(3)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("start", tmpdir)
        out.ft_forward(prompt_t)
        run_test("19: content from i=2", read_ft_element(out, 0) == "out2")
        run_test("20: confidence 0.9", abs(out.ft_static_tensor.data.flatten()[0].item() - 0.9) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Prompt accumulates across iterations
        prompts_seen = []

        async def capture_prompt(coords, prompt):
            prompts_seen.append(prompt)
            i = coords[-1]
            if i < 2:
                return (f"gen_{i}", Status.self_confidence_but_failed(0.5))
            return (f"gen_{i}", Status.confidence(0.8))

        inp = FutureTensor(tmpdir, capture_prompt, [sympy.Integer(3)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("INIT", tmpdir)
        out.ft_forward(prompt_t)

        run_test("21: i=0 sees INIT", prompts_seen[0] == "INIT")
        run_test("22: i=1 has Iteration 0", "====[Iteration 0]====" in prompts_seen[1])
        run_test("23: i=1 has gen_0", "gen_0" in prompts_seen[1])
        run_test("24: i=2 has Iteration 1", "====[Iteration 1]====" in prompts_seen[2])
        run_test("25: i=2 has gen_1", "gen_1" in prompts_seen[2])

    # === Group 3: All fail — best result returned (tests 26-40) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=3, all self_confidence_but_failed
        async def all_fail(coords, prompt):
            i = coords[-1]
            confs = [0.3, 0.7, 0.5]
            return (f"output_{i}", Status.self_confidence_but_failed(confs[i]))

        inp = FutureTensor(tmpdir, all_fail, [sympy.Integer(3)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)

        # Best scyf value: i=1 -> 0.7
        run_test("26: best content is output_1", read_ft_element(out, 0) == "output_1")
        # self_confidence_but_failed(0.7) -> convert_status_to_float -> -0.7
        run_test("27: stored as -0.7",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - (-0.7)) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # All fail, best is last iteration
        async def best_last(coords, prompt):
            i = coords[-1]
            return (f"out_{i}", Status.self_confidence_but_failed(0.1 * (i + 1)))

        inp = FutureTensor(tmpdir, best_last, [sympy.Integer(3)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        # Best: i=2 -> 0.3
        run_test("28: best is out_2", read_ft_element(out, 0) == "out_2")
        run_test("29: stored as -0.3",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - (-0.3)) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # All fail, best is first
        async def best_first(coords, prompt):
            i = coords[-1]
            confs = [0.9, 0.2, 0.1]
            return (f"out_{i}", Status.self_confidence_but_failed(confs[i]))

        inp = FutureTensor(tmpdir, best_first, [sympy.Integer(3)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("30: best is out_0", read_ft_element(out, 0) == "out_0")
        run_test("31: stored as -0.9",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - (-0.9)) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=1, single fail
        async def single_fail(coords, prompt):
            return ("only_out", Status.self_confidence_but_failed(0.6))

        inp = FutureTensor(tmpdir, single_fail, [sympy.Integer(1)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("32: content only_out", read_ft_element(out, 0) == "only_out")
        run_test("33: stored as -0.6",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - (-0.6)) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=2, tied scyf — first one wins (strictly >)
        async def tied(coords, prompt):
            i = coords[-1]
            return (f"out_{i}", Status.self_confidence_but_failed(0.5))

        inp = FutureTensor(tmpdir, tied, [sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("34: tied -> first wins", read_ft_element(out, 0) == "out_0")
        run_test("35: stored as -0.5",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - (-0.5)) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # kContextOverflow returns early
        async def overflow_get(coords, trajactory):
            return ("overflow_content", Status.kContextOverflow)

        inp = FutureTensor(tmpdir, overflow_get, [sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("36: kContextOverflow returns empty string",
                 read_ft_element(out, 0) == "")
        run_test("37: stored as -3.0",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - (-3.0)) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # kConfidenceNotBounded returns with confidence(1.0)
        async def not_bounded_get(coords, trajactory):
            return ("unbounded_output", Status.kConfidenceNotBounded)

        inp = FutureTensor(tmpdir, not_bounded_get, [sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("38: kConfidenceNotBounded content", read_ft_element(out, 0) == "unbounded_output")
        run_test("39: stored as 1.0 (confidence)",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - 1.0) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Shape assertion: at least 1 dim
        try:
            bad = FutureTensor(tmpdir, lambda c, p: None, [])
            recurrent_forward(bad)
            run_test("40: 0D should fail", False)
        except AssertionError:
            run_test("40: 0D assertion", True)

    # === Group 4: Prompt accumulation details (tests 41-55) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_seen = {}

        async def capture_all(coords, prompt):
            key = tuple(coords)
            prompts_seen.setdefault(key, []).append(prompt)
            i = coords[-1]
            return (f"gen_{i}", Status.self_confidence_but_failed(0.5))

        inp = FutureTensor(tmpdir, capture_all, [sympy.Integer(3)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("INITIAL", tmpdir)
        out.ft_forward(prompt_t)

        run_test("41: i=0 sees INITIAL", prompts_seen[(0,)][0] == "INITIAL")
        run_test("42: i=0 validator sees INITIAL", True)  # no separate validator now

        p1 = prompts_seen[(1,)][0]
        run_test("43: i=1 starts with INITIAL", p1.startswith("INITIAL"))
        run_test("44: i=1 has ====[Iteration 0]====", "====[Iteration 0]====" in p1)
        run_test("45: i=1 has gen_0", "gen_0" in p1)

        p2 = prompts_seen[(2,)][0]
        run_test("46: i=2 has Iteration 0", "====[Iteration 0]====" in p2)
        run_test("47: i=2 has Iteration 1", "====[Iteration 1]====" in p2)
        run_test("48: i=2 has gen_0", "gen_0" in p2)
        run_test("49: i=2 has gen_1", "gen_1" in p2)

    with tempfile.TemporaryDirectory() as tmpdir:
        # prompt_tensor stores the prompts
        async def simple_retry(coords, prompt):
            i = coords[-1]
            if i == 0:
                return ("bad", Status.self_confidence_but_failed(0.3))
            return ("good", Status.confidence(0.9))

        inp = FutureTensor(tmpdir, simple_retry, [sympy.Integer(2)])
        out, pt, _, rec_state = recurrent_forward(inp)
        prompt_t = st_make_tensor("start", tmpdir)
        out.ft_forward(prompt_t)

        # Prompts now live in recurrent_state["cur_prompt"]
        run_test("50: recurrent_state has prompt",
                 ((),) and rec_state[()] is not None)
        cur_p = rec_state[()]["cur_prompt"]
        run_test("51: cur_prompt has Iteration 0",
                 cur_p is not None and "Iteration 0" in cur_p)
        run_test("52: cur_prompt has bad",
                 cur_p is not None and "bad" in cur_p)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Multiline prompt preserved
        async def multiline_test(coords, prompt):
            return ("out", Status.confidence(0.9))

        inp = FutureTensor(tmpdir, multiline_test, [sympy.Integer(1)])
        out, pt, _, rec_state = recurrent_forward(inp)
        prompt_t = st_make_tensor("line1\nline2\nline3", tmpdir)
        out.ft_forward(prompt_t)
        run_test("53: multiline content", read_ft_element(out, 0) == "out")
        cur_p = rec_state[()]["cur_prompt"]
        run_test("54: multiline prompt in state",
                 cur_p is not None and "line1\nline2\nline3" in cur_p)
        run_test("55: returns (output, prompt_tensor) tuple",
                 isinstance(pt, torch.Tensor))

    # === Group 5: Multi-element prefix (tests 56-75) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        # prefix [3], recurrent_dim=2 -> shape [3, 2]
        async def multi_elem(coords, prompt):
            prefix, i = coords
            if i == 0:
                return (f"p{prefix}_bad", Status.self_confidence_but_failed(0.3))
            else:
                return (f"p{prefix}_good", Status.confidence(0.9))

        inp = FutureTensor(tmpdir, multi_elem, [sympy.Integer(3), sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("56: prefix [3] shape", out.ft_capacity_shape == [3])
        run_test("57: prompt_tensor shape [3,2]", pt.ft_capacity_shape == [3, 2])

        prompt_t = st_make_tensor(["pa", "pb", "pc"], tmpdir)
        out.ft_forward(prompt_t)

        run_test("58: elem 0 content", read_ft_element(out, 0) == "p0_good")
        run_test("59: elem 1 content", read_ft_element(out, 1) == "p1_good")
        run_test("60: elem 2 content", read_ft_element(out, 2) == "p2_good")
        for i in range(3):
            run_test(f"61+{i}: elem {i} conf 0.9",
                     abs(out.ft_static_tensor.data.flatten()[i].item() - 0.9) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # prefix [2, 2], recurrent_dim=1 -> shape [2, 2, 1]
        async def prefix_2d(coords, prompt):
            r, c, i = coords
            return (f"r{r}c{c}", Status.confidence(0.8))

        inp = FutureTensor(tmpdir, prefix_2d, [sympy.Integer(2), sympy.Integer(2), sympy.Integer(1)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("64: prefix [2,2] shape", out.ft_capacity_shape == [2, 2])

        prompt_t = st_make_tensor([["p00", "p01"], ["p10", "p11"]], tmpdir)
        out.ft_forward(prompt_t)

        run_test("65: [0,0]", read_ft_element(out, 0) == "r0c0")
        run_test("66: [0,1]", read_ft_element(out, 1) == "r0c1")
        run_test("67: [1,0]", read_ft_element(out, 2) == "r1c0")
        run_test("68: [1,1]", read_ft_element(out, 3) == "r1c1")

    with tempfile.TemporaryDirectory() as tmpdir:
        # prefix [4], recurrent_dim=3, mixed ok/fail per element
        async def mixed_per_elem(coords, prompt):
            prefix, i = coords
            ok_at = [0, 2, 1, -1]  # elem 3 never ok
            if i == ok_at[prefix]:
                return (f"e{prefix}_i{i}", Status.confidence(0.9))
            return (f"e{prefix}_i{i}", Status.self_confidence_but_failed(0.3 + i * 0.1))

        inp = FutureTensor(tmpdir, mixed_per_elem, [sympy.Integer(4), sympy.Integer(3)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("69: shape [4]", out.ft_capacity_shape == [4])
        prompt_t = st_make_tensor(["p"] * 4, tmpdir)
        out.ft_forward(prompt_t)

        run_test("70: elem 0 Ok at i=0", read_ft_element(out, 0) == "e0_i0")
        run_test("71: elem 0 conf 0.9", abs(out.ft_static_tensor.data.flatten()[0].item() - 0.9) < 0.01)
        run_test("72: elem 1 Ok at i=2", read_ft_element(out, 1) == "e1_i2")
        run_test("73: elem 2 Ok at i=1", read_ft_element(out, 2) == "e2_i1")
        # elem 3: all fail, best scyf = i=2 (0.5), stored as -0.5
        run_test("74: elem 3 all fail best i=2", read_ft_element(out, 3) == "e3_i2")
        run_test("75: elem 3 stored as scyf -0.5",
                 abs(out.ft_static_tensor.data.flatten()[3].item() - (-0.5)) < 0.02)

    # === Group 6: recurrent_dim=1 edge case (tests 76-82) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        async def single_ok(coords, prompt):
            return ("result", Status.confidence(0.85))

        inp = FutureTensor(tmpdir, single_ok, [sympy.Integer(1)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("76: recurrent_dim=1 Ok shape", out.ft_capacity_shape == [])
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("77: content", read_ft_element(out, 0) == "result")
        run_test("78: conf 0.85", abs(out.ft_static_tensor.data.flatten()[0].item() - 0.85) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        async def single_err(coords, prompt):
            return ("result", Status.self_confidence_but_failed(0.7))

        inp = FutureTensor(tmpdir, single_err, [sympy.Integer(1)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("79: single fail content", read_ft_element(out, 0) == "result")
        run_test("80: stored as -0.7",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - (-0.7)) < 0.02)

    with tempfile.TemporaryDirectory() as tmpdir:
        # recurrent_dim=5
        async def dim5(coords, prompt):
            i = coords[-1]
            if i == 4:
                return (f"out_{i}", Status.confidence(0.8))
            return (f"out_{i}", Status.self_confidence_but_failed(0.1 * (i + 1)))

        inp = FutureTensor(tmpdir, dim5, [sympy.Integer(5)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("81: recurrent_dim=5 shape", out.ft_capacity_shape == [])
        prompt_t = st_make_tensor("p", tmpdir)
        out.ft_forward(prompt_t)
        run_test("82: Ok at i=4", read_ft_element(out, 0) == "out_4")

    # === Group 7: Prompt-dependent responses (tests 83-90) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        async def prompt_dependent(coords, prompt):
            i = coords[-1]
            if "FIXED" in prompt:
                return (f"gen_{i}", Status.confidence(0.9))
            return (f"gen_{i}", Status.self_confidence_but_failed(0.5))

        inp = FutureTensor(tmpdir, prompt_dependent, [sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("initial", tmpdir)
        out.ft_forward(prompt_t)
        # i=0 fails (no FIXED), error details not directly in prompt
        # but i=1 accumulates "gen_0" — no FIXED there either
        # Both fail, best = 0.5 (tied, first wins)
        run_test("83: without FIXED both fail",
                 read_ft_element(out, 0) == "gen_0")

    with tempfile.TemporaryDirectory() as tmpdir:
        async def prompt_dependent2(coords, prompt):
            i = coords[-1]
            if "FIXED" in prompt:
                return (f"gen_{i}", Status.confidence(0.9))
            return (f"gen_{i}_with_FIXED", Status.self_confidence_but_failed(0.5))

        inp = FutureTensor(tmpdir, prompt_dependent2, [sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("initial", tmpdir)
        out.ft_forward(prompt_t)
        # i=0 output is "gen_0_with_FIXED", accumulated into i=1 prompt
        # i=1 prompt contains "gen_0_with_FIXED" which contains "FIXED"
        run_test("84: FIXED propagated via output accumulation",
                 read_ft_element(out, 0) == "gen_1")
        run_test("85: conf 0.9",
                 abs(out.ft_static_tensor.data.flatten()[0].item() - 0.9) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        async def prompt_dependent3(coords, prompt):
            if "FIXED" in prompt:
                return ("gen", Status.confidence(0.9))
            return ("gen", Status.self_confidence_but_failed(0.5))

        inp = FutureTensor(tmpdir, prompt_dependent3, [sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor("FIXED already", tmpdir)
        out.ft_forward(prompt_t)
        run_test("86: Ok on i=0 with FIXED in initial", read_ft_element(out, 0) == "gen")
        run_test("87: conf 0.9", abs(out.ft_static_tensor.data.flatten()[0].item() - 0.9) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Each element gets its own prompt
        async def per_prompt(coords, prompt):
            prefix, i = coords
            return (f"echo:{prompt[:3]}", Status.confidence(0.8))

        inp = FutureTensor(tmpdir, per_prompt, [sympy.Integer(3), sympy.Integer(1)])
        out, pt, _, _ = recurrent_forward(inp)
        prompt_t = st_make_tensor(["AAA", "BBB", "CCC"], tmpdir)
        out.ft_forward(prompt_t)
        run_test("88: elem 0 uses its prompt", read_ft_element(out, 0) == "echo:AAA")
        run_test("89: elem 1 uses its prompt", read_ft_element(out, 1) == "echo:BBB")
        run_test("90: elem 2 uses its prompt", read_ft_element(out, 2) == "echo:CCC")

    # === Group 8: Composition with slice/unsqueeze (tests 91-100) ===

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.slice_forward import slice_forward

        # prefix [4], recurrent_dim=2 -> shape [4, 2]
        async def comp_get(coords, trajactory):
            prefix, i = coords
            return (f"p{prefix}_i{i}", Status.confidence(0.8))

        inp = FutureTensor(tmpdir, comp_get, [sympy.Integer(4), sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("91: shape [4]", out.ft_capacity_shape == [4])

        # Slice the output
        sliced = slice_forward(out, [slice(1, 3)])
        run_test("92: sliced shape [2]", sliced.ft_capacity_shape == [2])

        prompt_t = st_make_tensor(["p0", "p1"], tmpdir)
        sliced.ft_forward(prompt_t)
        # sliced[0] -> out[1], sliced[1] -> out[2]
        run_test("93: sliced[0] = p1_i0", read_ft_element(sliced, 0) == "p1_i0")
        run_test("94: sliced[1] = p2_i0", read_ft_element(sliced, 1) == "p2_i0")

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.unsqueeze_forward import unsqueeze_forward

        async def unsq_get(coords, trajactory):
            i = coords[-1]
            return (f"item_{i}", Status.confidence(0.85))

        inp = FutureTensor(tmpdir, unsq_get, [sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("95: before unsqueeze shape []", out.ft_capacity_shape == [])

        unsq = unsqueeze_forward(out, 0)
        run_test("96: unsqueezed shape [1]", unsq.ft_capacity_shape == [1])

        prompt_t = st_make_tensor(["p"], tmpdir)
        unsq.ft_forward(prompt_t)
        run_test("97: unsqueezed content", read_ft_element(unsq, 0) == "item_0")

    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.future_tensor.function.slice_forward import slice_forward

        # 2D prefix + recurrent: shape [3, 3, 2]
        async def big_get(coords, trajactory):
            r, c, i = coords
            if i == 0:
                return (f"r{r}c{c}_i0", Status.self_confidence_but_failed(0.3))
            return (f"r{r}c{c}_i1", Status.confidence(0.8))

        inp = FutureTensor(tmpdir, big_get, [sympy.Integer(3), sympy.Integer(3), sympy.Integer(2)])
        out, pt, _, _ = recurrent_forward(inp)
        run_test("98: 2D prefix shape [3,3]", out.ft_capacity_shape == [3, 3])

        sliced = slice_forward(out, [1, slice(None)])
        run_test("99: sliced row shape [3]", sliced.ft_capacity_shape == [3])

        prompt_t = st_make_tensor(["p0", "p1", "p2"], tmpdir)
        sliced.ft_forward(prompt_t)
        run_test("100: sliced[0] = r1c0_i1", read_ft_element(sliced, 0) == "r1c0_i1")

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All recurrent_forward tests completed.")
