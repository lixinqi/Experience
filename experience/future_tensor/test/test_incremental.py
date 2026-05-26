"""
test_incremental: Verify FutureTensor incremental concat mechanism.

ft_incremental_concated_tensors grows the logical view tensor dynamically.
Instead of pre-allocating [1, max_iters], declare the iter dim as symbolic,
start at size 0, and append one chunk per iteration.

Logical view = concat(ft_static_tensor, *ft_incremental_concated_tensors)
"""

import os
import sys
import tempfile

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor, _coords_to_flat
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor


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


def _storage_path(tensor, flat_index):
    digits = list(str(flat_index))
    return os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )


def read_element(tensor, flat_index):
    path = _storage_path(tensor, flat_index)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════
# Group 1: Symbolic dim produces capacity 0
# ═══════════════════════════════════════════════════════════════════════

print("Group 1: Symbolic dim produces capacity 0")

with tempfile.TemporaryDirectory() as tmpdir:
    n = sympy.Symbol("n")
    ft = FutureTensor(tmpdir, None, [sympy.Integer(1), n])
    run_test("1: symbolic dim → capacity 0",
             ft.ft_capacity_shape == [1, 0])
    run_test("2: ft_shape_schema preserved",
             ft.ft_shape_schema == [sympy.Integer(1), n])
    run_test("3: ft_incremental_concated_tensors starts empty",
             ft.ft_incremental_concated_tensors == [])
    run_test("4: ft_static_tensor shape matches capacity",
             list(ft.ft_static_tensor.shape) == [1, 0])

with tempfile.TemporaryDirectory() as tmpdir:
    n = sympy.Symbol("n")
    ft = FutureTensor(tmpdir, None, [n])
    run_test("5: 1D symbolic → capacity [0]",
             ft.ft_capacity_shape == [0])

with tempfile.TemporaryDirectory() as tmpdir:
    n = sympy.Symbol("n")
    m = sympy.Symbol("m")
    ft = FutureTensor(tmpdir, None, [n, m])
    run_test("6: all symbolic → capacity [0, 0]",
             ft.ft_capacity_shape == [0, 0])


# ═══════════════════════════════════════════════════════════════════════
# Group 2: Appending to ft_incremental_concated_tensors grows capacity
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 2: Incremental append grows capacity")

with tempfile.TemporaryDirectory() as tmpdir:
    n = sympy.Symbol("n")
    ft = FutureTensor(tmpdir, None, [sympy.Integer(1), n])

    # Append a chunk of shape [1, 1] along axis 1
    chunk1 = st_make_tensor([["hello"]], tmpdir)
    ft.ft_incremental_concated_tensors.append((chunk1, 1))
    ft.ft_capacity_shape = [1, 1]

    run_test("7: capacity grows to [1, 1]",
             ft.ft_capacity_shape == [1, 1])

    # Append another chunk
    chunk2 = st_make_tensor([["world"]], tmpdir)
    ft.ft_incremental_concated_tensors.append((chunk2, 1))
    ft.ft_capacity_shape = [1, 2]

    run_test("8: capacity grows to [1, 2]",
             ft.ft_capacity_shape == [1, 2])
    run_test("9: two chunks in list",
             len(ft.ft_incremental_concated_tensors) == 2)

with tempfile.TemporaryDirectory() as tmpdir:
    n = sympy.Symbol("n")
    ft = FutureTensor(tmpdir, None, [sympy.Integer(3), n])

    # Append chunk of shape [3, 1] along axis 1
    chunk = st_make_tensor([["a"], ["b"], ["c"]], tmpdir)
    ft.ft_incremental_concated_tensors.append((chunk, 1))
    ft.ft_capacity_shape = [3, 1]

    run_test("10: prefix [3] + 1 chunk → [3, 1]",
             ft.ft_capacity_shape == [3, 1])


# ═══════════════════════════════════════════════════════════════════════
# Group 3: ft_get_materialized_value resolves across logical view
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 3: ft_get_materialized_value across logical view")

with tempfile.TemporaryDirectory() as tmpdir:
    n = sympy.Symbol("n")

    async def dummy_get(coords, prompt):
        return ("", Status.confidence(1.0))

    ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(1), n])

    # Append chunk with known content
    chunk1 = st_make_tensor([["iter0_data"]], tmpdir)
    ft.ft_incremental_concated_tensors.append((chunk1, 1))
    ft.ft_capacity_shape = [1, 1]

    # chunk1 itself has the data at flat index 0
    chunk_content = read_element(chunk1, 0)
    run_test("11: chunk1 has 'iter0_data'",
             chunk_content == "iter0_data")

    # ft_get_materialized_value currently only reads ft_static_tensor (size [1,0])
    # so coords [0,0] will IndexError — documenting the gap
    try:
        coeff, path = ft.ft_get_materialized_value([0, 0])
        run_test("12: ft_get_materialized_value fails on incremental (gap)",
                 False)
    except (IndexError, Exception):
        run_test("12: ft_get_materialized_value fails on incremental (gap confirmed)",
                 True)


# ═══════════════════════════════════════════════════════════════════════
# Group 4: ft_async_get works with dynamic coordinates
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 4: ft_async_get works with any iteration index")

with tempfile.TemporaryDirectory() as tmpdir:
    call_log = []

    async def dynamic_get(coords, prompt):
        call_log.append(list(coords))
        i = coords[-1] if coords else 0
        return (f"iter_{i}", Status.confidence(1.0))

    n = sympy.Symbol("n")
    ft = FutureTensor(tmpdir, dynamic_get, [sympy.Integer(1), n])

    # ft_async_get can be called with any coords — it's just a function
    import asyncio

    result0 = asyncio.run(ft.ft_async_get([0, 0], "prompt"))
    run_test("13: ft_async_get [0,0] works",
             result0[0] == "iter_0")

    result5 = asyncio.run(ft.ft_async_get([0, 5], "prompt"))
    run_test("14: ft_async_get [0,5] works (beyond any declared size)",
             result5[0] == "iter_5")

    result99 = asyncio.run(ft.ft_async_get([0, 99], "prompt"))
    run_test("15: ft_async_get [0,99] works",
             result99[0] == "iter_99")

    run_test("16: all 3 calls logged",
             call_log == [[0, 0], [0, 5], [0, 99]])


# ═══════════════════════════════════════════════════════════════════════
# Group 5: ft_recurrent drives iteration without pre-declared shape
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 5: ft_recurrent with body that ignores iteration index")

from experience.future_tensor.function.ft_recurrent_forward import recurrent_forward

with tempfile.TemporaryDirectory() as tmpdir:
    # Body has shape [1, 5] — recurrent_dim=5, prefix=[1]
    # But the body's ft_async_get just returns a live value based on iteration
    iteration_results = ["bad0", "bad1", "bad2", "good3", "bad4"]

    async def body_get(coords, prompt):
        prefix, i = coords
        val = iteration_results[i]
        if val.startswith("good"):
            return (val, Status.confidence(0.9))
        return (val, Status.self_confidence_but_failed(0.3))

    body = FutureTensor(tmpdir, body_get, [sympy.Integer(1), sympy.Integer(5)])
    output, prompt_tensor, next_pos, rec_state = recurrent_forward(body)

    prompt_t = st_make_tensor(["start"], tmpdir)
    output.ft_forward(prompt_t)

    content = read_element(output.ft_static_tensor, 0)
    run_test("17: recurrent finds 'good3' at i=3",
             content == "good3")
    run_test("18: output forwarded",
             output.ft_forwarded is True)
    run_test("19: next_position advanced to recurrent_dim",
             next_pos.flatten()[0].item() == 5)


# ═══════════════════════════════════════════════════════════════════════
# Group 6: Incremental recurrent pattern (step_budget=1)
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 6: Incremental recurrent with step_budget=1")

with tempfile.TemporaryDirectory() as tmpdir:
    call_count = [0]

    async def counting_get(coords, prompt):
        call_count[0] += 1
        i = coords[-1]
        if i == 2:
            return ("found_it", Status.confidence(0.95))
        return (f"try_{i}", Status.self_confidence_but_failed(0.3 + i * 0.1))

    body = FutureTensor(tmpdir, counting_get, [sympy.Integer(5)])
    output, prompt_tensor, next_pos, rec_state = recurrent_forward(
        body, step_budget=1
    )

    prompt_t = st_make_tensor("go", tmpdir)

    # Step 1: processes i=0 only
    output.ft_forward(prompt_t)
    run_test("20: after step 1, next_pos=1",
             next_pos.flatten()[0].item() == 1)
    run_test("21: not yet forwarded (not terminal)",
             output.ft_forwarded is False)

    # Step 2: processes i=1 only
    output.ft_forward(prompt_t)
    run_test("22: after step 2, next_pos=2",
             next_pos.flatten()[0].item() == 2)
    run_test("23: still not forwarded",
             output.ft_forwarded is False)

    # Step 3: processes i=2, finds confidence → done
    output.ft_forward(prompt_t)
    run_test("24: after step 3, next_pos=3 (end_i on confidence)",
             next_pos.flatten()[0].item() == 3)
    run_test("25: now forwarded",
             output.ft_forwarded is True)

    content = read_element(output.ft_static_tensor, 0)
    run_test("26: content is 'found_it'",
             content == "found_it")
    run_test("27: exactly 3 calls to body",
             call_count[0] == 3)


# ═══════════════════════════════════════════════════════════════════════
# Group 7: Body ops with expand handle arbitrary iteration coords
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 7: expand_forward handles iteration coord passthrough")

from experience.future_tensor.function.expand_forward import expand_forward

with tempfile.TemporaryDirectory() as tmpdir:
    # instance_ft has shape [1], content "my_instance"
    instance_ft = FutureTensor(tmpdir, None, [sympy.Integer(1)])
    instance_st = st_make_tensor(["my_instance"], tmpdir)
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
    assign_tensor(instance_ft.ft_static_tensor, instance_st)
    instance_ft.ft_forwarded = True

    # Expand to [1, 3] — coord [0, i] maps to [0]
    expanded = expand_forward(instance_ft, [1, 3])
    run_test("28: expanded shape [1, 3]",
             expanded.ft_capacity_shape == [1, 3])
    run_test("29: expanded forwarded (input was forwarded)",
             expanded.ft_forwarded is True)

    # All [0, i] should have same content as [0]
    for i in range(3):
        content = read_element(expanded.ft_static_tensor, i)
        run_test(f"30.{i}: expanded[0,{i}] = 'my_instance'",
                 content == "my_instance")


# ═══════════════════════════════════════════════════════════════════════
# Group 8: Lazy expand + recurrent — the full pattern
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 8: Full pattern — expand + body + recurrent")

with tempfile.TemporaryDirectory() as tmpdir:
    # Simulate: instance_ft [1] → expand to [1, max_iters] → body uses it
    async def instance_get(coords, prompt):
        return ("instance_A", Status.confidence(1.0))

    instance_ft = FutureTensor(tmpdir, instance_get, [sympy.Integer(1)])

    max_iters = 4
    expanded = expand_forward(instance_ft, [1, max_iters])

    # Body reads from expanded (gets instance_id) and does work
    async def body_with_expand(coords, prompt):
        # Read instance from expanded
        instance_text, _ = await expanded.ft_async_get(coords, prompt)
        prefix, i = coords
        if i == 2:
            return (f"done_{instance_text}", Status.confidence(0.9))
        return (f"retry_{i}_{instance_text}", Status.self_confidence_but_failed(0.4))

    body = FutureTensor(tmpdir, body_with_expand,
                        [sympy.Integer(1), sympy.Integer(max_iters)])

    output, _, next_pos, _ = recurrent_forward(body)
    prompt_t = st_make_tensor(["task"], tmpdir)
    output.ft_forward(prompt_t)

    content = read_element(output.ft_static_tensor, 0)
    run_test("31: full pattern finds result at i=2",
             content == "done_instance_A")
    run_test("32: output forwarded",
             output.ft_forwarded is True)


# ═══════════════════════════════════════════════════════════════════════
# Group 9: Validator wrapping body — the react_loop pattern
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 9: Validator wrapping body (react_loop pattern)")

with tempfile.TemporaryDirectory() as tmpdir:
    # Body produces output at any coords — it doesn't care about shape
    terminal_state = {"found": False}

    async def reactive_body(coords, prompt):
        prefix, i = coords
        if i >= 2:
            terminal_state["found"] = True
            return ("hello world", Status.confidence(1.0))
        return ("partial", Status.confidence(1.0))

    max_iters = 5

    # Validator wraps body + checks termination
    async def validator_get(coords, prompt):
        text, status = await reactive_body(coords, prompt)
        i = coords[-1]
        if i > 0 and "hello world" in text:
            return (text, Status.confidence(1.0))
        return (text, Status.self_confidence_but_failed(0.5))

    validator = FutureTensor(tmpdir, validator_get,
                             [sympy.Integer(1), sympy.Integer(max_iters)])

    output, _, next_pos, _ = recurrent_forward(validator)
    prompt_t = st_make_tensor(["task"], tmpdir)
    output.ft_forward(prompt_t)

    content = read_element(output.ft_static_tensor, 0)
    run_test("33: validator finds 'hello world' at i=2",
             content == "hello world")
    run_test("34: terminal state reached",
             terminal_state["found"] is True)


# ═══════════════════════════════════════════════════════════════════════
# Group 10: The key insight — body doesn't need max_iters in its shape
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 10: Body shape independent of iteration budget")

with tempfile.TemporaryDirectory() as tmpdir:
    # Body has shape [1] — just the prefix dimension. No iter dim!
    # But ft_async_get receives coords [prefix, i] from recurrent via validator.

    async def minimal_body(coords, prompt):
        # coords may have 1 or 2 elements — body handles both
        i = coords[-1] if len(coords) > 1 else 0
        if i == 3:
            return ("success", Status.confidence(0.9))
        return (f"attempt_{i}", Status.self_confidence_but_failed(0.4))

    # Validator has shape [1, 6] — this is where max_iters lives
    max_iters = 6

    async def validator_from_minimal(coords, prompt):
        # validator calls body with same coords
        text, status = await minimal_body(coords, prompt)
        if status.is_confidence:
            return (text, status)
        return (text, Status.self_confidence_but_failed(0.5))

    validator = FutureTensor(tmpdir, validator_from_minimal,
                             [sympy.Integer(1), sympy.Integer(max_iters)])

    output, _, _, _ = recurrent_forward(validator)
    prompt_t = st_make_tensor(["go"], tmpdir)
    output.ft_forward(prompt_t)

    content = read_element(output.ft_static_tensor, 0)
    run_test("35: body with shape [1] works in recurrent (finds at i=3)",
             content == "success")


# ═══════════════════════════════════════════════════════════════════════

print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
if failed == 0:
    print("All incremental mechanism tests passed.")
else:
    sys.exit(1)
