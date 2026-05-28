"""
test_switch_2nd.py — Individual 2nd-derivative test for ft_switch using natural PyTorch flow.

Pattern:
    mock_input = need_reflection(mock_input, backward_dispatch_start)
    loss = model.forward(mock_input)
    loss.backward(create_graph=True)
    with dispatch_policy(TracePolicy):
        backward_dispatch_start.grad.backward()

Groups:
  1. Model forward shape and autograd connectivity
  2. Natural 2nd-derivative flow dispatches switch_backward record
  3. ReflectionRecord fields verification

Run:
    python -m experience.future_tensor.backward_dispatch.test.test_switch_2nd
"""

import os
import sys
import tempfile

import sympy
import torch
import torch.nn as nn

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
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")


# ── imports ───────────────────────────────────────────────────────────────────

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.ft_switch import ft_switch
from experience.future_tensor.function.ft_mean import ft_mean
from experience.future_tensor.function.switch_backward import switch_backward

from experience.future_tensor.backward_dispatch import (
    need_reflection,
    dispatch_policy,
    TracePolicy,
)

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor


# ── helpers ───────────────────────────────────────────────────────────────────

def make_forwarded_ft(shape, data_list, tmpdir):
    def _unflatten(flat, shp):
        if not shp:
            return flat[0]
        if len(shp) == 1:
            return flat
        k = 1
        for s in shp[1:]:
            k *= s
        return [_unflatten(flat[i * k:(i + 1) * k], shp[1:]) for i in range(shp[0])]

    async def dummy_get(coords, trajactory):
        return ("unused", Status.confidence(1.0))

    ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(s) for s in shape])
    nested = _unflatten(data_list, shape) if shape else data_list[0]
    result_tensor = make_tensor(nested, tmpdir)
    assign_tensor(ft.ft_initial_static_tensor, result_tensor)
    ft.ft_forwarded = True
    return ft


def make_overflow_ft(shape, data_list, tmpdir):
    """FutureTensor with kContextOverflow — exits without calling LLM."""
    ft = make_forwarded_ft(shape, data_list, tmpdir)
    ft.ft_initial_static_tensor.data.fill_(
        Status.convert_status_to_float(Status.kContextOverflow)
    )
    return ft


def _condition_storage_path(ft, flat_index=0):
    digits = list(str(flat_index))
    return os.path.join(
        ft.ft_initial_static_tensor.st_relative_to, ft.ft_initial_static_tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )


def make_condition_ft(symbol, tmpdir):
    """Create a materialized condition FutureTensor with the symbol in storage."""
    async def symbol_get(coords, trajactory):
        return (symbol, Status.confidence(1.0))
    ft = FutureTensor(tmpdir, symbol_get, [sympy.Integer(1)])
    # Materialize: write symbol into element-0 storage
    path = _condition_storage_path(ft, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(symbol)
    ft.ft_initial_static_tensor.data[0] = Status.convert_status_to_float(Status.confidence(1.0))
    ft.ft_forwarded = True
    return ft


# ── SwitchOnlyModel ───────────────────────────────────────────────────────────

class SwitchOnlyModel(nn.Module):
    """Minimal model containing only ft_switch."""

    def forward(self, condition, branch_a, branch_b):
        cases = [
            ("A", "case A", "desc A", branch_a),
            ("B", "case B", "desc B", branch_b),
        ]
        return ft_switch(condition, cases)


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: Model forward — shape and autograd connectivity
# ══════════════════════════════════════════════════════════════════════════════
print("Group 1: Model forward — shape and autograd connectivity")

with tempfile.TemporaryDirectory() as tmpdir:
    condition = make_condition_ft("B", tmpdir)
    branch_a = make_overflow_ft([3], ["a0", "a1", "a2"], tmpdir)
    branch_b = make_overflow_ft([3], ["b0", "b1", "b2"], tmpdir)
    branch_b.requires_grad_(True)

    model = SwitchOnlyModel()
    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(branch_b, backward_dispatch_start)
    output = model(condition, branch_a, anchored)

    run_test("output ft_capacity_shape is [3]",
             output.ft_capacity_shape == [3])
    run_test("output requires_grad", output.requires_grad is True)
    run_test("output has grad_fn", output.grad_fn is not None)


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: Natural 2nd-derivative flow
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 2: Natural 2nd-derivative flow")

with tempfile.TemporaryDirectory() as tmpdir:
    condition = make_condition_ft("B", tmpdir)
    branch_a = make_overflow_ft([2], ["a0", "a1"], tmpdir)
    branch_b = make_overflow_ft([2], ["b0", "b1"], tmpdir)
    branch_b.requires_grad_(True)

    model = SwitchOnlyModel()
    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(branch_b, backward_dispatch_start)
    output = model(condition, branch_a, anchored)
    loss = ft_mean(output)

    run_test("loss is scalar", loss.shape == torch.Size([]))
    run_test("loss requires_grad", loss.requires_grad is True)

    # 1st backward
    loss.backward(create_graph=True)

    run_test("backward_dispatch_start.grad exists",
             backward_dispatch_start.grad is not None)
    run_test("backward_dispatch_start.grad has grad_fn",
             backward_dispatch_start.grad.grad_fn is not None)

    # 2nd backward with TracePolicy
    coll = []
    with dispatch_policy(TracePolicy(coll)):
        backward_dispatch_start.grad.backward()

    run_test("TracePolicy collected at least 1 record", len(coll) >= 1)
    if len(coll) >= 1:
        run_test("record fn is switch_backward", coll[0].fn is switch_backward)
        run_test("record has grad_output",
                 "grad_output" in coll[0].inputs)
        run_test("record has selected_index",
                 "selected_index" in coll[0].inputs)
        run_test("record has branches",
                 "branches" in coll[0].inputs)
        run_test("record selected_index value is 1",
                 coll[0].inputs.get("selected_index") == 1)
        run_test("record has grad_input",
                 "grad_input" in coll[0].inputs)


# ══════════════════════════════════════════════════════════════════════════════
# Group 3: ReflectionRecord fields
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 3: ReflectionRecord fields")

with tempfile.TemporaryDirectory() as tmpdir:
    condition = make_condition_ft("B", tmpdir)
    branch_a = make_overflow_ft([2], ["a0", "a1"], tmpdir)
    branch_b = make_overflow_ft([2], ["b0", "b1"], tmpdir)
    branch_b.requires_grad_(True)

    model = SwitchOnlyModel()
    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(branch_b, backward_dispatch_start)
    output = model(condition, branch_a, anchored)
    loss = ft_mean(output)
    loss.backward(create_graph=True)

    coll = []
    with dispatch_policy(TracePolicy(coll)):
        backward_dispatch_start.grad.backward()

    if len(coll) >= 1:
        rec = coll[0]
        run_test("record.fn is switch_backward", rec.fn is switch_backward)
        run_test("record.inputs is dict", isinstance(rec.inputs, dict))
        run_test("record.output is scalar tensor", rec.output.shape == torch.Size([]))
        run_test("record.output value is 1.0", rec.output.item() == 1.0)
        run_test("record.timestamp is float > 0",
                 isinstance(rec.timestamp, float) and rec.timestamp > 0)
        run_test("grad_output in inputs", "grad_output" in rec.inputs)
        run_test("grad_input in inputs", "grad_input" in rec.inputs)
        run_test("selected_index in inputs", "selected_index" in rec.inputs)
        run_test("branches in inputs", "branches" in rec.inputs)


# ══════════════════════════════════════════════════════════════════════════════
# Group 4: Strict selected_index verification — different symbols
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 4: Strict selected_index verification — different symbols")

for expected_symbol, expected_index in [("A", 0), ("B", 1)]:
    with tempfile.TemporaryDirectory() as tmpdir:
        condition = make_condition_ft(expected_symbol, tmpdir)
        branch_a = make_overflow_ft([2], ["a0", "a1"], tmpdir)
        branch_b = make_overflow_ft([2], ["b0", "b1"], tmpdir)
        branch_a.requires_grad_(True)
        branch_b.requires_grad_(True)

        backward_dispatch_start = torch.ones(
            (), dtype=torch.bfloat16, requires_grad=True
        )

        # Anchor need_reflection on the branch that WILL be selected
        if expected_index == 0:
            anchored_a = need_reflection(branch_a, backward_dispatch_start)
            cases = [
                ("A", "case A", "desc A", anchored_a),
                ("B", "case B", "desc B", branch_b),
            ]
        else:
            anchored_b = need_reflection(branch_b, backward_dispatch_start)
            cases = [
                ("A", "case A", "desc A", branch_a),
                ("B", "case B", "desc B", anchored_b),
            ]

        output = ft_switch(condition, cases)
        loss = ft_mean(output)
        loss.backward(create_graph=True)

        coll = []
        with dispatch_policy(TracePolicy(coll)):
            backward_dispatch_start.grad.backward()

        run_test(
            f"symbol {expected_symbol} collected 1 record",
            len(coll) == 1,
            expected=1,
            actual=len(coll),
        )
        if len(coll) == 1:
            run_test(
                f"symbol {expected_symbol} selected_index is {expected_index}",
                coll[0].inputs.get("selected_index") == expected_index,
                expected=expected_index,
                actual=coll[0].inputs.get("selected_index"),
            )


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
if failed == 0:
    print("All ft_switch 2nd-derivative tests passed.")
else:
    sys.exit(1)
