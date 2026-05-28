"""
test_recurrent_2nd.py — Individual 2nd-derivative test for ft_recurrent using natural PyTorch flow.

Pattern:
    mock_input = need_reflection(mock_input, backward_dispatch_start)
    loss = model.forward(mock_input)
    loss.backward(create_graph=True)
    with dispatch_policy(TracePolicy):
        backward_dispatch_start.grad.backward()

All FutureTensors carry kContextOverflow so ft_recurrent exits immediately
without calling any LLM — safe for unit tests.

Groups:
  1. Model forward shape and autograd connectivity
  2. Natural 2nd-derivative flow dispatches recurrent_backward record
  3. ReflectionRecord fields verification

Run:
    python -m experience.future_tensor.backward_dispatch.test.test_recurrent_2nd
"""

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
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_mean import ft_mean
from experience.future_tensor.function.ft_recurrent_backward import recurrent_backward

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
    """FutureTensor with kContextOverflow — ft_recurrent exits without calling LLM."""
    ft = make_forwarded_ft(shape, data_list, tmpdir)
    ft.ft_initial_static_tensor.data.fill_(
        Status.convert_status_to_float(Status.kContextOverflow)
    )
    return ft


# ── RecurrentOnlyModel ────────────────────────────────────────────────────────

class RecurrentOnlyModel(nn.Module):
    """Minimal model containing only ft_recurrent."""

    def forward(self, input_ft):
        output = ft_recurrent(input_ft, task_prompt="test")
        return output


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: Model forward — shape and autograd connectivity
# ══════════════════════════════════════════════════════════════════════════════
print("Group 1: Model forward — shape and autograd connectivity")

with tempfile.TemporaryDirectory() as tmpdir:
    input_ft = make_overflow_ft([2, 3], ["a", "b", "c", "d", "e", "f"], tmpdir)
    input_ft.requires_grad_(True)

    model = RecurrentOnlyModel()
    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(input_ft, backward_dispatch_start)
    output = model(anchored)

    run_test("output ft_capacity_shape is [2]",
             output.ft_capacity_shape == [2])
    run_test("output requires_grad", output.requires_grad is True)
    run_test("output has grad_fn", output.grad_fn is not None)


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: Natural 2nd-derivative flow
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 2: Natural 2nd-derivative flow")

with tempfile.TemporaryDirectory() as tmpdir:
    input_ft = make_overflow_ft([1, 2], ["x", "y"], tmpdir)
    input_ft.requires_grad_(True)

    model = RecurrentOnlyModel()
    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(input_ft, backward_dispatch_start)
    output = model(anchored)
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
        run_test("record fn is recurrent_backward", coll[0].fn is recurrent_backward)
        run_test("record has grad_output",
                 "grad_output" in coll[0].inputs)
        run_test("record has task_prompt",
                 coll[0].inputs.get("task_prompt") == "test")
        run_test("record has grad_input",
                 "grad_input" in coll[0].inputs)


# ══════════════════════════════════════════════════════════════════════════════
# Group 3: ReflectionRecord fields
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 3: ReflectionRecord fields")

with tempfile.TemporaryDirectory() as tmpdir:
    input_ft = make_overflow_ft([2, 2], ["a", "b", "c", "d"], tmpdir)
    input_ft.requires_grad_(True)

    model = RecurrentOnlyModel()
    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(input_ft, backward_dispatch_start)
    output = model(anchored)
    loss = ft_mean(output)
    loss.backward(create_graph=True)

    coll = []
    with dispatch_policy(TracePolicy(coll)):
        backward_dispatch_start.grad.backward()

    if len(coll) >= 1:
        rec = coll[0]
        run_test("record.fn is recurrent_backward", rec.fn is recurrent_backward)
        run_test("record.inputs is dict", isinstance(rec.inputs, dict))
        run_test("record.output is scalar tensor", rec.output.shape == torch.Size([]))
        run_test("record.output value is 1.0", rec.output.item() == 1.0)
        run_test("record.timestamp is float > 0",
                 isinstance(rec.timestamp, float) and rec.timestamp > 0)
        run_test("grad_output in inputs", "grad_output" in rec.inputs)
        run_test("grad_input in inputs", "grad_input" in rec.inputs)
        run_test("task_prompt in inputs", rec.inputs.get("task_prompt") == "test")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
if failed == 0:
    print("All ft_recurrent 2nd-derivative tests passed.")
else:
    sys.exit(1)
