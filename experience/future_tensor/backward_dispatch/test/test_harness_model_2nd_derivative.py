"""
test_harness_model_2nd_derivative.py — Toy harness model 2nd-derivative test.

Constructs a ToyHarnessModel (nn.Module) whose forward pass chains ft_unsqueeze,
ft_slice, and ft_recurrent.  Tests demonstrate the public 2nd-derivative API
using the natural PyTorch flow:

    mock_input = need_reflection(mock_input, backward_dispatch_start)
    loss = model.forward(mock_input)
    loss.backward(create_graph=True)
    with dispatch_policy(TracePolicy):
        backward_dispatch_start.grad.backward()

Architecture of ToyHarnessModel::

    input_ft  (2, 2)
      ft_unsqueeze(dim=1)           -> (2, 1, 2)
      ft_slice([:, 0, :])           -> (2, 2)
      ft_recurrent                  -> (2,)

Groups:
  1. ToyHarnessModel forward — shape and forwarded checks
  2. need_reflection — scalar assertions + requires_grad
  3. Natural 2nd-derivative flow — 3 records dispatched from model op GradFns
  4. ReflectionRecord fields from model ops
  5. Custom Policy — selective dispatch by fn identity
  6. dispatch_policy scope — default collector used outside block
  7. PolicyConflictError on nested dispatch_policy blocks

Run:
    python -m experience.future_tensor.backward_dispatch.test.test_harness_model_2nd_derivative
"""

import sys
import tempfile
import time

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
from experience.future_tensor.function.ft_slice import ft_slice
from experience.future_tensor.function.ft_unsqueeze import ft_unsqueeze
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_mean import ft_mean
from experience.future_tensor.function.ft_recurrent_backward import recurrent_backward
from experience.future_tensor.function.slice_backward import slice_backward
from experience.future_tensor.function.unsqueeze_forward import unsqueeze_forward

from experience.future_tensor.backward_dispatch import (
    need_reflection,
    dispatch_policy,
    TracePolicy,
    Policy,
    ReflectionRecord,
    PolicyConflictError,
)
from experience.future_tensor.backward_dispatch.context import _default_collector

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


# ── ToyHarnessModel ───────────────────────────────────────────────────────────

class ToyHarnessModel(nn.Module):
    """Minimal harness model: ft_unsqueeze → ft_slice → ft_recurrent.

    forward pipeline::

        input_ft  (2, 2)
          ft_unsqueeze(dim=1)           -> (2, 1, 2)
          ft_slice([:, 0, :])           -> (2, 2)
          ft_recurrent                  -> (2,)

    All FutureTensors carry kContextOverflow so ft_recurrent exits immediately
    without calling any LLM — safe for unit tests.
    """

    def forward(self, input_ft: FutureTensor) -> FutureTensor:
        x = ft_unsqueeze(input_ft, dim=1)                      # (2, 1, 2)
        x = ft_slice(x, [slice(None), 0, slice(None)])         # (2, 2)
        output = ft_recurrent(x, task_prompt="toy model")   # (2,)
        return output


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: ToyHarnessModel forward
# ══════════════════════════════════════════════════════════════════════════════
print("Group 1: ToyHarnessModel forward — shape and forwarded checks")

with tempfile.TemporaryDirectory() as tmpdir:
    input_ft = make_overflow_ft([2, 2], ["a0", "a1", "b0", "b1"], tmpdir)
    model = ToyHarnessModel()
    output = model(input_ft)

    run_test("output is a tensor", isinstance(output, torch.Tensor))
    run_test("output ft_capacity_shape is [2]", output.ft_capacity_shape == [2])

    prompts = make_tensor(["p0", "p1"], tmpdir)
    output.ft_forward(prompts)
    run_test("output forwarded", output.ft_forwarded is True)
    run_test("ft_initial_static_tensor shape [2]",
             list(output.ft_initial_static_tensor.shape) == [2])


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: need_reflection — scalar assertions + requires_grad
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 2: need_reflection — scalar assertions + requires_grad")

model2 = ToyHarnessModel()
backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
scalar = torch.zeros(())
result = need_reflection(scalar, backward_dispatch_start)

run_test("result is scalar (shape ())", result.shape == torch.Size([]))
run_test("requires_grad set to True", result.requires_grad is True)

# idempotent
scalar2 = torch.zeros((), requires_grad=True)
run_test("idempotent when already requires_grad",
         need_reflection(scalar2, backward_dispatch_start).requires_grad is True)

# non-scalar input raises
try:
    need_reflection(torch.zeros(3), backward_dispatch_start)
    run_test("non-scalar input raises AssertionError", False)
except AssertionError:
    run_test("non-scalar input raises AssertionError", True)

# non-scalar backward_dispatch_start raises
try:
    need_reflection(torch.zeros(()), torch.ones(3))
    run_test("non-scalar backward_dispatch_start raises AssertionError", False)
except AssertionError:
    run_test("non-scalar backward_dispatch_start raises AssertionError", True)


# ══════════════════════════════════════════════════════════════════════════════
# Group 3: Natural 2nd-derivative flow — 3 records dispatched
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 3: Natural 2nd-derivative flow — 3 records dispatched")

with tempfile.TemporaryDirectory() as tmpdir:
    input_ft = make_overflow_ft([2, 2], ["a0", "a1", "b0", "b1"], tmpdir)
    input_ft.requires_grad_(True)

    model = ToyHarnessModel()
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
    coll3 = []
    with dispatch_policy(TracePolicy(coll3)):
        backward_dispatch_start.grad.backward()

    run_test("3 records dispatched", len(coll3) == 3)

    if len(coll3) == 3:
        fns = [r.fn for r in coll3]
        # Backward traversal order: closest to backward_dispatch_start first
        run_test("record 0 fn: unsqueeze_forward", fns[0] is unsqueeze_forward)
        run_test("record 1 fn: slice_backward", fns[1] is slice_backward)
        run_test("record 2 fn: recurrent_backward", fns[2] is recurrent_backward)

        run_test("unsqueeze record has dim", "dim" in coll3[0].inputs)
        run_test("unsqueeze record has grad_input", "grad_input" in coll3[0].inputs)
        run_test("slice record has original_shape", "original_shape" in coll3[1].inputs)
        run_test("slice record has grad_input", "grad_input" in coll3[1].inputs)
        run_test("recurrent record has grad_input", "grad_input" in coll3[2].inputs)
        run_test("recurrent record has task_prompt",
                 coll3[2].inputs.get("task_prompt") == "toy model")
        run_test("recurrent record has topk_self_confidence_but_failed",
                 "topk_self_confidence_but_failed" in coll3[2].inputs)
        run_test("all outputs are scalar tensors",
                 all(r.output.shape == torch.Size([]) for r in coll3))


# ══════════════════════════════════════════════════════════════════════════════
# Group 4: ReflectionRecord fields
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 4: ReflectionRecord fields from model ops")

with tempfile.TemporaryDirectory() as tmpdir:
    input_ft = make_overflow_ft([2, 2], ["a0", "a1", "b0", "b1"], tmpdir)
    input_ft.requires_grad_(True)

    model = ToyHarnessModel()
    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(input_ft, backward_dispatch_start)
    output = model(anchored)
    loss = ft_mean(output)
    loss.backward(create_graph=True)

    coll4 = []
    with dispatch_policy(TracePolicy(coll4)):
        backward_dispatch_start.grad.backward()

    if len(coll4) == 3:
        # Backward traversal order: closest to backward_dispatch_start first
        rec_u = coll4[0]
        rec_s = coll4[1]
        rec_r = coll4[2]

        run_test("rec_u.fn is unsqueeze_forward", rec_u.fn is unsqueeze_forward)
        run_test("rec_s.fn is slice_backward", rec_s.fn is slice_backward)
        run_test("rec_r.fn is recurrent_backward", rec_r.fn is recurrent_backward)
        run_test("rec_r.inputs is dict", isinstance(rec_r.inputs, dict))
        run_test("rec_r.output is scalar tensor", rec_r.output.shape == torch.Size([]))
        run_test("rec_r.output value is 1.0", rec_r.output.item() == 1.0)
        run_test("rec_r.timestamp is float > 0",
                 isinstance(rec_r.timestamp, float) and rec_r.timestamp > 0)
        run_test("rec_r.timestamp recent", abs(rec_r.timestamp - time.monotonic()) < 5.0)
        run_test("rec_r.fn.__name__ accessible", rec_r.fn.__name__ == "recurrent_backward")
        run_test("rec_s has grad_output", "grad_output" in rec_s.inputs)
        run_test("rec_u has squeeze_slices", "squeeze_slices" in rec_u.inputs)


# ══════════════════════════════════════════════════════════════════════════════
# Group 5: Custom Policy — selective dispatch by fn identity
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 5: Custom Policy — selective dispatch by fn identity")


class SelectivePolicy(Policy):
    """Routes recurrent_backward to a dedicated handler; traces the rest."""

    def __init__(self):
        self.recurrent_calls = []
        self.other_calls = []

    def dispatch(self, fn, arg_name2inputs):
        if fn is recurrent_backward:
            self.recurrent_calls.append(arg_name2inputs)
        else:
            self.other_calls.append((fn, arg_name2inputs))
        return torch.ones(())


with tempfile.TemporaryDirectory() as tmpdir:
    input_ft = make_overflow_ft([2, 2], ["a0", "a1", "b0", "b1"], tmpdir)
    input_ft.requires_grad_(True)

    model = ToyHarnessModel()
    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(input_ft, backward_dispatch_start)
    output = model(anchored)
    loss = ft_mean(output)
    loss.backward(create_graph=True)

    sp = SelectivePolicy()
    with dispatch_policy(sp):
        backward_dispatch_start.grad.backward()

run_test("recurrent_calls has 1 entry", len(sp.recurrent_calls) == 1)
run_test("other_calls has 2 entries (slice + unsqueeze)", len(sp.other_calls) == 2)
run_test("recurrent call has task_prompt",
         sp.recurrent_calls[0].get("task_prompt") == "toy model")
other_fns = {fn for fn, _ in sp.other_calls}
run_test("other_calls contain slice_backward and unsqueeze_forward",
         slice_backward in other_fns and unsqueeze_forward in other_fns)


# ══════════════════════════════════════════════════════════════════════════════
# Group 6: dispatch_policy scope — outside block routes to default collector
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 6: dispatch_policy scope — outside block routes to default collector")

with tempfile.TemporaryDirectory() as tmpdir:
    input_ft = make_overflow_ft([2, 2], ["a0", "a1", "b0", "b1"], tmpdir)
    input_ft.requires_grad_(True)

    model = ToyHarnessModel()
    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(input_ft, backward_dispatch_start)
    output = model(anchored)
    loss = ft_mean(output)
    loss.backward(create_graph=True)

    coll6 = []
    with dispatch_policy(TracePolicy(coll6)):
        pass  # policy active but no backward inside

    pre_len = len(_default_collector)
    backward_dispatch_start.grad.backward()  # no dispatch_policy block

    run_test("inside-block collector still empty", len(coll6) == 0)
    run_test("outside block routes to default collector",
             len(_default_collector) > pre_len)


# ══════════════════════════════════════════════════════════════════════════════
# Group 7: PolicyConflictError on nested dispatch_policy blocks
# ══════════════════════════════════════════════════════════════════════════════
print("\nGroup 7: PolicyConflictError on nested dispatch_policy blocks")

try:
    with dispatch_policy(TracePolicy([])):
        with dispatch_policy(TracePolicy([])):
            pass
    run_test("nested dispatch_policy raises PolicyConflictError", False)
except PolicyConflictError:
    run_test("nested dispatch_policy raises PolicyConflictError", True)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
if failed == 0:
    print("All harness_model 2nd-derivative tests passed.")
else:
    sys.exit(1)
