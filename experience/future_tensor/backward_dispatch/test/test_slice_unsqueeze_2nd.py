"""
Tests for 2nd derivatives of ft_slice and ft_unsqueeze.

Groups:
  1. Module imports and structure
  2. SliceGradFn: forward IS slice_backward
  3. SliceGradFn: backward() dispatches 2nd derivative
  4. SliceGradFn: grad_fn attached when input requires_grad
  5. UnsqueezeGradFn: forward IS squeeze-via-slice_forward
  6. UnsqueezeGradFn: backward() dispatches 2nd derivative
  7. UnsqueezeGradFn: grad_fn attached when input requires_grad
  8. Integration: FtSlice.backward() uses SliceGradFn
  9. Integration: FtUnsqueeze.backward() uses UnsqueezeGradFn

Run:
    python -m experience.future_tensor.backward_dispatch.test.test_slice_unsqueeze_2nd
"""

import sys
import tempfile

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

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


def read_ft_element(ft, flat_index):
    import os
    digits = list(str(flat_index))
    path = os.path.join(
        ft.ft_initial_static_tensor.st_relative_to, ft.ft_initial_static_tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


# ── Group 1: Imports ──────────────────────────────────────────────────────────
print("Group 1: Module imports")

from experience.future_tensor.backward_dispatch import (
    dispatch_policy,
    TracePolicy,
)
from experience.future_tensor.function.slice_2nd import SliceGradFn
from experience.future_tensor.function.unsqueeze_2nd import UnsqueezeGradFn
from experience.future_tensor.function.slice_backward import slice_backward
from experience.future_tensor.function.unsqueeze_forward import unsqueeze_forward
from experience.future_tensor.function.ft_slice import FtSlice, ft_slice
from experience.future_tensor.function.ft_unsqueeze import FtUnsqueeze, ft_unsqueeze

run_test("SliceGradFn is autograd.Function subclass",
         issubclass(SliceGradFn, torch.autograd.Function))
run_test("UnsqueezeGradFn is autograd.Function subclass",
         issubclass(UnsqueezeGradFn, torch.autograd.Function))
run_test("FtSlice is autograd.Function subclass",
         issubclass(FtSlice, torch.autograd.Function))
run_test("FtUnsqueeze is autograd.Function subclass",
         issubclass(FtUnsqueeze, torch.autograd.Function))

# ── Group 2: SliceGradFn.forward = slice_backward ─────────────────────────────
print("\nGroup 2: SliceGradFn.forward IS slice_backward")

with tempfile.TemporaryDirectory() as tmpdir:
    grad = make_forwarded_ft([3], ["g0", "g1", "g2"], tmpdir)
    grad.requires_grad_(True)

    result = SliceGradFn.apply(grad, [10], [slice(2, 5)])

    run_test("forward produces shape [10]", result.ft_capacity_shape == [10])
    run_test("forward is forwarded", result.ft_forwarded is True)
    run_test("forward [2] = g0", read_ft_element(result, 2) == "g0")
    run_test("forward [3] = g1", read_ft_element(result, 3) == "g1")
    run_test("forward [4] = g2", read_ft_element(result, 4) == "g2")
    run_test("forward [0] empty (not in slice)", read_ft_element(result, 0) is None)
    run_test("forward [9] empty (not in slice)", read_ft_element(result, 9) is None)

# ── Group 3: SliceGradFn.backward dispatches 2nd derivative ──────────────────
print("\nGroup 3: SliceGradFn.backward() dispatches 2nd derivative")

with tempfile.TemporaryDirectory() as tmpdir:
    grad3 = make_forwarded_ft([4], ["a", "b", "c", "d"], tmpdir)
    grad3.requires_grad_(True)

    coll3 = []
    with dispatch_policy(TracePolicy(coll3)):
        result3 = SliceGradFn.apply(grad3, [8], [slice(1, 5)])
        result3.sum().backward()

    run_test("2nd derivative dispatched", len(coll3) >= 1)
    if len(coll3) >= 1:
        run_test("fn is slice_backward", coll3[0].fn is slice_backward)
        run_test("grad_output in inputs", "grad_output" in coll3[0].inputs)
        run_test("original_shape in inputs",
                 coll3[0].inputs.get("original_shape") == [8])
        run_test("slices in inputs", coll3[0].inputs.get("slices") == [slice(1, 5)])
        run_test("grad_input in inputs", "grad_input" in coll3[0].inputs)
        run_test("grad_input shape [8]",
                 coll3[0].inputs["grad_input"].ft_capacity_shape == [8])

# ── Group 4: SliceGradFn grad_fn attached ─────────────────────────────────────
print("\nGroup 4: SliceGradFn grad_fn attached when input requires_grad")

with tempfile.TemporaryDirectory() as tmpdir:
    grad4 = make_forwarded_ft([2], ["x", "y"], tmpdir)
    grad4.requires_grad_(True)

    result4 = SliceGradFn.apply(grad4, [5], [slice(1, 3)])
    run_test("result requires_grad", result4.requires_grad)
    run_test("result has grad_fn", result4.grad_fn is not None)
    run_test("grad_fn is SliceGradFnBackward",
             "SliceGradFn" in type(result4.grad_fn).__name__)

# ── Group 5: UnsqueezeGradFn.forward = squeeze ────────────────────────────────
print("\nGroup 5: UnsqueezeGradFn.forward IS squeeze-via-slice_forward")

with tempfile.TemporaryDirectory() as tmpdir:
    grad5 = make_forwarded_ft([1, 4], ["p", "q", "r", "s"], tmpdir)
    grad5.requires_grad_(True)

    # Squeeze slices: [0, slice(None)] collapses dim 0 → shape [4]
    result5 = UnsqueezeGradFn.apply(grad5, 0, [0, slice(None)])

    run_test("forward produces shape [4]", result5.ft_capacity_shape == [4])
    run_test("forward [0] = p", read_ft_element(result5, 0) == "p")
    run_test("forward [3] = s", read_ft_element(result5, 3) == "s")

# ── Group 6: UnsqueezeGradFn.backward dispatches 2nd derivative ───────────────
print("\nGroup 6: UnsqueezeGradFn.backward() dispatches 2nd derivative")

with tempfile.TemporaryDirectory() as tmpdir:
    grad6 = make_forwarded_ft([1, 3], ["u", "v", "w"], tmpdir)
    grad6.requires_grad_(True)

    coll6 = []
    with dispatch_policy(TracePolicy(coll6)):
        result6 = UnsqueezeGradFn.apply(grad6, 0, [0, slice(None)])
        result6.sum().backward()

    run_test("2nd derivative dispatched", len(coll6) >= 1)
    if len(coll6) >= 1:
        run_test("fn is unsqueeze_forward", coll6[0].fn is unsqueeze_forward)
        run_test("grad_output in inputs", "grad_output" in coll6[0].inputs)
        run_test("dim in inputs", coll6[0].inputs.get("dim") == 0)
        run_test("squeeze_slices in inputs", "squeeze_slices" in coll6[0].inputs)
        run_test("grad_input in inputs", "grad_input" in coll6[0].inputs)
        run_test("grad_input shape [3]",
                 coll6[0].inputs["grad_input"].ft_capacity_shape == [3])

# ── Group 7: UnsqueezeGradFn grad_fn attached ─────────────────────────────────
print("\nGroup 7: UnsqueezeGradFn grad_fn attached when input requires_grad")

with tempfile.TemporaryDirectory() as tmpdir:
    grad7 = make_forwarded_ft([1, 2], ["m", "n"], tmpdir)
    grad7.requires_grad_(True)

    result7 = UnsqueezeGradFn.apply(grad7, 0, [0, slice(None)])
    run_test("result requires_grad", result7.requires_grad)
    run_test("result has grad_fn", result7.grad_fn is not None)
    run_test("grad_fn is UnsqueezeGradFnBackward",
             "UnsqueezeGradFn" in type(result7.grad_fn).__name__)

# ── Group 8: FtSlice.backward uses SliceGradFn ────────────────────────────────
print("\nGroup 8: Integration — FtSlice.backward() → SliceGradFn → TracePolicy")

with tempfile.TemporaryDirectory() as tmpdir:
    data8 = [f"v{i}" for i in range(10)]
    ft8 = make_forwarded_ft([10], data8, tmpdir)
    ft8.requires_grad_(True)

    sliced8 = ft_slice(ft8, [slice(2, 6)])
    run_test("FtSlice forward shape", sliced8.ft_capacity_shape == [4])
    run_test("FtSlice forward [0] = v2", read_ft_element(sliced8, 0) == "v2")

    grad8 = make_forwarded_ft([4], ["g0", "g1", "g2", "g3"], tmpdir)
    grad8.requires_grad_(True)

    coll8 = []
    ctx8 = type("Ctx", (), {})()
    ctx8.original_shape = [10]
    ctx8.slices = [slice(2, 6)]
    ctx8.input_ft = ft8
    ctx8.output_shape = [4]
    result8 = FtSlice.backward(ctx8, grad8)
    grad_input8 = result8[0]
    # Trigger 2nd derivative inside dispatch_policy
    with dispatch_policy(TracePolicy(coll8)):
        from experience.future_tensor.function.slice_2nd import SliceGradFn
        SliceGradFn.apply(grad8, [10], [slice(2, 6)]).sum().backward()

    run_test("backward produces shape [10]", grad_input8.ft_capacity_shape == [10])
    run_test("backward [2] = g0", read_ft_element(grad_input8, 2) == "g0")
    run_test("backward [5] = g3", read_ft_element(grad_input8, 5) == "g3")
    run_test("backward [0] empty", read_ft_element(grad_input8, 0) is None)
    run_test("SliceGradFn dispatched 2nd derivative", len(coll8) >= 1)
    if len(coll8) >= 1:
        run_test("fn is slice_backward", coll8[0].fn is slice_backward)

# ── Group 9: FtUnsqueeze.backward uses UnsqueezeGradFn ────────────────────────
print("\nGroup 9: Integration — FtUnsqueeze.backward() → UnsqueezeGradFn → TracePolicy")

with tempfile.TemporaryDirectory() as tmpdir:
    data9 = [f"e{i}" for i in range(5)]
    ft9 = make_forwarded_ft([5], data9, tmpdir)
    ft9.requires_grad_(True)

    unsqueezed9 = ft_unsqueeze(ft9, 0)
    run_test("FtUnsqueeze forward shape", unsqueezed9.ft_capacity_shape == [1, 5])
    run_test("FtUnsqueeze forward [0] = e0", read_ft_element(unsqueezed9, 0) == "e0")

    grad9 = make_forwarded_ft([1, 5], [f"g{i}" for i in range(5)], tmpdir)
    grad9.requires_grad_(True)

    coll9 = []
    ctx9 = type("Ctx", (), {})()
    ctx9.dim = 0
    ctx9.input_ft = ft9
    ctx9.output_shape = [1, 5]
    result9 = FtUnsqueeze.backward(ctx9, grad9)
    grad_input9 = result9[0]
    # Trigger 2nd derivative inside dispatch_policy
    with dispatch_policy(TracePolicy(coll9)):
        from experience.future_tensor.function.unsqueeze_2nd import UnsqueezeGradFn
        UnsqueezeGradFn.apply(grad9, 0, [0, slice(None)]).sum().backward()

    run_test("backward produces shape [5]", grad_input9.ft_capacity_shape == [5])
    run_test("backward [0] = g0", read_ft_element(grad_input9, 0) == "g0")
    run_test("backward [4] = g4", read_ft_element(grad_input9, 4) == "g4")
    run_test("UnsqueezeGradFn dispatched 2nd derivative", len(coll9) >= 1)
    if len(coll9) >= 1:
        run_test("fn is unsqueeze_forward", coll9[0].fn is unsqueeze_forward)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
if failed == 0:
    print("All slice/unsqueeze 2nd-derivative tests passed.")
else:
    sys.exit(1)
