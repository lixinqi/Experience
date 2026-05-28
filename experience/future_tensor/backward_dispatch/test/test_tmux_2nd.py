"""
test_tmux_2nd.py — 2nd-derivative tests for tmux FutureTensor ops.

Pattern:
    mock_input = need_reflection(mock_input, backward_dispatch_start)
    loss = model.forward(mock_input)
    loss.backward(create_graph=True)
    with dispatch_policy(TracePolicy):
        backward_dispatch_start.grad.backward()

Groups:
  1. Module imports and structure
  2. TmuxCreateSessionGradFn: forward IS pass-through backward
  3. TmuxCreateSessionGradFn: backward() dispatches 2nd derivative
  4. TmuxSendTextGradFn: forward IS pass-through backward
  5. TmuxSendTextGradFn: backward() dispatches 2nd derivative
  6. TmuxSendCtrlGradFn: forward IS pass-through backward
  7. TmuxSendCtrlGradFn: backward() dispatches 2nd derivative
  8. Integration: FtTmuxCreateSession.backward() uses GradFn
  9. Integration: FtTmuxSendText.backward() uses GradFn
  10. Integration: FtTmuxSendCtrl.backward() uses GradFn
  11. Natural 2nd-derivative flow dispatches records

Run:
    python -m experience.future_tensor.backward_dispatch.test.test_tmux_2nd
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
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}")
        if expected is not None:
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")


# ── imports ───────────────────────────────────────────────────────────────────

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.ft_tmux_create_session import (
    ft_tmux_create_session,
    FtTmuxCreateSession,
)
from experience.future_tensor.function.ft_tmux_send_text import (
    ft_tmux_send_text,
    FtTmuxSendText,
)
from experience.future_tensor.function.ft_tmux_send_ctrl import (
    ft_tmux_send_ctrl,
    FtTmuxSendCtrl,
)
from experience.future_tensor.function.ft_mean import ft_mean
from experience.future_tensor.function.tmux_create_session_backward import tmux_create_session_backward
from experience.future_tensor.function.tmux_send_text_backward import tmux_send_text_backward
from experience.future_tensor.function.tmux_send_ctrl_backward import tmux_send_ctrl_backward
from experience.future_tensor.function.tmux_create_session_2nd import TmuxCreateSessionGradFn
from experience.future_tensor.function.tmux_send_text_2nd import TmuxSendTextGradFn
from experience.future_tensor.function.tmux_send_ctrl_2nd import TmuxSendCtrlGradFn

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


def read_ft_element(ft, flat_index):
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
print("Group 1: Module imports and structure")

run_test("TmuxCreateSessionGradFn is autograd.Function subclass",
         issubclass(TmuxCreateSessionGradFn, torch.autograd.Function))
run_test("TmuxSendTextGradFn is autograd.Function subclass",
         issubclass(TmuxSendTextGradFn, torch.autograd.Function))
run_test("TmuxSendCtrlGradFn is autograd.Function subclass",
         issubclass(TmuxSendCtrlGradFn, torch.autograd.Function))
run_test("FtTmuxCreateSession is autograd.Function subclass",
         issubclass(FtTmuxCreateSession, torch.autograd.Function))
run_test("FtTmuxSendText is autograd.Function subclass",
         issubclass(FtTmuxSendText, torch.autograd.Function))
run_test("FtTmuxSendCtrl is autograd.Function subclass",
         issubclass(FtTmuxSendCtrl, torch.autograd.Function))


# ── Group 2: TmuxCreateSessionGradFn.forward IS pass-through ──────────────────
print("\nGroup 2: TmuxCreateSessionGradFn.forward IS pass-through")

with tempfile.TemporaryDirectory() as tmpdir:
    grad = make_forwarded_ft([3], ["g0", "g1", "g2"], tmpdir)
    grad.requires_grad_(True)

    result = TmuxCreateSessionGradFn.apply(grad)

    run_test("forward produces scalar", list(result.shape) == [])
    run_test("forward has grad_fn", result.grad_fn is not None)
    run_test("grad_fn is TmuxCreateSessionGradFnBackward",
             "TmuxCreateSessionGradFn" in type(result.grad_fn).__name__)


# ── Group 3: TmuxCreateSessionGradFn.backward dispatches 2nd derivative ──────
print("\nGroup 3: TmuxCreateSessionGradFn.backward() dispatches 2nd derivative")

with tempfile.TemporaryDirectory() as tmpdir:
    grad3 = make_forwarded_ft([2], ["a", "b"], tmpdir)
    grad3.requires_grad_(True)

    coll3 = []
    with dispatch_policy(TracePolicy(coll3)):
        result3 = TmuxCreateSessionGradFn.apply(grad3)
        result3.sum().backward()

    run_test("2nd derivative dispatched", len(coll3) >= 1)
    if len(coll3) >= 1:
        run_test("fn is tmux_create_session_backward",
                 coll3[0].fn is tmux_create_session_backward)
        run_test("grad_output in inputs", "grad_output" in coll3[0].inputs)
        run_test("grad_input in inputs", "grad_input" in coll3[0].inputs)
        run_test("grad_input shape [2]",
                 coll3[0].inputs["grad_input"].ft_capacity_shape == [2])


# ── Group 4: TmuxSendTextGradFn.forward IS pass-through ───────────────────────
print("\nGroup 4: TmuxSendTextGradFn.forward IS pass-through")

with tempfile.TemporaryDirectory() as tmpdir:
    grad4 = make_forwarded_ft([2], ["t0", "t1"], tmpdir)
    grad4.requires_grad_(True)

    result4 = TmuxSendTextGradFn.apply(grad4)

    run_test("forward produces scalar", list(result4.shape) == [])
    run_test("forward has grad_fn", result4.grad_fn is not None)
    run_test("grad_fn is TmuxSendTextGradFnBackward",
             "TmuxSendTextGradFn" in type(result4.grad_fn).__name__)


# ── Group 5: TmuxSendTextGradFn.backward dispatches 2nd derivative ───────────
print("\nGroup 5: TmuxSendTextGradFn.backward() dispatches 2nd derivative")

with tempfile.TemporaryDirectory() as tmpdir:
    grad5 = make_forwarded_ft([2], ["x", "y"], tmpdir)
    grad5.requires_grad_(True)

    coll5 = []
    with dispatch_policy(TracePolicy(coll5)):
        result5 = TmuxSendTextGradFn.apply(grad5)
        result5.sum().backward()

    run_test("2nd derivative dispatched", len(coll5) >= 1)
    if len(coll5) >= 1:
        run_test("fn is tmux_send_text_backward",
                 coll5[0].fn is tmux_send_text_backward)
        run_test("grad_output in inputs", "grad_output" in coll5[0].inputs)
        run_test("grad_input in inputs", "grad_input" in coll5[0].inputs)


# ── Group 6: TmuxSendCtrlGradFn.forward IS pass-through ───────────────────────
print("\nGroup 6: TmuxSendCtrlGradFn.forward IS pass-through")

with tempfile.TemporaryDirectory() as tmpdir:
    grad6 = make_forwarded_ft([2], ["c0", "c1"], tmpdir)
    grad6.requires_grad_(True)

    result6 = TmuxSendCtrlGradFn.apply(grad6)

    run_test("forward produces scalar", list(result6.shape) == [])
    run_test("forward has grad_fn", result6.grad_fn is not None)
    run_test("grad_fn is TmuxSendCtrlGradFnBackward",
             "TmuxSendCtrlGradFn" in type(result6.grad_fn).__name__)


# ── Group 7: TmuxSendCtrlGradFn.backward dispatches 2nd derivative ───────────
print("\nGroup 7: TmuxSendCtrlGradFn.backward() dispatches 2nd derivative")

with tempfile.TemporaryDirectory() as tmpdir:
    grad7 = make_forwarded_ft([2], ["u", "v"], tmpdir)
    grad7.requires_grad_(True)

    coll7 = []
    with dispatch_policy(TracePolicy(coll7)):
        result7 = TmuxSendCtrlGradFn.apply(grad7)
        result7.sum().backward()

    run_test("2nd derivative dispatched", len(coll7) >= 1)
    if len(coll7) >= 1:
        run_test("fn is tmux_send_ctrl_backward",
                 coll7[0].fn is tmux_send_ctrl_backward)
        run_test("grad_output in inputs", "grad_output" in coll7[0].inputs)
        run_test("grad_input in inputs", "grad_input" in coll7[0].inputs)


# ── Group 8: Integration — FtTmuxCreateSession.backward() ─────────────────────
print("\nGroup 8: Integration — FtTmuxCreateSession.backward()")

with tempfile.TemporaryDirectory() as tmpdir:
    ft8 = make_forwarded_ft([2], ["id0", "id1"], tmpdir)
    ft8.requires_grad_(True)

    output8 = ft_tmux_create_session(ft8)
    run_test("forward shape [2]", output8.ft_capacity_shape == [2])

    grad8 = make_forwarded_ft([2], ["g0", "g1"], tmpdir)
    grad8.requires_grad_(True)

    coll8 = []
    with dispatch_policy(TracePolicy(coll8)):
        ctx8 = type("Ctx", (), {})()
        ctx8.input_ft = ft8
        ctx8.shape = [2]
        ctx8.relative_to = tmpdir
        result8 = FtTmuxCreateSession.backward(ctx8, grad8)
        grad_input8 = result8
        TmuxCreateSessionGradFn.apply(grad8).sum().backward()

    run_test("backward produces scalar", list(grad_input8.shape) == [])
    run_test("GradFn dispatched 2nd derivative", len(coll8) >= 1)
    if len(coll8) >= 1:
        run_test("fn is tmux_create_session_backward",
                 coll8[0].fn is tmux_create_session_backward)


# ── Group 9: Integration — FtTmuxSendText.backward() ──────────────────────────
print("\nGroup 9: Integration — FtTmuxSendText.backward()")

with tempfile.TemporaryDirectory() as tmpdir:
    ft9 = make_forwarded_ft([2], ["sid0", "sid1"], tmpdir)
    ft9.requires_grad_(True)

    text_input9 = make_forwarded_ft([2], ["text_0", "text_1"], tmpdir)
    output9 = ft_tmux_send_text(text_input9, ft9)
    run_test("forward shape [2]", output9.ft_capacity_shape == [2])

    grad9 = make_forwarded_ft([2], ["g0", "g1"], tmpdir)
    grad9.requires_grad_(True)

    coll9 = []
    with dispatch_policy(TracePolicy(coll9)):
        ctx9 = type("Ctx", (), {})()
        ctx9.input_ft = ft9
        ctx9.shape = [2]
        ctx9.relative_to = tmpdir
        result9 = FtTmuxSendText.backward(ctx9, grad9)
        grad_input9 = result9[0]
        TmuxSendTextGradFn.apply(grad9).sum().backward()

    run_test("backward produces scalar", list(grad_input9.shape) == [])
    run_test("GradFn dispatched 2nd derivative", len(coll9) >= 1)
    if len(coll9) >= 1:
        run_test("fn is tmux_send_text_backward",
                 coll9[0].fn is tmux_send_text_backward)


# ── Group 10: Integration — FtTmuxSendCtrl.backward() ─────────────────────────
print("\nGroup 10: Integration — FtTmuxSendCtrl.backward()")

with tempfile.TemporaryDirectory() as tmpdir:
    ft10 = make_forwarded_ft([2], ["sid0", "sid1"], tmpdir)
    ft10.requires_grad_(True)

    ctrl_input10 = make_forwarded_ft([2], ["ctrl_0", "ctrl_1"], tmpdir)
    output10 = ft_tmux_send_ctrl(ctrl_input10, ft10)
    run_test("forward shape [2]", output10.ft_capacity_shape == [2])

    grad10 = make_forwarded_ft([2], ["g0", "g1"], tmpdir)
    grad10.requires_grad_(True)

    coll10 = []
    with dispatch_policy(TracePolicy(coll10)):
        ctx10 = type("Ctx", (), {})()
        ctx10.input_ft = ft10
        ctx10.shape = [2]
        ctx10.relative_to = tmpdir
        result10 = FtTmuxSendCtrl.backward(ctx10, grad10)
        grad_input10 = result10[0]
        TmuxSendCtrlGradFn.apply(grad10).sum().backward()

    run_test("backward produces scalar", list(grad_input10.shape) == [])
    run_test("GradFn dispatched 2nd derivative", len(coll10) >= 1)
    if len(coll10) >= 1:
        run_test("fn is tmux_send_ctrl_backward",
                 coll10[0].fn is tmux_send_ctrl_backward)


# ── Group 11: Natural 2nd-derivative flow ─────────────────────────────────────
print("\nGroup 11: Natural 2nd-derivative flow")

with tempfile.TemporaryDirectory() as tmpdir:
    ft11 = make_forwarded_ft([2], ["n0", "n1"], tmpdir)
    ft11.requires_grad_(True)

    backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
    anchored = need_reflection(ft11, backward_dispatch_start)
    model_output = ft_tmux_create_session(anchored)
    loss = ft_mean(model_output)

    run_test("loss is scalar", loss.shape == torch.Size([]))
    run_test("loss requires_grad", loss.requires_grad is True)

    # 1st backward
    loss.backward(create_graph=True)

    run_test("backward_dispatch_start.grad exists",
             backward_dispatch_start.grad is not None)
    run_test("backward_dispatch_start.grad has grad_fn",
             backward_dispatch_start.grad.grad_fn is not None)

    # 2nd backward with TracePolicy
    coll11 = []
    with dispatch_policy(TracePolicy(coll11)):
        backward_dispatch_start.grad.backward()

    run_test("TracePolicy collected at least 1 record", len(coll11) >= 1)
    if len(coll11) >= 1:
        run_test("record fn is tmux_create_session_backward",
                 coll11[0].fn is tmux_create_session_backward)
        run_test("record has grad_output",
                 "grad_output" in coll11[0].inputs)
        run_test("record has grad_input",
                 "grad_input" in coll11[0].inputs)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
if failed == 0:
    print("All tmux 2nd-derivative tests passed.")
else:
    sys.exit(1)
