"""
ReActLoop: engine-agnostic framework loop that drives any engine.

Composes a static DAG of FutureTensor ops:
  capture → engine_op → ft_speculative_keystroke → send_keys → post_capture
  All driven by ft_recurrent.

The iteration dimension is symbolic (sympy.Symbol 'n'). All ops in the DAG
carry [1, n] shape. Only the validator resolves n to concrete max_iters
for ft_recurrent.
"""

import os
import tempfile
from typing import Optional

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.ft_make_forwarded import ft_make_forwarded
from experience.future_tensor.function.ft_expand import ft_expand
from experience.future_tensor.function.ft_tmux_create_session import ft_tmux_create_session
from experience.future_tensor.function.ft_tmux_send_keys import ft_tmux_send_keys
from experience.future_tensor.function.ft_tmux_capture_pane import ft_tmux_capture_pane
from experience.future_tensor.function.ft_sleep import ft_sleep
from experience.future_tensor.function.ft_sequential import ft_sequential
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor

from experience.react.types import ReactConfig, EngineStepFn, ValidatorFn
from experience.react.fixation import compute_fixation
from experience.react.ft_speculative_keystroke import ft_speculative_keystroke


def _make_fixation_op(capture_ft, tmpdir):
    """Create FutureTensor op: capture → fixation point string.

    Reads capture text at coords, computes fixation, outputs "row,col".
    Same shape as capture_ft.
    """
    shape = capture_ft.ft_capacity_shape
    schema = capture_ft.ft_shape_schema

    async def _fixation_get(coords, trajectory):
        if capture_ft.ft_forwarded:
            _, fpath = capture_ft.ft_get_materialized_value(coords)
            capture_text = open(fpath).read() if os.path.isfile(fpath) else ""
        else:
            capture_text, _ = await capture_ft.ft_async_get(coords, trajectory)
        if not capture_text:
            return ("0,0", Status.confidence(1.0))
        fix = compute_fixation(capture_text, (0, 0))
        return (f"{fix[0]},{fix[1]}", Status.confidence(1.0))

    ft = FutureTensor(tmpdir, _fixation_get, list(schema))
    ft.ft_capacity_shape = list(shape)
    return ft


def _make_mail_op(task, capture_ft, tmpdir):
    """Create FutureTensor op: mail tensor carrying the task string.

    Same shape as capture_ft (broadcastable). Returns task at any coords.
    """
    shape = capture_ft.ft_capacity_shape
    schema = capture_ft.ft_shape_schema

    async def _mail_get(coords, trajectory):
        return (task, Status.confidence(1.0))

    ft = FutureTensor(tmpdir, _mail_get, list(schema))
    ft.ft_capacity_shape = list(shape)
    return ft


def _make_engine_op(capture_ft, fixation_ft, mail_ft, engine_step_fn, task, tmpdir):
    """Create FutureTensor op wrapping engine_step → serialized KeystrokeNode tree.

    Passes DAG tensors + real coordinates to engine per engine_step.viba contract.
    """
    shape = capture_ft.ft_capacity_shape
    schema = capture_ft.ft_shape_schema

    async def _engine_get(coords, trajectory):
        output = await engine_step_fn(capture_ft, fixation_ft, mail_ft, coords, task)
        if output.plan:
            return (output.plan.serialize(), Status.confidence(1.0))
        return ("", Status.self_confidence_but_failed(0.5))

    engine_ft = FutureTensor(tmpdir, _engine_get, list(schema))
    engine_ft.ft_capacity_shape = list(shape)
    engine_ft.requires_grad_(True)
    return engine_ft


def _make_validator(body, validator_fn, tmpdir):
    """Create validator FutureTensor with symbolic iteration dim."""
    n = sympy.Symbol("n")

    async def _validator_get(coords, trajectory):
        i = coords[-1]
        text, _ = await body.ft_async_get(coords, trajectory)
        if validator_fn(text, i):
            return (text, Status.confidence(1.0))
        return ("", Status.self_confidence_but_failed(0.5))

    v = FutureTensor(tmpdir, _validator_get, [sympy.Integer(1), n])
    v.ft_capacity_shape = [1, 0]
    v.requires_grad_(True)
    return v


def _setup_session(instance_ft, tmpdir):
    """Create tmux session and wait for shell init."""
    setup = ft_sequential(
        ft_tmux_create_session(instance_ft),
        ft_sleep(instance_ft, 0.5),
    )
    setup.ft_forward(st_make_tensor(["setup"], tmpdir))


def _compose_dag(instance_ft, engine_step_fn, task, tmpdir, config):
    """Compose the static DAG, return (output_ft, prompt_tensor)."""
    n = sympy.Symbol("n")
    expanded = ft_expand(instance_ft, [sympy.Integer(1), n])
    capture = ft_tmux_capture_pane(expanded)
    fixation = _make_fixation_op(capture, tmpdir)
    mail = _make_mail_op(task, capture, tmpdir)
    engine_op = _make_engine_op(capture, fixation, mail, engine_step_fn, task, tmpdir)
    cmd_spec = ft_speculative_keystroke(engine_op, capture)
    send_keys = ft_tmux_send_keys(cmd_spec, expanded)
    post_capture = ft_sequential(
        ft_sleep(expanded, config.settle_delay),
        ft_tmux_capture_pane(expanded),
    )
    body = ft_sequential(send_keys, post_capture)
    validator = _make_validator(body, config.validator_fn, tmpdir)
    output = ft_recurrent(validator, step_budget=config.step_budget)
    prompt = st_make_tensor([task], tmpdir)
    return output, prompt


def _drive_loop(output, prompt, max_iters):
    """Drive ft_forward in a sync loop until completion or exhaustion."""
    for step in range(max_iters):
        output.ft_forward(prompt)
        if output.ft_forwarded:
            print(f"  react_loop step {step}: completed")
            return True
        print(f"  react_loop step {step}: continuing")
    return False


def react_loop(
    instance_id: str,
    engine_step_fn: EngineStepFn,
    task: str,
    validator_fn: ValidatorFn,
    config: Optional[ReactConfig] = None,
) -> bool:
    """Run the ReAct loop as a static DAG driven by ft_recurrent."""
    if config is None:
        config = ReactConfig()
    config.validator_fn = validator_fn
    tmpdir = tempfile.mkdtemp(prefix="react_loop_")
    instance_ft = ft_make_forwarded(tmpdir, [1], [instance_id])
    _setup_session(instance_ft, tmpdir)
    output, prompt = _compose_dag(instance_ft, engine_step_fn, task, tmpdir, config)
    return _drive_loop(output, prompt, config.max_iterations)

