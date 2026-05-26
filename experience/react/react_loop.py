"""
ReActLoop: engine-agnostic framework loop that drives any engine.

Composes a static DAG of FutureTensor ops:
  capture → engine_op → ft_speculative_keystroke → send_keys → post_capture
  All driven by ft_recurrent.
"""

import os
import tempfile
from typing import List, Optional

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.ft_make_forwarded import ft_make_forwarded
from experience.future_tensor.function.ft_tmux_create_session import ft_tmux_create_session
from experience.future_tensor.function.ft_tmux_send_keys import ft_tmux_send_keys
from experience.future_tensor.function.ft_tmux_capture_pane import ft_tmux_capture_pane
from experience.future_tensor.function.ft_sleep import ft_sleep
from experience.future_tensor.function.ft_sequential import ft_sequential
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_expand import ft_expand
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor

from experience.react.types import ReactConfig, EngineStepFn, ValidatorFn
from experience.react.fixation import compute_fixation
from experience.react.ft_speculative_keystroke import ft_speculative_keystroke



async def _read_capture(capture_ft, coords, trajectory):
    """Read capture text from upstream FutureTensor."""
    if capture_ft.ft_forwarded:
        _, fpath = capture_ft.ft_get_materialized_value(coords)
        return open(fpath).read() if os.path.isfile(fpath) else ""
    text, _ = await capture_ft.ft_async_get(coords, trajectory)
    return text


async def _call_engine(engine_step_fn, capture_text, task, tmpdir):
    """Package args and call engine_step, return EngineOutput."""
    fixation = compute_fixation(capture_text, (0, 0))
    fixation_str = f"{fixation[0]},{fixation[1]}"
    capture_arg = ft_make_forwarded(tmpdir, [1], [capture_text])
    fixation_arg = ft_make_forwarded(tmpdir, [1], [fixation_str])
    mail_arg = ft_make_forwarded(tmpdir, [1], [task])
    return await engine_step_fn(capture_arg, fixation_arg, mail_arg, [0], task)


def _make_engine_op(capture_ft, engine_step_fn, task, tmpdir, max_iters):
    """Create FutureTensor op wrapping engine_step → serialized KeystrokeNode tree."""
    shape = capture_ft.ft_capacity_shape

    async def _engine_get(coords, trajectory):
        capture_text = await _read_capture(capture_ft, coords, trajectory)
        if not capture_text:
            return ("", Status.self_confidence_but_failed(0.5))
        output = await _call_engine(engine_step_fn, capture_text, task, tmpdir)
        if output.plan:
            return (output.plan.serialize(), Status.confidence(1.0))
        return ("", Status.self_confidence_but_failed(0.5))

    engine_ft = FutureTensor(tmpdir, _engine_get,
                             [sympy.Integer(s) for s in shape])
    engine_ft.ft_capacity_shape = list(shape)
    engine_ft.requires_grad_(True)
    return engine_ft


def _make_validator(body, validator_fn, max_iters, tmpdir):
    """Create validator FutureTensor wrapping user's validator_fn."""
    async def _validator_get(coords, trajectory):
        i = coords[-1]
        text, _ = await body.ft_async_get(coords, trajectory)
        if validator_fn(text, i):
            return (text, Status.confidence(1.0))
        return ("", Status.self_confidence_but_failed(0.5))

    v = FutureTensor(tmpdir, _validator_get,
                     [sympy.Integer(1), sympy.Integer(max_iters)])
    v.ft_capacity_shape = [1, max_iters]
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
    max_iters = config.max_iterations
    expanded = ft_expand(instance_ft, [1, max_iters])
    capture = ft_tmux_capture_pane(expanded)
    engine_op = _make_engine_op(capture, engine_step_fn, task, tmpdir, max_iters)
    cmd_spec = ft_speculative_keystroke(engine_op, capture)
    send_keys = ft_tmux_send_keys(cmd_spec, expanded)
    post_capture = ft_sequential(
        ft_sleep(expanded, config.settle_delay),
        ft_tmux_capture_pane(expanded),
    )
    body = ft_sequential(send_keys, post_capture)
    validator = _make_validator(body, config.validator_fn, max_iters, tmpdir)
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
