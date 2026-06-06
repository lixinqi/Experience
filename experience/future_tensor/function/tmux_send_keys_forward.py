"""
tmux_send_keys_forward :=
    FutureTensor
    <- $input FutureTensor
    <- $session_name FutureTensor  # broadcastable to $input
    # inline

Unified send-keys forward: parses "T " / "C " prefix to decide literal vs key-name.
$input elements contain prefixed commands ("T echo hello" or "C Enter").
$session_name elements contain the instance_id for session lookup.
Output shape = $input shape.

Infrastructure constraints:
- Payload is truncated to MAX_SEND_KEYS_LEN (16) characters.
- If TMUX_OUTPUTS_DIR is set and contains a .git repo, each send_keys
  triggers a pane capture + git commit (audit trail for validators).
"""

import os
import subprocess
from typing import List, Tuple

import libtmux
import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.function.tmux_send_text_forward import _broadcast_coords

MAX_SEND_KEYS_LEN = 256


def _parse_prefix(raw: str) -> Tuple[str, bool]:
    """Parse prefix → (payload, literal).
    TMUX-TEXT-<p> or T-<p> or T <p> = literal text.
    TMUX-CTRL-<k> or C-<k> or C <k> = ctrl key.
    """
    low = raw.lower()
    if low.startswith("tmux-text-"):
        return raw[10:], True
    if low.startswith("tmux-ctrl-"):
        return raw[10:], False
    if low.startswith("t-") or low.startswith("t "):
        return raw[2:], True
    if low.startswith("c-") or low.startswith("c "):
        return raw[2:], False
    return raw, True


def _get_pane(instance_id: str):
    """Find tmux pane by instance_id. Returns None if not found."""
    session_name = f"{tmux_session_prefix}{instance_id}"
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == session_name:
            return s.active_window.active_pane
    return None


async def _read_ft(ft, coordinates, shape, trajactory):
    """Read text content from a FutureTensor at given coordinates."""
    if ft.ft_forwarded:
        _coeff, filepath = ft.ft_get_materialized_value(coordinates)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    text, _status = await ft.ft_async_get(coordinates, trajactory)
    return text


def _tmux_outputs_commit(session_name, payload, pane):
    """Capture pane and commit to TMUX_OUTPUTS_DIR git repo (no-op if unconfigured).

    Pane captures go to .captures/ (gitignored) — they're audit trail only.
    The git diff tracks only user-created files in the repo root (typed code).
    """
    outputs_dir = os.environ.get("TMUX_OUTPUTS_DIR")
    if not outputs_dir or not os.path.isdir(f"{outputs_dir}/.git"):
        return
    # Write pane capture to .captures/ (excluded from diff measurement)
    captures_dir = f"{outputs_dir}/.captures/{session_name}"
    os.makedirs(captures_dir, exist_ok=True)
    seq = len([f for f in os.listdir(captures_dir) if f.endswith(".txt")])
    captured = pane.capture_pane()
    text = "\n".join(captured) if isinstance(captured, list) else str(captured)
    with open(f"{captures_dir}/{seq:04d}.txt", "w") as f:
        f.write(text)
    # Commit everything (code files in root + captures)
    subprocess.run(["git", "add", "."], cwd=outputs_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"{session_name}: {payload[:16]}", "--allow-empty"],
        cwd=outputs_dir, capture_output=True,
    )


def tmux_send_keys_forward(
    input_ft: FutureTensor,
    session_name_ft: FutureTensor,
) -> FutureTensor:
    """Forward: create a lazy FutureTensor whose ft_async_get sends keys to tmux."""
    shape = input_ft.ft_capacity_shape
    schema = input_ft.ft_shape_schema
    relative_to = input_ft.ft_initial_static_tensor.st_relative_to
    session_shape = session_name_ft.ft_capacity_shape

    async def send_async_get(coordinates: List[int], trajactory: str):
        raw = (await _read_ft(input_ft, coordinates, shape, trajactory))
        if not raw or not raw.strip():
            return ("", Status.confidence(1.0))

        # Trailing newline = auto-Enter (merged by engine for single-keystroke bash)
        enter_after = raw.endswith("\n")
        raw = raw.strip()

        payload, literal = _parse_prefix(raw)
        if not payload.strip():
            return ("", Status.confidence(1.0))

        session_coords = _broadcast_coords(coordinates, session_shape, shape)
        instance_id = (await _read_ft(session_name_ft, session_coords, session_shape, trajactory)).strip()

        pane = _get_pane(instance_id)
        if pane is None:
            return ("", Status.self_confidence_but_failed(0.0))

        if literal and len(payload) > MAX_SEND_KEYS_LEN:
            payload = payload[:MAX_SEND_KEYS_LEN]

        pane.send_keys(payload, literal=literal, enter=enter_after)
        _tmux_outputs_commit(instance_id, payload, pane)
        return ("", Status.confidence(1.0))

    result = FutureTensor(
        relative_to,
        send_async_get,
        list(schema),
    )
    result.ft_capacity_shape = list(shape)
    return result
