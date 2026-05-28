"""
CodingAgentEngine: General-purpose engine functor for coding agents.

Wraps any interactive coding agent (Claude Code, OpenClaw, OpenCode, Ducc, Hermes)
running in an "inner tmux" (the brain). Communicates via file-based IPC.

__call__ returns a FutureTensor — a lazy op that, when evaluated at coordinates,
triggers the inner agent step and yields EngineOutput.

ADT:
    CodingAgentEngine :=
        FutureTensor[
          Awaitable[$output EngineOutput]
            <- $coordinates list[int]
            <- $prompt str
        ]
        # __call__
        <- $task str
        <- $mail FutureTensor
        <- $fixation FutureTensor
        <- $capture FutureTensor
        # __init__
        <- ($launch_cmd str | void)
        <- $work_dir str
        <- $inner_session_id str
        <- $agent_type str
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional

import libtmux

from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.react.fixation import extract_foveal
from experience.react.react_types import EngineOutput, KeystrokeNode


LAUNCH_COMMANDS = {
    "claude_code": "claude",
    "open_claw": "openclaw",
    "open_code": "opencode",
    "ducc": "ducc",
    "hermes": "hermes",
}


def _parse_fixation(fixation_text: str):
    try:
        parts = fixation_text.strip().split(",")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (0, 0)


async def _read_ft(ft: FutureTensor, coordinates: List[int]) -> str:
    if ft.ft_forwarded:
        try:
            _, filepath = ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except (IndexError, FileNotFoundError, OSError):
            pass
    text, _ = await ft.ft_async_get(coordinates, "")
    return text


def _parse_keystroke_node(d: dict) -> KeystrokeNode:
    """Recursively parse a dict into a KeystrokeNode tree."""
    node = KeystrokeNode(
        current_fixation=tuple(d.get("current_fixation", (0, 0))),
        current_foveal=d.get("current_foveal", ""),
        keystrokes=d.get("keystrokes", ""),
        predicted_next_fixation=tuple(d.get("predicted_next_fixation", (0, 0))),
        predicted_next_foveal=d.get("predicted_next_foveal", ""),
    )
    for child in d.get("children", []):
        node.children.append(_parse_keystroke_node(child))
    return node


class CodingAgentEngine:
    """Engine functor that delegates to a coding agent running in an inner tmux."""

    def __init__(self, agent_type: str, inner_session_id: str, work_dir: str,
                 launch_cmd: Optional[str] = None):
        self.agent_type = agent_type
        self.inner_session_id = inner_session_id
        self.session_name = f"{tmux_session_prefix}{inner_session_id}"
        self.work_dir = Path(work_dir)
        self.launch_cmd = launch_cmd or LAUNCH_COMMANDS.get(agent_type, agent_type)
        self.step_counter = 0
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, capture: FutureTensor, fixation: FutureTensor,
                 mail: FutureTensor, task: str) -> FutureTensor:
        self._ensure_launched()
        shape = list(capture.ft_capacity_shape)
        schema = list(capture.ft_shape_schema)

        async def _engine_get(coordinates, trajectory):
            capture_text = await _read_ft(capture, coordinates)
            fixation_text = await _read_ft(fixation, coordinates)
            step_n = self.step_counter
            self.step_counter += 1
            self._write_step_input(capture_text, fixation_text, task, step_n)
            self._send_step_command(step_n)
            self._wait_for_signal(step_n, timeout=120)
            output = self._read_step_output(step_n)
            if output.plan:
                return (output.plan.serialize(), Status.confidence(1.0))
            return ("", Status.self_confidence_but_failed(0.5))

        ft = FutureTensor(str(self.work_dir), _engine_get, schema)
        ft.ft_capacity_shape = shape
        ft.requires_grad_(True)
        return ft

    def _get_inner_pane(self):
        """Get the active pane of the inner agent tmux session."""
        server = libtmux.Server()
        session = server.sessions.get(session_name=self.session_name)
        if session is None:
            return None
        return session.active_window.active_pane

    def _ensure_launched(self):
        server = libtmux.Server()
        for s in server.sessions:
            if s.session_name == self.session_name:
                return
        session = server.new_session(session_name=self.session_name, attach=False)
        time.sleep(0.5)
        pane = session.active_window.active_pane
        # Unset env vars that prevent nested agent launch
        pane.send_keys("unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT", enter=True)
        time.sleep(0.3)
        pane.send_keys(self.launch_cmd, enter=True)
        # Wait for agent REPL prompt (❯ or > in last few lines)
        deadline = time.time() + 30
        while time.time() < deadline:
            time.sleep(2)
            lines = pane.capture_pane()
            tail = "\n".join(lines[-5:]) if isinstance(lines, list) else str(lines)
            if "\u276f" in tail or ">" in tail:
                return
        # Proceed anyway after timeout

    def _write_step_input(self, capture_text: str, fixation_text: str,
                          task: str, step_n: int):
        input_file = self.work_dir / f"step_{step_n}_input.json"
        data = {
            "step": step_n,
            "task": task,
            "capture": capture_text,
            "fixation": fixation_text,
            "output_file": str(self.work_dir / f"step_{step_n}_output.json"),
            "signal_file": str(self.work_dir / f"step_{step_n}_done"),
        }
        input_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    _STUCK_KEYWORDS = [
        "Do you want to create",
        "Do you want to proceed",
        "Do you want to edit",
        "Do you want to delete",
        "Do you want to overwrite",
        "Do you want to run",
        "Do you want to execute",
        "Do you want to replace",
        "Do you want to write",
        "Permit field",
    ]

    def _detect_stuck(self, pane) -> bool:
        """Check whether the inner agent is stuck at a permission prompt."""
        lines = pane.capture_pane()
        tail = "\n".join(lines[-8:]) if isinstance(lines, list) else str(lines)
        return any(kw in tail for kw in self._STUCK_KEYWORDS)

    def _auto_approve(self, pane):
        """Send Enter to select the default Yes option on a permission prompt."""
        pane.send_keys("Enter")
        time.sleep(0.3)

    def _send_step_command(self, step_n: int):
        pane = self._get_inner_pane()
        input_path = self.work_dir / f"step_{step_n}_input.json"
        prompt_schema = (
            '{\n'
            '  "current_fixation": [row, col],\n'
            '  "current_foveal": "...",\n'
            '  "keystrokes": "T <type text here>",\n'
            '  "predicted_next_fixation": [row, col],\n'
            '  "predicted_next_foveal": "...",\n'
            '  "children": [\n'
            '    {\n'
            '      "current_fixation": [row, col],\n'
            '      "current_foveal": "...",\n'
            '      "keystrokes": "C Enter",\n'
            '      "predicted_next_fixation": [row, col],\n'
            '      "predicted_next_foveal": "...",\n'
            '      "children": []\n'
            '    }\n'
            '  ]\n'
            '}'
        )
        prompt = (
            f"Read {input_path} and execute the next step. "
            f"Write a KeystrokeNode plan to the output_file. "
            f"Use exactly this JSON schema:\n{prompt_schema}\n\n"
            f"Keystroke format: T for type text, C for control key (C Enter, C c, etc). "
            f"Then touch the signal_file. "
            f"After touching the signal_file, do absolutely nothing else."
        )
        pane.send_keys(prompt, enter=True)

    def _wait_for_signal(self, step_n: int, timeout: float):
        signal_file = self.work_dir / f"step_{step_n}_done"
        deadline = time.time() + timeout
        pane = self._get_inner_pane()
        while time.time() < deadline:
            if signal_file.exists():
                return
            if pane is not None and self._detect_stuck(pane):
                self._auto_approve(pane)
                time.sleep(0.5)
                continue
            time.sleep(0.5)
        raise TimeoutError(f"Step {step_n} signal not received within {timeout}s")

    def _read_step_output(self, step_n: int) -> EngineOutput:
        output_file = self.work_dir / f"step_{step_n}_output.json"
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            node = _parse_keystroke_node(data)
            return EngineOutput(plan=node)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return EngineOutput(mail="Failed to parse agent output")


# Module-level engine_step factory for convenience
def make_engine_step(agent_type: str, inner_session_id: str, work_dir: str,
                     launch_cmd: Optional[str] = None):
    """Create an engine_step function from a CodingAgentEngine instance."""
    return CodingAgentEngine(agent_type, inner_session_id, work_dir, launch_cmd)
