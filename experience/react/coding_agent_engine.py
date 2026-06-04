"""
CodingAgentEngine: General-purpose engine functor for coding agents.

Wraps an interactive coding agent (ducc/claude) running in an inner tmux session.
Communicates via file-based IPC: writes step_N_input.json, reads step_N_output.json.
Agent stays warm across steps — launched once, reused until session is killed.

__call__ returns a FutureTensor — a lazy op that, when evaluated at coordinates,
triggers the inner agent step and yields KeystrokeNode.
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
from experience.react.react_types import KeystrokeNode


LAUNCH_COMMANDS = {
    "claude_code": "claude",
    "open_claw": "openclaw",
    "open_code": "opencode",
    "ducc": "ducc",
    "hermes": "hermes",
}

_PROMPT_SCHEMA = (
    '{\n'
    '  "current_fixation": [row, col],\n'
    '  "current_foveal": "exact text at fixation from capture (3-10 chars)",\n'
    '  "keystrokes": "T <type text here>",\n'
    '  "predicted_next_fixation": [row, col],\n'
    '  "predicted_next_foveal": "exact text expected after keystroke (3-10 chars)",\n'
    '  "children": [\n'
    '    {\n'
    '      "current_fixation": [row, col],\n'
    '      "current_foveal": "exact text at fixation from capture (3-10 chars)",\n'
    '      "keystrokes": "C Enter",\n'
    '      "predicted_next_fixation": [row, col],\n'
    '      "predicted_next_foveal": "exact text expected after keystroke (3-10 chars)",\n'
    '      "children": []\n'
    '    }\n'
    '  ]\n'
    '}'
)

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
    ks = d.get("keystrokes", "")
    # Handle LLM occasionally outputting array instead of string
    if isinstance(ks, list):
        ks = ks[0] if ks else ""
    node = KeystrokeNode(
        current_fixation=tuple(d.get("current_fixation", (0, 0))),
        current_foveal=d.get("current_foveal", ""),
        keystrokes=ks,
        predicted_next_fixation=tuple(d.get("predicted_next_fixation", (0, 0))),
        predicted_next_foveal=d.get("predicted_next_foveal", ""),
    )
    for child in d.get("children", []):
        node.children.append(_parse_keystroke_node(child))
    return node


class CodingAgentEngine:
    """Engine functor that delegates to a coding agent running in an inner tmux.

    Features:
      - Interactive agent stays warm across steps (launched once per session)
      - Flash model for speed (sourced from ~/.llm-config/flash-env.sh)
      - --add-dir pre-authorizes the work_dir to skip file permission prompts
      - Auto-approve handles stray permission prompts
      - Array→string normalization for LLM output quirks
    """

    def __init__(self, agent_type: str, inner_session_id: str, work_dir: str,
                 launch_cmd: Optional[str] = None):
        self.agent_type = agent_type
        self.inner_session_id = inner_session_id
        self.session_name = f"{tmux_session_prefix}{inner_session_id}"
        self.work_dir = Path(work_dir)
        binary = launch_cmd or LAUNCH_COMMANDS.get(agent_type, agent_type)
        self.launch_cmd = f"{binary} --add-dir {self.work_dir}"
        self.step_counter = 0
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ── Public interface ──────────────────────────────────────────────

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

            logical_views = {
                "capture": capture.ft_describe_logical_view(),
                "fixation": fixation.ft_describe_logical_view(),
                "mail": mail.ft_describe_logical_view(),
            }
            self._write_step_input(capture_text, fixation_text, task,
                                   logical_views, step_n)
            self._send_step_command(step_n)
            if not self._wait_for_signal(step_n, timeout=300):
                return ("", Status.self_confidence_but_failed(0.5))
            node = self._read_step_output(step_n)
            if node.keystrokes:
                return (node.serialize(), Status.confidence(1.0))
            return (node.serialize(), Status.self_confidence_but_failed(0.5))

        ft = FutureTensor(str(self.work_dir), _engine_get, schema)
        ft.ft_capacity_shape = shape
        ft.requires_grad_(True)
        return ft

    # ── Tmux session management ────────────────────────────────────────

    def _pane(self):
        """Get the active pane of the inner agent tmux session."""
        server = libtmux.Server()
        session = server.sessions.get(session_name=self.session_name)
        return session.active_window.active_pane if session else None

    def _ensure_launched(self):
        """Launch the inner agent if not already running."""
        server = libtmux.Server()
        for s in server.sessions:
            if s.session_name == self.session_name:
                return  # already warm — reuse

        session = server.new_session(session_name=self.session_name, attach=False)
        time.sleep(0.3)
        pane = session.active_window.active_pane

        # Clean env: unset vars that interfere with nested agent launch
        pane.send_keys("unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT", enter=True)
        time.sleep(0.1)
        # Source flash model for speed
        flash_env = os.path.expanduser("~/.llm-config/flash-env.sh")
        if os.path.isfile(flash_env):
            pane.send_keys(f"source {flash_env}", enter=True)
            time.sleep(0.1)
        # Launch the agent
        pane.send_keys(self.launch_cmd, enter=True)

        # Wait for REPL prompt
        deadline = time.time() + 30
        while time.time() < deadline:
            time.sleep(1)
            lines = pane.capture_pane()
            tail = "\n".join(lines[-5:]) if isinstance(lines, list) else str(lines)
            if "❯" in tail or ">" in tail:
                break

        # Readiness check: confirm agent processes commands
        marker = "__ENGINE_READY__"
        pane.send_keys(f"echo {marker}", enter=True)
        deadline2 = time.time() + 30
        while time.time() < deadline2:
            time.sleep(0.5)
            lines = pane.capture_pane()
            full = "\n".join(lines) if isinstance(lines, list) else str(lines)
            if marker in full and ("❯" in full or ">" in full):
                return
        # Proceed anyway after timeout — agent may still be usable

    # ── File-based IPC ─────────────────────────────────────────────────

    def _write_step_input(self, capture_text: str, fixation_text: str,
                          task: str, logical_views: dict, step_n: int):
        input_file = self.work_dir / f"step_{step_n}_input.json"
        data = {
            "step": step_n,
            "task": task,
            "capture": capture_text,
            "fixation": fixation_text,
            "logical_views": logical_views,
            "output_file": str(self.work_dir / f"step_{step_n}_output.json"),
            "signal_file": str(self.work_dir / f"step_{step_n}_done"),
        }
        input_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _send_step_command(self, step_n: int):
        pane = self._pane()
        input_path = self.work_dir / f"step_{step_n}_input.json"
        prompt = (
            f"Read {input_path} and plan ALL keystrokes to complete the task. "
            f"Write a KeystrokeNode JSON to the output_file using this schema:\n"
            f"{_PROMPT_SCHEMA}\n\n"
            f'Keystrokes: "T text" for typing, "C key" for ctrl keys. '
            f"current_foveal and predicted_next_foveal: 3-10 char substrings "
            f"from the terminal capture. "
            f"Then touch the signal_file. No extra output."
        )
        pane.send_keys(prompt, enter=True)
        time.sleep(0.2)
        pane.enter()  # ducc paste fix

    def _read_step_output(self, step_n: int) -> KeystrokeNode:
        output_file = self.work_dir / f"step_{step_n}_output.json"
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            return _parse_keystroke_node(data)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return KeystrokeNode()

    # ── Wait loop with auto-approve ────────────────────────────────────

    def _detect_stuck(self, pane) -> bool:
        """Check if agent is stuck at a permission prompt."""
        lines = pane.capture_pane()
        tail = "\n".join(lines[-8:]) if isinstance(lines, list) else str(lines)
        return any(kw in tail for kw in _STUCK_KEYWORDS)

    def _auto_approve(self, pane):
        """Send Enter to accept the default (Yes) on a permission prompt."""
        pane.send_keys("Enter")
        time.sleep(0.2)

    def _wait_for_signal(self, step_n: int, timeout: float) -> bool:
        signal_file = self.work_dir / f"step_{step_n}_done"
        deadline = time.time() + timeout
        pane = self._pane()
        while time.time() < deadline:
            if signal_file.exists():
                return True
            if pane is not None and self._detect_stuck(pane):
                self._auto_approve(pane)
                time.sleep(0.3)
                continue
            time.sleep(0.3)
        return False


def make_engine_step(agent_type: str, inner_session_id: str, work_dir: str,
                     launch_cmd: Optional[str] = None):
    """Create an engine_step function from a CodingAgentEngine instance."""
    return CodingAgentEngine(agent_type, inner_session_id, work_dir, launch_cmd)
