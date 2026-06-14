"""Shared types for the react package (MVP)."""

import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ── KeystrokeNode (deprecated in MVP — kept for legacy engine compat) ──

@dataclass
class KeystrokeNode:
    """Recursive tree of keystrokes (legacy — not used in MVP)."""
    current_fixation: Tuple[int, int] = (0, 0)
    current_foveal: str = ""
    keystrokes: str = ""
    predicted_next_fixation: Tuple[int, int] = (0, 0)
    predicted_next_foveal: str = ""
    children: List["KeystrokeNode"] = field(default_factory=list)
    comments: str = ""

    def serialize(self) -> str:
        d = {
            "current_fixation": list(self.current_fixation),
            "current_foveal": self.current_foveal,
            "keystrokes": self.keystrokes,
            "predicted_next_fixation": list(self.predicted_next_fixation),
            "predicted_next_foveal": self.predicted_next_foveal,
            "children": [json.loads(c.serialize()) for c in self.children],
        }
        if self.comments:
            d["comments"] = self.comments
        return json.dumps(d)

    @staticmethod
    def deserialize(text: str) -> Optional[dict]:
        if not text or not text.strip():
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None


# ── MVP types ──────────────────────────────────────────────────────────

@dataclass
class ReactConfig:
    """Configuration for the ReAct loop."""
    max_iterations: int = 20
    settle_delay: float = 0.3
    step_budget: int = 8
    llm_model: Optional[str] = None  # override ANTHROPIC_MODEL env var


# engine_step: (capture: str, task: str) -> list[str]
# Returns a list of keystroke DSL statements (tmux-text / tmux-ctrl).
EngineStepFn = Callable[[str, str], List[str]]

ValidatorFn = Callable[[str, int], bool]
