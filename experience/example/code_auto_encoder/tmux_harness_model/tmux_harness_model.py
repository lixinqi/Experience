"""TmuxHarnessModel: Simplified code auto-encoder without tool-use loop.

Architecture:
  1. Read .cloze_task.json + target file (Python I/O, no LLM)
  2. ft_legacy_expert with trainable experience (code generation)
  3. ft_recurrent (retry until valid)

Key simplification vs HarnessModel:
  - No Stage 1 nested ft_recurrent for context gathering
  - Context gathered directly via Python file reads
  - Same ft_legacy_expert + ft_recurrent pipeline for trainable generation
"""

import json
import os
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import sympy

from experience.future_tensor.future_tensor import FutureTensor, _read_element
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_legacy_expert import ft_legacy_expert
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


_SYSTEM_GENERATION = """\
You are a coding agent that recovers masked code from a codebase.

Given the full file content with a masked region, output ONLY the missing source code
that was in the masked region. No explanations, no markdown fences, no extra comments.
Do NOT output function signatures, decorators, or docstrings unless they were in the masked region.

The masked region is marked with <AUTOENCODER-CLOZE-MASK-PLACEHOLDER> in the file.
Replace that placeholder with the original code. The mask may span multiple lines — output ALL
lines that were removed, preserving original indentation exactly.
"""


class TmuxHarnessModel(nn.Module):
    def __init__(
        self,
        n_experience: int = 64,
        experience_dir: Optional[str] = None,
        max_codegen_steps: int = 4,
        topk: int = 2,
        task_prompt: str = "",
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        if n_experience > 0:
            if experience_dir is None:
                experience_dir = tempfile.mkdtemp()
            self.experience = make_tensor(
                [["", "", ""]] * n_experience,
                experience_dir,
            )
            self.experience.requires_grad_(True)
        else:
            self.experience = None
        self.max_codegen_steps = max_codegen_steps
        self.topk = topk
        self.task_prompt = task_prompt
        self.llm_method = llm_method
        self.llm_env = llm_env
        self.last_output_ft = None

    def forward(self, worktree_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = worktree_tensor.shape[0]
        tmpdir = worktree_tensor.st_relative_to

        # Step 1: Gather context synchronously (Python file I/O, no LLM)
        contexts = []
        for b in range(batch_size):
            wt_path = _read_element(worktree_tensor, b)
            task_json_path = os.path.join(wt_path, ".cloze_task.json")
            with open(task_json_path) as f:
                task = json.load(f)

            target_path = os.path.join(wt_path, task["target_file"])
            with open(target_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # Add line numbers (cat -n style)
            numbered = "\n".join(
                f"{i+1:>6}\t{line}"
                for i, line in enumerate(file_content.splitlines())
            )

            # Build task header with mask metadata
            mask_start = task.get("mask_start_1idx")
            mask_end = task.get("mask_end_1idx")
            mask_line = task.get("mask_line", "?")
            if isinstance(mask_start, int) and isinstance(mask_end, int):
                n_lines = mask_end - mask_start + 1
                task_header = (
                    f"CLOZE_TASK: target_file={task['target_file']}, "
                    f"mask_line={mask_line}, mask_lines={n_lines} ({mask_start}-{mask_end})"
                )
            else:
                task_header = (
                    f"CLOZE_TASK: target_file={task['target_file']}, mask_line={mask_line}"
                )

            contexts.append(task_header + "\n\n" + numbered)

        # Step 2: Build [batch, max_codegen_steps] FutureTensor for ft_legacy_expert
        ft_context = FutureTensor(
            tmpdir,
            self._make_context_repeater(contexts),
            [sympy.Integer(batch_size), sympy.Integer(self.max_codegen_steps)],
        )
        # Materialize context before ft_legacy_expert reads it
        ctx_prompts = make_tensor(
            [["ctx"] * self.max_codegen_steps] * batch_size,
            tmpdir,
        )
        ft_context.ft_forward(ctx_prompts)

        # Step 3: ft_legacy_expert with trainable experience
        ft_gen = ft_legacy_expert(
            ft_context,
            self.experience if self.experience is not None else make_tensor([["", "", ""]], tmpdir),
            output_prompt=self._make_code_gen_output_prompt(),
            task_prompt=self.task_prompt,
            topk=self.topk,
            llm_method=self.llm_method,
            llm_env=self.llm_env,
        )

        # Step 4: ft_recurrent for retry
        output_ft = ft_recurrent(
            ft_gen,
            task_prompt=self.task_prompt,
            llm_method=self.llm_method,
        )

        # Materialize output
        gen_prompts = make_tensor(["generate code"] * batch_size, tmpdir)
        output_ft.ft_forward(gen_prompts)
        self.last_output_ft = output_ft
        return output_ft.ft_initial_static_tensor

    def _make_context_repeater(self, contexts: List[str]):
        """Build ft_async_get for [batch, L] that broadcasts context strings."""
        async def context_repeater(coords: List[int], prompt: str) -> Tuple[str, Status]:
            batch_idx = coords[0]
            return (contexts[batch_idx], Status.confidence(0.9))

        return context_repeater

    def _make_code_gen_output_prompt(self):
        """Return output_prompt callable for ft_legacy_expert's code generation stage."""
        def output_prompt(task_prompt, workspace_dir, exp_view_dir, input_view_dir, output_dir):
            return (
                _SYSTEM_GENERATION.strip() + "\n\n"
                "The file content (with line numbers and CLOZE_TASK metadata) is in const_input_view/. "
                "The mask is marked with <AUTOENCODER-CLOZE-MASK-PLACEHOLDER>. "
                "Similar past code recoveries are in const_experiance_view/ as examples "
                "(the value files contain past recovery results).\n\n"
                "Write ONLY the exact original lines of code that were removed, "
                "matching the line count from CLOZE_TASK. "
                "Preserve original indentation exactly."
            )
        return output_prompt
