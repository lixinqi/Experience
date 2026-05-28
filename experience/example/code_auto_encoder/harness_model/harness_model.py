"""Harness Model: Simulate Claude Code with SymbolicTensor Framework.

Two-stage architecture:
  1. code_context_gather: ft_recurrent with accumulate_output=concat
  2. code_gen: ft_recurrent with validation
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

from experience.future_tensor.future_tensor import FutureTensor, _read_element, _coords_to_flat
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_legacy_expert import ft_legacy_expert
from experience.symbolic_tensor.function.select_qkv_indexes import select_qkv_indexes
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.llm_client.raw_llm_query import raw_llm_query
from experience.llm_client.agent_config_factory import AgentConfigFactory

from experience.example.code_auto_encoder.harness_model.function import (
    ALL_OPS,
    ALL_VALIDATORS,
    ft_unary,
)
from experience.example.code_auto_encoder.harness_model.function.harness_validator_op import HarnessValidatorOp


_SYSTEM_TOOL_SCHEMA = """\
You are a coding agent that gathers context from a codebase to recover masked code.

Available tools:
- glob(pattern: str) — Find files matching a glob pattern.
- grep(pattern: str, glob: str | None = None) — Search file contents by regex.
- read(file_path: str, offset: int = 0, limit: int = 200) — Read a file with line numbers.

Call a tool by outputting EXACTLY:
  tool_name(arg1="value1", arg2="value2")

If you have gathered enough context, output:
  CONTEXT_SUFFICIENT

Important: when reading the target file, read a range that includes the masked line and
several lines before and after it (e.g., offset=mask_line-5, limit=15) to get sufficient context.

Past successful tool traces for similar tasks may be provided for reference.
"""

_SYSTEM_GENERATION = """\
You are a coding agent that recovers masked code from a codebase.

Given the codebase context gathered from tool calls, output ONLY the missing source code
that was in the masked region. No explanations, no markdown fences, no extra comments.
Do NOT output function signatures, decorators, or docstrings unless they were in the masked region.

The masked region is marked with <AUTOENCODER-CLOZE-MASK-PLACEHOLDER> in the gathered context.
Replace that placeholder with the original code. The mask may span multiple lines — output ALL
lines that were removed, preserving original indentation exactly.
"""


async def _call_llm(system: str, user: str, llm_method: str = "raw_llm_api") -> str:
    """Async LLM call."""
    prompt = f"{system}\n\n{user}"
    if llm_method == "raw_llm_api":
        config = AgentConfigFactory.create_raw_llm_config()
        return await raw_llm_query(prompt, config=config)
    raise ValueError(f"Unsupported llm_method: {llm_method}")


def _parse_tool_call(response: str) -> Tuple[str, dict]:
    """Parse 'tool_name(arg="val")' into (tool_name, kwargs dict)."""
    import re
    response = response.strip()
    if response.startswith("CONTEXT_SUFFICIENT"):
        return ("CONTEXT_SUFFICIENT", {})
    # Match tool_name(key="value", ...)
    m = re.match(r"(\w+)\s*\((.*)\)", response)
    if not m:
        return ("", {})
    tool_name = m.group(1)
    args_str = m.group(2)
    kwargs = {}
    # Simple key="value" parser
    for key, val in re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_str):
        kwargs[key] = val
    for key, val in re.findall(r"(\w+)\s*=\s*'([^']*)'", args_str):
        kwargs[key] = val
    # Numeric args
    for key, val in re.findall(r'(\w+)\s*=\s*(\d+)', args_str):
        kwargs[key] = int(val)
    return tool_name, kwargs


_CONTEXT_BUDGET_CHARS = 12000
_MIDDLE_PIECE_KEEP_CHARS = 500


def _extract_clean_read(cur: str) -> str:
    """Return the clean file content from a tool trace, or '' if not a valid read result."""
    if cur.startswith("[") and "\n" in cur:
        header, result = cur.split("\n", 1)
        if not header.startswith("[read("):
            return ""
        _errors = ("ERROR:", "(file not found", "(read error", "(no matches", "(empty", "(regex error")
        if any(result.startswith(e) for e in _errors):
            return ""
        return result
    if cur.startswith("[invalid tool"):
        return ""
    return cur


def _concat_context(acc: str, cur: str) -> str:
    """Accumulate clean read results, discarding grep/glob noise and errors."""
    clean = _extract_clean_read(cur)
    if not clean:
        return acc
    if not acc:
        return clean
    return acc + "\n\n---\n\n" + clean


def _concat_context_weighted(acc: str, cur: str) -> str:
    """Weight-decayed accumulation: adds [Step N] labels; compresses middle pieces when over budget.

    Strategy when budget exceeded (len > _CONTEXT_BUDGET_CHARS):
    - Keep first piece intact  (bootstrap = class structure / imports)
    - Keep last piece intact   (most recent = most targeted)
    - Truncate middle pieces to _MIDDLE_PIECE_KEEP_CHARS chars each
    """
    clean = _extract_clean_read(cur)
    if not clean:
        return acc

    step_idx = acc.count("\n\n---\n\n") + 1 if acc else 0
    labeled = f"[Step {step_idx}]\n{clean}"

    if not acc:
        return labeled

    combined = acc + "\n\n---\n\n" + labeled

    if len(combined) > _CONTEXT_BUDGET_CHARS:
        pieces = combined.split("\n\n---\n\n")
        if len(pieces) >= 3:
            middle = [
                p[:_MIDDLE_PIECE_KEEP_CHARS] + "\n...(truncated)" if len(p) > _MIDDLE_PIECE_KEEP_CHARS else p
                for p in pieces[1:-1]
            ]
            combined = "\n\n---\n\n".join([pieces[0]] + middle + [pieces[-1]])
        elif len(pieces) == 2:
            keep = max(_CONTEXT_BUDGET_CHARS - len(pieces[-1]) - 20, 500)
            combined = pieces[0][:keep] + "\n...(truncated)\n\n---\n\n" + pieces[-1]

    return combined



async def _build_context_query_llm(task: dict, target_lines: list, llm_method: str) -> str:
    """Use LLM to generate semantic keywords for experience retrieval.

    Produces keywords in the same vocabulary space as the Q column (generated by StSGD via
    get_query_tensor), enabling meaningful Jaccard similarity in select_qkv_indexes.
    """
    content = "\n".join(target_lines[:200])
    system = (
        "You are a semantic grep keyword generator. "
        "Analyze the code and output 8-15 semantic keywords that summarize what this code does. "
        "Each line must contain exactly one keyword (snake_case or CamelCase, no spaces). "
        "Focus on conceptual meaning, not variable names."
    )
    user = (
        f"File: {task.get('target_file', '')}\n"
        f"Code:\n{content}\n\n"
        "Output semantic keywords, one per line."
    )
    response = await _call_llm(system, user, llm_method)
    keywords = [
        line.strip() for line in response.splitlines()
        if line.strip() and not line.startswith("```") and not line.startswith("#")
    ]
    return "\n".join(keywords[:20])


def _fetch_experience_snippets(
    experience: Optional[torch.Tensor], query_str: str, topk: int
) -> str:
    """Retrieve top-k value entries from experience tensor and format as few-shot prompt.

    Returns "" when experience is None, all entries are TODO, or any error occurs.
    """
    if experience is None:
        return ""
    try:
        idx_tensors = select_qkv_indexes(experience, query_str, topk=topk, random_noise=True)
        if not idx_tensors:
            return ""
        row_indices = idx_tensors[0].tolist()
        exp_shape = list(experience.shape)
        snippets = []
        for i, row in enumerate(row_indices):
            flat = _coords_to_flat([row, 2], exp_shape)
            val = _read_element(experience, flat)
            if not val.strip() or "TODO" in val:
                continue
            snippets.append(f"[Example {i + 1}]\n{val}")
        if not snippets:
            return ""
        return "=== Similar past tool traces ===\n" + "\n".join(snippets)
    except Exception as e:
        print(f"[DEBUG experience retrieval error: {e}]")
        return ""


class HarnessModel(nn.Module):
    def __init__(
        self,
        n_experience: int = 256,
        experience_dir: Optional[str] = None,
        max_codegen_steps: int = 4,
        max_context_collects: int = 5,
        max_tool_call_retries: int = 2,
        topk: int = 2,
        task_prompt: str = "",
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
        accumulate_mode: str = "naive",
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
        self.max_context_collects = max_context_collects
        self.max_tool_call_retries = max_tool_call_retries
        self.topk = topk
        self.task_prompt = task_prompt
        self.llm_method = llm_method
        self.llm_env = llm_env
        self.accumulate_mode = accumulate_mode
        self.ops = ALL_OPS
        self.validators = ALL_VALIDATORS
        self.last_context_tensor = None
        self.last_output_ft = None

    def forward(self, worktree_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = worktree_tensor.shape[0]
        tmpdir = worktree_tensor.st_relative_to

        # ── Stage 1: code_context_gather ──
        # Nested recurrent:
        #   ft_raw [batch, C, R]
        #   -> ft_unary(validate)
        #   -> ft_recurrent(inner)  [batch, C]   # retry until valid tool result
        #   -> ft_unary(check context sufficiency)
        #   -> ft_recurrent(outer)  [batch]      # accumulate clean context across steps
        ft_raw = FutureTensor(
            tmpdir,
            self._make_tool_use(worktree_tensor),
            [sympy.Integer(batch_size), sympy.Integer(self.max_context_collects), sympy.Integer(self.max_tool_call_retries)],
        )
        ft_validated = ft_unary(ft_raw, self._validate_tool_result)

        # Inner recurrent: retry individual tool call until valid
        ft_tool_result = ft_recurrent(
            ft_validated,
            task_prompt=self.task_prompt,
            llm_method=self.llm_method,
        )

        # Check context sufficiency per step
        ft_checked = ft_unary(
            ft_tool_result,
            lambda c, p, o, s: self._check_context_sufficiency(c, p, o, s, worktree_tensor),
        )

        # Outer recurrent: accumulate clean context across collection steps
        accum_fn = (
            _concat_context_weighted if self.accumulate_mode == "weighted"
            else _concat_context
        )
        context_ft = ft_recurrent(
            ft_checked,
            accumulate_output=accum_fn,
            task_prompt=self.task_prompt,
            llm_method=self.llm_method,
        )

        # Materialize context
        context_prompts = make_tensor(["gather context"] * batch_size, tmpdir)
        context_ft.ft_forward(context_prompts)
        context_tensor = context_ft.ft_initial_static_tensor
        self.last_context_tensor = context_tensor

        # ── Stage 2: code_gen via ft_legacy_expert ──
        # Build [batch, L] input FutureTensor from materialized context + task metadata.
        ft_context_broadcast = FutureTensor(
            tmpdir,
            self._make_context_repeater(context_tensor, worktree_tensor),
            [sympy.Integer(batch_size), sympy.Integer(self.max_codegen_steps)],
        )
        # Materialize before ft_legacy_expert reads ft_initial_static_tensor.
        context_broadcast_prompts = make_tensor(
            [["ctx"] * self.max_codegen_steps] * batch_size,
            tmpdir,
        )
        ft_context_broadcast.ft_forward(context_broadcast_prompts)

        # Wire experience into the backward graph via ft_legacy_expert.
        ft_gen, _, _ = ft_legacy_expert(
            ft_context_broadcast,
            self.experience if self.experience is not None else make_tensor([["", "", ""]], tmpdir),
            output_prompt=self._make_code_gen_output_prompt(),
            task_prompt=self.task_prompt,
            topk=self.topk,
            llm_method=self.llm_method,
            llm_env=self.llm_env,
        )

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

    def _make_tool_use(self, worktree_tensor: torch.Tensor):
        """Build ft_async_get for a single raw tool-use step.

        Pipeline: LLM decides tool → parse → execute → return raw trace.
        No validation, no context sufficiency check. Returns scbf(0.5) for all
        non-confidence outputs so the caller can decide what to do next.
        """
        _experience_cache: Dict[int, str] = {}

        async def tool_use(coords: List[int], prompt: str) -> Tuple[str, Status]:
            batch_idx = coords[0]
            collect_idx = coords[1]
            retry_idx = coords[2]

            worktree_path = _read_element(worktree_tensor, batch_idx)

            # Read task metadata
            task_json_path = os.path.join(worktree_path, ".cloze_task.json")
            with open(task_json_path) as f:
                task = json.load(f)

            # Build user prompt for tool selection
            already_read_target = "read(" in prompt and task["target_file"] in prompt
            if already_read_target:
                read_hint = (
                    f"NOTE: You have already read {task['target_file']}. "
                    f"Do NOT read it again. Instead, search for related files in the same directory "
                    f"or grep for function/variable names from the mask region, then read the most "
                    f"relevant files. Or declare CONTEXT_SUFFICIENT if you have enough context."
                )
            else:
                read_hint = (
                    f"Strategy: start by reading the target file around the mask (offset=0, limit=200). "
                    f"If the mask is inside a function whose body is unclear, search for "
                    f"similar patterns in other files and read the most relevant files."
                )
            user_prompt = (
                f"Task: Recover the masked code in {task['target_file']} "
                f"(line {task['mask_line']}).\n"
                f"Current context:\n{prompt}\n\n"
                f"{read_hint}\n"
                f"Decide the next action."
            )

            # Inject experience snippets at first non-bootstrap step (collect_idx >= 1).
            # Cache per batch_idx so retrieval runs at most once per sample per forward pass.
            if self.experience is not None and collect_idx >= 1 and retry_idx == 0 \
                    and batch_idx not in _experience_cache:
                target_path = os.path.join(worktree_path, task["target_file"])
                try:
                    with open(target_path, "r", encoding="utf-8") as fh:
                        target_lines = [line.rstrip() for line in fh.readlines()]
                except (OSError, UnicodeDecodeError):
                    target_lines = []
                query_str = await _build_context_query_llm(task, target_lines, self.llm_method)
                _experience_cache[batch_idx] = _fetch_experience_snippets(
                    self.experience, query_str, self.topk
                )
                print(f"[DEBUG exp b{batch_idx} c{collect_idx}] snippets_len={len(_experience_cache[batch_idx])}")
            exp_snippets = _experience_cache.get(batch_idx, "")
            if exp_snippets:
                user_prompt = exp_snippets + "\n\n" + user_prompt

            # Bootstrap: on first collect step, first retry, auto-read target file
            if collect_idx == 0 and retry_idx == 0:
                tool_name = "read"
                kwargs = {"file_path": task["target_file"], "offset": 0, "limit": 200}
                print(f"[DEBUG b{batch_idx} c{collect_idx} r{retry_idx}] bootstrap read: {kwargs}")
                response = ""
            else:
                response = await _call_llm(_SYSTEM_TOOL_SCHEMA, user_prompt, self.llm_method)
                print(f"[DEBUG b{batch_idx} c{collect_idx} r{retry_idx}] LLM response: {repr(response[:120])}")
                tool_name, kwargs = _parse_tool_call(response)

            if tool_name == "CONTEXT_SUFFICIENT":
                print(f"[DEBUG b{batch_idx} c{collect_idx} r{retry_idx}] CONTEXT_SUFFICIENT")
                return ("", Status.confidence(0.9))

            if tool_name not in self.ops:
                return (
                    f"[invalid tool: {tool_name}]\n{response}",
                    Status.self_confidence_but_failed(0.5),
                )

            # Execute tool (raw, no validation here)
            result = self.ops[tool_name].forward(worktree_path, **kwargs)
            trace = f"[{tool_name}({kwargs})]\n{result}"
            print(f"[DEBUG b{batch_idx} c{collect_idx} r{retry_idx}] tool trace len={len(trace)}")
            return (trace, Status.self_confidence_but_failed(0.5))

        return tool_use

    def _validate_tool_result(
        self, coords: List[int], prompt: str, output: str, status: Status
    ) -> Tuple[str, Status]:
        """Pure ft_elementwise step: validate raw tool output.

        Downgrades status for invalid tools or empty/error results.
        Passes through confidence unchanged.
        """
        if status.is_confidence:
            return (output, status)

        if output.startswith("[invalid tool"):
            return (output, Status.self_confidence_but_failed(0.1))

        # Extract result from trace format "[tool(...)]\nresult"
        if "\n" in output and output.startswith("["):
            _, result = output.split("\n", 1)
            tool_validator = self.validators.get("validate_tool_result")
            if tool_validator is not None:
                ok, msg = tool_validator.validate(result)
                if not ok:
                    return (
                        f"{output}\nERROR: {msg}\n",
                        Status.self_confidence_but_failed(0.2),
                    )

        return (output, status)

    def _check_context_sufficiency(
        self,
        coords: List[int],
        prompt: str,
        output: str,
        status: Status,
        worktree_tensor: torch.Tensor,
    ) -> Tuple[str, Status]:
        """Pure ft_elementwise step: check if accumulated context is sufficient.

        Upgrades to confidence(0.9) when context validator passes.
        Passes through confidence unchanged.
        """
        if status.is_confidence:
            return (output, status)

        batch_idx = coords[0]
        collect_idx = coords[1] if len(coords) > 1 else 0
        worktree_path = _read_element(worktree_tensor, batch_idx)
        task_json_path = os.path.join(worktree_path, ".cloze_task.json")
        with open(task_json_path) as f:
            task = json.load(f)

        context_validator = self._context_validator(worktree_path, task)
        ok, cv_msg = context_validator.validate(prompt + "\n" + output)
        print(f"[DEBUG b{batch_idx} c{collect_idx}] context_validator: ok={ok} msg={cv_msg}")
        if ok:
            print(f"[DEBUG b{batch_idx} c{collect_idx}] context sufficient -> confidence")
            return (output, Status.confidence(0.9))

        return (output, Status.self_confidence_but_failed(0.5))

    def _context_validator(self, worktree_path: str, task: dict) -> HarnessValidatorOp:
        """Build a context validator that checks if target file was READ with mask region."""
        target_file = task["target_file"]
        mask_line = task.get("mask_line", 0)
        target_path = os.path.join(worktree_path, target_file)
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                target_lines = [line.rstrip() for line in f.readlines()]
        except (OSError, UnicodeDecodeError):
            target_lines = []

        class ContextValidator(HarnessValidatorOp):
            name = "context_validator"
            description = "Check if accumulated context contains read result for target file around mask and at least one other file."

            def validate(self, content: str) -> tuple:
                # Must have a read() result for the target file
                if "read(" not in content or target_file not in content:
                    return (False, f"target file {target_file} not yet read")

                # Must contain lines around the mask (mask_line +/- 3)
                start = max(0, mask_line - 3)
                end = min(len(target_lines), mask_line + 4)
                found = 0
                for line in target_lines[start:end]:
                    stripped = line.strip()
                    if len(stripped) > 3 and stripped in content:
                        found += 1
                if found < 2:
                    return (False, f"read result missing mask region lines")

                return (True, "")

            def schema(self) -> str:
                return "context_validator()"

        return ContextValidator()

    def _make_context_repeater(self, context_tensor: torch.Tensor, worktree_tensor: torch.Tensor):
        """Build ft_async_get for [batch, L] FutureTensor that broadcasts gathered context.

        Each element at [batch_idx, gen_step_idx] returns the gathered context for batch_idx,
        prefixed with CLOZE_TASK metadata so ft_legacy_expert's output_prompt can access it.
        """
        async def context_repeater(coords: List[int], prompt: str) -> Tuple[str, Status]:
            batch_idx = coords[0]
            context = _read_element(context_tensor, batch_idx)
            worktree_path = _read_element(worktree_tensor, batch_idx)
            task_json_path = os.path.join(worktree_path, ".cloze_task.json")
            with open(task_json_path) as f:
                task = json.load(f)
            mask_start = task.get("mask_start_1idx")
            mask_end = task.get("mask_end_1idx")
            mask_line = task.get("mask_line", "?")
            if isinstance(mask_start, int) and isinstance(mask_end, int):
                num_lines = mask_end - mask_start + 1
                task_header = (
                    f"CLOZE_TASK: target_file={task['target_file']}, "
                    f"mask_line={mask_line}, mask_lines={num_lines} ({mask_start}-{mask_end})\n"
                )
            else:
                task_header = f"CLOZE_TASK: target_file={task['target_file']}, mask_line={mask_line}\n"
            return (task_header + context, Status.confidence(0.9))

        return context_repeater

    def _make_code_gen_output_prompt(self):
        """Return an output_prompt callable for ft_legacy_expert's code generation stage.

        The workspace passed to output_prompt contains:
          const_input_view/  — gathered context + CLOZE_TASK metadata
          const_experiance_view/ — top-k experience (query/key/value files)
          mutable_output_dir/ — where the LLM writes the generated code (initially TODO)
        """
        def output_prompt(task_prompt, workspace_dir, exp_view_dir, input_view_dir, output_dir):
            return (
                _SYSTEM_GENERATION.strip() + "\n\n"
                "The gathered context (including the CLOZE_TASK metadata line and tool call results) "
                "is in const_input_view/. The mask is marked with "
                "<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>. "
                "Similar past code recoveries are in const_experiance_view/ as examples "
                "(the value files contain past tool traces and recovery results).\n\n"
                "Write ONLY the exact original lines of code that were removed, "
                "matching the line count from CLOZE_TASK. "
                "Preserve original indentation exactly."
            )
        return output_prompt
