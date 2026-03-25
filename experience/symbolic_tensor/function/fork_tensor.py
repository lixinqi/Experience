import os
import itertools
import shutil
import tempfile
import torch
from typing import Any, List, Tuple, Union

from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.tensor_util.dump_view import dump_view
from experience.llm_client.agent_task import AgentTask
from experience.llm_client.task_handler import TaskHandler


def _scalar_slice_indices(shape: torch.Size) -> List[List[int]]:
    ranges = [range(s) for s in shape]
    return [list(coord) for coord in itertools.product(*ranges)]


def fork_tensor_forward(
    input: torch.Tensor,
    num_outputs: int = 2,
) -> List[torch.Tensor]:
    """Replicate input into num_outputs identical views (symlinked).

    Each output is a slice_view with a full-slice on every dimension,
    so all outputs share the same underlying storage.

    Args:
        input: A symbolic tensor to replicate.
        num_outputs: Number of output copies.

    Returns:
        A list of num_outputs symbolic tensor views, all pointing to input's storage.
    """
    slices = [slice(None, None, None)] * len(input.size())
    return [slice_view(input, slices) for _ in range(num_outputs)]


def fork_tensor_backward(
    grad_outputs: List[torch.Tensor],
    input: torch.Tensor,
    outputs: List[torch.Tensor],
    forward_prompt: str = "",
    llm_method: str = "raw_llm_api",
) -> Union[torch.Tensor, None]:
    """Backward pass of fork_tensor: merge multiple grad_outputs into one grad_input.

    During forward, input was replicated to multiple identical outputs.
    Now given multiple output gradients, the LLM merges them into a single
    input gradient.

    Args:
        grad_outputs: List of gradient tensors (one per forked output).
        input: Original input tensor (saved from forward ctx).
        outputs: Original output tensors (saved from forward ctx).
        forward_prompt: Optional prompt guidance.
        llm_method: LLM backend to use.

    Returns:
        grad_input, or None if input doesn't require grad.
    """
    if not input.requires_grad:
        return None

    grad_input = todo_tensor_like(input)

    # ── Numeric channel ──
    # Sum numeric gradients from all outputs
    grad_input.data.zero_()
    for grad_out in grad_outputs:
        grad_input.data.add_(grad_out.data)

    # ── Symbolic channel ──
    input_shape = list(input.size())
    coords_list = _scalar_slice_indices(input.size())

    all_tasks = []
    task_contexts = []

    for coords in coords_list:
        int_slices = [c for c in coords]

        scalar_grad_outputs = [slice_view(g, int_slices) for g in grad_outputs]
        scalar_outputs = [slice_view(o, int_slices) for o in outputs]
        scalar_input = slice_view(input, int_slices)
        scalar_grad_input_view = slice_view(grad_input, int_slices)
        scalar_grad_input_value = slice_tensor(grad_input, int_slices)

        workspace_dir = tempfile.mkdtemp()

        # Dump each grad_output and output as numbered subdirs
        for i, sg in enumerate(scalar_grad_outputs):
            dump_view(sg, os.path.join(workspace_dir, f"const_grad_outputs_view/{i}"), "txt")
        dump_view(scalar_input, os.path.join(workspace_dir, "const_input_view"), "txt")
        for i, so in enumerate(scalar_outputs):
            dump_view(so, os.path.join(workspace_dir, f"const_outputs_view/{i}"), "txt")
        dump_view(scalar_grad_input_value, os.path.join(workspace_dir, "mutable_grad_input_dir"), "txt")

        grad_outputs_dir = os.path.join(workspace_dir, "const_grad_outputs_view")
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        outputs_view_dir = os.path.join(workspace_dir, "const_outputs_view")
        grad_input_dir = os.path.join(workspace_dir, "mutable_grad_input_dir")

        prompt = (
            "You are a symbolic gradient collector for backward pass.\n\n"
            f"{forward_prompt}\n\n"
            "During forward pass, the input was replicated to multi identical outputs.\n"
            "Now given the output gradients (how output should change), merge gradient for\n"
            "input.\n\n"
            "Context (read-only):\n"
            f"- Output gradients (text diff): \"{grad_outputs_dir}\"\n"
            f"- Original input: \"{input_view_dir}\"\n"
            f"- Original outputs: \"{outputs_view_dir}\"\n\n"
            "Compute and write:\n"
            f"1. Input gradient in \"{grad_input_dir}\":\n"
            "   How should the input text change to improve the output?\n"
            f"2. File \"{grad_input_dir}/<xxx>/data\" must be a better version of \"{input_view_dir}/<xxx>/data\"\n\n"
            "Replace all TODO with source semantic files.\n"
        )

        agent_task = AgentTask(
            workspace_dir=workspace_dir,
            output_relative_dir=["mutable_grad_input_dir"],
            prompt=prompt,
        )
        all_tasks.append(agent_task)
        task_contexts.append((workspace_dir, scalar_grad_input_view, scalar_input, scalar_grad_input_value))

    # Batch LLM call
    if all_tasks:
        TaskHandler()(all_tasks, llm_method)

    # Compute diff between original input and LLM-written improved version,
    # assign diff back to grad_input view
    for workspace_dir, scalar_grad_input_view, scalar_input, scalar_grad_input_value in task_contexts:
        diff = get_diff_tensor(scalar_input, scalar_grad_input_value)
        assign_tensor(scalar_grad_input_view, diff)
        shutil.rmtree(workspace_dir)

    return grad_input


if __name__ == "__main__":
    import subprocess
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    # Source anthropic env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    print("Running fork_tensor tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_storage(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to, tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )
        with open(path) as f:
            return f.read()

    # Test 1: fork_tensor_forward basics
    print("Test 1: fork_tensor_forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["hello", "world"], tmpdir)
        forks = fork_tensor_forward(t, num_outputs=3)
        run_test("Returns 3 tensors", len(forks) == 3, 3, len(forks))
        run_test("Each shape matches input", all(list(f.shape) == [2] for f in forks))
        # All forks should read the same content
        for i, f in enumerate(forks):
            content = read_storage(f, 0)
            run_test(f"Fork {i} elem 0 == 'hello'", content == "hello", "hello", content)
        # Forks are views (symlinks)
        fork_path = os.path.join(tmpdir, forks[0].st_tensor_uid, "storage", "0", "data")
        run_test("Fork is symlink", os.path.islink(fork_path))

    # Test 2: fork_tensor_forward with 2D tensor
    print("Test 2: fork_tensor_forward 2D")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor([["a", "b"], ["c", "d"]], tmpdir)
        forks = fork_tensor_forward(t, num_outputs=2)
        run_test("Returns 2 tensors", len(forks) == 2)
        run_test("Shape [2, 2]", list(forks[0].shape) == [2, 2], [2, 2], list(forks[0].shape))
        run_test("Elem (0,0) == 'a'", read_storage(forks[0], 0) == "a")
        run_test("Elem (1,1) == 'd'", read_storage(forks[0], 3) == "d")

    # Test 3: fork_tensor_forward default num_outputs=2
    print("Test 3: fork_tensor_forward default")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["x"], tmpdir)
        forks = fork_tensor_forward(t)
        run_test("Default 2 outputs", len(forks) == 2)

    # Test 4: fork_tensor_backward — numeric channel sums gradients
    print("Test 4: Numeric channel (sum)")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["text_a", "text_b"], tmpdir)
        t.requires_grad_(True)
        forks = fork_tensor_forward(t, num_outputs=3)

        grad_outs = []
        for i in range(3):
            g = make_tensor(["grad_a", "grad_b"], tmpdir)
            g.data.fill_(float(i + 1))
            grad_outs.append(g)

        grad_input = fork_tensor_backward(
            grad_outs, t, forks,
            forward_prompt="test",
            llm_method="raw_llm_api",
        )

        run_test("grad_input shape matches", list(grad_input.shape) == [2])
        expected_sum = 1.0 + 2.0 + 3.0
        run_test("Numeric sum correct", grad_input.data[0].item() == expected_sum,
                 expected_sum, grad_input.data[0].item())

    # Test 5: fork_tensor_backward returns None when no grad needed
    print("Test 5: No grad needed")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["text"], tmpdir)
        # requires_grad is False by default
        forks = fork_tensor_forward(t, num_outputs=2)
        grad_outs = [make_tensor(["g1"], tmpdir), make_tensor(["g2"], tmpdir)]
        result = fork_tensor_backward(grad_outs, t, forks)
        run_test("Returns None", result is None)

    print("\nAll tests completed.")
