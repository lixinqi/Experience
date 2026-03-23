import os
import shutil
import tempfile
import itertools
import torch
from typing import Callable, List, Optional

from symbolic_tensor.tensor_util.slice_view import slice_view
from symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from symbolic_tensor.tensor_util.dump_view import dump_view
from symbolic_tensor.llm_client.agent_task import AgentTask
from symbolic_tensor.llm_client.task_handler import TaskHandler


def _scalar_slice_indices(shape: torch.Size) -> List[List[int]]:
    """Generate all coordinate tuples for iterating over each scalar element."""
    ranges = [range(s) for s in shape]
    return [list(coord) for coord in itertools.product(*ranges)]


def _build_nested_result(flat_results: list, shape: List[int]):
    """Reshape a flat list of results into a nested list matching the given shape."""
    if not shape:
        return flat_results[0]
    if len(shape) == 1:
        return flat_results
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    return [
        _build_nested_result(flat_results[i * chunk_size:(i + 1) * chunk_size], shape[1:])
        for i in range(shape[0])
    ]


def _copy_back_to_storage_view(mutable_dir: str, view_tensor: torch.Tensor) -> None:
    """Copy LLM results from mutable workspace dir back through view tensor's symlinks."""
    coords_list = [list(coord) for coord in itertools.product(*[range(s) for s in view_tensor.size()])]
    for coords in coords_list:
        flat_index = sum(c * s for c, s in zip(coords, view_tensor.stride()))
        digits = list(str(flat_index))
        view_storage_path = os.path.join(
            view_tensor.st_relative_to,
            view_tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        real_storage_path = os.path.realpath(view_storage_path)
        if coords:
            coord_dirs = os.path.join(*[str(c) for c in coords])
            mutable_file = os.path.join(mutable_dir, coord_dirs, "data.txt")
        else:
            mutable_file = os.path.join(mutable_dir, "data.txt")
        if os.path.isfile(mutable_file):
            with open(mutable_file, "r", encoding="utf-8") as f:
                content = f.read()
            with open(real_storage_path, "w", encoding="utf-8") as f:
                f.write(content)


def _reset_grad_text_to_todo(param: torch.Tensor) -> None:
    """Reset all text storage files of param.grad to 'TODO'."""
    grad = param.grad
    if grad is None or not hasattr(grad, "st_tensor_uid"):
        return
    coords_list = [list(coord) for coord in itertools.product(*[range(s) for s in grad.size()])]
    for coords in coords_list:
        flat_index = sum(c * s for c, s in zip(coords, grad.stride()))
        digits = list(str(flat_index))
        storage_path = os.path.join(
            grad.st_relative_to,
            grad.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        real_path = os.path.realpath(storage_path)
        if os.path.isfile(real_path):
            with open(real_path, "w", encoding="utf-8") as f:
                f.write("TODO")


class SymbolicSGD(torch.optim.Optimizer):
    """
    Symbolic SGD optimizer. Two-channel update:
      a) Numeric (coefficient): standard SGD param.data -= lr * grad.data
      b) Symbolic (text): LLM applies text diffs from grad to param storage

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 0.01).
        step_prompt: Optional context prompt for the LLM during step.
    """

    def __init__(self, params, lr: float = 0.01, step_prompt: str = "", llm_method: str = "raw_llm_api"):
        defaults = dict(lr=lr, step_prompt=step_prompt, llm_method=llm_method)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step.

        For each parameter with gradients:
          1. Numeric: param.data -= lr * grad.data
          2. Symbolic: per scalar element, LLM applies text diff to param text

        Args:
            closure: Optional closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            step_prompt = group["step_prompt"]
            llm_method = group["llm_method"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # ── Numeric channel ──
                # standard SGD: param.data -= lr * grad.data
                param.data.add_(grad.data, alpha=-lr)

                # ── Symbolic channel ──
                # Skip if grad doesn't have symbolic storage
                if not hasattr(grad, "st_tensor_uid"):
                    continue

                coords_list = _scalar_slice_indices(param.size())
                flat_tasks: List[AgentTask] = []
                flat_copyback_info = []

                for coords in coords_list:
                    int_slices = [c for c in coords]

                    # View (symlink) for copy-back, value (copy) for LLM to modify
                    param_view = slice_view(param, int_slices)
                    param_value = slice_tensor(param, int_slices)
                    grad_view = slice_view(grad, int_slices)

                    # Get coefficient for signal strength
                    grad_coeff = grad[tuple(coords)].item()

                    workspace_dir = tempfile.mkdtemp()
                    grad_view_dir = os.path.join(workspace_dir, "const_grad_view")
                    param_dir = os.path.join(workspace_dir, "mutable_param_dir")

                    # Dump const grad view (what should change)
                    dump_view(grad_view, grad_view_dir, "txt")
                    # Dump mutable param copy (LLM modifies in-place)
                    dump_view(param_value, param_dir, "txt")

                    prompt = (
                        "You are a parameter updater for a symbolic optimizer.\n\n"
                        f"{step_prompt}\n\n"
                        "The gradient describes how the parameter should change.\n"
                        f"The update strength (learning_rate * |gradient_coefficient|) is {lr * abs(grad_coeff):.4f}.\n"
                        "Higher values mean the gradient signal is stronger — apply changes more confidently.\n"
                        "Lower values mean subtle adjustments.\n\n"
                        f"Read the gradient (how to change) at \"{grad_view_dir}\".\n"
                        f"Apply the suggested changes to the current parameter at \"{param_dir}\".\n"
                        "Modify the parameter text in-place. Do not add commentary, just update the text.\n\n"
                        "If the gradient says \"TODO\" or \"No change needed\", leave the parameter unchanged.\n"
                    )

                    flat_tasks.append(AgentTask(
                        workspace_dir=workspace_dir,
                        output_relative_dir="mutable_param_dir",
                        prompt=prompt,
                    ))
                    flat_copyback_info.append((param_dir, param_view))

                # Build nested task list matching param shape and dispatch
                all_tasks = _build_nested_result(flat_tasks, list(param.size()))
                TaskHandler()(all_tasks, llm_method)

                # Copy back results and cleanup
                for param_dir, param_view in flat_copyback_info:
                    _copy_back_to_storage_view(param_dir, param_view)

                for task in flat_tasks:
                    shutil.rmtree(task.workspace_dir, ignore_errors=True)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset gradients. If set_to_none=False, also resets grad text storage to 'TODO'."""
        if not set_to_none:
            # Reset text storage before super().zero_grad zeros the coefficients
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        _reset_grad_text_to_todo(param)
        super().zero_grad(set_to_none=set_to_none)


if __name__ == "__main__":
    import subprocess
    from symbolic_tensor.tensor_util.make_tensor import make_tensor
    from symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
    from symbolic_tensor.function.symbolic_transform_forward import symbolic_transform_forward
    from symbolic_tensor.function.symbolic_transform_backward import symbolic_transform_backward

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

    print("Running SymbolicSGD tests...\n")

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
            tensor.st_relative_to,
            tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        with open(path) as f:
            return f.read()

    # Test 1: Constructor
    print("Test 1: Constructor")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["q", "k", "v"]], tmpdir)
        exp.requires_grad_(True)
        opt = SymbolicSGD([exp], lr=0.1, step_prompt="test")
        run_test("param_groups has 1 group", len(opt.param_groups) == 1)
        run_test("lr is 0.1", opt.param_groups[0]["lr"] == 0.1)
        run_test("step_prompt is 'test'", opt.param_groups[0]["step_prompt"] == "test")

    # Test 2: Numeric channel only (no symbolic storage on grad)
    print("Test 2: Numeric channel (coefficient update)")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["q", "k", "v"]], tmpdir)
        exp.requires_grad_(True)
        opt = SymbolicSGD([exp], lr=0.5)

        # Manually set a plain gradient (no st_ attrs)
        exp.grad = torch.ones_like(exp) * 2.0
        orig_data = exp.data.clone()
        opt.step()
        # param.data -= lr * grad.data => orig - 0.5 * 2.0 = orig - 1.0
        expected = orig_data - 1.0
        run_test("Coefficient updated", torch.allclose(exp.data, expected))

    # Test 3: zero_grad with set_to_none=True
    print("Test 3: zero_grad(set_to_none=True)")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["q", "k", "v"]], tmpdir)
        exp.requires_grad_(True)
        exp.grad = torch.ones_like(exp)
        opt = SymbolicSGD([exp], lr=0.1)
        opt.zero_grad(set_to_none=True)
        run_test("grad is None", exp.grad is None)

    # Test 4: zero_grad with set_to_none=False resets text
    print("Test 4: zero_grad(set_to_none=False) resets text")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["q", "k", "v"]], tmpdir)
        exp.requires_grad_(True)
        # Create a symbolic grad with non-TODO text
        grad = make_tensor([["grad_q", "grad_k", "grad_v"]], tmpdir)
        grad.data.fill_(1.0)
        exp.grad = grad
        run_test("grad text before reset", read_storage(exp.grad, 0) == "grad_q")
        opt = SymbolicSGD([exp], lr=0.1)
        opt.zero_grad(set_to_none=False)
        run_test("grad coeff zeroed", exp.grad.data[0, 0].item() == 0.0)
        run_test("grad text is TODO", read_storage(exp.grad, 0) == "TODO")

    # Test 5: Full forward -> backward -> step
    for method in ["coding_agent", "raw_llm_api"]:
        print(f"Test 5: Full training step (llm_method={method})")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_tensor = make_tensor(["Hello world in English"], tmpdir)

            experience_data = [
                ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
                ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
            ]
            experience_tensor = make_tensor(experience_data, tmpdir)
            experience_tensor.requires_grad_(True)

            optimizer = SymbolicSGD([experience_tensor], lr=1.0,
                step_prompt="You are updating translation experience entries (query keywords, key text, value text).",
                llm_method=method)

            # Read experience before
            exp_val_before = read_storage(experience_tensor, 2)  # value of first entry
            print(f"  experience[0].value before: {repr(exp_val_before[:80])}")

            # Forward
            output, selected_indexes = symbolic_transform_forward(
                input_tensor, experience_tensor,
                forward_prompt="Translate the English text to French.",
                topk=2,
                llm_method=method,
            )

            # Backward with a quality signal
            grad_output = make_tensor(
                ["The translation should use formal French: 'Bonjour le monde' -> 'Bonjour au monde'"],
                tmpdir,
            )
            grad_output.data.fill_(1.0)

            grad_input, grad_experience = symbolic_transform_backward(
                grad_output, input_tensor, output, experience_tensor,
                selected_experience_qkv_indexes_list=selected_indexes,
                forward_prompt="Translate the English text to French.",
                topk=2,
                llm_method=method,
            )

            # Assign grad to experience tensor
            experience_tensor.grad = grad_experience

            # Check grad text before step
            grad_val_text = read_storage(grad_experience, 2)
            print(f"  grad_experience[0].value: {repr(grad_val_text[:80])}")

            # Step
            optimizer.step()

            # Check experience after
            exp_val_after = read_storage(experience_tensor, 2)
            print(f"  experience[0].value after: {repr(exp_val_after[:80])}")
            run_test("experience text updated", exp_val_after != exp_val_before)
            run_test("experience coeff updated", experience_tensor.data[0, 0].item() != 1.0)

    print("\nAll tests completed.")
