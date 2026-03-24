import os
import subprocess
import tempfile
import itertools
import torch
from typing import Callable, List, Optional

from symbolic_tensor.function import symbolic_grad_registry


def _scalar_slice_indices(shape: torch.Size) -> List[List[int]]:
    """Generate all coordinate tuples for iterating over each scalar element."""
    ranges = [range(s) for s in shape]
    return [list(coord) for coord in itertools.product(*ranges)]


def _get_storage_elem_relative_paths(tensor: torch.Tensor) -> List[str]:
    """Get all storage element relative paths (e.g., '0/data', '1/2/data') for a tensor."""
    paths = []
    coords_list = _scalar_slice_indices(tensor.size())
    for coords in coords_list:
        flat_index = sum(c * s for c, s in zip(coords, tensor.stride()))
        digits = list(str(flat_index))
        paths.append(os.path.join("storage", os.path.join(*digits), "data"))
    return paths


def _reset_grad_text_to_todo(param: torch.Tensor) -> None:
    """Reset all text storage files of param.grad to 'TODO'."""
    grad = param.grad
    if grad is None or not hasattr(grad, "st_tensor_uid"):
        return
    root = os.path.join(grad.st_relative_to, grad.st_tensor_uid)
    for rel_path in _get_storage_elem_relative_paths(grad):
        storage_path = os.path.join(root, rel_path)
        real_path = os.path.realpath(storage_path)
        if os.path.isfile(real_path):
            with open(real_path, "w", encoding="utf-8") as f:
                f.write("TODO")


class SymbolicSGD(torch.optim.Optimizer):
    """
    Symbolic SGD optimizer. Two-channel update:
      a) Numeric (coefficient): param.data = (1 - lr) * param.data + lr * grad.data
      b) Symbolic (text): apply unified diff patches from grad storage to param storage

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 0.01).
    """

    def __init__(self, params, lr: float = 0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step.

        For each parameter with gradients:
          1. Numeric: param.data = (1 - lr) * param.data + lr * grad.data
          2. Symbolic: apply patch -i grad_file param_file for each storage element

        Args:
            closure: Optional closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # Restore symbolic attributes stripped by autograd
                if hasattr(param, "st_tensor_uid"):
                    symbolic_grad = symbolic_grad_registry.pop(param.st_tensor_uid)
                    if symbolic_grad is not None:
                        grad = symbolic_grad
                        param.grad = grad

                # ── Numeric channel ──
                # param.data = (1 - lr) * param.data + lr * grad.data
                param.data.mul_(1.0 - lr).add_(grad.data, alpha=lr)

                # ── Symbolic channel ──
                # Apply unified diff patches from grad storage to param storage
                if not hasattr(grad, "st_tensor_uid"):
                    continue

                param_storage_root = os.path.join(param.st_relative_to, param.st_tensor_uid)
                grad_storage_root = os.path.join(grad.st_relative_to, grad.st_tensor_uid)

                for rel_path in _get_storage_elem_relative_paths(param):
                    param_file = os.path.realpath(os.path.join(param_storage_root, rel_path))
                    grad_file = os.path.realpath(os.path.join(grad_storage_root, rel_path))

                    if not os.path.isfile(grad_file):
                        continue

                    # Skip if grad is TODO or empty
                    with open(grad_file, "r", encoding="utf-8") as f:
                        grad_content = f.read().strip()
                    if not grad_content or grad_content == "TODO":
                        continue

                    # Ensure param file ends with newline (patch requires it)
                    with open(param_file, "r", encoding="utf-8") as f:
                        param_content = f.read()
                    if not param_content.endswith("\n"):
                        with open(param_file, "w", encoding="utf-8") as f:
                            f.write(param_content + "\n")

                    # Write normalized diff to temp file (ensure trailing newline)
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding="utf-8") as pf:
                        pf.write(grad_content if grad_content.endswith("\n") else grad_content + "\n")
                        patch_path = pf.name

                    try:
                        result = subprocess.run(
                            ["patch", "--no-backup-if-mismatch", "-i", patch_path, param_file],
                            capture_output=True, text=True,
                        )
                        if result.returncode != 0:
                            print(f"patch failed for {param_file}: {result.stderr.strip()}")
                    finally:
                        os.unlink(patch_path)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset gradients. If set_to_none=False, also resets grad text storage to 'TODO'."""
        if not set_to_none:
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        _reset_grad_text_to_todo(param)
        super().zero_grad(set_to_none=set_to_none)


if __name__ == "__main__":
    import tempfile
    from symbolic_tensor.tensor_util.make_tensor import make_tensor
    from symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like

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
        opt = SymbolicSGD([exp], lr=0.1)
        run_test("param_groups has 1 group", len(opt.param_groups) == 1)
        run_test("lr is 0.1", opt.param_groups[0]["lr"] == 0.1)

    # Test 2: Numeric channel (coefficient update)
    print("Test 2: Numeric channel")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["q", "k", "v"]], tmpdir)
        exp.requires_grad_(True)
        opt = SymbolicSGD([exp], lr=0.5)

        # Manually set a plain gradient (no st_ attrs)
        exp.grad = torch.ones_like(exp) * 2.0
        orig_data = exp.data.clone()
        opt.step()
        # param.data = (1 - 0.5) * orig + 0.5 * 2.0 = 0.5 * orig + 1.0
        expected = 0.5 * orig_data + 1.0
        run_test("Coefficient updated", torch.allclose(exp.data, expected),
                 expected.tolist(), exp.data.tolist())

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
        grad = make_tensor([["grad_q", "grad_k", "grad_v"]], tmpdir)
        grad.data.fill_(1.0)
        exp.grad = grad
        run_test("grad text before reset", read_storage(exp.grad, 0) == "grad_q")
        opt = SymbolicSGD([exp], lr=0.1)
        opt.zero_grad(set_to_none=False)
        run_test("grad coeff zeroed", exp.grad.data[0, 0].item() == 0.0)
        run_test("grad text is TODO", read_storage(exp.grad, 0) == "TODO")

    # Test 5: Symbolic channel (patch application)
    print("Test 5: Patch application")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create param with known content
        exp = make_tensor([["hello world"]], tmpdir)
        exp.requires_grad_(True)
        param_text_before = read_storage(exp, 0)
        run_test("param before", param_text_before == "hello world")

        # Create grad with a unified diff that changes "hello" to "goodbye"
        diff_content = (
            "--- data\n"
            "+++ data\n"
            "@@ -1 +1 @@\n"
            "-hello world\n"
            "+goodbye world\n"
        )
        grad = make_tensor([[diff_content]], tmpdir)
        grad.data.fill_(1.0)
        exp.grad = grad

        opt = SymbolicSGD([exp], lr=0.5)
        opt.step()

        param_text_after = read_storage(exp, 0)
        run_test("param text patched", param_text_after.strip() == "goodbye world",
                 "goodbye world", repr(param_text_after))

    # Test 6: Skip TODO and empty grads
    print("Test 6: Skip TODO grads")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["original text"]], tmpdir)
        exp.requires_grad_(True)

        grad = make_tensor([["TODO"]], tmpdir)
        grad.data.fill_(1.0)
        exp.grad = grad

        opt = SymbolicSGD([exp], lr=0.5)
        opt.step()

        param_text = read_storage(exp, 0)
        run_test("param unchanged with TODO grad", param_text == "original text")

    print("\nAll tests completed.")
