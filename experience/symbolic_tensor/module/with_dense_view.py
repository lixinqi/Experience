import torch
import torch.nn as nn
from typing import Callable

from experience.symbolic_tensor.function.with_dense_view import with_dense_view


class WithDenseView(nn.Module):
    """Module wrapper for with_dense_view.

    Stores a dense_handler callable and applies it on sparse input
    via dense conversion in forward().

    Args:
        dense_handler: Callable that takes a dense symbolic tensor
            and returns a dense symbolic tensor.
    """

    def __init__(
        self,
        dense_handler: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.dense_handler = dense_handler

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return with_dense_view(self.dense_handler, input)


if __name__ == "__main__":
    import os
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running WithDenseView module tests...\n")

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
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # Test 1: Identity module
    print("Test 1: Identity module")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = WithDenseView(dense_handler=lambda x: x)
        t = make_tensor(["alpha", "beta"], tmpdir)
        result = module(t)
        run_test("result has 2 elements", result.numel() == 2)
        run_test("result[0] == 'alpha'", read_storage(result, 0) == "alpha")
        run_test("result[1] == 'beta'", read_storage(result, 1) == "beta")

    # Test 2: Module with zeroing handler
    print("Test 2: Module with zeroing handler")
    with tempfile.TemporaryDirectory() as tmpdir:
        def zero_first(dense):
            dense.data[0] = 0.0
            return dense

        module = WithDenseView(dense_handler=zero_first)
        t = make_tensor(["x", "y", "z"], tmpdir)
        result = module(t)
        run_test("result has 2 elements", result.numel() == 2)
        run_test("result[0] == 'y'", read_storage(result, 0) == "y")
        run_test("result[1] == 'z'", read_storage(result, 1) == "z")

    # Test 3: Module is nn.Module
    print("Test 3: nn.Module compatibility")
    module = WithDenseView(dense_handler=lambda x: x)
    run_test("is nn.Module", isinstance(module, nn.Module))
    run_test("has forward", hasattr(module, "forward"))

    print("\nAll tests completed.")
