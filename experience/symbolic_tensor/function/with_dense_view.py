import torch
from typing import Callable, List

from experience.symbolic_tensor.tensor_util.sparse_to_dense import sparse_to_dense
from experience.symbolic_tensor.tensor_util.dense_to_sparse import dense_to_sparse


def with_dense_view(
    dense_handler: Callable[[torch.Tensor], torch.Tensor],
    input: torch.Tensor,
) -> torch.Tensor:
    """Apply a dense_handler on a sparse symbolic tensor via dense view.

    Converts sparse input to dense, applies dense_handler, converts back.

    Args:
        dense_handler: Callable that takes a dense symbolic tensor and returns
            a dense symbolic tensor (same or different shape).
        input: A sparse symbolic tensor (1D with associated indexes/shape).

    Returns:
        Sparse symbolic tensor after dense_handler transformation.
    """
    # Convert sparse -> dense
    sparse_data, indexes, shape = dense_to_sparse(input, view=True)
    dense = sparse_to_dense(sparse_data, indexes, shape)

    # Apply handler on dense view
    result_dense = dense_handler(dense)

    # Convert dense -> sparse
    result_sparse, _, _ = dense_to_sparse(result_dense, view=False)
    return result_sparse


if __name__ == "__main__":
    import os
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.dense_to_sparse import dense_to_sparse as d2s

    print("Running with_dense_view tests...\n")

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

    # Test 1: Identity handler — roundtrip preserves content
    print("Test 1: Identity handler roundtrip")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["hello", "world", "foo"], tmpdir)
        original.data[1] = 0.0  # zero out "world"

        result = with_dense_view(lambda x: x, original)
        run_test("result has 2 elements", result.numel() == 2)
        run_test("result[0] == 'hello'", read_storage(result, 0) == "hello")
        run_test("result[1] == 'foo'", read_storage(result, 1) == "foo")

    # Test 2: Handler that zeros out an element
    print("Test 2: Handler zeros out element")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["a", "b", "c"], tmpdir)

        def zero_middle(dense):
            dense.data[1] = 0.0
            return dense

        result = with_dense_view(zero_middle, original)
        run_test("result has 2 elements", result.numel() == 2)
        run_test("result[0] == 'a'", read_storage(result, 0) == "a")
        run_test("result[1] == 'c'", read_storage(result, 1) == "c")

    # Test 3: 2D tensor through dense view
    print("Test 3: 2D identity roundtrip")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor([["x", "y"], ["z", "w"]], tmpdir)
        original.data[0, 1] = 0.0  # zero out "y"

        result = with_dense_view(lambda x: x, original)
        run_test("result has 3 elements", result.numel() == 3)

    # Test 4: All-nonzero tensor
    print("Test 4: All-nonzero identity")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["p", "q"], tmpdir)

        result = with_dense_view(lambda x: x, original)
        run_test("result has 2 elements", result.numel() == 2)
        run_test("result[0] == 'p'", read_storage(result, 0) == "p")
        run_test("result[1] == 'q'", read_storage(result, 1) == "q")

    # Test 5: Symbolic content preserved through dense view
    print("Test 5: Content preserved through dense view")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["a", "b"], tmpdir)
        original.data[0] = 0.0  # zero out "a", only "b" is nonzero

        result = with_dense_view(lambda x: x, original)
        run_test("result has 1 element", result.numel() == 1)
        run_test("result[0] == 'b'", read_storage(result, 0) == "b")

    print("\nAll tests completed.")
