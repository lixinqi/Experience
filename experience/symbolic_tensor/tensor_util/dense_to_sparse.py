import os
import torch
from pathlib import Path
from typing import List, Tuple

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


def _get_storage_path(tensor: torch.Tensor, flat_index: int) -> Path:
    """Get the storage file Path for a given flat index."""
    digits = list(str(flat_index))
    return Path(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )


def dense_to_sparse(
    input: torch.Tensor,
    view: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    """Convert a dense symbolic tensor to sparse representation.

    Finds nonzero elements (elements with content on disk) and extracts
    them into a compact 1D tensor.

    Args:
        input: A dense symbolic tensor.
        view: If True, output shares storage with input via symlinks.
            If False, output has independent copies.

    Returns:
        A tuple of:
        - output: 1D symbolic tensor containing only the nonzero elements.
        - indexes: list[torch.Tensor[int]], one per dimension of input,
            containing the coordinates of the nonzero elements.
        - shape: Original dense shape as list[int].
    """
    shape = list(input.shape)
    indexes = list(torch.nonzero(input.data, as_tuple=True))

    n = indexes[0].numel()
    if n == 0:
        from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor
        output = make_none_tensor([0], input.st_relative_to)
        return output, indexes, shape

    # Compute flat indices for zip-style coordinate tuples
    stride = input.stride()
    flat_indices = sum(
        idx * s for idx, s in zip(indexes, stride)
    )  # element-wise: tensor of flat indices

    # Collect storage paths for each nonzero element
    paths = [_get_storage_path(input, int(fi)) for fi in flat_indices]
    output = make_tensor(paths, input.st_relative_to, symlink=view)

    return output, indexes, shape


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor

    print("Running dense_to_sparse tests...\n")

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

    # Test 1: All nonzero 1D
    print("Test 1: All nonzero 1D")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["a", "b", "c"], tmpdir)
        output, indexes, shape = dense_to_sparse(t)
        run_test("shape preserved", shape == [3])
        run_test("3 indexes", indexes[0].numel() == 3)
        run_test("output has 3 elements", output.numel() == 3)
        run_test("output[0] == 'a'", read_storage(output, 0) == "a")
        run_test("output[2] == 'c'", read_storage(output, 2) == "c")

    # Test 2: Mixed zero/nonzero 1D
    print("Test 2: Mixed zero/nonzero 1D")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["hello", "world"], tmpdir)
        # Zero out element 0 to simulate empty
        t.data[0] = 0.0
        output, indexes, shape = dense_to_sparse(t)
        run_test("shape preserved", shape == [2])
        run_test("1 nonzero element", indexes[0].numel() == 1)
        run_test("index is [1]", indexes[0].tolist() == [1])
        run_test("output[0] == 'world'", read_storage(output, 0) == "world")

    # Test 3: 2D tensor
    print("Test 3: 2D tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor([["a", "b"], ["c", "d"]], tmpdir)
        t.data[0, 1] = 0.0  # zero out (0,1)
        t.data[1, 0] = 0.0  # zero out (1,0)
        output, indexes, shape = dense_to_sparse(t)
        run_test("shape preserved", shape == [2, 2])
        run_test("2 nonzero elements", indexes[0].numel() == 2)
        run_test("dim0 indexes", indexes[0].tolist() == [0, 1])
        run_test("dim1 indexes", indexes[1].tolist() == [0, 1])
        run_test("output[0] == 'a'", read_storage(output, 0) == "a")
        run_test("output[1] == 'd'", read_storage(output, 1) == "d")

    # Test 4: All zeros
    print("Test 4: All zeros (empty tensor)")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_none_tensor([3], tmpdir)
        output, indexes, shape = dense_to_sparse(t)
        run_test("shape preserved", shape == [3])
        run_test("0 nonzero elements", indexes[0].numel() == 0)
        run_test("output is empty", output.numel() == 0)

    # Test 5: view=True shares storage
    print("Test 5: view=True shares storage")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["x", "y"], tmpdir)
        output, indexes, shape = dense_to_sparse(t, view=True)
        run_test("output[0] == 'x'", read_storage(output, 0) == "x")
        # Check that view's storage is symlinked
        path = os.path.join(
            output.st_relative_to, output.st_tensor_uid,
            "storage", "0", "data",
        )
        run_test("output storage is symlink", os.path.islink(path))

    # Test 6: view=False has independent copy
    print("Test 6: view=False has independent copy")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["x", "y"], tmpdir)
        output, indexes, shape = dense_to_sparse(t, view=False)
        run_test("output[0] == 'x'", read_storage(output, 0) == "x")
        path = os.path.join(
            output.st_relative_to, output.st_tensor_uid,
            "storage", "0", "data",
        )
        run_test("output storage is NOT symlink", not os.path.islink(path))

    print("\nAll tests completed.")
