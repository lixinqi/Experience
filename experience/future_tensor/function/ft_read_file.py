"""
FtReadFile := torch.autograd.Function[
    $forward  Import[{future_tensor function read_file_forward.viba}],
    $backward Import[{future_tensor function read_file_backward.viba}]
]

ft_read_file = FtReadFile.apply

Read file contents from the local filesystem, wrapped as a FutureTensor
autograd Function. Each input element contains a file path; forward reads
that file and returns its content. This is an "observe the world" op —
backward is a pass-through (no meaningful reverse for file reading),
but 2nd-derivative dispatch is supported.
"""

import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.read_file_forward import read_file_forward
from experience.future_tensor.function.read_file_backward import read_file_backward


class FtReadFile(torch.autograd.Function):
    """Autograd Function for reading file contents per FutureTensor element."""

    @staticmethod
    def forward(ctx, input_ft: FutureTensor):
        ctx.input_ft = input_ft
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_initial_static_tensor.st_relative_to
        return read_file_forward(input_ft)

    @staticmethod
    def backward(ctx, grad_output):
        return read_file_backward(ctx, grad_output)


def ft_read_file(input_ft: FutureTensor) -> FutureTensor:
    """Read file contents for each element of ``input_ft``.

    Each element's symbolic content is used as the file path to read.
    The file's text content is stored as the output element.

    Args:
        input_ft: FutureTensor whose elements contain file paths.

    Returns:
        A FutureTensor with the same shape as ``input_ft``, where each element
        contains the text content of the file at the corresponding path.
    """
    return FtReadFile.apply(input_ft)


if __name__ == "__main__":
    import os
    import sys
    import tempfile

    import sympy

    from experience.future_tensor.status import Status
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running tests for ft_read_file...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
            print(f"  \u2713 {name}")
        else:
            failed += 1
            print(f"  \u2717 {name}")
            if expected is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def make_forwarded_ft(shape, data_list, tmpdir):
        async def dummy_get(coords, trajactory):
            return ("unused", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(s) for s in shape])
        nested = _unflatten_data(data_list, shape)
        result_tensor = st_make_tensor(nested, tmpdir)
        assign_tensor(ft.ft_initial_static_tensor, result_tensor)
        ft.ft_forwarded = True
        return ft

    def _unflatten_data(flat_list, shape):
        if not shape:
            return flat_list[0] if flat_list else None
        if len(shape) == 1:
            return flat_list
        chunk_size = 1
        for s in shape[1:]:
            chunk_size *= s
        return [
            _unflatten_data(flat_list[i * chunk_size : (i + 1) * chunk_size], shape[1:])
            for i in range(shape[0])
        ]

    def _storage_path(ft, flat_index):
        digits = list(str(flat_index))
        return os.path.join(
            ft.ft_initial_static_tensor.st_relative_to, ft.ft_initial_static_tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )

    def read_ft_element(ft, flat_index):
        path = _storage_path(ft, flat_index)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # === Group 1: Forward shape and laziness ===
    print("Group 1: Forward shape and laziness")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3], ["/tmp/a", "/tmp/b", "/tmp/c"], tmpdir)
        output = ft_read_file(ft)
        run_test("output shape matches input", output.ft_capacity_shape == [3])
        run_test("output is lazy", output.ft_forwarded is False)

    # === Group 2: Read real files ===
    print("\nGroup 2: Read real files")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file_a = os.path.join(tmpdir, "test_a.txt")
        test_file_b = os.path.join(tmpdir, "test_b.txt")
        with open(test_file_a, "w") as f:
            f.write("content_of_file_a\nline2\n")
        with open(test_file_b, "w") as f:
            f.write("content_of_file_b\n")

        ft = make_forwarded_ft([2], [test_file_a, test_file_b], tmpdir)
        output = ft_read_file(ft)

        prompt_t = st_make_tensor(["p", "p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("output forwarded", output.ft_forwarded is True)
        c0 = read_ft_element(output, 0)
        c1 = read_ft_element(output, 1)
        run_test("elem 0 contains file_a content",
                 c0 is not None and "content_of_file_a" in c0,
                 "contains 'content_of_file_a'", repr(c0[:80] if c0 else None))
        run_test("elem 1 contains file_b content",
                 c1 is not None and "content_of_file_b" in c1,
                 "contains 'content_of_file_b'", repr(c1[:80] if c1 else None))
        run_test("confidence > 0 for elem 0",
                 output.ft_initial_static_tensor.data.flatten()[0].item() > 0)
        run_test("confidence > 0 for elem 1",
                 output.ft_initial_static_tensor.data.flatten()[1].item() > 0)

    # === Group 3: Non-existent file ===
    print("\nGroup 3: Non-existent file")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1], ["/tmp/nonexistent_file_xyz_123.txt"], tmpdir)
        output = ft_read_file(ft)

        prompt_t = st_make_tensor(["p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("nonexistent file: forwarded", output.ft_forwarded is True)
        run_test("nonexistent file: status < 0 (failed)",
                 output.ft_initial_static_tensor.data.flatten()[0].item() < 0)

    # === Group 4: Autograd connectivity ===
    print("\nGroup 4: Autograd connectivity")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["/tmp/a", "/tmp/b"], tmpdir)
        ft.requires_grad_(True)
        output = ft_read_file(ft)
        run_test("output has grad_fn", output.grad_fn is not None)
        run_test("output requires_grad", output.requires_grad is True)

    # === Group 5: Backward produces grad ===
    print("\nGroup 5: Backward produces grad")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["/tmp/a", "/tmp/b"], tmpdir)
        ft.requires_grad_(True)
        output = ft_read_file(ft)
        loss = output.sum()
        loss.backward()
        run_test("input has grad", ft.grad is not None)

    # === Group 6: 2nd derivative support ===
    print("\nGroup 6: 2nd derivative support")

    from experience.future_tensor.backward_dispatch import (
        need_reflection,
        dispatch_policy,
        TracePolicy,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["/tmp/a", "/tmp/b"], tmpdir)
        ft.requires_grad_(True)

        backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
        anchored = need_reflection(ft, backward_dispatch_start)
        output = ft_read_file(anchored)
        loss = output.sum()
        loss.backward(create_graph=True)

        run_test("2nd: grad exists", backward_dispatch_start.grad is not None)
        run_test("2nd: grad has grad_fn", backward_dispatch_start.grad.grad_fn is not None)

        records = []
        with dispatch_policy(TracePolicy(records)):
            backward_dispatch_start.grad.backward()

        run_test("2nd: TracePolicy collected records", len(records) >= 1)
        if records:
            run_test("2nd: record fn is read_file_backward",
                     records[0].fn is read_file_backward)
            run_test("2nd: record has grad_output",
                     "grad_output" in records[0].inputs)
            run_test("2nd: record has grad_input",
                     "grad_input" in records[0].inputs)
        else:
            for name in ["fn", "grad_output", "grad_input"]:
                run_test(f"2nd: record has {name}", False)

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All ft_read_file tests passed.")
    else:
        sys.exit(1)
