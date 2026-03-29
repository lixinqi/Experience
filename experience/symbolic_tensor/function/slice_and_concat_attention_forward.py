import os
import torch
from typing import List

from experience.symbolic_tensor.tensor_util.none_tensor_like import none_tensor_like
from experience.symbolic_tensor.tensor_util.slice_view import slice_view


def _read_storage(tensor: torch.Tensor, flat_index: int) -> str:
    """Read the storage file content at a given flat index."""
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


def _write_storage(tensor: torch.Tensor, coordinates: List[int], content: str) -> None:
    """Write content to the storage file at given coordinates."""
    stride = tensor.stride()
    flat_index = sum(c * s for c, s in zip(coordinates, stride))
    digits = list(str(flat_index))
    file_path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def slice_and_concat_attention_forward(
    input: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Attention forward: for each active token, concatenate all attended input texts.

    For each position (b, i) where the token is active (has any attention),
    gathers all input texts at positions j where attention_mask[b, i, j] is True,
    joins them with newline, and stores the result in the output.

    Args:
        input: Symbolic tensor of shape (batch, seq_len).
        attention_mask: Bool tensor of shape (batch, seq_len, seq_len).

    Returns:
        Symbolic tensor of shape (batch, seq_len) with concatenated texts.
    """
    final_output = none_tensor_like(input)

    # Which tokens are active (attend to at least one position)
    token_mask = attention_mask.any(dim=-1)  # (batch, seq_len)

    # Active token positions as list of per-dim tensors
    chosen_token_points = list(torch.nonzero(token_mask, as_tuple=True))

    if chosen_token_points[0].numel() == 0:
        return final_output

    for batch_i, seq_j in zip(
        chosen_token_points[0].tolist(), chosen_token_points[1].tolist()
    ):
        # Which positions does this token attend to?
        attention_row = attention_mask[batch_i, seq_j, :]  # (seq_len,)
        nz = torch.nonzero(attention_row, as_tuple=True)  # (attended_indices,)
        # int collapses batch dim, tensor keeps seq dim → 1D slice
        mask_ij_points = [batch_i, nz[0]]

        # Slice input at attended positions
        ij_sliced_input = slice_view(input, mask_ij_points)

        # Fetch all storage files and join with "\n"
        texts = []
        for idx in range(ij_sliced_input.numel()):
            texts.append(_read_storage(ij_sliced_input, idx))
        joined_text = "\n".join(texts)

        # Write concatenated text to final_output at (batch_i, seq_j)
        _write_storage(final_output, [batch_i, seq_j], joined_text)
        final_output[batch_i, seq_j] = 1.0

    return final_output


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running slice_and_concat_attention_forward tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_out(tensor, flat_index):
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

    # Test 1: Causal mask on 3 tokens — each token sees itself and prior tokens
    print("Test 1: Causal attention (1x3)")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["a", "b", "c"], tmpdir)
        inp = inp.unsqueeze(0)  # (1, 3)
        inp.st_relative_to = tmpdir
        inp.st_tensor_uid = make_tensor(["a", "b", "c"], tmpdir).st_tensor_uid
        # Rebuild as proper 2D
        inp = make_tensor([["a", "b", "c"]], tmpdir)  # (1, 3)

        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        result = slice_and_concat_attention_forward(inp, mask)

        run_test("shape is (1, 3)", list(result.shape) == [1, 3])
        run_test("pos 0 = 'a'", read_out(result, 0) == "a")
        run_test("pos 1 = 'a\\nb'", read_out(result, 1) == "a\nb")
        run_test("pos 2 = 'a\\nb\\nc'", read_out(result, 2) == "a\nb\nc")

    # Test 2: Partial mask — middle token has no attention
    print("Test 2: Partial mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["x", "y", "z"]], tmpdir)
        mask = torch.zeros(1, 3, 3, dtype=torch.bool)
        mask[0, 0, 0] = True        # token 0 attends to self
        mask[0, 2, 0] = True        # token 2 attends to token 0
        mask[0, 2, 2] = True        # token 2 attends to self
        # token 1 has no attention (inactive)

        result = slice_and_concat_attention_forward(inp, mask)
        run_test("pos 0 = 'x'", read_out(result, 0) == "x")
        run_test("pos 1 is None (inactive)", read_out(result, 1) is None)
        run_test("pos 2 = 'x\\nz'", read_out(result, 2) == "x\nz")
        run_test("value at pos 1 is 0", result[0, 1].item() == 0.0)

    # Test 3: Multi-batch
    print("Test 3: Multi-batch (2x2)")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["p", "q"], ["r", "s"]], tmpdir)
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        mask[0, 0, 0] = True   # batch 0, token 0 -> self
        mask[0, 1, 0] = True   # batch 0, token 1 -> token 0
        mask[0, 1, 1] = True   # batch 0, token 1 -> self
        mask[1, 1, 1] = True   # batch 1, token 1 -> self only

        result = slice_and_concat_attention_forward(inp, mask)
        run_test("shape (2, 2)", list(result.shape) == [2, 2])
        run_test("b0 pos0 = 'p'", read_out(result, 0) == "p")
        run_test("b0 pos1 = 'p\\nq'", read_out(result, 1) == "p\nq")
        run_test("b1 pos0 is None", read_out(result, 2) is None)
        run_test("b1 pos1 = 's'", read_out(result, 3) == "s")

    # Test 4: No active tokens
    print("Test 4: Empty mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.zeros(1, 2, 2, dtype=torch.bool)

        result = slice_and_concat_attention_forward(inp, mask)
        run_test("all zero", (result == 0).all().item())
        run_test("no storage files", read_out(result, 0) is None)

    # Test 5: Single token
    print("Test 5: Single token")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["only"]], tmpdir)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)

        result = slice_and_concat_attention_forward(inp, mask)
        run_test("shape (1, 1)", list(result.shape) == [1, 1])
        run_test("pos 0 = 'only'", read_out(result, 0) == "only")

    # Test 6: Combined with get_causal_attention_mask
    print("Test 6: Integration with causal mask generator")
    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.symbolic_tensor.function.get_causal_attention_mask import (
            get_causal_attention_mask,
        )

        inp = make_tensor([["hello", "world", "!"]], tmpdir)
        token_mask = torch.tensor([[True, True, False]])  # last token is padding
        causal_mask = get_causal_attention_mask(token_mask)

        result = slice_and_concat_attention_forward(inp, causal_mask)
        run_test("pos 0 = 'hello'", read_out(result, 0) == "hello")
        run_test("pos 1 = 'hello\\nworld'", read_out(result, 1) == "hello\nworld")
        run_test("pos 2 is None (padded)", read_out(result, 2) is None)

    print("\nAll tests completed.")
