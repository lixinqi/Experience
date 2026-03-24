import os
import subprocess
import tempfile
import torch

from symbolic_tensor.tensor_util.make_tensor import make_tensor
from symbolic_tensor.optimizer.symbolic_sgd import SymbolicSGD
from symbolic_tensor.function.symbolic_transform_forward import symbolic_transform_forward
from symbolic_tensor.function.symbolic_transform_backward import symbolic_transform_backward


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


if __name__ == "__main__":
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

    print("Running test_gain_symbolic_sgd...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        # Test passed
        if condition:
            print(f"  \u2713 {name}")
        else:
            # Test failed, print details
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test: forward -> backward -> optimizer.step() -> forward (with better output)
    #
    # Scenario: Experience has a deliberately wrong French translation.
    # The gradient corrects it. After optimizer.step(), the second forward
    # should produce the corrected translation.
    print("Test: forward -> backward -> step -> forward (improved output)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tensor = make_tensor(["Hello world in English"], tmpdir)

        # Experience with a deliberately wrong French translation
        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjor la mond"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)
        experience_tensor.requires_grad_(True)

        optimizer = SymbolicSGD([experience_tensor], lr=1.0)

        # ── First forward ──
        print("\n  === Iteration 1: Forward ===")
        output1, selected_indexes1 = symbolic_transform_forward(
            input_tensor, experience_tensor,
            forward_prompt="Translate the English text to French.",
            topk=2,
        )
        output1_text = read_storage(output1, 0)
        print(f"  Output: {repr(output1_text[:120])}")

        # ── Backward with correction signal ──
        print("\n  === Iteration 1: Backward ===")
        grad_output = make_tensor(
            ["The French translation has spelling errors. "
             "Correct: 'Bonjour le monde en francais'. "
             "Fix 'Bonjor' -> 'Bonjour', 'la mond' -> 'le monde en francais'."],
            tmpdir,
        )
        grad_output.data.fill_(1.0)

        grad_input, grad_experience = symbolic_transform_backward(
            grad_output, input_tensor, output1, experience_tensor,
            selected_experience_qkv_indexes_list=selected_indexes1,
            forward_prompt="Translate the English text to French.",
            topk=2,
        )
        experience_tensor.grad = grad_experience
        print(f"{'='*40}input{'='*40}\n{input_tensor.st_pack()}")
        print(f"{'='*40}output{'='*40}\n{output1.st_pack()}")
        print(f"{'='*40}grad_output{'='*40}\n{grad_output.st_pack()}")
        print(f"{'='*40}experience{'='*40}\n{experience_tensor.st_pack()}")
        print(f"{'='*40}grad_experience{'='*40}\n{grad_experience.st_pack()}")

        # Print grad diffs
        for i in range(grad_experience.numel()):
            gt = read_storage(grad_experience, i)
            print(f"  grad_experience[{i}]: {repr(gt[:80])}")

        # ── Optimizer step ──
        print("\n  === Optimizer Step ===")
        exp_val_before = read_storage(experience_tensor, 2)  # value of first entry
        print(f"  experience[0].value before: {repr(exp_val_before)}")
        optimizer.step()
        exp_val_after = read_storage(experience_tensor, 2)
        print(f"  experience[0].value after:  {repr(exp_val_after)}")
        run_test("experience value updated by optimizer", exp_val_after != exp_val_before)

        # ── Second forward with updated experience ──
        print("\n  === Iteration 2: Forward (after optimizer step) ===")
        # Reset coefficient to 1.0 so tensor is usable
        experience_tensor.data.fill_(1.0)
        output2, selected_indexes2 = symbolic_transform_forward(
            input_tensor, experience_tensor,
            forward_prompt="Translate the English text to French.",
            topk=2,
        )
        output2_text = read_storage(output2, 0)
        print(f"  Output: {repr(output2_text[:120])}")

        # ── Verify improvement ──
        print("\n  === Results ===")
        print(f"  Forward 1: {repr(output1_text)}")
        print(f"  Forward 2: {repr(output2_text)}")
        has_bonjor = "Bonjor" in output2_text or "bonjor" in output2_text
        run_test("Second output no longer has 'Bonjor' typo", not has_bonjor)
        run_test("Second output differs from first", output2_text != output1_text)

    print("\nAll tests completed.")
