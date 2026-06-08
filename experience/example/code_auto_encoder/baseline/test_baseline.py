"""Test baseline model for auto-encoder cloze experiment.

Generated from test_baseline.viba.

Viba DSL specification:
  kMaskedHint := "<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>"

  test_baseline :=
    IterationList[$baseline_loss list[float]]
    <- $total_batch_size int # default 16
    <- $num_iterations int # default 1
    <- $llm_method str # default "raw_llm_api"
    <- $workspace_dir (str | None) # None means temp directory
    <- $dataset_dir <- { ./codebase/ }
    <- Import[./parepare_dataset]
    <- IterationList[int] <- range <- $num_iterations
    <- Import[./baseline_agent_model.viba].BaselineAgentModel <- $llm_method
    <- Import[function/get_edit_distance]
"""

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional, List

import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio_impl

from experience.example.code_auto_encoder.baseline.prepare_dataset import parepare_dataset
from experience.example.code_auto_encoder.baseline.baseline_agent_model import BaselineAgentModel
from experience.example.code_auto_encoder.baseline.organize_results import ExperimentTracker, print_results_summary


# <- ($dataset_dir <- { ./codebase/ })
CODEBASE_DIR = os.path.join(os.path.dirname(__file__), "..", "codebase")


def _read_storage(tensor, flat_index: int) -> str:
    """Read the content of a symbolic tensor element from disk."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    with open(path) as f:
        return f.read()


@dataclass
class IterationResult:
    """Result of a single iteration."""
    losses: List[float]
    file_info: List[str]
    masked_path_tensor: torch.Tensor
    masked_content_tensor: torch.Tensor
    gt_tensor: torch.Tensor
    output_tensor: torch.Tensor


def _run_single_iteration(
    iteration_idx: int,
    total_batch_size: int,
    dataset_dir: str,
    tmpdir: str,
    llm_method: str,
    interactive: bool = False,
    auto_confirm: bool = True,
    tmux_session: str = None,
    dataset_cache_dir: str = None,
    seed: int = None,
    tracker: ExperimentTracker = None,
) -> IterationResult:
    """Run a single iteration of the baseline test."""
    print(f"\n{'='*50}")
    print(f"Iteration {iteration_idx + 1}")
    print(f"{'='*50}")

    # <- Import[./parepare_dataset]
    masked_path_tensor, masked_content_tensor, gt_tensor, file_info = parepare_dataset(
        total_batch_size, dataset_dir, tmpdir,
        cache_dir=dataset_cache_dir,
        seed=seed,
    )
    print(f"Batch={total_batch_size}, files={masked_path_tensor.shape[1]}")
    for i, info in enumerate(file_info):
        gt_preview = _read_storage(gt_tensor, i)[:60].replace("\n", "\\n")
        print(f"  [{i}] {info} -> {gt_preview}...")

    # <- Import[./baseline_agent_model.viba].BaselineAgentModel <- $llm_method
    mode_str = f"interactive={interactive}" if llm_method == "tmux_cc" else ""
    print(f"\nRunning BaselineAgentModel (llm_method={llm_method} {mode_str})...")
    model = BaselineAgentModel(
        llm_method=llm_method,
        interactive=interactive,
        auto_confirm=auto_confirm,
        tmux_session=tmux_session,
    )
    output = model(masked_path_tensor, masked_content_tensor)

    # <- Import[function/get_edit_distance]
    loss = get_edit_distance_ratio_impl(output, gt_tensor)

    # <- { print $baseline_output.st_uid ground_truth_content_tensor.st_uid }
    print(f"\noutput tensor uid: {output.st_tensor_uid}")
    print(f"ground_truth tensor uid: {gt_tensor.st_tensor_uid}")

    # Register tensors with tracker if available
    if tracker is not None:
        tracker.set_tensors(masked_path_tensor, masked_content_tensor, gt_tensor, output)

    print(f"\nResults (edit_distance_ratio, lower=better):")
    baseline_loss = []
    for i in range(total_batch_size):
        actual = _read_storage(output, i)
        gt = _read_storage(gt_tensor, i)
        r = loss[i].item()
        baseline_loss.append(r)
        newline = '\n'
        print(f"  [{i}] {file_info[i]}  loss={r:.4f}")
        print(f"       gt:     {gt[:80].replace(newline, chr(92) + 'n')}")
        print(f"       actual: {actual[:80].replace(newline, chr(92) + 'n')}")

        # Update tracker for each task (incremental update)
        if tracker is not None:
            tracker.update_task(
                task_index=i,
                file_info=file_info[i],
                loss=r,
                gt_preview=gt,
                output_content=actual,  # Full output content, will be saved to file
            )

    mean_loss = loss.float().mean().item()
    print(f"\nIteration {iteration_idx + 1} mean loss: {mean_loss:.4f}")
    return IterationResult(
        losses=baseline_loss,
        file_info=file_info,
        masked_path_tensor=masked_path_tensor,
        masked_content_tensor=masked_content_tensor,
        gt_tensor=gt_tensor,
        output_tensor=output,
    )


def test_baseline(
    total_batch_size: int = 16,
    num_iterations: int = 1,
    llm_method: str = "raw_llm_api",
    interactive: bool = False,
    auto_confirm: bool = True,
    tmux_session: Optional[str] = None,
    seed: Optional[int] = None,
    experiment_dir: Optional[str] = None,
) -> List[List[float]]:
    """test_baseline from test_baseline.viba.

    Args:
        total_batch_size: default 16
        num_iterations: default 1
        llm_method: default "raw_llm_api", also supports "coding_agent" and "tmux_cc"
        interactive: If True and llm_method="tmux_cc", run in tmux for visual observation.
            Use `tmux attach -t <session>` to watch Claude in real-time.
        auto_confirm: If True (and interactive), auto-confirm prompts in tmux.
        tmux_session: Custom tmux session name (interactive mode only).
        seed: Random seed for dataset generation reproducibility.
        experiment_dir: Organize all results into structured directory:
            experiment_dir/
            ├── dataset/              # Input data cache (auto-managed)
            ├── runs/
            │   ├── raw_llm_api/      # Grouped by llm_method
            │   ├── coding_agent/
            │   └── tmux_cc/
            │       └── run_YYYYMMDD_HHMMSS/
            │           ├── config.json
            │           ├── results.json
            │           ├── output/
            │           │   └── 0/
            │           │       ├── prediction.txt
            │           │       └── ground_truth.txt
            │           └── logs/
            └── latest -> runs/...    # Symlink to latest run

    Returns:
        List of baseline_loss per iteration
    """
    # <- $dataset_dir <- { ./codebase/ }
    dataset_dir = os.path.realpath(CODEBASE_DIR)
    print(f"Dataset: {dataset_dir}")

    # Handle experiment_dir: dataset cache is always under experiment_dir/dataset
    dataset_cache_dir = None
    if experiment_dir is not None:
        os.makedirs(experiment_dir, exist_ok=True)
        dataset_cache_dir = os.path.join(experiment_dir, "dataset")

    # Temporary directory for symbolic tensor storage
    tmpdir = tempfile.mkdtemp()

    # Create experiment tracker if experiment_dir is specified
    tracker = None
    if experiment_dir is not None:
        config = {
            "llm_method": llm_method,
            "total_batch_size": total_batch_size,
            "num_iterations": num_iterations,
            "seed": seed,
            "interactive": interactive,
        }
        tracker = ExperimentTracker(
            experiment_dir=experiment_dir,
            config=config,
            batch_size=total_batch_size,
            seed=seed,
            dataset_cache_dir=dataset_cache_dir,
            llm_method=llm_method,
        )
        
        # Log LLM configuration
        from experience.llm_client.config import get_config_summary
        tracker.log(f"LLM Config ({llm_method}): {get_config_summary(llm_method)}")

    # <- IterationList[int] <- range <- $num_iterations
    all_iteration_results: List[IterationResult] = []
    for iteration_idx in range(num_iterations):
        result = _run_single_iteration(
            iteration_idx, total_batch_size, dataset_dir, tmpdir, llm_method,
            interactive=interactive,
            auto_confirm=auto_confirm,
            tmux_session=tmux_session,
            dataset_cache_dir=dataset_cache_dir,
            seed=seed,
            tracker=tracker,
        )
        all_iteration_results.append(result)

    # Extract losses for return value
    all_iteration_losses = [r.losses for r in all_iteration_results]

    # Summary
    if num_iterations > 1:
        print(f"\n{'='*50}")
        print(f"Summary ({num_iterations} iterations):")
        print(f"{'='*50}")
        mean_losses = [sum(l) / len(l) for l in all_iteration_losses]
        print(f"Mean losses per iteration: {['%.4f' % m for m in mean_losses]}")
        overall_mean = sum(mean_losses) / len(mean_losses)
        print(f"Overall mean loss: {overall_mean:.4f}")

    # Finalize experiment tracker
    if tracker is not None:
        run_dir = tracker.finalize()
        results_path = os.path.join(run_dir, "results.json")
        print_results_summary(results_path)

    return all_iteration_losses


if __name__ == "__main__":
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Test baseline model for auto-encoder cloze experiment.")
    parser.add_argument(
        "--total-batch-size", type=int, default=16,
        help="Total batch size (default: 16)"
    )
    parser.add_argument(
        "--num-iterations", type=int, default=1,
        help="Number of iterations (default: 1)"
    )
    parser.add_argument(
        "--llm-method", type=str, default="raw_llm_api",
        choices=["raw_llm_api", "coding_agent", "tmux_cc"],
        help="LLM method to use (default: raw_llm_api)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive tmux mode for visual observation (only for llm_method=tmux_cc)"
    )
    parser.add_argument(
        "--no-auto-confirm", action="store_true",
        help="Disable auto-confirm prompts in interactive mode (manual operation required)"
    )
    parser.add_argument(
        "--tmux-session", type=str, default=None,
        help="Custom tmux session name (only for interactive mode)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for dataset generation reproducibility."
    )
    parser.add_argument(
        "--experiment-dir", type=str, default=None,
        help="Experiment directory for organized results. Creates: dataset/, runs/, latest symlink."
    )

    args = parser.parse_args()

    # Setup environment from ~/.experience.json based on llm_method
    from experience.llm_client.config import setup_env_for_method, get_config_summary
    setup_env_for_method(args.llm_method)
    print(f"[Config] {args.llm_method}: {get_config_summary(args.llm_method)}")

    # Run the test
    test_baseline(
        total_batch_size=args.total_batch_size,
        num_iterations=args.num_iterations,
        llm_method=args.llm_method,
        interactive=args.interactive,
        auto_confirm=not args.no_auto_confirm,
        tmux_session=args.tmux_session,
        seed=args.seed,
        experiment_dir=args.experiment_dir,
    )
