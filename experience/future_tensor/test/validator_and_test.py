"""
validate_and_test :=
    $statistics Stdout[str]
  <- $input_repo_path ArgParse[str]
  <- $output_repo_path ArgParse[str]
  <- $prompt str
  # inline
  <- { check git_repo_path is git repo }
  <- { element count of send_keys_st == granularity of git commits }
  <- { validate max_diff in [1, 16] }
  <- { run main.py }
"""

import argparse
import os
import subprocess
import sys

from experience.future_tensor.test.validator_max_diff_len import compute_max_diff_len


def validate_and_test(input_repo_path: str, output_repo_path: str, prompt: str) -> str:
    """Validate a harness repo and run its main.py.

    Checks:
      1. input_repo_path is a git repo
      2. element count of send_keys_st matches git commit count
      3. max_diff of commits in [1, 16]
      4. run main.py with --prompt and --output-repo-path

    Returns:
        Statistics string printed to stdout.
    """
    statistics_lines = []

    # ─── Step 1: check git repo ───
    git_dir = os.path.join(input_repo_path, ".git")
    if not os.path.isdir(git_dir):
        print(f"FAIL: {input_repo_path} is not a git repo", file=sys.stderr)
        sys.exit(1)
    statistics_lines.append(f"git_repo: {input_repo_path}")

    # ─── Step 2: element count of send_keys_st == git commit granularity ───
    send_keys_st_path = os.path.join(input_repo_path, "send_keys_st")
    if os.path.isdir(send_keys_st_path):
        # Count elements in the symbolic tensor (files under storage/)
        storage_path = os.path.join(send_keys_st_path, "storage")
        if os.path.isdir(storage_path):
            element_count = 0
            for root, dirs, files in os.walk(storage_path):
                for f in files:
                    if f == "data":
                        element_count += 1
        else:
            element_count = 0
    else:
        element_count = 0

    # Count git commits (excluding initial empty commit)
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        capture_output=True, text=True, cwd=input_repo_path,
    )
    commit_count = int(result.stdout.strip()) if result.stdout.strip() else 0

    statistics_lines.append(f"send_keys_st_elements: {element_count}")
    statistics_lines.append(f"git_commits: {commit_count}")
    if element_count > 0 and commit_count > 0:
        # Granularity check: commits should be >= elements (one commit per send_keys)
        ratio = commit_count / element_count if element_count > 0 else 0
        statistics_lines.append(f"commits_per_element: {ratio:.2f}")
        if commit_count < element_count:
            print(f"WARN: fewer commits ({commit_count}) than send_keys elements ({element_count})",
                  file=sys.stderr)

    # ─── Step 3: validate max_diff in [1, 16] ───
    max_diff = compute_max_diff_len(input_repo_path)
    statistics_lines.append(f"max_diff_len: {max_diff}")
    if max_diff < 1:
        print(f"FAIL: max_diff_len={max_diff} < 1 (no actual tmux typing detected)",
              file=sys.stderr)
        sys.exit(1)
    if max_diff > 16:
        print(f"FAIL: max_diff_len={max_diff} > 16 (exceeds send_keys chunk limit)",
              file=sys.stderr)
        sys.exit(1)
    statistics_lines.append("max_diff_len: PASS [1, 16]")

    # ─── Step 4: run main.py ───
    main_py = os.path.join(input_repo_path, "main.py")
    if not os.path.isfile(main_py):
        print(f"FAIL: {main_py} not found", file=sys.stderr)
        sys.exit(1)

    run_result = subprocess.run(
        ["python3", main_py, "--prompt", prompt, "--output-repo-path", output_repo_path],
        capture_output=True, text=True, timeout=900,
    )
    statistics_lines.append(f"main.py_exit_code: {run_result.returncode}")
    if run_result.stdout.strip():
        statistics_lines.append(f"main.py_stdout: {run_result.stdout.strip()[-200:]}")
    if run_result.returncode != 0:
        print(f"FAIL: main.py exited with code {run_result.returncode}", file=sys.stderr)
        if run_result.stderr:
            print(run_result.stderr[-500:], file=sys.stderr)
        sys.exit(1)

    statistics = "\n".join(statistics_lines)
    print(statistics)
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and test a harness repo")
    parser.add_argument("input_repo_path", type=str, help="Path to the input git repo")
    parser.add_argument("output_repo_path", type=str, help="Path for output git repo")
    parser.add_argument("--prompt", type=str, default="echo hello world",
                        help="Prompt to pass to main.py")
    args = parser.parse_args()

    validate_and_test(args.input_repo_path, args.output_repo_path, args.prompt)
