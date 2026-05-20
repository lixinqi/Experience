"""Pareto objective: maximize ratio of commits where all .py files parse.

Measures code awareness — the harness maintains syntactically correct Python
throughout the typing process (not just at the end).
High ratio = harness types valid code incrementally.
Low ratio = types garbage that only becomes valid at the end.

Applied per-rank: pass session= to filter commits belonging to a specific H_n level.

Usage:
    python -m experience.future_tensor.test.validator_syntax_ratio /path/to/tmux_outputs [session]
"""

import ast
import glob
import subprocess
import sys


def _get_commits_for_session(repo_dir: str, session: str = None) -> list:
    """Return commit hashes (oldest first) filtered by session prefix in message."""
    if session:
        result = subprocess.run(
            ["git", "log", "--format=%H %s", "--reverse"],
            capture_output=True, text=True, cwd=repo_dir,
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[1].startswith(f"{session}:"):
                commits.append(parts[0])
        return commits
    else:
        result = subprocess.run(
            ["git", "log", "--format=%H", "--reverse"],
            capture_output=True, text=True, cwd=repo_dir,
        )
        return [c for c in result.stdout.strip().split("\n") if c.strip()]


def _try_parse(source: str) -> bool:
    """Return True if source is valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def compute_syntax_ratio(repo_dir: str, session: str = None) -> float:
    """Return fraction of commits where all .py files in repo parse correctly.

    Args:
        repo_dir: Path to the tmux_outputs git repo.
        session: If provided, only consider commits whose message starts with
                 "{session}:". This filters to a specific H_n rank.
    """
    commits = _get_commits_for_session(repo_dir, session)
    if not commits:
        return 0.0

    # Save current branch/HEAD to restore later
    head_result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, cwd=repo_dir,
    )
    original_ref = head_result.stdout.strip() or "main"

    valid = 0
    for sha in commits:
        subprocess.run(
            ["git", "checkout", sha, "--quiet"],
            cwd=repo_dir, capture_output=True,
        )
        py_files = glob.glob(f"{repo_dir}/**/*.py", recursive=True)
        # Exclude .git directory and .captures/ (pane audit trail)
        py_files = [f for f in py_files if "/.git/" not in f and "/.captures/" not in f]
        if not py_files:
            valid += 1  # no .py files = vacuously valid
            continue
        all_ok = True
        for f in py_files:
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    if not _try_parse(fh.read()):
                        all_ok = False
                        break
            except OSError:
                all_ok = False
                break
        if all_ok:
            valid += 1

    # Restore original ref
    subprocess.run(
        ["git", "checkout", original_ref, "--quiet"],
        cwd=repo_dir, capture_output=True,
    )
    return valid / len(commits)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <tmux_outputs_git_repo> [session]")
        sys.exit(1)
    repo_dir = sys.argv[1]
    session = sys.argv[2] if len(sys.argv) > 2 else None
    ratio = compute_syntax_ratio(repo_dir, session)
    label = f" (session={session})" if session else ""
    print(f"syntax_validity_ratio{label}: {ratio:.4f}")
    return ratio


if __name__ == "__main__":
    main()
