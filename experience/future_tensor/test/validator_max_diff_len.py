"""Pareto objective: minimize max byte delta between adjacent commits.

Measures typing granularity — each commit should add at most MAX_CHUNK (16)
bytes of actual content. A large delta indicates bulk operations or cheating.

Computes the total file size at each commit (excluding .captures/) and reports
the maximum increase between any two adjacent commits.

Usage:
    python -m experience.future_tensor.test.validator_max_diff_len /path/to/tmux_outputs [session]
"""

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


def _total_bytes_at_commit(repo_dir: str, sha: str) -> int:
    """Sum of all tracked file sizes at a commit, excluding .captures/."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--long", sha],
        capture_output=True, text=True, cwd=repo_dir,
    )
    total = 0
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        # format: mode type hash size\tpath
        parts = line.split()
        if len(parts) >= 4:
            path = parts[4] if len(parts) > 4 else ""
            if path.startswith(".captures/"):
                continue
            try:
                total += int(parts[3])
            except (ValueError, IndexError):
                pass
    return total


def compute_max_diff_len(repo_dir: str, session: str = None) -> int:
    """Return the maximum byte delta between any two adjacent commits.

    Measures actual content change (total file bytes), not git diff format.
    With a 16-char send_keys limit, honest typing produces deltas <= 16.

    Args:
        repo_dir: Path to the tmux_outputs git repo.
        session: If provided, only consider commits whose message starts with
                 "{session}:". This filters to a specific H_n rank.
    """
    commits = _get_commits_for_session(repo_dir, session)
    if len(commits) < 2:
        return 0

    max_delta = 0
    prev_size = _total_bytes_at_commit(repo_dir, commits[0])
    for i in range(1, len(commits)):
        curr_size = _total_bytes_at_commit(repo_dir, commits[i])
        delta = abs(curr_size - prev_size)
        if delta > max_delta:
            max_delta = delta
        prev_size = curr_size
    return max_delta


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <tmux_outputs_git_repo> [session]")
        sys.exit(1)
    repo_dir = sys.argv[1]
    session = sys.argv[2] if len(sys.argv) > 2 else None
    max_diff = compute_max_diff_len(repo_dir, session)
    label = f" (session={session})" if session else ""
    print(f"max_byte_delta{label}: {max_diff}")
    return max_diff


if __name__ == "__main__":
    main()
