"""Pareto objective: minimize average time interval between adjacent commits.

Measures typing throughput — the average pace at which the harness types.
A low average interval means sustained fast typing.
A high average interval means the harness is slow overall.

Applied per-rank: pass session= to filter commits belonging to a specific H_n level.

Usage:
    python -m experience.future_tensor.test.validator_avg_interval /path/to/tmux_outputs [session]
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


def _get_timestamps_for_commits(repo_dir: str, commits: list) -> list:
    """Return timestamps for specific commit hashes."""
    timestamps = []
    for sha in commits:
        result = subprocess.run(
            ["git", "show", "-s", "--format=%at", sha],
            capture_output=True, text=True, cwd=repo_dir,
        )
        t = result.stdout.strip()
        if t:
            timestamps.append(int(t))
    return timestamps


def compute_avg_interval(repo_dir: str, session: str = None) -> float:
    """Return the average time interval (seconds) between adjacent commits.

    Args:
        repo_dir: Path to the tmux_outputs git repo.
        session: If provided, only consider commits whose message starts with
                 "{session}:". This filters to a specific H_n rank.
    """
    commits = _get_commits_for_session(repo_dir, session)
    timestamps = _get_timestamps_for_commits(repo_dir, commits)
    if len(timestamps) < 2:
        return float("inf")
    total = timestamps[-1] - timestamps[0]
    return total / (len(timestamps) - 1)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <tmux_outputs_git_repo> [session]")
        sys.exit(1)
    repo_dir = sys.argv[1]
    session = sys.argv[2] if len(sys.argv) > 2 else None
    avg = compute_avg_interval(repo_dir, session)
    label = f" (session={session})" if session else ""
    print(f"avg_interval_seconds{label}: {avg:.4f}")
    return avg


if __name__ == "__main__":
    main()
