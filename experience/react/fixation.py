"""Fixation utilities: extract foveal, compute fixation, match predictions."""

from typing import Tuple


def _nearest_word(words: list, col: int, start: int) -> str:
    """Find the word nearest to col position."""
    best = words[0]
    best_dist = abs(col - start)
    pos = start
    for w in words:
        dist = abs(col - (pos + len(w) // 2))
        if dist < best_dist:
            best = w
            best_dist = dist
        pos += len(w) + 1
    return best


def extract_foveal(capture: str, fixation: Tuple[int, int], window: int = 8) -> str:
    """Extract the foveal word at the fixation point from captured screen."""
    lines = capture.split("\n")
    row, col = fixation
    if row < 0 or row >= len(lines):
        return ""
    line = lines[row]
    start = max(0, col - window // 2)
    end = min(len(line), col + window // 2)
    words = line[start:end].strip().split()
    if not words:
        return ""
    return _nearest_word(words, col, start)


def compute_fixation(capture: str, prev_fixation: Tuple[int, int]) -> Tuple[int, int]:
    """Compute next fixation: last non-empty line, end of content."""
    lines = capture.split("\n")
    last_row = 0
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_row = i
            break
    col = len(lines[last_row].rstrip()) if last_row < len(lines) else 0
    return (last_row, col)


def foveal_matches(actual: str, predicted: str) -> bool:
    """Check if actual foveal matches predicted (fuzzy substring match)."""
    if not predicted or not actual:
        return True
    return predicted in actual or actual in predicted
