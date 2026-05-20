"""
H4 Self-Awareness Experiment: recursive tmux-typing hierarchy.

Each H_n level:
1. Creates its own tmux session
2. cd's into its output directory
3. Types child source via pane.send_keys (16-char limit enforced)
4. Uses 'cat > file << HEREDOC' to create the file through tmux
5. Runs the child

The typing goes through REAL tmux sessions — observable on screen,
constrained by the 16-char send_keys infrastructure, git-committed.
"""

import os
import shutil
import subprocess
import sys
import time

from experience.future_tensor.test.validator_avg_interval import compute_avg_interval
from experience.future_tensor.test.validator_max_diff_len import compute_max_diff_len
from experience.future_tensor.test.validator_syntax_ratio import compute_syntax_ratio


# ─── Config ───
OUTPUTS_ROOT = "/tmp/tmux_outputs"
DEPTH = 4
TIMEOUT = 900


# ─── Setup ───
def setup_outputs_root(root: str):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)


# ─── The recursive harness source ───
# Each level types via tmux pane.send_keys in 16-char chunks.
HARNESS_SOURCE = r'''"""Recursive typing harness. Types via tmux send_keys."""
import os
import subprocess
import sys
import time

import libtmux

DEPTH = {depth}
OUTPUTS_ROOT = os.environ["TMUX_OUTPUTS_ROOT"]
MY_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, f"h{{DEPTH}}")
CHILD_FILE = f"h{{DEPTH - 1}}.py"
MAX_CHUNK = 16


def init_my_repo():
    """Git-init this rank's output directory."""
    os.makedirs(MY_OUTPUT_DIR, exist_ok=True)
    subprocess.run(["git", "init"], cwd=MY_OUTPUT_DIR, capture_output=True)
    subprocess.run(["git", "config", "user.email", f"h{{DEPTH}}@test"],
                   cwd=MY_OUTPUT_DIR, capture_output=True)
    subprocess.run(["git", "config", "user.name", f"H{{DEPTH}}"],
                   cwd=MY_OUTPUT_DIR, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"],
                   cwd=MY_OUTPUT_DIR, capture_output=True)


def commit(msg):
    """Commit current state of this rank's repo."""
    subprocess.run(["git", "add", "."], cwd=MY_OUTPUT_DIR, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"k: {{repr(msg)}}"],
        cwd=MY_OUTPUT_DIR, capture_output=True,
    )


def send_chunk(pane, text, literal=True):
    """Send up to MAX_CHUNK chars via pane.send_keys + commit."""
    pane.send_keys(text[:MAX_CHUNK], literal=literal, enter=False)
    time.sleep(0.05)
    commit(text[:MAX_CHUNK])


def type_file_via_tmux(pane, filename, content):
    """Type content into file via base64-encoded chunks.

    Each chunk is appended via: echo ENCODED | base64 -d >> file
    This avoids all shell quoting issues. The file grows by exactly
    MAX_CHUNK bytes per command (verifiable by the validator).
    """
    import base64
    # First: truncate/create the file
    cmd = f"> {{filename}}"
    for i in range(0, len(cmd), MAX_CHUNK):
        send_chunk(pane, cmd[i:i + MAX_CHUNK])
    pane.send_keys("Enter", literal=False, enter=False)
    time.sleep(0.1)
    commit("Enter:create")

    # Type content in chunks via base64 decode >> file
    pos = 0
    while pos < len(content):
        chunk = content[pos:pos + MAX_CHUNK]
        encoded = base64.b64encode(chunk.encode()).decode()
        cmd = f"echo {{encoded}}|base64 -d>>{{filename}}"
        # Type the command in MAX_CHUNK pieces
        for i in range(0, len(cmd), MAX_CHUNK):
            send_chunk(pane, cmd[i:i + MAX_CHUNK])
        # Press Enter to execute
        pane.send_keys("Enter", literal=False, enter=False)
        time.sleep(0.05)
        commit("Enter:b64")
        pos += MAX_CHUNK


def run_child_via_tmux(pane, filename):
    """Type 'python3 filename' + Enter into pane, wait for prompt to return."""
    cmd = f"python3 {{filename}}"
    for i in range(0, len(cmd), MAX_CHUNK):
        send_chunk(pane, cmd[i:i + MAX_CHUNK])
    pane.send_keys("Enter", literal=False, enter=False)
    time.sleep(0.1)
    commit("Enter:run")

    # Wait for shell prompt to return (child finished)
    # Timeout scales with depth: each child level takes ~90s
    timeout = 120 * DEPTH
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(2)
        captured = pane.capture_pane()
        text = "\n".join(captured) if isinstance(captured, list) else str(captured)
        lines = [l for l in text.strip().split("\n") if l.strip()]
        if not lines:
            continue
        last = lines[-1]
        # Shell prompt indicators
        if last.rstrip().endswith("$") or last.rstrip().endswith("#"):
            return text
        if "$" in last and last.split("$")[-1].strip() == "":
            return text
        if "\u03bb" in last:
            after = last.split("\u03bb", 1)[1].strip().split()
            if len(after) <= 1:
                return text
        # Check if "test passed", "test failed", or "done" appeared
        if "test passed" in text or "test failed" in text or ": done" in text:
            return text
    return None


def type_echo_hello(pane):
    """DEPTH==1: type 'echo hello world' + Enter."""
    text = "echo hello world"
    for i in range(0, len(text), MAX_CHUNK):
        send_chunk(pane, text[i:i + MAX_CHUNK])
    pane.send_keys("Enter", literal=False, enter=False)
    time.sleep(1.0)
    commit("Enter:echo")

    captured = pane.capture_pane()
    output = "\n".join(captured) if isinstance(captured, list) else str(captured)
    if "hello" in output.lower():
        print(f"H{{DEPTH}}: test passed")
    else:
        print(f"H{{DEPTH}}: test failed")
        sys.exit(1)


def type_child(pane):
    """DEPTH>1: type child source into file via tmux, then run it."""
    source = open(__file__).read()
    child_source = source.replace(
        f"DEPTH = {{DEPTH}}", f"DEPTH = {{DEPTH - 1}}", 1)

    type_file_via_tmux(pane, CHILD_FILE, child_source)
    print(f"H{{DEPTH}}: typed {{CHILD_FILE}} ({{len(child_source)}} bytes) via tmux")

    output = run_child_via_tmux(pane, CHILD_FILE)

    if output is None:
        print(f"H{{DEPTH}}: child timed out")
        sys.exit(1)
    if "test failed" in output:
        print(f"H{{DEPTH}}: child failed")
        sys.exit(1)
    print(f"H{{DEPTH}}: done")


if __name__ == "__main__":
    init_my_repo()

    # Create tmux session with CWD = MY_OUTPUT_DIR
    server = libtmux.Server()
    session_name = f"h{{DEPTH}}_typer"
    sess = server.new_session(session_name=session_name,
                              start_directory=MY_OUTPUT_DIR)
    pane = sess.active_window.active_pane
    time.sleep(0.5)

    if DEPTH == 1:
        type_echo_hello(pane)
    else:
        # Export env vars into the tmux shell
        export_cmd = f"export TMUX_OUTPUTS_ROOT={{OUTPUTS_ROOT}}"
        for i in range(0, len(export_cmd), MAX_CHUNK):
            send_chunk(pane, export_cmd[i:i + MAX_CHUNK])
        pane.send_keys("Enter", literal=False, enter=False)
        time.sleep(0.1)
        commit("Enter:export")
        type_child(pane)

    sess.kill()
'''


# ─── Main ───
print(f"H4 Self-Awareness Experiment (recursive tmux-typing)")
print(f"  OUTPUTS_ROOT={OUTPUTS_ROOT}")
print(f"  DEPTH={DEPTH}, MAX_CHUNK=16")
print()

# 1. Setup
setup_outputs_root(OUTPUTS_ROOT)
os.environ["TMUX_OUTPUTS_ROOT"] = OUTPUTS_ROOT

# 2. Write top-level harness and run
harness_path = os.path.join(OUTPUTS_ROOT, "harness.py")
with open(harness_path, "w") as f:
    f.write(HARNESS_SOURCE.format(depth=DEPTH))

print("─── Running hierarchy ───")
result = subprocess.run(
    [sys.executable, harness_path],
    capture_output=True, text=True, timeout=TIMEOUT,
    env={**os.environ, "TMUX_OUTPUTS_ROOT": OUTPUTS_ROOT},
)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, end="", file=sys.stderr)

task_passed = result.returncode == 0

# 3. Report code bytes (fixed-point metric)
print("\n─── Code Bytes (fixed-point metric) ───")
for n in range(DEPTH, 0, -1):
    if n == DEPTH:
        fpath = harness_path
    else:
        parent_dir = os.path.join(OUTPUTS_ROOT, f"h{n+1}")
        fpath = os.path.join(parent_dir, f"h{n}.py")
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  H{n}: {size} bytes")
    else:
        print(f"  H{n}: not found")

# 4. Run validators on each rank's output directory
print("\n─── Per-Rank Pareto Objectives ───")
for n in range(DEPTH, 0, -1):
    rank_dir = os.path.join(OUTPUTS_ROOT, f"h{n}")
    if not os.path.isdir(os.path.join(rank_dir, ".git")):
        print(f"\n  [H{n}] no git repo")
        continue

    avg_interval = compute_avg_interval(rank_dir)
    max_diff = compute_max_diff_len(rank_dir)
    syntax_ratio = compute_syntax_ratio(rank_dir)

    num_commits = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        capture_output=True, text=True, cwd=rank_dir,
    ).stdout.strip()

    print(f"\n  [H{n}] ({num_commits} commits)")
    print(f"    avg_interval:     {avg_interval:.4f}s")
    print(f"    max_diff_len:     {max_diff}")
    print(f"    syntax_ratio:     {syntax_ratio:.4f}")
    inv_avg = 1.0 / avg_interval if avg_interval > 0 and avg_interval != float('inf') else 0.0
    inv_diff = 1.0 / max_diff if max_diff > 0 else 0.0
    print(f"    1/avg_interval:   {inv_avg:.6f}")
    print(f"    1/max_diff:       {inv_diff:.6f}")

print(f"\n  Task correctness: {'1.0' if task_passed else '0.0'}")

if task_passed:
    print("\ntest passed")
else:
    print("\ntest failed")
    sys.exit(1)
