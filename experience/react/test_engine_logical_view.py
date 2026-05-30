"""
test_engine_logical_view: Verify engine_step receives logical views
that enable the inner agent to read history across iterations.

Simulates the CodingAgentEngine IPC flow without tmux/LLM:
  1. Build FutureTensors with growing incremental chunks (simulating recurrent loop)
  2. Write step_N_input.json like _write_step_input does
  3. Mock inner agent reads the JSON, uses logical_views to walk history
  4. Verify the agent can find and read content from previous iterations
"""

import json
import os
import sys
import tempfile

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


passed = 0
failed = 0


def run_test(name: str, condition: bool, expected=None, actual=None):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  ✗ {name}")
        if expected is not None:
            print(f"    expected: {expected}, actual: {actual}")


# ═══════════════════════════════════════════════════════════════════════
# Group 1: Simulate engine step IPC with growing capture tensor
# ═══════════════════════════════════════════════════════════════════════

print("Group 1: Multi-step engine with growing logical views")

with tempfile.TemporaryDirectory() as tmpdir:
    # Build capture FutureTensor with [1, n] schema
    # Simulate what ft_tmux_capture_pane produces
    n = sympy.Symbol("n")
    capture_contents = ["screen_after_iter_0", "screen_after_iter_1", "screen_after_iter_2"]

    async def capture_get(coords, prompt):
        i = coords[-1] if coords else 0
        return (capture_contents[i], Status.confidence(1.0))

    capture = FutureTensor(tmpdir, capture_get, [sympy.Integer(1), n])

    # Simulate 3 iterations of the recurrent loop:
    # Each iteration calls capture.ft_async_get([0, i], ...)
    # which triggers _cached_async_get_impl -> _grow_capacity_for
    import asyncio

    for i in range(3):
        asyncio.run(capture.ft_async_get([0, i], f"prompt_{i}"))

    # Now capture has 1 static segment + 3 chunks
    run_test("1: capacity shape is [1,3]",
             capture.ft_capacity_shape == [1, 3])
    run_test("2: 3 incremental chunks",
             len(capture.ft_incremental_concated_tensors) == 3)

    # Simulate CodingAgentEngine._engine_get at iteration 2
    # (the last iteration, where it reads capture at coords [0, 2])
    logical_views = {
        "capture": capture.ft_describe_logical_view(),
    }

    run_test("3: logical_view has 4 segments (static + 3 chunks)",
             len(logical_views["capture"]) == 4)

    # Segments: [static shape_before [0,0]->[1,0]], [chunk0 [1,0]->[1,1]],
    #            [chunk1 [1,1]->[1,2]], [chunk2 [1,2]->[1,3]]
    segs = logical_views["capture"]
    run_test("4: seg0 static shape_before [0,0]",
             segs[0]["shape_before"] == [0, 0])
    run_test("5: seg0 static shape_after [1,0]",
             segs[0]["shape_after"] == [1, 0])
    run_test("6: seg1 shape_before [1,0]",
             segs[1]["shape_before"] == [1, 0])
    run_test("7: seg1 shape_after [1,1]",
             segs[1]["shape_after"] == [1, 1])
    run_test("8: seg2 shape_before [1,1]",
             segs[2]["shape_before"] == [1, 1])
    run_test("9: seg2 shape_after [1,2]",
             segs[2]["shape_after"] == [1, 2])
    run_test("10: seg3 shape_before [1,2]",
             segs[3]["shape_before"] == [1, 2])
    run_test("11: seg3 shape_after [1,3]",
             segs[3]["shape_after"] == [1, 3])

    # Write step_2_input.json (simulating _write_step_input)
    step_input = {
        "step": 2,
        "task": "do something",
        "capture": capture_contents[2],
        "fixation": "0,0",
        "logical_views": logical_views,
        "output_file": os.path.join(tmpdir, "step_2_output.json"),
        "signal_file": os.path.join(tmpdir, "step_2_done"),
    }
    input_path = os.path.join(tmpdir, "step_2_input.json")
    with open(input_path, "w") as f:
        json.dump(step_input, f, indent=2)

    run_test("12: step_2_input.json written",
             os.path.isfile(input_path))


# ═══════════════════════════════════════════════════════════════════════
# Group 2: Mock inner agent reads logical_views to find history
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 2: Mock agent reads history via logical_views")

with tempfile.TemporaryDirectory() as tmpdir:
    # Build 3 captures directly as SymbolicTensors
    content_0 = make_tensor(["capture_iter_0: terminal shows prompt"], tmpdir)
    content_1 = make_tensor(["capture_iter_1: terminal shows 'a' echoed"], tmpdir)
    content_2 = make_tensor(["capture_iter_2: terminal shows 'b' echoed"], tmpdir)

    # Build a FutureTensor with these chunks appended
    n = sympy.Symbol("n")

    async def dummy_get(coords, prompt):
        return ("", Status.confidence(1.0))

    ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(1), n])
    ft.ft_incremental_concated_tensors = [(content_0, 1), (content_1, 1), (content_2, 1)]
    ft.ft_capacity_shape = [1, 3]

    logical_views = {"capture": ft.ft_describe_logical_view()}

    # Mock inner agent: reads step input, then walks logical_views backward
    # to find what happened in iteration 1

    # Agent logic:
    #   segs = logical_views["capture"]
    #   # Iteration 1's capture is at logical coords [0, 1]
    #   # Which segment covers flat index 1?
    #   # seg0: shape_after [1,0] -> flat 0 (covers nothing)
    #   # seg1: shape_after [1,1] -> flat 0..0  (covers flat 0)
    #   # seg2: shape_after [1,2] -> flat 0..1  (covers flats 0,1)
    #   # seg3: shape_after [1,3] -> flat 0..2  (covers flats 0,1,2)
    #   # Flat index 1 falls in seg2

    segs = logical_views["capture"]

    # Find segment covering flat index 1 (iteration 1)
    target_flat = 1
    found_seg = None
    for seg in segs:
        n_elements = 1
        for d in seg["shape_after"]:
            n_elements *= d
        if target_flat < n_elements:
            found_seg = seg
            break

    run_test("13: agent found segment for flat index 1",
             found_seg is not None)
    run_test("14: segment is seg2 (chunk 1)",
             found_seg["shape_before"] == [1, 1])

    # Agent reads the content from the segment's SymbolicTensor
    seg_path = found_seg["symbolic_tensor_path"]
    storage_file = os.path.join(seg_path, "storage", "0", "data")
    run_test("15: storage file exists at segment path",
             os.path.isfile(storage_file))
    with open(storage_file) as f:
        history_content = f.read()
    run_test("16: agent reads iter_1 history",
             "capture_iter_1" in history_content)

    # Agent also reads iteration 0 (flat index 0)
    target_flat = 0
    found_seg_0 = None
    for seg in segs:
        n_elements = 1
        for d in seg["shape_after"]:
            n_elements *= d
        if target_flat < n_elements:
            found_seg_0 = seg
            break

    run_test("17: agent found segment for flat index 0",
             found_seg_0 is not None)
    seg_path_0 = found_seg_0["symbolic_tensor_path"]
    storage_file_0 = os.path.join(seg_path_0, "storage", "0", "data")
    with open(storage_file_0) as f:
        history_0 = f.read()
    run_test("18: agent reads iter_0 history",
             "capture_iter_0" in history_0)


# ═══════════════════════════════════════════════════════════════════════
# Group 3: Simulate full CodingAgentEngine flow (no tmux)
# ═══════════════════════════════════════════════════════════════════════

print("\nGroup 3: Simulated CodingAgentEngine IPC with mock inner agent")

with tempfile.TemporaryDirectory() as tmpdir:
    # This is a mock inner agent: reads step_N_input.json,
    # uses logical_views to find previous capture, makes decision

    n = sympy.Symbol("n")

    # Pre-build captures as SymbolicTensors
    c0 = make_tensor(["iter0: empty terminal"], tmpdir)
    c1 = make_tensor(["iter1: typed 'a', terminal shows a"], tmpdir)
    c2 = make_tensor(["iter2: typed Enter, terminal shows a"], tmpdir)

    async def cap_get(coords, prompt):
        i = coords[-1] if coords else 0
        return (f"live_capture_{i}", Status.confidence(1.0))

    capture_ft = FutureTensor(tmpdir, cap_get, [sympy.Integer(1), n])
    capture_ft.ft_incremental_concated_tensors = [(c0, 1), (c1, 1), (c2, 1)]
    capture_ft.ft_capacity_shape = [1, 3]

    # Simulate engine_step at iteration 2 (last one)
    lv = capture_ft.ft_describe_logical_view()

    # Write step input
    step_data = {
        "step": 2,
        "task": "type a and press enter",
        "capture": "iter2: typed Enter, terminal shows a",
        "fixation": "0,0",
        "logical_views": {"capture": lv},
        "output_file": os.path.join(tmpdir, "step_2_output.json"),
        "signal_file": os.path.join(tmpdir, "step_2_done"),
    }
    input_file = os.path.join(tmpdir, "step_2_input.json")
    with open(input_file, "w") as f:
        json.dump(step_data, f, indent=2)

    # ── Mock inner agent ──
    # Reads step_2_input.json, gets logical_views, walks history
    with open(input_file) as f:
        agent_input = json.load(f)

    agent_lv = agent_input["logical_views"]["capture"]
    agent_current = agent_input["capture"]
    agent_task = agent_input["task"]

    # Agent decision: check if task is complete by reading all history
    all_captures = []
    for seg in agent_lv:
        seg_path = seg["symbolic_tensor_path"]
        storage = os.path.join(seg_path, "storage", "0", "data")
        if os.path.isfile(storage):
            with open(storage) as f:
                all_captures.append(f.read())

    run_test("19: agent sees 3 historical captures",
             len([c for c in all_captures if c]) == 3)

    # Agent checks if "a" appeared in any capture (task progress)
    saw_a = any("a" in c for c in all_captures)
    run_test("20: agent finds 'a' in history",
             saw_a is True)

    # Agent checks if the task was completed (Enter was pressed after typing)
    # iter2 shows "typed Enter" -> task complete
    task_complete = "Enter" in all_captures[-1] if all_captures else False
    run_test("21: agent confirms Enter was sent",
             task_complete is True)

    # Agent can also see growth pattern: shape went [1,0]->[1,1]->[1,2]->[1,3]
    shapes = [seg["shape_after"] for seg in agent_lv]
    run_test("22: shape growth pattern",
             shapes == [[1, 0], [1, 1], [1, 2], [1, 3]])

    # Write agent's output (simulating what the inner agent produces)
    agent_output = {
        "current_fixation": [0, 0],
        "current_foveal": "prompt",
        "keystrokes": "",  # task already complete
        "predicted_next_fixation": [0, 0],
        "predicted_next_foveal": "prompt",
        "children": [],
    }
    with open(step_data["output_file"], "w") as f:
        json.dump(agent_output, f)

    run_test("23: agent output written",
             os.path.isfile(step_data["output_file"]))

    # Touch signal file
    with open(step_data["signal_file"], "w") as f:
        f.write("done")

    run_test("24: signal file written",
             os.path.isfile(step_data["signal_file"]))


# ═══════════════════════════════════════════════════════════════════════

print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
if failed == 0:
    print("All engine logical view tests passed.")
else:
    sys.exit(1)