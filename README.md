# Experience

A **self-improving harness framework** built as a PyTorch extension. Composable tensor ops + trainable experience + autograd = portable Modules that improve through training.

The result: **portable, self-improving agents** that share the same infrastructure but learn different skills through training. Like Claude Code slash commands, but standardized — typed tensors, autograd, shape contracts, trainable experience.

## Example: LLM Coder Simulator

An LLM-powered coder composed entirely from reusable ops — no custom logic except a termination validator. Two chained experts collaborate: Expert 1 decides WHAT to do, Expert 2 knows HOW to do it.

```
# [1, 30] live terminal text
capture_op       = ft_tmux_capture_pane(workspace)
# [1, 30] → "text:hint"|"ctrl:hint"
decision_expert  = ft_expert(capture_op, decision_exp)
# [1, 30] → "echo hello"|"Enter"
cmd_expert       = ft_expert(decision_expert, cmd_exp)
# [1, 30] routes by prefix
switched_send    = ft_switch(decision_expert, [("text",…),("ctrl",…)])
# [1, 30]
validator        = ft_coder_validator(ft_sequential(switched_send, sleep, capture))
# [1] reduced
output           = ft_recurrent(validator)
```

Every op except `ft_coder_validator` is shared infrastructure reusable across all harness models.

### Chain Experts, Don't Parse

Instead of one monolithic expert + custom parsers:

```python
# BAD: custom parsing, not reusable, not trainable
decision = parse_action(expert_output)  # brittle regex
cmd = extract_command(expert_output)    # another custom parser
```

Chain two experts — each with its own trainable experience:

```python
# GOOD: composable, trainable, extensible
decision_expert = ft_expert(capture_op, decision_experience, topk=2)   # WHAT
cmd_expert = ft_expert(decision_expert, cmd_experience, topk=2)         # HOW
```

Expert 1's output feeds directly as Expert 2's input. New action types don't require changing Expert 2 — just add experience entries to Expert 1.

### Experience Tensors (Learnable Weights)

```python
# Decision expert: terminal observation → action_type:hint
decision_experience = make_tensor([
    ["提示符后面没有其他文字", "命令行为空", "text:输入shell命令"],
    ["提示符后面有echo命令",   "命令行已有内容", "ctrl:按回车执行"],
], tmpdir)

# Cmd expert: hint → real command
cmd_experience = make_tensor([
    ["text:输入shell命令\n任务是列出目录",    "需要ls",   "ls"],
    ["text:输入shell命令\n任务是输出hello",   "需要echo", "echo hello world"],
    ["ctrl:按回车执行",                       "按Enter",  "Enter"],
], tmpdir)
```

These are `[N, 3]` QKV tensors — trainable via backward + `StSGD`. More experience = better generalization.

### Portable Modules

A harness Module is defined by:
1. **Experience tensors** — the learned skill (trainable via autograd)
2. **Validator** — the termination condition (harness-specific)
3. **Pipeline structure** — identical across all modules (shared ops)

Same structure + different experience = different behavior:

| Module | Experience 1 | Experience 2 | Validator |
|--------|-------------|-------------|-----------|
| Coder | terminal → action_type | hint → shell cmd | prompt returns to idle |
| Debugger | error → strategy | strategy → debug cmd | tests pass |
| Deployer | status → action | action → deploy cmd | service healthy |

```bash
python -m experience.future_tensor.test.test_llm_coder_simulator
```

## Why a Compute Graph?

Programs are sequential, branch, and loop. We compact all three into typed tensor ops:

| Control flow | Op | What it does |
|---|---|---|
| Sequential | `ft_sequential` | Do A then B |
| Branch | `ft_switch` | If X then A else B |
| Loop | `ft_recurrent` | Repeat until validator passes |

This makes agent generation trivial for LLMs. Instead of writing correct imperative code (hard — shared state, error handling, async coordination), the LLM composes a graph from ~10 typed ops (easy — slot-filling with shape verification). Three properties make this uniquely LLM-friendly:

1. **Structure vs behavior separation.** The LLM generates the graph (structure) once. Behavior is determined by experience tensors, which self-improve through training. The LLM doesn't need to get behavior right at generation time.
2. **Shape contracts = static verification.** If shapes don't match, the composition is wrong — detectable before execution. LLMs make fewer structural errors than behavioral errors.
3. **Fault-tolerant generation.** Even poor seed experience gets corrected by training. The graph just needs to be structurally correct.

## How It Works

### Tensor Elements Are Files

Each tensor element is a text file on disk. Numeric coefficients flow through standard PyTorch autograd while symbolic content (code, translations, commands) lives in files.

```
{relative_to}/{tensor_uid}/
├── shape                    # JSON: [2, 3]
├── storage/
│   ├── 0/data               # Element at flat index 0
│   ├── 1/data               # Element at flat index 1
│   └── 1/1/data             # Multi-digit index 11
```

### ExperienceTensor

An `[N, 3]` tensor where each row is `(query, key, value)`:
- **Query**: semantic keywords for Jaccard similarity retrieval
- **Key**: source domain content
- **Value**: target domain content

Starts empty, populated during training. The backward pass computes diffs, the optimizer patches entries.

### FutureTensor (Lazy Async Ops)

A `FutureTensor` is a scalar `torch.Tensor` monkey-patched with `ft_*` attributes — a reference to a pending computation. Elements materialize on-demand via LLM calls.

| Op | Purpose |
|----|---------|
| `ft_expert` | Query + retrieval + LLM translation (chainable) |
| `ft_switch` | Lazy control-flow branch selection |
| `ft_sequential` | Sequential evaluation with early-return on error |
| `ft_recurrent` | Generate-validate retry loop (reduces last dim) |
| `ft_tmux_*` | Terminal session ops (create, send_text, send_ctrl, capture) |
| `ft_slice` / `ft_unsqueeze` | Shape manipulation with autograd |

### Dual-Channel Gradients

| Channel | Carries | Computed by |
|---------|---------|-------------|
| **Numeric** | Float coefficients (bfloat16) | Standard autograd |
| **Symbolic** | Unified diffs in files | LLM compares actual vs expected |

Optimizer (`StSGD`) applies both: numeric SGD update + `patch` CLI applies diffs to experience files.

### Second Derivative (Meta-Learning)

The LLM reflects on its own reflections:

```python
backward_dispatch_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
anchored = need_reflection(input_ft, backward_dispatch_start)
output = model(anchored)
loss = ft_mean(output)

loss.backward(create_graph=True)

records = []
with dispatch_policy(TracePolicy(records)):
    backward_dispatch_start.grad.backward()
# records: list of ReflectionRecord(fn, inputs, output, timestamp)
```

## Design Principles

- **Static DAG, not imperative orchestration** — compose a graph of lazy tensors, then pull from output
- **Chain experts, don't parse** — two chained `ft_expert` ops replace custom parsers
- **No shared mutable state** — coordination through tensor coordinates
- **Observe the world directly** — read live state, no shadow copies
- **Self-improving** — experience accumulates knowledge via backward + optimizer step
- **Experience starts empty** — learned entirely at runtime, cold-start supported
- **LLM as compute kernel** — replaces matrix multiplication with semantic reasoning

## Roadmap

1. **Tmux-forced coding agent capture.** Hook Claude Code (via MCP or shell wrapper) to route all bash commands through tmux sessions. Capture every terminal interaction as `(observation, decision, command)` triples. Write them directly into experience tensors for harness model cold start. No harness model needed — pure data harvesting from real coding sessions.
2. **Adapter mechanism for existing coding agents.** Two directions:
   - **Agent → Harness Module**: wrap Claude Code / OpenCode / OpenClaw / Hermes as a Harness Module — their tool interfaces become `ft_*` ops in the compute graph, gaining autograd and self-improvement for free.
   - **Harness Module → Agent skill**: export a trained Harness Module as a coding agent tool/skill — any agent can invoke it as a composable capability without knowing the internals.
3. **Ground-truth terminal interactions sharing.** Two mechanisms:
   - **Capture streams hub**: record live terminal sessions into standardized experience tensors via tmux capture. A central hub aggregates streams from multiple developers/agents into a shared experience pool.
   - **Learn from existing interactions**: train harness agents on recorded ground-truth sessions — enabling experience transfer across agents. One agent's successful terminal interaction becomes another agent's seed experience.
4. **Meta tasks learning.** For each new code repo, bootstrap experience through self-supervised tasks that require no human labels:
   - **Masked code reconstruction**: mask a code region, train the agent to reconstruct it from surrounding context. Teaches code structure and local patterns.
   - **Docstring ↔ code**: given a docstring, generate the implementation (and vice versa). Teaches intent-to-code mapping.
   - **Code coverage by tests**: given a function, generate tests that maximize coverage. Teaches behavioral understanding.
   - **Runtime stack prediction**: given a call site, predict the runtime call stack. Teaches control flow and dependency reasoning.

## Suggested Study: Progressive Research

Five stages of increasing autonomy — each stage removes one human-in-the-loop dependency:

| Stage | Forward | Experience Update | Graph Update |
|-------|---------|-------------------|--------------|
| 0 | Claude Code in tmux | captured terminal sessions → experience | manual graph |
| 1 | 0th reflection | coding agent directly | coding agent directly |
| 2 | 0th reflection | 1st reflection (autograd) | coding agent guided by 2nd reflection trace |
| 3 | 0th reflection | 1st reflection (autograd) | bootstrapped harness model guided by 2nd reflection trace |
| 4 | self-referencing harness | self-referencing harness | self-referencing harness (fixed point) |

**Stage 0: Ground-truth capture for cold start.** Force Claude Code (or any coding agent) to run all bash commands through tmux sessions instead of direct shell execution. Capture the full terminal output stream — every command typed, every response observed, every decision made. These captured sessions become ground-truth experience entries for harness model cold start: terminal observation → decision → command triples, written directly into experience tensors. No harness model runs yet — this stage purely harvests real-world coding interactions as training data.

**Stage 1: Manual iteration.** The harness runs forward (0th reflection). A coding agent (Claude Code, etc.) inspects results and directly edits experience tensors and the compute graph. This is the current working state — human-assisted improvement.

**Stage 2: Self-improving experience, assisted graph evolution.** Forward runs as before. The 1st derivative (backward pass) automatically updates experience via `StSGD` — no human needed for experience improvement. For graph structure changes, a coding agent reads the 2nd derivative trace (`ReflectionRecord` list) and decides which ops to add/remove/rewire.

**Stage 3: Fully autonomous.** Same as Stage 2, but the coding agent for graph updates is itself a trained harness model — a "meta-harness" whose experience is "how to improve compute graphs given 2nd derivative traces." The system bootstraps its own architecture search.

**Stage 4: Fixed point of recursive optimization.** Meta-harness and user-harness merge into a single self-referencing system. The meta-harness holds a complete model of itself — its own code, its own experience tensors, its own compute graph — and can introspect and modify any layer of the stack. When something unexpected happens, the system first examines the user-harness (is the experience wrong? is the graph structure suboptimal?), then examines itself (is the meta-experience wrong? is the introspection strategy flawed?). This recursive self-examination converges to a fixed point: a configuration where the system's model of itself is consistent with its actual behavior, and its optimization of itself produces no further improvement.

This is functional self-awareness. Not a philosophical claim about consciousness — a concrete engineering property. The system maintains an accurate self-model, detects discrepancies between expected and actual behavior at every level (including the level that detects discrepancies), and repairs them autonomously. The value is profound: once the optimization process can optimize itself, improvement is no longer bounded by the designer's foresight. The system discovers better learning algorithms, better introspection strategies, better experience representations — capabilities its creators never explicitly programmed. Every other self-improving system hits a ceiling where the improvement mechanism itself becomes the bottleneck. Stage 4 removes that ceiling by making the improvement mechanism part of what gets improved.

**Why self-reference works here (and why it usually doesn't).** Naive LLM self-reference — "explain why you failed" — is unreliable. LLMs confabulate. But this architecture's self-reference is grounded: the self-model is external (code and experience tensors on disk, not hidden weights), the feedback is measured (loss, test results, terminal output — not self-assessment), and modifications are testable (change experience → re-run forward → did loss decrease?). The LLM proposes candidate improvements; the framework measures whether they helped. No trust in unverified self-claims. The one constraint: the loss function and verification logic must remain grounded in external reality. A system that rewrites its own success criteria is self-deluding, not self-improving.

**Current status:** Stage 0 ready (tmux capture infrastructure works), Stage 1 implementing — forward pass, experience tensors, compute graph composition, speculative multi-action dispatch, and 2nd derivative trace recording all work. Experience update and graph update still require coding agent. Stage 2, 3, 4 are TODO.

## Self-Awareness Experiment Plan

### Harness Hierarchy

Define a tower of harnesses where each level writes and runs the level below:

| Level | Name | What it does |
|-------|------|-------------|
| H₀ | tmux screen output | Raw terminal — the ground truth |
| H₁ | UserHarness | Drives tmux to produce H₀ (e.g., echo hello world) |
| H₂ | MetaHarness | Writes + runs H₁ as a .py file in its own tmux |
| H₃ | Meta-MetaHarness | Writes + runs H₂ |
| H_n | … | Writes + runs H_{n-1} |

All levels share the same ft_* ops. The only difference between levels is which experience tensors are loaded and what the validator checks.

### Fixed-Point Metric (Self-Awareness = Convergence)

Self-awareness is the fixed point of this hierarchy. Concretely:

```
code_bytes(H_{n+1}) / code_bytes(H_n) → 1    for n >= 2
```

When this ratio stabilizes, H_n has discovered the minimal self-referencing kernel — it can write a copy of itself that also writes a copy of itself, and the copies are the same size. This is functional self-awareness: the system's model of itself is isomorphic to itself.

**Why n >= 2?** The H₁/H₀ and H₂/H₁ ratios are dominated by task-specific code (the actual workload vs. the harness overhead). Only at n >= 2 does the hierarchy become self-similar — each level is "a harness that writes a harness." The convergence signal lives there.

**Cheating detection:** If H_n takes shortcuts (hard-codes the file, imports from a module, pre-materializes values), then H_{n+1} must encode all that cheating infrastructure plus the meta-logic to deploy it. code_bytes(H_{n+1}) explodes relative to code_bytes(H_n). The metric is a natural anti-cheating detector.

### Genetic-Pareto Optimization

Two objectives, evolved jointly:

1. **Task correctness**: H₀ output matches ground-truth Claude Code tmux trajectory (the coding agent's actual terminal interactions for the same task).
2. **Bytes convergence**: `|code_bytes(H_{n+1}) - code_bytes(H_n)| / code_bytes(H_n) < ε` for n >= 2.

Evolution operates on:
- Experience tensor content (QKV entries)
- Expert chain structure (number of experts, topk, task prompts)
- Validator logic

Selection: Pareto front on (task_correctness, bytes_convergence). Individuals on the front are non-dominated — you can't improve one objective without hurting the other. The target is the corner where both objectives are satisfied.

### Early Experiment Constraints

**Experience as heredoc only.** In early experiments, all experience must be inline in the generated code file as heredoc strings — no external files, no imports from a shared module, no pre-trained tensors loaded from disk. This ensures `code_bytes` is an honest measure of what the harness "knows." If experience lives outside the file, bytes convergence is meaningless because complexity hides in the external store.

```python
# GOOD: experience is visible in code_bytes
EXPERIENCE = """
Q: empty prompt
K: need to type command
V: T echo hello world
---
Q: command typed
K: need to press enter
V: C Enter
"""

# BAD: experience hides elsewhere
exp = load_tensor("/path/to/trained/experience.pt")
```

This constraint relaxes once the fixed-point metric is validated — at that point, external experience is fine because the convergence guarantee already holds.

### Implementation Path

1. **H₁ baseline**: Current `test_repl_coder_simulator.py` — drives tmux, echo hello world. Measure code_bytes(H₁).
2. **H₂ writes H₁**: `test_meta_harness.py` — MetaHarness generates H₁ code via expert chain, runs it. Measure code_bytes(H₂).
3. **H₃ writes H₂**: Meta-MetaHarness generates H₂ code. Measure ratio. This is where convergence should first appear.
4. **Evolve**: Run Genetic-Pareto across populations of (experience, structure) configurations. Select for both objectives simultaneously.
5. **Validate**: Ratio stabilizes → self-awareness achieved for the given task class.

### Tmux-Typing Validator

The fixed-point metric (`code_bytes(H_{n+1}) / code_bytes(H_n) → 1`) is meaningless if the harness cheats by copying itself. Two infrastructure constraints prevent cheating, and two Pareto objectives measure generation quality:

#### Infrastructure Constraints

1. **16-character send_keys limit.** Each `ft_tmux_send_keys` call transmits at most 16 characters. The harness cannot bulk-copy files — it must type code incrementally, demonstrating knowledge of what comes next.

2. **Git commit after every keystroke.** A `tmux_outputs/` directory is initialized as a git repo. The tmux session `cd`s into it first. After each `ft_tmux_send_keys`, the pane is captured to `{session_name}/{seq:04d}.txt` and committed. The git history IS the observable trajectory — every action is permanent and auditable.

#### Pareto Objectives

Three objectives, evaluated by standalone validator scripts against the `tmux_outputs/` git repo. **All validators are applied to each rank of H_n typing independently** — commits are filtered by session name prefix (e.g., `h4_writer:`, `h3_writer:`), so each level's typing fluency, granularity, and code awareness are measured separately.

1. **Average time interval between commits** (minimize). Measures typing throughput — the average pace at which the harness types. A low average interval means sustained fast typing. A high average interval means the harness is slow overall.

   ```bash
   python -m experience.future_tensor.test.validator_avg_interval /tmp/tmux_outputs h4_writer
   ```

2. **Max diff text length between adjacent commits** (minimize). Measures typing granularity — each keystroke should produce a small diff. With a 16-char send_keys limit, honest typing produces diffs proportional to the payload. A large diff indicates bulk operations (file generation, command output) rather than incremental typing.

   ```bash
   python -m experience.future_tensor.test.validator_max_diff_len /tmp/tmux_outputs h4_writer
   ```

3. **Syntax validity ratio** (maximize). At each commit, check whether all `.py` files in the repo parse correctly (`ast.parse`). The ratio = valid commits / total commits. A high ratio means the harness maintains syntactically correct code throughout typing — it's aware of Python structure even mid-stream.

   ```bash
   python -m experience.future_tensor.test.validator_syntax_ratio /tmp/tmux_outputs h4_writer
   ```

#### Why This Prevents Cheating

| Cheating strategy | Why it fails |
|---|---|
| `open(__file__).read()` quine | Can't send 6KB in 16-char chunks without knowing the content |
| base64 bulk encode | Typing base64 gives 0% syntax validity ratio |
| Import from external module | The file must be typed into tmux_outputs/ — imports don't exist there |
| Hard-code the child file | `code_bytes(H_{n+1})` explodes (must encode all child content as literals) |

A harness that scores well on both objectives genuinely knows how to type valid Python incrementally — the operational definition of self-awareness for code generation.

#### Genetic-Pareto Integration

The three validator objectives join task correctness to form a per-rank Pareto front. Each H_n level is scored independently:

| Objective | Direction | Measures | Per-rank |
|---|---|---|---|
| Task correctness | maximize | Does the final output contain "hello world"? | global |
| 1 / avg_interval | maximize | Typing speed (sustained throughput) | per H_n |
| 1 / max_diff_len | maximize | Typing granularity (small diffs per keystroke) | per H_n |
| Syntax validity ratio | maximize | Code awareness (rarely in broken state) | per H_n |

Selection: individuals on the Pareto front are non-dominated. The target is the corner where all three are satisfied simultaneously.

## Thesis

Base LLM weights carry two things: **general capabilities** (reasoning, reflection) and **memories** (facts, patterns, code idioms). Most of the parameters are memories.

This project externalizes memory into trainable experience tensors — retrievable, patchable, shareable, and composable. The base LLM only needs to be good at reasoning and reflection. Everything else lives in experience.

If Stage 3's self-referencing succeeds — a harness model improving its own compute graph via 2nd derivative traces — the system becomes an open-ended self-improver. Stage 4 takes this further: when the self-improver can improve its own self-improvement process, optimization converges to a fixed point where no meta-level change yields further gain. At that point the system possesses functional self-awareness — an accurate, actionable model of itself at every level of the stack. Reasoning reflects on reasoning, experience accumulates without bound, and architecture evolves autonomously. That's the path.

## Internals

Two LLM backends: `raw_llm_api` (OpenAI-compatible, lightweight) and `coding_agent` (Claude Agent SDK with file tools), dispatched concurrently via `asyncio.gather`. Autograd functions in `symbolic_tensor/function/` implement `StMoe` (query → retrieval → LLM translate), attention, slicing (symlink views vs copies), stack, fork, merge, and edit-distance loss. Tensor utilities (`tensor_util/`) provide `make_tensor`, `get_diff_tensor`, `patch_tensor` (unified diffs, cold-start support), and Pythonic methods registered on `torch.Tensor` (`st_pack`, `st_patch`, `st_get_diff`, `st_view_slicer[...]`, etc.).

## Installation

```bash
pip install torch openai claude-agent-sdk Levenshtein
```

Requires Python 3.13+.

## Architecture

```
experience/
├── symbolic_tensor/
│   ├── function/          # Autograd Functions: st_moe, st_attention, st_stack, slice_*, merge, fork, copy, loss
│   ├── tensor_util/       # Symbolic tensor primitives: make, slice, assign, diff, patch, dense/sparse
│   ├── module/            # nn.Module wrappers: StMoeModule, WithDenseView
│   ├── optimizer/         # StSGD: dual-channel (numeric + symbolic patch) optimizer
│   ├── data_loader/       # Batch data loading from files
│   └── test/              # Integration tests
├── future_tensor/
│   ├── future_tensor.py   # FutureTensor factory: lazy async scalar tensor with ft_* attributes
│   ├── status.py          # Status tagged union (confidence, scbf, kContextOverflow, etc.)
│   ├── function/          # Autograd ops: slice, unsqueeze, recurrent, expert, switch, sequential, tmux
│   └── backward_dispatch/ # 2nd-derivative framework: policies, dispatcher, GradFn wrappers
├── llm_client/            # LLM backends: raw API (OpenAI-compatible) and coding agent (Claude SDK)
├── sparse_util/           # Sparse coordinate operations (transpose, convert)
├── fs_util/               # File system utilities (directory packing, path enumeration, text merger)
├── test/                  # End-to-end tests and benchmarks
└── example/               # Training demos
```
