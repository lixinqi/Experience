# Experience Framework: A Technical Overview

This document is for developers who want to understand `harness_model`. It walks through the full stack from `symbolic_tensor` at the bottom to `harness_model` at the top, explaining both the *why* and the *how* of each layer.

---

## 1. SymbolicTensor: The Filesystem as Tensor Storage

### 1.1 Core Idea

`SymbolicTensor` is a PyTorch extension where each tensor element is **a text file on disk** rather than a float. This means:

- Elements can hold arbitrary text: Python code, natural language, Viba DSL, unified diffs, etc.
- The tensor still maintains **numeric coefficients** (bfloat16) that are fully compatible with standard autograd.
- Tensor operations (slice, stack, merge, etc.) manipulate **file paths and symlinks**, not in-memory values.

```
{relative_to}/{tensor_uid}/
├── shape                    # JSON: [2, 3]
├── storage/
│   ├── 0/data               # content at flat index 0
│   ├── 1/data               # content at flat index 1
│   └── 1/1/data             # content at flat index 11
```

### 1.2 Key Fields

A `SymbolicTensor` is a regular `torch.Tensor` monkey-patched with custom attributes:

| Field | Type | Description |
|---|---|---|
| `st_relative_to` | `str` | Storage root directory. All tensor files live under this path. |
| `st_tensor_uid` | `str` | Unique identifier (UUID). `{st_relative_to}/{st_tensor_uid}/` is the full storage path. |
| `data` | `torch.Tensor` | Numeric coefficients (bfloat16), fully compatible with autograd. |
| `shape` | `torch.Size` | Tensor shape. Each element corresponds to one file on disk. |

Each element's on-disk path follows this rule:

```
{st_relative_to}/{st_tensor_uid}/storage/{flat_index}/data
```

The flat index digits are split into single-level subdirectories. For example, flat index 11 → `storage/1/1/data`; flat index 123 → `storage/1/2/3/data`.

`register_tensor_ops.py` registers symbolic operation methods on `torch.Tensor`:

| Method | Purpose |
|---|---|
| `st_assign(rvalue)` | Copy symbolic content from rvalue into this tensor |
| `st_assign_view(rvalue)` | Create a view via symlinks (shared storage) |
| `st_get_diff(rvalue)` | Compute unified diff between this tensor and rvalue |
| `st_patch(rvalue)` | Apply diffs in rvalue to this tensor |
| `st_view_slicer[...]` | Pythonic slicing returning a symlink view |
| `st_value_slicer[...]` | Pythonic slicing returning an independent copy |

### 1.3 Dual-Channel Gradient System

Gradient propagation in SymbolicTensor has two channels:

| Channel | Carries | Computed By |
|---|---|---|
| **Numeric** (coefficient) | Float values (bfloat16) | Standard autograd / SGD arithmetic |
| **Symbolic** | Unified diff text | LLM computes `diff -u`, stored as files |

During the forward pass, the LLM reads input files and writes output files. During the backward pass, the LLM computes "how to improve the output," producing a diff. The `StSGD` optimizer then applies that diff via `patch` to update the parameters (i.e., the experience).

### 1.4 ExperienceTensor

`ExperienceTensor` is a `[N, 3]` SymbolicTensor where each row is a `(query, key, value)` triple:

- **Query** (index 0): semantic keywords, used for Jaccard-similarity retrieval
- **Key** (index 1): source-domain content (e.g., Python code)
- **Value** (index 2): target-domain content (e.g., Viba code)

This is the model's **learnable weight** — empty at the start of training, incrementally updated each round via patch. The model doesn't "memorize" training data; it **accumulates experience at runtime**.

### 1.5 Autograd Functions

Core SymbolicTensor operations are wrapped as `torch.autograd.Function` subclasses:

| Function | Purpose |
|---|---|
| `StMoe` | Mixture-of-experts: generate query → retrieve top-k → LLM translate → write back |
| `st_attention` | Attention (slice_attention + merge) |
| `st_stack` | Stack symbolic tensors along a new axis |
| `slice_view` | Create a view via symlinks (shared storage) |
| `slice_tensor` | Create an independent copy |
| `GetEditDistanceRatio` | Loss function: Levenshtein edit-distance ratio |

---

## 2. FutureTensor: Scalar Reference to Lazy Async Storage

### 2.1 Why FutureTensor?

SymbolicTensor's forward pass is **synchronous and blocking**: every autograd Function calls the LLM inline and waits for the result. This works fine in a training loop, but is too rigid for **agent orchestration**:

- Agents need **multi-round retry** (generate → validate → fail → retry).
- Each agent step should execute **asynchronously**, with retry counts **encoded into tensor dimensions**.
- During backprop, the LLM needs to **concurrently reflect** on all failed attempts.

FutureTensor solves these problems.

### 2.2 Definition

`FutureTensor` is an **interface** — a plain `torch.Tensor` of shape `()`, dtype `bfloat16`,
scalar value always `1`, monkey-patched with `ft_*` attributes. It is **not** a subclass of
`SymbolicTensor`.

```
FutureTensor :=
    torch.Tensor[(), bfloat16, value=1]           # scalar reference — not a SymbolicTensor
    * ft_initial_static_tensor        SymbolicTensor       # base storage; make_none_tensor before materialization,
                                                   # zero-sized tensor in dynamic cases
    * ft_incremental_concated_tensors              # list[(SymbolicTensor, concat_axis int)]
                              list[(SymbolicTensor, int)]
                                                   # appended in dynamic cases
    * ft_shape_schema         list[sympy.Symbol]   # declared shape schema (may be symbolic)
    * ft_capacity_shape       list[int]            # concrete shape; always equals
                                                   # concat(ft_initial_static_tensor, *ft_incremental_concated_tensors).shape
    * ft_forwarded            bool
    * ft_forward              void <- prompt FutureTensor
    * ft_async_get            (str, Status) <- coordinates list[int] <- prompt str
    * ft_get_materialized_value   (float, str) <- coordinates list[int]
    * ft_reset_materialized_value void <- coordinates list[int]
                                       <- coefficient float
                                       <- filepath str
                                       <- symlink bool = False
```

The **logical view tensor** is always:
```
concat(ft_initial_static_tensor, *ft_incremental_concated_tensors)
```

`Status` and coefficient semantics live inside the `SymbolicTensor`s of the logical view.
The scalar `1` on the `FutureTensor` handle itself carries no semantic meaning.

Key methods:

| Method | Purpose |
|---|---|
| `ft_forward(prompt)` | Materialize — fires all `ft_async_get` calls concurrently |
| `ft_async_get(coordinates, prompt)` | Async element generator: returns `(content, Status)` |
| `ft_get_materialized_value(coordinates)` | Returns `(coefficient, file_path)` at coordinates |
| `ft_reset_materialized_value(...)` | Overwrites an element by copy or symlink |

### 2.3 Forward: Async + Loops

FutureTensor's forward has two key properties:

**1) Async concurrency**

`ft_forward` uses `asyncio.gather` to run all `ft_async_get` calls concurrently:

```python
async def _gather():
    tasks = [self.ft_async_get(coords, prompt) for coords, prompt in ...]
    return await asyncio.gather(*tasks)
```

A logical view of capacity `[batch=8, retries=5]` can fire all 40 LLM requests simultaneously.

**2) Loops encoded as dimensions — because LLMs are probabilistic**

The design of `ft_recurrent` starts from a core insight: **LLM APIs are not deterministic matrix multiplications; they are probabilistic compute models**. Any call may produce errors, hallucinations, or malformed output. A mechanism is therefore needed to:

- **Retry with confirmation**: generate → validate → on failure, retry.
- **Filter and select**: among multiple attempts, pick the highest-confidence success, or fall back to the best failure.
- **Accumulate context**: retain prior failure traces as input to the next retry so the LLM knows "what went wrong last time."

`ft_recurrent` encodes this generate-validate-retry **for loop into the last dimension of `ft_capacity_shape`**. Input capacity is `(*prefix_dims, recurrent_dim)`; output capacity is `(*prefix_dims)`.

For each prefix coordinate, it iterates `i in range(recurrent_dim)`:
- Call `input.ft_async_get([*prefix, i], prompt_i)`
- Receive `(output_i, status_i)`
- If `status.is_confidence`: return immediately (success)
- If failed: append `output_i` to the prompt and continue

```
input  ft_capacity_shape=[batch, retries]  --ft_recurrent-->  output ft_capacity_shape=[batch]
                                                 prompt_tensor ft_capacity_shape=[batch, retries]
```

The `accumulate_output` parameter allows outputs to be accumulated across iterations (e.g., concatenating each round's read results into a growing context).

### 2.4 Backward: Async + Concurrent

`ft_recurrent` is a `torch.autograd.Function`. During the backward pass:

- **`ft_initial_static_tensor` is already materialized** (since `ft_forward` has completed).
- The backward directly manipulates `ft_initial_static_tensor`'s storage files.
- For all elements needing reflection, it **concurrently constructs AgentTasks** and submits them to the LLM via `TaskHandler()` in a single batch.
- The LLM writes improved content for each element; the framework computes the `diff` and stores it as the gradient.

This is the essence of **"forward is lazy sequential retry; backward is eager concurrent reflection."**

### 2.5 Status: Control Flow in the Logical View

`Status` encodes success/failure/retry as numeric coefficients **inside `ft_initial_static_tensor`
and `ft_incremental_concated_tensors`** — not on the `FutureTensor` scalar itself (which is always `1`):

| Status | Meaning | Stored as float |
|---|---|---|
| `confidence(v)` | Success, confidence v | `+v` |
| `self_confidence_but_failed(v)` | Failure, self-assessed confidence v | `-v` |
| `kConfidenceNotBounded` | Confidence out of bounds | `-2.0` |
| `kContextOverflow` | Context too long | `-3.0` |

`ft_recurrent` reads these coefficients from the logical view to determine success vs. failure and whether to exit the loop early.

---

## 3. Future Ops: How Loops and Traces Are Encoded

### 3.1 `ft_recurrent`: For Loop = Last Dimension

```python
# Input capacity  [batch, retries]
# Meaning: batch samples, each with up to retries attempts
ft_input = FutureTensor(tmpdir, ft_async_get, ft_shape_schema=[batch_sym, retries_sym])
output, prompt_tensor = ft_recurrent(ft_input, accumulate_output=concat_fn)
# output ft_capacity_shape       [batch]
# prompt_tensor ft_capacity_shape [batch, retries]
```

The loop is fully encoded in the capacity dimension:
- **Loop variable `i`** = coordinate along the last dimension.
- **Loop count** = last element of `ft_capacity_shape`.
- **Trace (accumulated prompt history)** = files in `prompt_tensor.ft_initial_static_tensor` at each `[*prefix, i]`.

The signature `ft_async_get(coordinates, prompt)` carries the trace:
- `coordinates` tells you which retry round you're on.
- `prompt` is the concatenated output of all prior rounds.

**Sparse support for variable-length loops**: if a round returns `status=confidence`, `ft_recurrent` exits early. Later dimension slots are never accessed (their coefficients remain 0 and their content remains TODO).

### 3.2 `ft_moe`: MoE Encoded as Element-Level Async Calls

```python
ft_input = FutureTensor(tmpdir, ft_async_get, ft_shape_schema=[batch_sym])
output, prompt_tensor, indexes = ft_moe(ft_input, experience_tensor, topk=16)
# output ft_capacity_shape [batch]
```

Each element's `ft_async_get` runs the full MoE pipeline:
1. Generate a query from the input content
2. Retrieve top-k entries from experience
3. Build a workspace and call the LLM
4. Return the translation result

Here **the batch dimension is the for loop**: 8 batch elements = 8 independent MoE calls, all executed concurrently by `ft_forward`.

### 3.3 `ft_unary`: Pure Functional Mapping

```python
def ft_unary(input_ft, fn) -> FutureTensor:
    async def wrapped(coords, prompt):
        output, status = await input_ft.ft_async_get(coords, prompt)
        return fn(coords, prompt, output, status)
    return FutureTensor(
        input_ft.ft_initial_static_tensor.st_relative_to,
        wrapped,
        ft_shape_schema=input_ft.ft_shape_schema,
    )
```

`ft_unary` preserves `ft_capacity_shape` and only transforms each element's `(output, status)`. It is the **functional composition primitive** for FutureTensors.

### 3.4 Summary: Dimensions as Control Flow

| Concept | Traditional code | FutureTensor encoding |
|---|---|---|
| For loop | `for i in range(R)` | Last dim of `ft_capacity_shape` = `R` |
| Nested loops | Outer `for` + inner `for` | `ft_capacity_shape = [..., C, R]` |
| Loop variable `i` | Local variable | `coordinates[-1]` |
| Trace / history | String concatenation | Files in `prompt_tensor.ft_initial_static_tensor` |
| Early exit | `break` | `status=confidence` |
| Retry on failure | `continue` | `status=scbf` + prompt accumulation |

---

## 4. Column-Wise Execution of Stacked Future Ops

### 4.1 Symbolic Ops Execute Row-Wise

SymbolicTensor autograd Functions (e.g., `StMoe`) execute **synchronously, row-wise**:

```python
# Inside st_moe_forward
for each_batch_element:
    query = llm_generate_query(input[b])
    topk = select_experience(query)
    workspace = build_workspace(topk)
    output[b] = llm_translate(workspace)   # blocks until LLM returns
```

A single Function call iterates over the batch with a Python `for` loop. Each element is processed **serially** (though `TaskHandler` can parallelize requests within the batch, the Function's semantics are "compute the entire tensor in one shot").

Row-wise execution characteristics:
- **The same operator finishes all batch elements** before moving to the next operator.
- Cannot express "element 0 runs through A→B→C, then element 1 runs through A→B→C" — no **cross-layer column-wise pipelining**.
- Retry logic must be written inside the Function; it cannot be expressed as a tensor dimension.

### 4.2 Stacked Future Ops Execute Column-Wise

When multiple FutureTensors are composed, `ft_forward`'s materialization mechanism produces a **column-wise** execution order.

Consider the Stage 1 pipeline in `harness_model`:

```
ft_raw        ft_capacity_shape=[batch, C, R]
  → ft_unary(validate)          [batch, C, R]
  → ft_recurrent(inner)         [batch, C]     # inner: retry until valid tool result
  → ft_unary(sufficiency)       [batch, C]
  → ft_recurrent(outer)         [batch]        # outer: accumulate context
```

When `context_ft.ft_forward(prompt)` is called, `context_ft.ft_capacity_shape = [batch]`. For each `b`, the call doesn't trigger a single LLM call — it triggers **an entire nested future op chain**.

Actual execution order (for batch element 0):

```
# outer recurrent processes b=0, iterates c=0,1,2,...
outer[0]:
  c=0:
    # inner recurrent processes (b=0,c=0), iterates r=0,1,2,...
    inner[0,0]:
      r=0: ft_unary(validate) -> ft_raw.ft_async_get([0,0,0]) -> LLM(bootstrap read)
             ↓ returns (trace, scbf)
      r=1: ft_unary(validate) -> ft_raw.ft_async_get([0,0,1]) -> LLM(grep)
             ↓ returns (trace, scbf)
      ...
      r=k: inner returns confidence -> exit, output accumulator
    ↓
    ft_unary(sufficiency) checks (prompt + accumulator)
    -> returns confidence (context is sufficient)
  ↓
  outer receives confidence, writes to accumulate_output
  -> outer exits, b=0 complete

# then process b=1
outer[1]: ...
```

The execution order is **not**:
```
❌ compute all ft_raw elements → compute all ft_unary elements → compute all ft_recurrent elements
```

It is:
```
✅ (ft_raw[0,0,0] → ft_unary[0,0,0] → ... → inner[0,0] → sufficiency[0,0] → outer[0])
   then move to the next batch element.
```

**This is column-wise execution**: for each "column" (outer coordinate `b`), run the entire chain `fop0[b] → fop1[b] → ... → fopn[b]`, get the final result, then move to column `b+1`.

### 4.3 Why Column-Wise Execution Matters for Agent Orchestration

**1) Agents have internal state (trace / accumulated prompt)**

Each agent step depends on the traces from all prior steps. In `harness_model`:
- The inner recurrent's `prompt_tensor.ft_initial_static_tensor` records why `r=0` failed, feeding it into `r=1`.
- The outer recurrent's `accumulate_output` records the `c=0` read result as the context baseline for `c=1`.

**2) Subagents must complete inside the parent agent**

```
Agent (outer recurrent) decides "what context to gather next"
  -> Launches Subagent (inner recurrent) to execute the tool call
     -> Subagent retries internally until a valid result is obtained
     -> Subagent returns result to the Agent
  -> Agent runs the context validator to check sufficiency
  -> Insufficient? Agent enters round c+1, launches another Subagent
```

This nesting is **naturally encoded by dimensions**:
- Agent step = outer recurrent coordinate `c`
- Subagent retry = inner recurrent coordinate `r`
- Subagent trace = files in inner `prompt_tensor.ft_initial_static_tensor`
- Agent trace = accumulated result in outer `accumulate_output`

**3) `ft_forward` provides unified materialization scheduling**

The framework only needs to iterate over `context_ft.ft_capacity_shape` coordinates, calling `ft_async_get` for each. The nested chain of ops resolves recursively and lazily — no explicit orchestration code required.

### 4.4 Comparison Summary

| Property | Symbolic Ops (row-wise) | Stacked Future Ops (column-wise) |
|---|---|---|
| Execution granularity | One layer finishes all batch elements | One batch element runs all layers |
| Loop expression | Python `for` / `while` | Last dim of `ft_capacity_shape` |
| Trace passing | Explicit args or global state | Files in `prompt_tensor.ft_initial_static_tensor` / `accumulate_output` |
| Retry mechanism | Hardcoded inside Function | Handled automatically by `ft_recurrent` |
| Agent / Subagent | Hard to express structurally | Outer / inner recurrent map naturally |
| Scheduling | Synchronous blocking | `ft_forward` async concurrent scheduling |

---

## 5. HarnessModel: Simulating Claude Code with FutureTensor

### 5.1 Goal

`harness_model` is a **pure-framework simulation of Claude Code**:
- No calls to the Claude Code CLI.
- The LLM acquires codebase context only through explicit tool calls (read/grep/glob).
- All execution traces are stored in the experience tensor for future learning.

### 5.2 Two-Stage Architecture

```
worktree_tensor [batch]
    │
    ▼
Stage 1: code_context_gather
    ft_raw        ft_capacity_shape=[batch, C, R]
        → ft_unary(validate_tool_result)
        → ft_recurrent(inner)  [batch, C]      # inner: retry until valid tool result
        → ft_unary(check_context_sufficiency)
        → ft_recurrent(outer, accumulate_output) [batch]  # outer: accumulate clean context
    │
    ▼
context_tensor [batch]   (= context_ft.ft_initial_static_tensor)
    │
    ▼
Stage 2: code_gen
    ft_gen        ft_capacity_shape=[batch, L]
        → ft_recurrent         [batch]          # generate → validate syntax → retry
    │
    ▼
output_tensor [batch]   (= output_ft.ft_initial_static_tensor)
```

- **C = `max_context_collects`**: maximum rounds of context gathering.
- **R = `max_tool_call_retries`**: maximum retries per round (in case the LLM outputs an invalid tool name).
- **L = `max_codegen_steps`**: maximum retries for code generation (on syntax validation failure).

### 5.3 Key Design Decisions

**1) Bootstrap read**

The first iteration (`c=0, r=0`) bypasses the LLM entirely — it directly force-reads the target file at `offset=0, limit=200`. This is the single biggest performance win. Without it, the LLM wastes retries on tiny offset/limit windows that miss the mask region, causing loss to spike from ~0.2 to ~0.7+.

**2) Clean accumulator**

`accumulate_output=_concat_context` keeps only `[read(...)]` results and discards `[grep(...)]` and `[glob(...)]` noise. The execution trace goes into the experience tensor; only the clean accumulator feeds into generation.

**3) Syntax validation with indentation search**

After code generation, the candidate snippet is inserted at the actual masked position in the file and `ast.parse()` is run. On failure, indentations of 0, 2, 4, 6, 8, 10, and 12 spaces are tried before re-parsing. This handles cases where the snippet is syntactically valid in isolation but uses the wrong relative indentation.

**4) Deterministic context validator**

Rather than asking the LLM to self-report "I have enough context," a **deterministic validator** checks:
- Has the target file been read at least once?
- Does the read result include at least 2 lines of code before and after the mask region?

Only when both conditions are met does the validator emit `confidence`; otherwise context gathering continues.

### 5.4 Pipeline-Style Forward Pass

```python
def forward(self, worktree_tensor):
    # Stage 1
    ft_raw = FutureTensor(tmpdir, self._make_tool_use(worktree_tensor),
                          ft_shape_schema=[batch_sym, C_sym, R_sym])
    ft_validated = ft_unary(ft_raw, self._validate_tool_result)
    ft_tool_result, _ = ft_recurrent(ft_validated)           # inner recurrent
    ft_checked = ft_unary(ft_tool_result, self._check_context_sufficiency)
    context_ft, _ = ft_recurrent(ft_checked, accumulate_output=_concat_context)  # outer recurrent

    # Materialize
    context_ft.ft_forward(prompt_ft)
    context_tensor = context_ft.ft_initial_static_tensor

    # Stage 2
    ft_gen = FutureTensor(tmpdir, self._make_code_gen(context_tensor, worktree_tensor),
                          ft_shape_schema=[batch_sym, L_sym])
    output_ft, _ = ft_recurrent(ft_gen)
    output_ft.ft_forward(prompt_ft)
    return output_ft.ft_initial_static_tensor
```

The entire forward is a **declarative dataflow**: FutureTensor → ft_unary → ft_recurrent → ... → SymbolicTensor (via `.ft_initial_static_tensor`).

### 5.5 Validation Results

On a mask-recovery task at `symbolic_tensor/tensor_util/assign_tensor.py:13-18`:
- Loss: **0.3164** (seed 42, deepseek-chat)
- Stage 1: bootstrap read → grep related files → context sufficient
- Stage 2: one generation pass through AST syntax check

---

## 6. Summary

| Layer | Core Abstraction | Key Properties |
|---|---|---|
| **SymbolicTensor** | Filesystem as tensor storage | Dual-channel gradients, patch-based optimization, ExperienceTensor |
| **FutureTensor** | Scalar reference to lazy async storage | Scalar `()` bfloat16 handle; logical view via `ft_initial_static_tensor` + incremental tensors; `ft_capacity_shape` tracks concrete shape |
| **Future Ops** | Dimensions as control flow | For loop = last dim of `ft_capacity_shape`, trace = `prompt_tensor.ft_initial_static_tensor`, nested recurrent |
| **HarnessModel** | Agent orchestration | Two-stage pipeline, bootstrap read, clean accumulator, syntax validation |

The core philosophy of this framework is: **encode an agent's loops, branches, and traces entirely into tensor shapes and storage** rather than hiding them in Python control flow. This makes agent behavior schedulable by the framework, traceable by autograd, and learnable by the optimizer.
