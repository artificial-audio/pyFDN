# Recursive DSP (modular, PyTorch)

This folder implements a small **block-based, modular framework for recursive DSP systems** (FDN-like structures) where signals and state are represented as **PyTorch tensors**.

The package entrypoint, [`__init__.py`](__init__.py), mainly **exposes** the building blocks. The implementation lives in the stage modules listed below.

---

## Design goals

- **Modular pipeline**: compose a recursive system as an ordered list of stages.
- **Block processing**: operate on blocks `T = block_size` for efficiency.
- **Torch-native**: `torch.Tensor` everywhere (CPU/GPU), so you can use autograd, batching, and `torch.profiler`.
- **Explicit state**: persistent recursion state is stored in a global dict of tensors and updated once per block.

---

## Tensor conventions

Stages use consistent shapes:

- `x_block`: external input block, shape **`[B, N_in, T]`**
- `lines`: internal feedback-line block (the “loop signal”), shape **`[B, N, T]`**
- `y_block`: output block, shape **`[B, N_out, T]`**

Dimension meaning:

- `B`: batch size (parallel signals)
- `N_in`: number of input channels
- `N`: number of feedback lines (a.k.a. delay lines)
- `N_out`: number of output channels
- `T`: samples per processing block (always `block_size` inside the core)

The coordinator accepts input as:

- `[T_total, N_in]` (legacy / convenience), or
- `[B, N_in, T_total]`

and returns the matching output format (see [`RecursionCore.process`](core.py)).

---

## Core abstractions

### 1) `Stage`: the module interface

Defined in [`stage.py`](stage.py).

Every processing unit is a `Stage` with two methods:

- `init_state(batch_size, block_size, device) -> Dict[str, Tensor]`
  - allocates the stage’s **persistent** tensors and returns them in a dict
- `step_block(lines, state_t, next_state, block_size, x_block=None) -> (new_lines, y_block?)`
  - processes exactly one block and returns:
    - the updated `lines` tensor for downstream stages
    - optionally an output `y_block` (typically only the output stage produces it)

**State protocol**

- `state_t` is the **read-only** global state at the start of the block.
- `next_state` is a dict that accumulates **updated state tensors** for the next block.
- A stage should **not mutate** tensors inside `state_t`. Instead, clone, update, and write into `next_state`.
- `Stage.state_keys` documents which keys a stage “owns”, but this is not enforced at runtime—treat it as a contract.

Important: the coordinator only merges keys that already exist in `state` (i.e., were created by `init_state`). If a stage writes a brand-new key into `next_state` that wasn’t initialized, it will not be carried forward.

### 2) `RecursionCore`: the coordinator

Defined in [`core.py`](core.py).

`RecursionCore` orchestrates a list of stages over an input signal:

1. **Normalize input shape** to `[B, N_in, T_total]` and move it to `device`
2. **Initialize global state** by merging `Stage.init_state(...)` outputs
3. Split input into blocks; for each block:
   - extract `x_block`
   - **pad the final block** with zeros to exactly `block_size` (the core always calls stages with `T = block_size`)
   - reset block-local `lines=None` (so block-local tensors never “leak” across blocks)
   - run all stages in order:
     - each stage transforms `lines` and optionally emits `y_block`
   - merge `next_state` into the global `state`
4. Concatenate `y_block` outputs across blocks and trim padding back to `T_total`

This yields a clean separation:

- **Within a block**: computation flows through stages via `lines`
- **Across blocks**: recursion happens only through the persistent `state` dict

---

## Provided stages (what `__init__.py` exports)

### Delay lines: stateful recursion memory

Defined in [`delay_lines.py`](delay_lines.py).

#### `DelayRead` + `DelayWrite` (two-stage delay bank)

- `DelayRead` reads from circular buffers in `state` and *produces* `lines` for the current block.
- `DelayWrite` writes the processed `lines` block back into the same circular buffers and advances the pointer.

State keys (per `state_key` prefix):

- `"{state_key}_buffers"`: `[B, N, L]`
- `"{state_key}_pointer"`: `[B, N]`

Accuracy note:

- With the split `DelayRead`/`DelayWrite` approach, the newly written samples are not visible to `DelayRead` until the *next* block.
- This is correct when the **minimum delay length is at least `block_size`**.
- If you need sample-accurate behavior for delays **smaller than `block_size`**, use `DiagonalDelay` (below) or reduce `block_size`.

#### `DiagonalDelay` (combined read+write)

`DiagonalDelay` combines write-then-read inside a single stage, so within-block reads can “see” within-block writes. This is useful for cascaded filter-feedback-matrix (FFM) constructions and small delays.

`Delay` is currently an alias for `DiagonalDelay` for backwards compatibility.

### Feedback mixing: matrix operations on the loop signal

Defined in [`feedback_mix.py`](feedback_mix.py).

`FeedbackMix` applies a mixing operator across the **line dimension** `N`:

- `"dense"`: `lines @ Aᵀ` via `torch.einsum`
- `"hadamard"`: fast Walsh–Hadamard transform (requires `N` power of two)
- `"householder"`: Householder reflection defined by a vector

This stage is stateless; it only transforms `lines`.

### Filtering: per-line IIR biquads

Defined in [`biquads.py`](biquads.py).

`Biquads` applies a parallel bank of (optionally cascaded) DF2T biquad sections to each feedback line independently and stores its persistent IIR state:

- `biquad_state`: shape `[B, N, num_sections, 2]` (the two DF2T delay elements per section)

It supports two internal implementations:

- `use_vectorized=False` (default): per-sample Python loop over `t` inside the block
- `use_vectorized=True`: block vectorized path using `torch.linalg.eig` + `conv1d` (no Python loop over time)

### Input and output taps

Defined in [`input_tap.py`](input_tap.py) and [`output_tap.py`](output_tap.py).

- `InputTap` computes an injection from the external input and **adds it into `lines`**:
  - `inject = x_block @ Bᵀ`
  - `lines = (lines or 0) + inject`
  - Practical implication: place `InputTap` **after** something that creates `lines` (typically `DelayRead`) if you want it added to the delayed signal for that block.
- `OutputTap` produces the final `y_block` from `lines` and (optionally) a direct path from `x_block`:
  - `y = lines @ Cᵀ [+ x_block @ Dᵀ]`

### FFM helper: building cascaded structures

Defined in [`ffm_builders.py`](ffm_builders.py).

`build_ffm_stages(...)` builds a stage list for a cascaded filter-feedback-matrix (FFM) factorization:

- starts with `FeedbackMix(U0)`
- then repeats `DiagonalDelay(m_k)` + `FeedbackMix(U_k)` for each delay/mix pair

---

## Putting it together: a typical pipeline

A common feedback-delay-network style layout:

```python
from pyFDN.recursive import (
    RecursionCore, DelayRead, InputTap, FeedbackMix, Biquads, DelayWrite, OutputTap
)

stages = [
    DelayRead(delay_lengths=[81, 100, 121, 169]),
    InputTap(input_matrix=B),          # add input into the loop signal
    FeedbackMix(feedback_matrix=A),    # mix across lines
    Biquads(num_lines=4),              # per-line filtering
    DelayWrite(),                      # write back to delay buffers
    OutputTap(output_matrix=C),        # produce y_block
]

core = RecursionCore(stages, block_size=512, device=torch.device("cpu"))
y = core.process(x)  # x: [T_total, N_in] or [B, N_in, T_total]
```

Notes:

- You can reorder/insert stages as long as each stage’s expectations about `lines` and `x_block` are satisfied.
- `OutputTap` must appear somewhere in the list; otherwise `RecursionCore.process` raises an error when no `y_block` is produced.

---

## Profiling and analytical cost

This folder includes two complementary performance tools:

- **Profiler capture** in [`core.py`](core.py):
  - call `core.process(x, profile=True)` to capture a `torch.profiler` trace and aggregate per-stage buckets (see [`profile.py`](profile.py))
  - `core.last_profile_report` stores the last structured report
- **Analytical cost model** in [`cost.py`](cost.py):
  - `estimate_cost_from_shape(...)` and `estimate_cost_from_input(...)` compute FLOPs/bytes estimates per stage and for the whole graph
  - `derive_metrics(...)` converts a cost + runtime into GFLOP/s and GB/s metrics

The profile report can optionally attach analytical per-stage estimates (see `ProcessProfileConfig(include_analytical=True)`).

---

## Extending the system: writing a custom stage

Minimal template:

```python
import torch
from pyFDN.recursive.stage import Stage

class MyStage(Stage):
    def __init__(self):
        super().__init__(state_keys={"my_state"})

    def init_state(self, batch_size: int, block_size: int, device: torch.device):
        return {"my_state": torch.zeros(batch_size, device=device)}

    def step_block(self, lines, state_t, next_state, block_size: int, x_block=None):
        s = state_t["my_state"]
        # ... compute new_lines ...
        next_state["my_state"] = s + 1  # write updated state
        return lines, None
```

Guidelines:

- Keep `lines` shape `[B, N, T]` and `x_block` shape `[B, N_in, T]` consistent.
- Clone before in-place ops if the tensor came from `state_t` (so you don’t mutate the “read-only” view).
- Initialize every persistent key in `init_state`, and only update existing keys in `next_state`.
