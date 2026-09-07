# `pyfdn-system`: one JSON document for every pyFDN backend

**Status:** proposal, not implemented
**Date:** 2026-09-07
**Branch:** `docs/system-json-format`
**Supersedes:** nothing. `pyfdn-fdn-build` v2 stays valid and unchanged.

---

## 1. The problem

pyFDN can render the same feedback delay network three ways:

| backend | entry point | domain | covers |
|---|---|---|---|
| `process_fdn` | `pyFDN.process_fdn` | NumPy, block-processed time domain | vanilla FDN with three filter hooks |
| `td` graph | `pyFDN.td.{Series,Parallel,Recursion}` + operators | NumPy, block-processed time domain | arbitrary topology, including time-varying and nonlinear |
| FLAMO | `pyFDN.dss_to_flamo`, `pyFDN.build_to_flamo` | torch, FFT domain | arbitrary topology, differentiable |

`process_fdn` is not really a fourth thing: it drives `td.RecursionState` and
`td.MatrixFIR` under the hood, so it is the vanilla special case of the `td`
graph with a hand-written loop.

Exactly one of those three is saveable today, and only for vanilla networks.
Everything that is not a vanilla FDN — a Kronecker or otherwise time-varying
feedback matrix, a shimmer FDN, an SDN, a reverberation-enhancement system, a
coupled-room model — exists only as live Python objects. It cannot be written
to disk, handed to a collaborator, cited by a paper, replayed a year later, or
compiled to FAUST.

The ask is a settings format that carries **both** the baked parameters (what a
renderer needs) and the meta-parameters (what a *re*-generator or re-trainer
needs), and that is reusable across all three backends plus downstream
consumers.

## 2. What exists today

### 2.1 In pyFDN

- **`src/pyFDN/build.py`** — `FDNBuild` and the `pyfdn-fdn-build` v2 JSON
  format: `feedback_matrix`, `input_matrix`, `output_matrix`,
  `direct_matrix`, `delays`, `sample_rate`, and the three optional SOS hooks
  `post_delay`, `post_matrix`, `post_output`. Validated on both write and
  read (`fdn_build_to_dict` deliberately round-trips through
  `fdn_build_from_dict` before returning).
- **`src/pyFDN/preset.py`** — `FDNPreset` = `metadata` + `design` + `build`.
  `design` is already the meta-parameter layer, and is validated against the
  real generator vocabularies (`FEEDBACK_MATRIX_TYPES`, `IO_MATRIX_TYPES`,
  `DELAY_DISTRIBUTIONS`, `EQ_DESIGNS`).
- **`src/pyFDN/auxiliary/flamo_graph.py`** — `flamo_model_to_nodes` walks a
  FLAMO model into a node tree; `extract_build` bakes such a tree down to an
  `FDNBuild`, refusing (rather than silently dropping) any hook that does not
  bake to one SOS bank.
- **`src/pyFDN/auxiliary/flamo.py`** — `assemble_fdn_core` builds the graph in
  the other direction, with the node names `extract_build` keys off.
- **`src/pyFDN/td/`** — no serialisation of any kind.

So the *shape* of the answer already exists: baked parameters separated from
design information, with a document wrapper carrying metadata. It only covers
one topology.

### 2.2 In adac

[`adac`](https://github.com/cucuwritescode/adac) (local checkout: `rt-fdn`)
compiles a FLAMO model to FAUST through a JSON intermediate representation.
Relevant prior art:

- `adac/codegen/flamo_to_json.py` is a fork of pyFDN's `flamo_model_to_nodes`
  with parameter extraction added. adac does **not** depend on pyFDN — its
  core is numpy-only — so the traversal is duplicated.
- Its IR has no `format` or `version` field.
- Node types are FLAMO class names (`parallelDelay`, `parallelSOSFilter`),
  including a `SOSFilter` → `parallelSOSFilter` alias to absorb FLAMO version
  drift. A library implementation detail is therefore part of the wire format.
- Two things it gets right and this proposal keeps: **effective vs. raw
  parameters** (serialise `map(param)`, keep the trainable weights separately
  under `flamo.param_raw`) and per-node constructor metadata for
  reconstruction.
- `json_to_faust` already dispatches on types `flamo_to_json` never emits —
  `variableDelay`, `fractionalDelay`, `onePole`, `allpassComb`, `dcBlocker`.
  The FAUST side is *already* speaking a broader, backend-neutral vocabulary.
  It has simply never been named or written down.

## 3. Non-goals

- Replacing `pyfdn-fdn-build`. It stays, byte-compatible, forever.
- A general dataflow IR. This describes the systems pyFDN builds, not
  arbitrary DSP graphs.
- Model weights for training checkpoints. A document describes a *system*;
  optimiser state belongs in a torch checkpoint.
- A binary format. Human-readable and diffable is a feature; a network with a
  100 000-tap FIR matrix is a legitimate reason to reach for `.npz` instead.

## 4. Design

### 4.1 Two tiers, both permanent

| tier | used when | read by |
|---|---|---|
| `build` (exists) | the system is a vanilla FDN | `process_fdn`, `build_to_flamo`, `build_to_impz`, adac |
| `graph` (new) | anything else | `graph_to_td`, `graph_to_flamo`, adac |

The flat `build` is what keeps the common case readable, diffable, and
citable — a reviewer can see the whole network in one screen. A graph IR must
not be allowed to eat that. `graph` is the escape hatch for everything the
flat form cannot express.

A document carries exactly one of the two. `build_to_graph` lowers; `graph_to_build`
lifts and **refuses** a graph that is not vanilla, in the same spirit as
`extract_build` refusing an un-bakeable hook today.

### 4.2 Document layout

```json
{
  "format": "pyfdn-system",
  "version": 1,
  "metadata": {"name": "...", "description": "...", "tags": []},
  "sample_rate": 48000,
  "render": {"nfft": 16384, "alias_decay_db": 0.0, "block_size": 64},
  "build":  { ... },
  "graph":  { ... }
}
```

- `format` / `version` are mandatory and checked on load. (adac's IR has
  neither, which is the single cheapest thing to fix in this whole proposal.)
- `sample_rate` is top-level and singular. Nothing below it repeats a sample
  rate.
- `metadata` is an open object. As in `FDNPreset`, `tags` is a list of
  strings when present so catalogues can filter consistently.
- Exactly one of `build` / `graph`.

### 4.3 One `type` namespace, owned by pyFDN

Every node has a `type` drawn from a single flat namespace. Container types
carry children; leaf types carry `params`.

**Containers** — `series`, `parallel`, `recursion`.

**Leaves** — `gain`, `matrix`, `matrix_fir`, `delay`, `sos`, `identity`,
`time_varying_matrix`, `pitch_shift`, `dc_blocker`, … as the registry grows.

This is a departure from adac's two-field split (`type` for structure,
`module_type` for leaves). One field is enough: a node has children or it has
params, never both, and a single namespace means a single registry.

The names are pyFDN's, not FLAMO's. `sos`, not `parallelSOSFilter`.
`delay`, not `parallelDelay`. A FLAMO class rename must not be able to
invalidate documents on disk.

The registry is **open**, not an enum: `td.TimeVaryingMatrix` has no FLAMO
*or* FAUST equivalent, and `td.GranularPitchShift` has neither either. Each
backend owns its own map from `type` to its construction, and a backend that
cannot build a type fails loudly, naming the type — never silently drops the
node. (adac's FAUST emitter currently emits a `//warning:` comment for an
unknown type, which produces a plausible-looking `.dsp` that is quietly the
wrong system. That is the failure mode to avoid.)

### 4.4 `params` is baked, `design` is how it was made

The `build` / `design` split that `FDNPreset` already has, pushed down one
level and applied per node:

- **`params`** — the effective values a renderer needs, post-`map`. For a
  FLAMO `Matrix(matrix_type="orthogonal")` that is the matrix exponential of
  the stored skew-symmetric weights, not the weights.
- **`design`** — how the numbers were chosen, and the raw trainable
  parametrisation they came from: `{"type": "orthogonal", "raw": [[...]]}`,
  `{"type": "first_order_shelf", "rt": [1.8, 0.4]}`,
  `{"type": "coprime", "range_ms": [20, 50], "seed": 42}`.

A renderer ignores `design` entirely. Re-generation, re-training, and
round-tripping back into a trainable FLAMO model read it. This subsumes
adac's `flamo.param_raw` without pinning the format to FLAMO, and it reuses
the vocabularies `preset.py` already validates against.

`design` is optional on every node. A node without it is fully renderable and
merely does not remember its provenance — which is exactly the status quo for
a bare `FDNBuild`.

### 4.5 Units and array conventions

One canonical form per quantity, chosen so that no backend's convention wins:

| quantity | wire form | note |
|---|---|---|
| delays | samples, float permitted | FLAMO stores seconds internally; that is a FLAMO detail, converted in `graph_to_flamo`. adac currently emits both `samples` and `samples_fractional`; one field with a float is enough. |
| SOS | `(sections, 6, channels)`, `[b0,b1,b2,a0,a1,a2]` | pyFDN's existing layout. **Do not** normalise `a0` to 1 in the file — that is a FAUST `fi.tf2` concern and belongs in the emitter. |
| gains / matrices | nested lists, row-major, `(out, in)` | matches `FDNBuild` |
| FIR matrices | `(out, in, order)`, `z^-1` convention | matches `process_fdn`'s 3-D `A` |
| frequencies | Hz | |
| times | seconds | |
| levels | dB, sign as pyFDN uses it elsewhere | |

All arrays are finite. `allow_nan=False` on write, as `save_fdn_build`
already does.

### 4.6 `render` is not the system

`nfft` and `alias_decay_db` are FLAMO evaluation settings and are meaningless
in `td`. `block_size` is a `td` scheduling setting and is meaningless in
FLAMO. None of them changes *what system this is* — they change how one
backend approximates it.

They therefore live in one top-level `render` block, and every field in it is
optional with a documented default. adac currently attaches `nfft` and
`alias_decay_db` to individual nodes, which makes a document unrenderable by
`td` without stripping fields.

`alias_decay_db` in particular must not leak into `params`: `extract_build`
already takes care to return the *undamped* `A`/`B`/`C`, and a document
records the system, not the numerically-nudged version of it FLAMO evaluated.

### 4.7 The loop-delay convention (decided)

`td.Recursion(block_size=n)` inserts `n` samples of delay into its loop, and
today the caller compensates by hand — `examples/example_reverberation_enhancement.py`
shortens its delay lines with `td.Delay(delays - block_size)`.

If a document stored those pre-shortened lengths, a `td` render and a FLAMO
render of the same file would be different systems.

**The document stores the true delay lengths. `graph_to_td` performs the
block compensation.** An author never writes `delays - block_size` into a
file, and `td_to_graph` adds the compensation back when reading a live graph
whose `Recursion` has a non-zero `block_size`. This is the one place the
format takes a position on a backend's internals, and it is worth it: without
it the format silently means different things to different readers.

### 4.8 Node names are part of the contract

`extract_build` recognises a vanilla FDN by node name, and `assemble_fdn_core`
produces those names. They are normative in a `graph` document, not
decorative — they are what makes `graph_to_build` possible at all:

```
parallel(sum_output=true)
├── brA: series
│   ├── input_gain
│   ├── feedback_loop: recursion
│   │   ├── forward: series[ delay, post_delay ]      (or a bare delay)
│   │   └── feedback: series[ mixing_matrix, post_matrix ]  (or a bare matrix)
│   ├── output_gain
│   └── post_output
└── brB: direct_gain
```

A hook given several modules keeps the existing `post_delay_0`,
`post_delay_1`, … convention from `hook_module`.

`recursion` uses the field names `forward` / `feedback` in the document.
FLAMO's `fF` / `fB` are accepted as aliases on read for compatibility with
adac's existing IR, and never written.

## 5. Backend mapping

Each backend owns one table. Sketch, to be completed during implementation:

| `type` | `td` | FLAMO | FAUST (adac) |
|---|---|---|---|
| `series` | `td.Series` | `system.Series` | `:` |
| `parallel` | `td.Parallel` | `system.Parallel` | `,` / `:>` |
| `recursion` | `td.Recursion` | `system.Recursion` | `~` |
| `gain` | `td.Gain` | `dsp.Gain` / `dsp.parallelGain` | `*(g)` |
| `matrix` | `td.Gain` | `dsp.Matrix` | matrix expansion |
| `matrix_fir` | `td.MatrixFIR` / `td.MatrixConvolver` | — | — |
| `delay` | `td.Delay` | `dsp.parallelDelay` | `@(n)` / `de.fdelay` |
| `sos` | `td.SOSBank` | `dsp.parallelSOSFilter` | `fi.tf2` |
| `time_varying_matrix` | `td.TimeVaryingMatrix` | — | — |
| `pitch_shift` | `td.PitchShift` | — | — |

The empty cells are the point. They are not gaps to be filled; they are
systems that backend genuinely cannot render, and it must say so by name.

## 6. Ownership

**pyFDN owns the schema and every conversion in and out of it.** Nothing in
`pyFDN` core imports or depends on `adac`.

- pyFDN: schema, validation, `flamo ↔ graph`, `td ↔ graph`, `build ↔ graph`,
  and the packaged presets.
- adac: FAUST emission, stability certificate, JUCE export — reading
  `pyfdn-system` documents. Since adac's numpy-only core is a selling point,
  it vendors a small reader rather than taking a pyFDN dependency; the schema
  is a versioned document format, so vendoring a reader is legitimate in a way
  that forking the traversal was not.

This is the direction the dependency already wants to run: adac's FAUST
dispatcher is the most complete leaf vocabulary that exists anywhere today,
and it is downstream of a traversal it copied from pyFDN.

## 7. Implementation plan

Four PRs, each independently useful and independently revertable.

### PR 1 — the schema (this branch's follow-up)

`src/pyFDN/system.py`, pure data, no backend imports:

- `FDNSystemDoc` dataclass (or plain dicts — decide during review)
- `system_to_dict` / `system_from_dict`, `save_fdn_system` / `load_fdn_system`
- validation: `format`/`version`, exactly one of `build`/`graph`, node types
  against the registry, container/leaf arity, finite arrays
- `build_to_graph` / `graph_to_build`
- `FDNPreset` accepts `build` **or** `graph`; every existing preset file on
  disk stays valid and every existing test keeps passing

Tests: round-trip each packaged preset through `build → graph → build`;
`graph_to_build` raises on a non-vanilla graph, naming what it could not bake.

### PR 2 — the `td` backend

`graph_to_td` / `td_to_graph`. This is the tier that can currently be saved in
no form at all, and it is where the Kronecker / time-varying feedback work
lands. Includes the §4.7 block compensation and its round-trip test.

Tests: `graph_to_td(td_to_graph(g))` renders sample-identical output to `g`
for the FDN, shimmer, and RES graphs the examples already build.

### PR 3 — the FLAMO backend

`flamo_to_graph` / `graph_to_flamo`, generalising `extract_build` and
`build_to_flamo`. adac's `_serialise_leaf` is the reference for the extraction
logic (effective-vs-raw, channel counts, SOS handling), ported to the neutral
type names. `extract_build` becomes `graph_to_build(flamo_to_graph(model))`
and keeps its current signature and error messages.

Tests: `flamo_to_graph → graph_to_flamo` reproduces the impulse response of
every model `example_train_fdn_to_rir` and `example_train_colorless_FDN`
build; a trained model round-trips with its `design.raw` weights intact and
still trains.

### PR 4 — adac (separate repo)

`graph_to_faust` reading the neutral types. adac's dispatcher already nearly
speaks them, so this is mostly a rename plus a reader. `flamo_to_json` stays
as a deprecated shim.

## 8. Open questions

1. **Does `graph` need explicit channel counts per node?** adac stores
   `input_channels` / `output_channels` on every leaf. They are derivable from
   the parameter shapes for every type in the table above, and a stored value
   that disagrees with the shape is a new failure mode. Proposal: derive, do
   not store — but check this against `parallelFilter` and the nested-hook
   cases before committing.
2. **Large arrays.** A 100 000-tap `matrix_fir` in JSON is unpleasant. Option:
   allow a `params` value to be `{"$ref": "sidecar.npz#A"}`. Worth designing
   now so the escape hatch is not bolted on later, but not worth implementing
   until something needs it.
3. **`FDNSystem` name collision.** `pyFDN.FDNSystem` already exists
   (`generate/fdn_build_gallery.py`). Pick a different name for the document
   type, or rename — decide in PR 1 review.
4. **Should `render` be per-backend?** `{"flamo": {...}, "td": {...}}` is more
   honest than one flat block with fields that only some backends read. Flat
   is simpler and the field names do not currently collide. Revisit if they do.

---

## Appendix A — worked example

A 6-line FDN with a trained orthogonal feedback matrix, a designed
first-order-shelf absorption, and a time-varying matrix on the feedback path —
i.e. a system that has no representation on disk today.

```json
{
  "format": "pyfdn-system",
  "version": 1,
  "metadata": {
    "name": "tv-kronecker-N6",
    "description": "Time-varying FDN, Kronecker feedback modulation.",
    "tags": ["time-varying", "kronecker"]
  },
  "sample_rate": 48000,
  "render": {"nfft": 16384, "alias_decay_db": 0.0, "block_size": 64},
  "graph": {
    "type": "parallel",
    "name": "root",
    "sum_output": true,
    "children": [
      {
        "type": "series",
        "name": "brA",
        "children": [
          {
            "type": "gain",
            "name": "input_gain",
            "params": {"matrix": [[0.408], [0.408], [0.408], [0.408], [0.408], [0.408]]}
          },
          {
            "type": "recursion",
            "name": "feedback_loop",
            "forward": {
              "type": "series",
              "children": [
                {
                  "type": "delay",
                  "name": "delay",
                  "params": {"samples": [593, 743, 929, 1153, 1399, 1699]},
                  "design": {"type": "coprime", "range_ms": [12, 36], "seed": 42}
                },
                {
                  "type": "sos",
                  "name": "post_delay",
                  "params": {"sos": [[[0.98, "..."], "..."]]},
                  "design": {"type": "first_order_shelf", "rt": [1.8, 0.4]}
                }
              ]
            },
            "feedback": {
              "type": "series",
              "children": [
                {
                  "type": "matrix",
                  "name": "mixing_matrix",
                  "params": {"matrix": [["..."]]},
                  "design": {"type": "orthogonal", "raw": [["..."]]}
                },
                {
                  "type": "time_varying_matrix",
                  "name": "post_matrix",
                  "params": {"matrix": [["..."]], "rate_hz": 1.2, "depth": 0.15}
                }
              ]
            }
          },
          {
            "type": "gain",
            "name": "output_gain",
            "params": {"matrix": [[0.408, 0.408, 0.408, 0.408, 0.408, 0.408]]}
          }
        ]
      },
      {
        "type": "gain",
        "name": "brB",
        "params": {"matrix": [[0.0]]}
      }
    ]
  }
}
```

Rendering this file:

- `graph_to_td` builds it in full.
- `graph_to_flamo` raises, naming `time_varying_matrix`.
- `graph_to_build` raises, naming `time_varying_matrix` as the reason it does
  not bake to an `FDNBuild`.
- adac's `graph_to_faust` raises, naming `time_varying_matrix`.

Three of the four fail, all four say exactly why, and none of them silently
emits the wrong system. That is the whole design in one example.

## Appendix B — the same network as a `build`

Drop the `time_varying_matrix` node and the identical network is expressible
in the existing flat tier, which is what should be written whenever it is
possible:

```json
{
  "format": "pyfdn-system",
  "version": 1,
  "metadata": {"name": "static-N6"},
  "sample_rate": 48000,
  "build": {
    "format": "pyfdn-fdn-build",
    "version": 2,
    "feedback_matrix": [["..."]],
    "input_matrix": [["..."]],
    "output_matrix": [["..."]],
    "direct_matrix": [[0.0]],
    "delays": [593, 743, 929, 1153, 1399, 1699],
    "sample_rate": 48000.0,
    "post_delay": [[["..."]]],
    "post_matrix": null,
    "post_output": null
  }
}
```

Byte-for-byte the `build` object that `fdn_build_to_dict` writes today.
