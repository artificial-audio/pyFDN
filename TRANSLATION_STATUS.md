# pyFDN — Translation & Modernisation Status

This file tracks the translation of [fdnToolbox (MATLAB)](https://github.com/SebastianJiroSchlecht/fdnToolbox)
to pyFDN (Python).  The goal is modernisation, not a 1-to-1 port: architecture has
shifted to NumPy/SciPy primitives, FLAMO for the differentiable DSP graph, and
marimo for interactive examples.

**Environment:** `conda activate seb312` · Python 3.12  
**Run tests:** `/opt/homebrew/anaconda3/envs/seb312/bin/python -m pytest tests/ --ignore=tests/test_marimo_examples_run.py`

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Done — translated and tested |
| 🔜 | Remaining — planned or in-progress |
| ⏭ | Superseded — covered by FLAMO or another module |
| ⛔ | Skip — symbolic math, external tools, or out of scope |

---

## `GraphicEQ/` — 6/6 done ✅

| MATLAB | Python | Notes |
|--------|--------|-------|
| `shelvingFilter.m` | `graphicEQ/shelving_filter.py` | Eqs (18)/(20) Välimäki & Reiss 2016 |
| `bandpassFilter.m` | `graphicEQ/bandpass_filter.py` | Eq (29) Välimäki & Reiss 2016 |
| `graphicEQ.m` | `graphicEQ/graphic_eq.py` | 11 biquad sections: flat + low-shelf + 8 peaking + high-shelf |
| `probeSOS.m` | `graphicEQ/probe_sos.py` | `freqz` returns `(w, h)` order (not `(h, w)`) |
| `designGEQ.m` | `graphicEQ/design_geq.py` | `lsqlin` → `scipy.optimize.lsq_linear` |
| `absorptionGEQ.m` | `graphicEQ/absorption_geq.py` | Output shape `(num_delays, 1, 11, 6)`, `a0` normalised to 1 |

`hertz2rad.m` is covered by `auxiliary/utils.py::hertz_to_rad`.

---

## `Generate/` — 20 done, 2 superseded ✅/🔜

| MATLAB | Python | Notes |
|--------|--------|-------|
| `randomOrthogonal.m` | `generate/random_orthogonal.py` | ✅ |
| `householderMatrix.m` | `generate/householder_matrix.py` | ✅ |
| `nearestOrthogonal.m` | `generate/nearest_orthogonal.py` | ✅ SVD Procrustes |
| `nearestSignAgnosticOrthogonal.m` | `generate/nearest_sign_agnostic_orthogonal.py` | ✅ Inline Sinkhorn-Knopp |
| `completeOrthogonal.m` | `generate/complete_orthogonal.py` | ✅ Uses SVD of A for consistency |
| `degreeOneLossless.m` | `generate/degree_one_lossless.py` | ✅ Output shape `(N, N, 2)` |
| `fdnMatrixGallery.m` | `generate/fdn_matrix_gallery.py` | ✅ 10/12 types implemented |
| `AndersonMatrix.m` | `generate/anderson_matrix.py` | ✅ |
| `interpolateOrthogonal.m` | `auxiliary/math.py::interpolate_orthogonal` | ✅ |
| `constructCascadedParaunitaryMatrix.m` | `generate/construct_cascaded_paraunitary_matrix.py` | ✅ |
| `constructVelvetFeedbackMatrix.m` | `generate/construct_velvet_feedback_matrix.py` | ✅ |
| `constructParaunitaryFromElementals.m` | `generate/construct_paraunitary_from_elementals.py` | ✅ 2026-06-10, uses `degree_one_lossless` + `matrix_convolution` |
| `shiftMatrix.m` | `generate/shift_matrix.py` | ✅ |
| `shiftMatrixDistribute.m` | `generate/shift_matrix_distribute.py` | ✅ |
| `randomMatrixShift.m` | `generate/random_matrix_shift.py` | ✅ |
| `isAlmostZero.m` | `generate/is_almost_zero.py` | ✅ |
| `isAllpass.m` | `auxiliary/allpass.py::is_allpass` | ✅ |
| `isParaunitary.m` | `auxiliary/allpass.py::is_paraunitary` | ✅ |
| `isUniallpass.m` | `auxiliary/allpass.py::is_uniallpass` | ✅ |
| `homogeneousAllpassFDN.m` | `generate/allpass_FDN/homogeneous_allpass_fdn.py` | ✅ |
| `constructDelayMatrix.m` | — | ⏭ Use FLAMO `delay_module()` directly |
| `constructDelayFeedbackMatrix.m` | — | ⏭ Use FLAMO concatenation |

**In-gallery types deferred (raise `NotImplementedError`):**

| Gallery type | Reason |
|---|---|
| `allpassInFDN` | Use `generate/allpass_FDN/` submodule instead |
| `SchroederReverberator` | Use `generate/allpass_FDN/` submodule instead |

**Remaining generate functions:**

| MATLAB | Python target | Priority |
|--------|--------------|---------|
| `diagonallyEquivalent.m` | `generate/diagonally_equivalent.py` | 🔜 P2 |
| `isDiagonallyEquivalentToOrthogonal.m` | `generate/is_diagonally_equivalent_to_orthogonal.py` | 🔜 P2 |
| `isDiagonallySimilarToOrthogonal.m` | `generate/is_diagonally_similar_to_orthogonal.py` | 🔜 P2 |
| `mroots.m` | extend `auxiliary/math.py` | 🔜 P2 |
| `minRankChoiceBruteForce.m` | — | ⛔ research-specific |

---

## `Translate/` — 9/11 done, 1 remaining, 1 skip ✅/🔜

| MATLAB | Python | Notes |
|--------|--------|-------|
| `dss2impz.m` | `translate/dss_to_impz.py` | ✅ also accepts FIR (N,N,L) feedback matrices since 2026-06-10 |
| `dss2pr.m` | `translate/dss_to_pr_flamo.py` | ✅ FLAMO-based path |
| `dss2pr_direct.m` | `translate/dss_to_pr_direct.py` | ✅ |
| `dss2ss.m` | `translate/dss_to_ss.py` | ✅ |
| `dss2tf.m` | `translate/dss_to_tf.py` | ✅ |
| `impz2res.m` | `translate/impz_to_res.py` | ✅ |
| `mtf2impz.m` | `translate/mtf_to_impz.py` | ✅ |
| `pr2impz.m` | `translate/pr_to_impz.py` | ✅ |
| `dss2impzTransposed.m` | — | ⏭ Superseded by FLAMO's flexible graph |
| `dss2res.m` | `translate/dss_to_res.py` | 🔜 **P1** — modal excitation: pre-computed poles → residues (2024 paper) |
| `dss2tfSym.m` | — | ⛔ symbolic, skip |

---

## `Auxiliary/` — partial, ongoing

Key functions mapped:

| MATLAB | Python | Notes |
|--------|--------|-------|
| `absorptionFilters.m` | `auxiliary/acoustics.py::absorption_filters` | ✅ |
| `onePoleAbsorption.m` | `auxiliary/acoustics.py::one_pole_absorption` | ✅ |
| `absorption2T60.m` | `auxiliary/acoustics.py::absorption_to_rt` | ✅ |
| `RT602slope.m` | `auxiliary/acoustics.py::rt_to_slope` | ✅ |
| `EDC.m` | `auxiliary/acoustics.py::edc` | ✅ |
| `processFDN.m` | `process.py::process_fdn` | ✅ extended 2026-06-10: FIR feedback matrices, `absorption_filters`, `extra_matrix` |
| `matrixConvolution.m` | `auxiliary/math.py::matrix_convolution` | ✅ |
| `matrixDelayApproximation.m` | `auxiliary/delay.py::matrix_delay_approximation` | ✅ |
| `mgrpdelay.m` | `auxiliary/delay.py::mgrpdelay` | ✅ |
| `ms2smp.m` | `auxiliary/delay.py::ms_to_smp` | ✅ |
| `detPolynomial.m` | `auxiliary/math.py::det_polynomial` | ✅ |
| `generalCharPoly.m` | `auxiliary/math.py::general_char_poly` | ✅ |
| `matrix_polyder.m` | `auxiliary/math.py::matrix_polyder` | ✅ |
| `matrix_polyval.m` | `auxiliary/math.py::matrix_polyval` | ✅ |
| `negpolyder.m` | `auxiliary/math.py::negpolyder` | ✅ |
| `outerSumApproximation.m` | `auxiliary/math.py::outer_sum_approximation` | ✅ |
| `polyDegree.m` | `auxiliary/math.py::poly_degree` | ✅ |
| `polydiag.m` | `auxiliary/math.py::polydiag` | ✅ |
| `poleBoundaries.m` | `auxiliary/utils.py::pole_boundaries` | ✅ bug-fixed 2026-06-10 (`freqz` return order) |
| `skew.m` | `auxiliary/utils.py::skew` | ✅ |
| `hertz2unit.m` / `rad2hertz.m` | `auxiliary/utils.py` | ✅ |
| `plotImpulseResponseMatrix.m` | `auxiliary/plot.py::plot_impulse_response_matrix` | ✅ |
| `plotSystemMatrix.m` | `auxiliary/plot.py::plot_system_matrix` | ✅ |
| `tinyRotationMatrix.m` | `auxiliary/tiny_rotation_matrix.py` | ✅ (torch-based) |

**Remaining auxiliary functions (audit FLAMO coverage first):**

| MATLAB | Python target | Priority | Notes |
|--------|--------------|---------|-------|
| `loopTF.m` | `auxiliary/loop_tf.py` | 🔜 P2 | Check FLAMO frequency-response first |
| `tfMatrix.m` | `auxiliary/math.py` (extend) | 🔜 P2 | Check FLAMO |
| `firMatrix.m` | `auxiliary/math.py` (extend) | 🔜 P2 | Check FLAMO |
| `poleQuality.m` | `auxiliary/poles.py` (extend) | 🔜 P2 | Audit FLAMO optimisation coverage |
| `refinePolePositions.m` | `auxiliary/poles.py` (extend) | 🔜 P2 | As above |
| `toDiagonalSimilarCanonicalForm.m` | `auxiliary/poles.py` (extend) | 🔜 P2 | As above |
| `realLogOfNormalMatrix.m` | `auxiliary/math.py` (extend) | 🔜 P2 | |
| `lowRankApprox.m` | `auxiliary/math.py` (extend) | 🔜 P3 | |
| `adjPoly.m` / `adjugate.m` | `auxiliary/math.py` (extend) | 🔜 P3 | |
| `zFDNloop.m` / `zFDNloopSimple.m` | — | ⏭ FLAMO graph |
| `zSOS.m` / `zFilter.m` / `zDelay.m` etc. | — | ⏭ FLAMO modules |
| `processFDN.m` / `processTransposedFDN.m` | `process.py` | ✅ / ⏭ |
| `decayFitNet2InitialLevel.m` | — | ⛔ Standalone Python package exists |
| `printMatLatex.m` / `printMatSyntax.m` | — | ⛔ skip |
| `generalCharPolySym.m` / `mpoly2sym.m` | — | ⛔ symbolic, skip |

---

## `DSP/` — 2/4 done 🔜

| MATLAB | Python | Priority | Notes |
|--------|--------|---------|-------|
| `feedbackDelay.m` | `dsp/feedback_delay.py` | ✅ | |
| `dfiltMatrix.m` | `dsp/dfilt_matrix.py` | ✅ 2026-06-10 | `FIRMatrixFilter` (FIR case only — scalar matrices are plain matmuls in `process_fdn`, per-line IIR absorption is SOS-based) |
| `complexOscillatorBank.m` | `dsp/complex_oscillator_bank.py` | 🔜 P1 | Modulation bank for time-varying FDNs |
| `timeVaryingMatrix.m` | `dsp/time_varying_matrix.py` | 🔜 P1 | Class with `.filter(block)`; plugs into `process_fdn(extra_matrix=...)` |

---

## `dspGraph/` — superseded ⏭

All 29 files superseded by FLAMO (`auxiliary/flamo.py`, `auxiliary/flamo_graph.py`,
`translate/dss_to_flamo.py`).  No translation needed.

---

## Examples — 23 + 9 allpass done ✅

| Example | Covers |
|---------|--------|
| `example_vanilla_FDN.py` | Basic FDN |
| `example_colorless_FDN.py` | Colorless FDN design |
| `example_process_fdn.py` | `process_fdn`, `dss_to_impz` |
| `example_coupled_rooms.py` | Coupled rooms FDN |
| `example_absorption_filters.py` | FIR / one-pole absorption |
| `example_one_pole_absorption.py` | `one_pole_absorption` |
| `example_delay_matrix_density.py` | Echo density |
| `example_interpolate_matrix.py` | `interpolate_orthogonal` |
| `example_dss_to_pr_direct.py` | DSS → poles/residues (direct) |
| `example_dss_to_pr_flamo.py` | DSS → poles/residues (FLAMO) |
| `example_dss_to_ss.py` | DSS → state-space |
| `example_dss_to_tf.py` | DSS → transfer function |
| `example_sdn_coefficients.py` | Scattering delay network |
| `example_graphic_eq.py` | `design_geq` — ±10 dB target + freq response |
| `example_absorption_geq.py` | `absorption_geq` — FDN simulation + EDC/T60 |
| `example_fdn_matrix_gallery.py` | All 12 gallery types — heatmaps + losslessness |
| `example_nearest_sign_agnostic_orthogonal.py` | Sign-agnostic Procrustes vs. naive |
| `example_tradeoff.py` | Modal/echo density vs complexity (3×3 grid) |
| `example_spread_fdn_poles.py` | Proportional vs spread modal decay |
| `example_random_fdn_statistics.py` | Pole angle / residue magnitude statistics |
| `example_fdn_eigenvectors.py` | Mode shapes from left/right eigenvectors |
| `example_pole_boundaries.py` | Frequency-dependent pole magnitude bounds |
| `example_scattering_fdn.py` | FIR scattering matrices + echo density |
| `allpass_FDN/` (9 files) | All allpass FDN variants |

**Remaining examples:** see migration log below.

---

## Migration log — remaining examples (last update 2026-06-10)

Working order is dependency-driven (examples whose helpers all exist come first).
Status: ✅ done · 🔨 in progress · 🔜 not started. Update this table after every example.

**How to resume:** each Python example is a marimo notebook in `examples/` (plotly,
`pio.renderers.default = "sphinx_gallery"`, cell-local variables need a `_` prefix —
marimo requires unique names across cells). Smoke-test a single example with
`/opt/homebrew/anaconda3/envs/seb312/bin/python -m pytest tests/test_marimo_examples_run.py -q -k <name>`;
run the full suite with the command at the top of this file. New library code added for
this migration is tested in `tests/test_process_extensions.py`. Keep examples small
enough that the whole smoke suite stays in CI budget (eigendecompositions ≲ 3000×3000,
`pr_to_impz` in `lowMemory` mode for long IRs).

| # | MATLAB example | Python example | Status | New library code needed |
|---|----------------|----------------|--------|------------------------|
| 1 | `example_tradeoff.m` | `examples/example_tradeoff.py` | ✅ | none |
| 2 | `example_spreadFDNpoles.m` | `examples/example_spread_fdn_poles.py` | ✅ | none |
| 3 | `example_randomFDNstatistics.m` | `examples/example_random_fdn_statistics.py` | ✅ | none (`dss_to_pr_direct` meta has `undrivenResidues`) |
| 4 | `example_FDNEigenvectors.m` | `examples/example_fdn_eigenvectors.py` | ✅ | none (`dss_to_pr_direct` meta has `eigenvectors`) |
| 5 | `example_poleBoundaries.m` | `examples/example_pole_boundaries.py` | ✅ | none (FIR absorption as SOS in the FLAMO loop: `dss_to_flamo(sos_filter=...)` + `flamo_to_pr`) |
| 6 | `example_scatteringFDN.m` | `examples/example_scattering_fdn.py` | ✅ | `generate/construct_paraunitary_from_elementals.py`; FIR feedback matrix in `process_fdn`/`dss_to_impz` |
| 7 | `example_paraunitaryFDN.m` | `examples/example_paraunitary_fdn.py` | ✅ | FIR feedback in `process_fdn`; polynomial-A in `dss_to_flamo` + `flamo_to_pr(num_poles=...)` |
| 8 | `example_decorrelation.m` | `examples/example_decorrelation.py` | 🔜 | `auxiliary/math.py::adjugate`, `adj_poly`, `loop_tf`; `auxiliary/utils.py::max_corr` |
| 9 | `example_timeVaryingFDN.m` | `examples/example_time_varying_fdn.py` | 🔜 | `dsp/complex_oscillator_bank.py`, `dsp/time_varying_matrix.py`; `process_fdn` extensions (`absorption_filters`, `extra_matrix`) |
| 10 | `example_RIR2FDN.m` | `examples/example_rir_to_fdn.py` | 🔜 | uses `estimate_rt_bands` instead of DecayFitNet (architecture decision); copy `s3_r4_o.wav` from fdnToolbox |

**Shared infrastructure added 2026-06-10** (for examples 5–7, 9):

- `dsp/dfilt_matrix.py::FIRMatrixFilter` — FIR filter matrix with persistent state
  (MATLAB `dfiltMatrix`, FIR case only).
- `process_fdn` extended with FIR (N,N,L) feedback matrices, per-delay-line SOS
  `absorption_filters`, and `extra_matrix` (object with `.filter(block)`), matching the
  MATLAB `processFDN` loop ordering (delay → absorption → C; absorption → A → extra → +B).
  Validated against FLAMO (1e-9) and frequency-domain inversion (1e-13).
- Polynomial (FIR) feedback matrices go through the FLAMO path, not
  `dss_to_pr_direct` (which is numeric-static only): `dss_to_flamo` accepts a
  (N,N,L) feedback matrix (placed as a FLAMO `Filter` via
  `auxiliary/flamo.py::fir_matrix_module`), and `flamo_to_pr` gained a
  `num_poles` override for loops whose pole count exceeds sum(m) — set it to
  the degree of `general_char_poly(delays, A)`. Validated to 1e-11 on a
  cascaded paraunitary FDN.
- **Bug fix** in `general_char_poly` (polyphase branch): determinant coefficients were
  accumulated at `p_ind − j` instead of `p_ind + j`, so the GCP was wrong for every
  polynomial feedback matrix. Now matches `det(diag(z^m) − A(z))` exactly
  (regression test in `tests/test_process_extensions.py`).
- Known limitation: Hadamard-based cascaded paraunitary matrices have structural double
  poles at z = ±1 (Hadamard eigenvalue multiplicity); the simple-pole residue formula is
  inaccurate there (MATLAB has the same issue — its example only asserts 1e-3).
  Use `matrix_type="random"` for exact modal decomposition.

**Per-example notes** (translation decisions, deviations from MATLAB):

- **example_scattering_fdn** — all four matrix types run through the time-domain
  recursion (`process_fdn` with `FIRMatrixFilter`), unlike MATLAB which wraps them in
  `zFIR`. Mixing times printed per type; echo density overlaid on the IRs.
- **example_paraunitary_fdn** — N = 4, K = 3 cascade with
  `matrix_type="random"` (NOT the Hadamard default — Hadamard stages create structural
  double poles at z = ±1 where the simple-pole residue formula fails); delays scaled
  down to 150–500 samples (MATLAB uses 1250–6500; pole refinement on ≈15k poles
  would be far too slow). Modal decomposition via `dss_to_flamo` (FIR matrix as
  FLAMO `Filter` feedback) + `flamo_to_pr` with `num_poles` set to the GCP degree —
  the FIR feedback adds deg(det A) poles beyond sum(m), which the default seeding
  would miss. IR-vs-modal match ~1e-11 (MATLAB only reaches 1e-3 with its Hadamard
  cascade). `is_paraunitary` and `plot_impulse_response_matrix` expect time-first
  shape → pass `fbm.transpose(2, 0, 1)`.
- **example_pole_boundaries** — found + fixed a bug in `pole_boundaries`: `freqz` was
  unpacked in MATLAB order (`h, w`) instead of scipy's `(w, h)`, so the absorption term
  and the frequency axis were garbage. Poles are computed via FLAMO: the two-tap FIR
  absorption is one SOS section per delay line in the loop (`dss_to_flamo(sos_filter=...)`)
  and `flamo_to_pr` refines sum(m) seeds (matching MATLAB's `dss2pr` + `zTF`). The loop's
  N extra near-defective, zero-residue roots at the absorption FIR's zero (z = −b1/b0)
  are never seeded, so no exclusion is needed.
- **example_fdn_eigenvectors** — pyFDN eigenvectors are raw SVD null vectors (MATLAB
  normalises them by the derivative denominator), so the compact residue formula includes
  the `undrivenResidues` factor: `res = undriven * (c·rv) * (lv^H·b)`. Verified to machine
  precision.
- **example_random_fdn_statistics** — delays scaled down (100–400 vs MATLAB 500–2000);
  uses `meta["undrivenResidues"]` from `dss_to_pr_direct` for the residue factorisation
  histograms, matching the MATLAB metaData.
- **example_spread_fdn_poles** — delays scaled down (100–300 vs MATLAB 500–2000) so the
  `dss_to_pr_direct` eigendecomposition stays fast in CI; `pr_to_impz` uses `lowMemory`
  mode to avoid the (ir_len × num_poles) complex matrix.
- **example_tradeoff** — added echo-density overlay (Abel & Huang) per panel to make the
  tradeoff visible (the MATLAB script only renders the IRs); added audio playback of the
  two extreme settings.

**Environment note (2026-06-10):** conda env moved to `/opt/homebrew/anaconda3/envs/seb312`.
The env had lost its packages; reinstalled `pyFDN -e .`, `flamo`, `marimo`, and replaced the
broken `pysoundfile` with `soundfile` (pysoundfile 0.9 cannot find libsndfile on this machine —
if a later `pip install` pulls pysoundfile back in, run
`pip uninstall -y pysoundfile && pip install --force-reinstall --no-deps soundfile`).

---

## Priority summary

### P1 — implement next

1. Finish the remaining-examples migration (see migration log above): #7 paraunitary
   (in progress), #8 decorrelation, #9 time-varying, #10 RIR2FDN
2. `dsp/complex_oscillator_bank.py` + `dsp/time_varying_matrix.py` — needed by example #9
3. `auxiliary/math.py::adjugate` / `adj_poly` / `loop_tf` + `auxiliary/utils.py::max_corr` — needed by example #8
4. `translate/dss_to_res.py` — modal excitation (pre-computed poles → residues, 2024 paper)

### P2 — after FLAMO audit

- `tfMatrix`, `firMatrix` — first check how much FLAMO's frequency evaluation covers
- Pole analysis: `poleQuality`, `refinePolePositions`, `toDiagonalSimilarCanonicalForm`
- Matrix property checks: `diagonallyEquivalent`, `isDiagonallyEquivalentToOrthogonal`, `isDiagonallySimilarToOrthogonal`

### P3 / skip

- Symbolic math (`dss2tfSym`, `generalCharPolySym`, `mpoly2sym`, `msym2poly`) — skip
- `minRankChoiceBruteForce` — research-specific, skip
- `External/` (DecayFitNet, Violinplot) — standalone Python equivalents exist
- `OPCAR/` — evaluation scripts, out of scope

---

## Architecture decisions

| Decision | Rationale |
|----------|-----------|
| `dspGraph/` → FLAMO | FLAMO provides a more flexible, differentiable graph; no translation needed |
| `constructDelayMatrix` / `constructDelayFeedbackMatrix` → FLAMO | Use `delay_module()` and FLAMO graph concatenation |
| `dss2impzTransposed` → FLAMO | FLAMO's graph handles arbitrary topologies |
| `allpassInFDN` / `SchroederReverberator` in gallery → `NotImplementedError` | Use `generate/allpass_FDN/` submodule |
| `timeVaryingMatrix` → processing script | Not a general class; FDN-specific processing script is cleaner |
| `complete_orthogonal` uses SVD of A | Separate eigendecompositions of `I − A Aᵀ` / `I − AᵀA` are inconsistent when eigenvalues are degenerate |
| SOS convention | `a0 = 1` throughout (scipy convention); `absorption_geq` normalises on output |
| Examples → marimo | Interactive notebooks; plotly for visualisation |
