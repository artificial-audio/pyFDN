Lindsey–Fox on the modal examples
========================================

Current validated API
------------------------------------------------------------

PR 266 now provides adaptive Lindsey-Fox decomposition through both entry
points. Existing eig/EAI defaults are unchanged::

   residues, poles, direct, pairs, meta = pyFDN.dss_to_pr(
       delays, A, B, C, D, mode="lindsey_fox")

   residues, poles, direct, pairs, meta = pyFDN.flamo_to_pr(
       model, mode="lindsey_fox", num_poles=known_order)

The static route remains NumPy-only when LF succeeds. The FLAMO route requires
a real float64 canonical FDN; give ``num_poles`` explicitly for FIR matrix
loops. For parallel SOS loops, the detected order includes filter dynamics.
The scalar polynomial is constructed once. LF doubles its grid up to ten
attempts and at most 2097152 FFT samples. ``lf_max_seconds=3.0`` bounds root
search between numerical operations; it is not a hard wall-clock timeout.
Coefficient construction, matrix refinement, validation, and EAI fallback
are outside that budget. Roots are refined independently against the original
matrix before residues are extracted.

Both routes require full pole counts and validate every internal input/output
port. FLAMO also checks the external transfer, rejecting dynamics that the
modal representation does not contain. Validation uses at least 256 off-grid
points, increasing to ``2*order+1``, outside the unit circle and every returned
pole. This corresponds to exponential windowing over at least 4096 samples;
it does not establish a uniform error bound on the unit circle. Both relative
L2 and normalized peak errors must be below ``response_tolerance=1e-8``.

Search, count, pairing, residue, or response failures trigger EAI with a
warning. The fallback must pass the same acceptance checks. Inspect
``meta['solver']``, ``fallbackReason``, ``expectedPoles``, ``returnedPoles``,
``complete``, ``approximate``, and ``responseValidation`` to distinguish the
actual outcome. If neither solver passes, the call raises instead of returning
an unchecked decomposition.

Incomplete results are rejected by default. For an explicitly approximate
FLAMO decomposition, pass ``allow_approximate=True``. This permits an incomplete
EAI fallback only after both response checks pass, emits a warning, and reports
``complete=False`` and ``omittedPoleCount``. Acceptance here is numerical and
does not certify where every omitted pole lies. The separately benchmarked
certified filter-zero group shortcut below is not used automatically.

``tests/benchmark_validated_lindsey_fox.py`` checks all 30 example
configurations through the public APIs and compares one-second impulse
responses. All pass: 28 complete LF decompositions, including every Kronecker
snapshot and the paraunitary FIR model, and two explicitly approximate EAI
fallbacks for absorption. The latter retain 1578/1586 and 1958/1962 poles;
their all-port relative errors are ``1.8e-13`` and ``2.2e-13``. Their single
repetitions take about 8--9 seconds including reference rendering, so this
conservative fallback is slower than the separate group-factor experiment.
Raw results are in ``docs/benchmarks/lindsey_fox_validated_examples.jsonl``.

Upper-size stress tests: 100000 poles and 128 delay lines
------------------------------------------------------------

The current implementation is not yet practical for a complete 100000-pole
decomposition. The one-time coefficient FFT scales to this size, but root
polishing and postprocessing remain major limits. EAI fallback shares the
pairing/residue/validation costs; it does not solve these scaling problems.

``tests/benchmark_lindsey_fox_large.py`` runs isolated workers for every
combination of orders 1000, 10000, 100000 and matrix sizes 4, 16, 64, 128.
Matrices are seeded dense orthogonal matrices times 0.9; unequal integer
delays sum exactly to the requested order. Thirty-two independent off-grid
determinants validate the recovered polynomial. Three separate polynomials
``z**order - 0.9`` provide analytic root oracles. All timings below are single
trials with one BLAS thread, not medians or timing assertions.

At order 100000, all four coefficient constructions pass:

.. list-table:: One-time coefficient construction at degree 100000
   :header-rows: 1

   * - Matrix size
     - Construction seconds
     - Off-grid normalized maximum error
   * - 4
     - 0.033
     - 6.33e-12
   * - 16
     - 0.167
     - 3.08e-12
   * - 64
     - 2.75
     - 1.34e-12
   * - 128
     - 10.17
     - 5.78e-13

The coefficient FFT has 131072 samples and returns exactly 100001
coefficients. The 128-line worker's peak process RSS after construction is
198 MiB, including imports. Its peak after the interrupted root search is
272 MiB; these figures do not include a full modal decomposition.

All four full 1000-pole modal decompositions pass the independent 4096-sample
IR comparison and the production all-port check. Total stage times, including
that independent IR check, are 0.208, 0.412, 2.618, and 12.356 seconds as matrix
size grows from 4 to 128. The 128-line case peaks at 932 MiB process RSS and
has relative IR error ``9.72e-15``.

At order 10000, all four dense scalar root searches exceed the default
three-second cooperative budget. With a 30-second budget and a 45-second
external deadline, 4-line and 128-line scalar searches recover all 10000
roots. In a follow-up including 32 original-matrix checks, they take 7.90
and 13.09 seconds. Maximum sampled ``sigma_min(P)/sigma_max(P)`` values
are ``6.7e-13`` and ``3.4e-10``. These are scalar root sets, not complete
modal decompositions or verified response reconstructions. The sparse
10000-root oracle finishes in 2.26 seconds with maximum pole error ``9.6e-16``.

At order 100000, all four dense searches and the sparse oracle reach the
15-second external deadline inside ``polyval``. The sparse oracle already
has all 100000 grid candidates: full-degree Horner/Laguerre polishing is
the immediate bottleneck even in that simple case. The cooperative budget
cannot interrupt a single long polynomial evaluation. No complete 100000-
root set was verified within these deadlines; timeout is not an accuracy
failure or proof that more time could never complete the root search.

Full decompositions above order 1000 are deliberately not attempted by this
stress runner. It preflights order and memory before calling EAI or Hungarian
pairing. This guard belongs to the benchmark, not the production API:

* At 100000 poles, current conjugate pairing forms an 80 GB float cost array
  and a 160 GB complex subtraction temporary, about 240 GB together. At
  10000 poles the corresponding estimate is 2.4 GB. These are source-based
  allocation estimates, not measured allocations.
* At 100000 poles / 128 ports, one expanded complex all-port residue array
  alone is 26.2 GB; other arrays and temporaries add to that.
* The sampled modal-response summation uses O(order**2 * ports**2) work.
  Replacing pairing alone would therefore not make the whole pipeline scale.

Large-scale support needs changes to full-degree polynomial polishing,
quadratic-memory pairing, and dense all-port validation. The present method
should not be advertised as a tested 100000-pole modal solver.

Reproduce the default and extended-budget root trials::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_lindsey_fox_large.py
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_lindsey_fox_large.py --orders 10000 --lines 4 128 --root-seconds 30 --root-timeout 45

Raw stage measurements, process RSS, skipped stages, and timeout locations
are in ``docs/benchmarks/lindsey_fox_large_default.jsonl`` and
``docs/benchmarks/lindsey_fox_large_relaxed.jsonl``. Lightweight CI cases
independently verify exact cyclic-matrix characteristic coefficients at
degree 100000 and matrix size 128, without allocating a large root assignment.

Initial four-grid experiment
------------------------------------------------------------

The following historical experiment uses the original four-grid root finder on
the actual numerical setups in eight example notebooks. It tests both speed
and completeness; a small impulse-response error alone does not establish
that every pole was recovered.

Reproduce from an editable checkout with the examples dependencies installed::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_modal_examples.py --repeat 3 > modal-examples.jsonl

The script evaluates 30 model configurations across the eight notebooks.
It executes their setup cells, without their plotting or
audio cells. The Kronecker case uses the same delay vector, static matrix,
and all 21 frozen modulation snapshots. It reports JSON lines, including
verification failures and a 60-second per-repetition timeout. Failed searches
are not replaced by another solver or excluded from the coverage counts.

Method and scope
----------------

At this stage, ``dss_to_pr(..., mode="lindsey_fox")`` accepted static
feedback matrices. To test
the three filtered examples, the benchmark has a separate experimental
adapter. No public FLAMO solver API or production root-finding algorithm
was changed at that stage. The baseline benchmark retains the four-grid cap
and disables fallback so its coverage and timing comparisons remain explicit.

For these finite FIR / per-channel SOS loops, the adapter samples the full
FLAMO characteristic matrix ``P(w) = I - F(w) B(w)``. It multiplies its
determinant by the known SOS denominators before Fourier coefficient recovery.
Sampling a rational determinant directly would not recover a finite
characteristic polynomial. The expected degree includes filter dynamics;
``sum(delays)`` alone is insufficient.

The FIR paraunitary example uses its original independently computed
polynomial degree. For the absorption examples, the expected degree is the
total delay plus the section orders. Specifically, their one-section loops have

.. math::

   q(w)=\det\left(\operatorname{diag}(a_{0,i}+a_{1,i}w)
   -\operatorname{diag}(w^{m_i}(b_{0,i}+b_{1,i}w))A\right).

All their ``b1`` coefficients and ``det(A)`` are nonzero, so the coefficient
of degree ``sum(delays) + N`` is nonzero. The expected pole counts are
therefore independently justified, not inferred from either solver.
The stable examples use a unit-disc
upper search bound and a reciprocal Fujiwara lower bound; the lossless FIR
example uses the unit circle. These bounds do not come from EAI's results.
After the unchanged LF search, the adapter uses the same FLAMO SVD refinement,
conjugate symmetrization, and residue extraction as EAI. Those stages, and
coefficient recovery, are included in its timing. Both filtered backends
receive the same prebuilt model. Static DSS ``mode="eai"`` includes its
model construction, as in that API.

Each successful result is checked for the expected expanded pole count,
agreement with the reference poles where a complete reference is available,
and an independent time-domain impulse response. The IR length is
``min(20000, max(2048, 2 * degree))`` samples, rather than every notebook's
original playback duration. Setup, rendering of the time-domain reference,
and verification are outside the timed decomposition. Imports are warmed up.

Results
-------

The results below are median seconds over three repeats on macOS arm64,
Python 3.11.15, with ``OPENBLAS_NUM_THREADS=1`` and ``OMP_NUM_THREADS=1``.

.. list-table::
   :header-rows: 1

   * - Example / variant
     - Degree
     - eig / roots
     - EAI
     - Lindsey–Fox
     - LF coverage
   * - DSS solver comparison
     - 240
     - eig: 0.0678
     - 0.3269
     - 0.0130
     - Complete
   * - Eigenvectors
     - 55
     - eig: 0.00180
     - 0.0506
     - 0.00171
     - Complete
   * - Random FDN statistics
     - 1663
     - eig: 4.2197
     - 1.7449
     - 0.3545
     - Complete
   * - Proportional decay
     - 1734
     - eig: 5.1741
     - 1.6273
     - 0.2163
     - Complete
   * - Spread decay
     - 1734
     - eig: 4.6746
     - 1.6551
     - 0.2191
     - Complete
   * - Kronecker: static + 21 snapshots
     - 694 each
     - roots: 0.2680–0.2848
     - 0.8998–0.9684
     - 0.0510–0.1032 on successes
     - 12/22 complete

The ten Kronecker verification failures are benchmark indices
``00, 01, 05, 07, 08, 09, 13, 16, 20, 21``. Index ``00`` is the static
configuration; indices ``01`` through ``21`` are the modulation snapshots
in notebook order. Each failed LF search returned an explicit verification
error, not an impulse response. Roots and EAI recovered all 694 poles in all
22 cases. The successful timing range must not be read as full coverage.

.. list-table:: Full FLAMO examples (same prebuilt model for both methods)
   :header-rows: 1

   * - Example
     - Expected poles
     - EAI seconds
     - LF adapter seconds
     - Completeness
   * - Paraunitary FIR feedback
     - 1003
     - 6.6284
     - 0.4682
     - Both complete
   * - Two-tap FIR absorption
     - 1586
     - 5.4110
     - >60 (timeout)
     - EAI returns 1578
   * - First-order IIR absorption
     - 1962
     - 4.1781
     - 47.5745 (residue SVD fails)
     - EAI returns 1958

For the paraunitary example, LF's maximum absolute IR error was
``4.09e-14`` and its maximum matched pole difference from EAI was
``4.85e-16``. Both complete decompositions pass the checks.
For the two-tap absorption example, EAI's IR error was ``1.21e-14`` despite
returning eight fewer poles than the expected polynomial degree. It is marked
as an incomplete decomposition, not a complete success.

For the first-order IIR example, EAI's IR error was ``8.45e-15`` despite
returning four fewer poles than the expected degree. The LF adapter reached
the shared FLAMO residue stage, where an SVD failed; no complete modal
decomposition or reconstructed IR is claimed for that result. The reported
time is the elapsed time to failure, not a successful median.

A separate diagnostic confirmed that LF recovered all 1962 roots of the
sampled scalar polynomial. Its four roots with radius below 0.5 are near
``[-0.10552, -0.16232, -0.15222, -0.13954]``. FLAMO's characteristic-matrix
probe was non-finite at all four, so residue extraction could not evaluate
them. Their accuracy against the original matrix model is not certified by
the scalar polynomial's verification. This is a limitation of the complete
model-driven decomposition, beyond the LF root search itself.

Raw per-configuration measurements are saved in
``docs/benchmarks/lindsey_fox_modal_examples.jsonl`` alongside this report.
The small-root diagnostic is saved in
``docs/benchmarks/lindsey_fox_iir_diagnostic.jsonl``.
Successful timings are medians of three runs; a failed method stops at its
first failure or timeout. The public LF kernel is unchanged.

Across the 30 configurations, LF completed 18 modal decompositions, had ten
root-verification failures, one timeout, and one residue-extraction failure
after scalar root finding. EAI completed 28 decompositions and returned an
incomplete pole set in the two absorption cases.

Interpretation
--------------

The static API and the benchmark-only FLAMO adapter have different coverage.
Successful filtered cases demonstrate that the scalar LF search can be used
with those loop polynomials while retaining the model's filters in residue
calculation. They do not establish support for arbitrary FLAMO graphs,
nested recursion, external IIR paths, or time-varying systems.

A plausible source of difficulty is small-radius modes close to the absorption
filter zeros. In the two-tap FIR example, all eight filter zeros coincide at
approximately ``z = -0.461538``. In the first-order IIR example, the four zeros
are approximately ``[-0.13954, -0.16232, -0.15222, -0.10552]``. The exact
matrix-model locations of the difficult modes were not established by a
high-precision calculation in this benchmark. A complete pole count and an
accurate rendered IR must therefore be reported separately.

Pragmatic approximation for two-tap absorption
------------------------------------------------------------

For the common filter ``h(z) = 0.65 + 0.3 z^-1``, approximate the unresolved
eight-root cluster by the known group factor ``(z - z0)^8``, where
``z0 = -0.3/0.65``. Remove that factor using circle samples, run the unchanged
LF kernel on the remaining degree-1578 polynomial, and compute its residues
against the original FLAMO model. The eight cluster modes are omitted, not
claimed as recovered. No high-precision arithmetic is used in this prototype.

In the inverse variable, the group removal is just division of the sampled
characteristic polynomial by ``(1 - z0*w)^8``, followed by inverse FFT.
For this example, a matrix norm bound on the circle ``|z| = 0.9`` is
``5.93e-6 < 1``; comparison with the group-factor matrix places exactly eight
poles inside that circle. This provides an independent count for the omitted
group and an inner bound for the retained-root search.

Reproduce the self-checking prototype::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_two_tap_approximation.py

The benchmark checks all 1578 retained poles, one second of time-domain IR,
and 32769 frequency samples between DC and Nyquist. Error normalization uses
the wet response, excluding the direct impulse, so the direct path cannot
hide the error. It compares against the original FDN, without fitting the
residues to the reference IR. The measured error is at the numerical-noise
level for this SISO example; this is not a general bound for omitted modes
in other models. Solver defaults and strict completeness checks are unchanged.

On the same macOS arm64/Python 3.11.15 setup, three runs took a median
``2.1281 s``, versus ``5.4110 s`` for EAI's 1578-mode result. The quotient's
discarded coefficient tail was ``1.20e-14``. Measured response differences:

.. list-table:: Approximate 1578-mode reconstruction versus the original FDN
   :header-rows: 1

   * - Metric
     - Error
   * - Maximum absolute IR error, 48000 samples
     - ``9.15e-15``
   * - Relative wet IR L2 error
     - ``5.98e-14`` (about -264 dB)
   * - Maximum absolute complex frequency-response error, 32769 points
     - ``3.81e-13``
   * - Maximum frequency-response error / peak wet response
     - ``2.39e-13``

Measurements are saved in
``docs/benchmarks/lindsey_fox_two_tap_approximation.jsonl``.

Follow-up on all twelve problematic configurations
------------------------------------------------------------

The group-factor approximation was extended to the first-order IIR absorption
example, removing its four known numerator-zero factors together. Both filtered
cases now choose a tighter inner search radius from the same matrix norm
bound, retaining a bound below 0.5. This avoids scanning the large flat
interior that made the conservative ``|z| >= 0.9`` IIR run take 21.7 seconds.
The root finder is unchanged. Reproduce both absorption cases with::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_two_tap_approximation.py --all-absorption

.. list-table:: Absorption approximations, median of three runs
   :header-rows: 1

   * - Example
     - Retained / omitted poles
     - Inner radius
     - Seconds
     - Relative wet IR L2 error (48000 samples)
     - All-port relative IR error (4096 samples)
   * - Two-tap FIR
     - 1578 / 8
     - 0.982917
     - 1.4105
     - ``6.85e-14``
     - ``1.86e-13``
   * - First-order IIR
     - 1958 / 4
     - 0.995904
     - 1.4501
     - ``2.30e-13``
     - ``2.30e-13``

Both pass the response checks. On the same 32769-point unit-circle frequency
grid, maximum errors normalized by peak wet response are ``3.34e-13`` (FIR)
and ``8.09e-13`` (IIR). The all-port test replaces the notebook's input and
output gains with identities and its direct gain with zero, checking the
entire internal-to-output transfer matrix rather than one selected channel.

The ten failing Kronecker configurations have no absorption-zero group to
remove. A separate diagnostic tested the same pragmatic question: would
accepting their partial root sets and omitting the unresolved modes be safe?
It inspects LF's final candidates after the verification exception, without
changing the kernel, and checks the retained roots against an independent
reference. No reference roots are inserted into the approximation. Run::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_problematic_cases.py

.. list-table:: Omitting unresolved Kronecker modes is not acceptable
   :header-rows: 1

   * - Configuration index
     - Omitted poles
     - Original-I/O IR relative L2 error
     - All-port IR relative L2 error
   * - 00 (static)
     - 2
     - 4.140%
     - 5.653%
   * - 01
     - 2
     - 4.140%
     - 5.653%
   * - 05
     - 2
     - 0.0156%
     - 5.477%
   * - 07
     - 2
     - 10.986%
     - 5.311%
   * - 08
     - 2
     - 11.320%
     - 5.320%
   * - 09
     - 2
     - 6.689%
     - 6.244%
   * - 13
     - 2
     - 11.865%
     - 5.325%
   * - 16
     - 2
     - 6.469%
     - 4.989%
   * - 20
     - 2
     - 10.654%
     - 5.298%
   * - 21
     - 4
     - 12.582%
     - 8.319%

The omitted Kronecker poles lie on the unit circle, unlike the rapidly
decaying absorption-zero modes. All ten configurations fail the partial-mode
response test. Case 05 also shows why a selected input/output channel can
hide a failure that becomes substantial when the gains are changed.
The original-I/O test uses 48000 samples; the all-port test uses 4096 samples.
Since the lossless model has unit-circle poles, the frequency check uses
``|z| = exp(1/4800)`` (an exponential window), not a claimed finite uniform
error bound on the unit circle. Its relative frequency errors range from
``1.75e-4`` to ``0.130``, likewise rejecting the omission.

The practical policy supported by these initial runs was: use the certified filter-zero
group approximation for these absorption examples, and retain EAI as the
fallback for the ten unresolved Kronecker configurations. Do not accept an
incomplete modal decomposition merely because its current single-channel IR
looks adequate. No automatic fallback or approximation API is introduced by
these diagnostics.

Raw data:

* ``docs/benchmarks/lindsey_fox_absorption_approximation_all_ports.jsonl``
* ``docs/benchmarks/lindsey_fox_absorption_conservative.jsonl``
* ``docs/benchmarks/lindsey_fox_problematic_partial_modes.jsonl``

Why the Kronecker search misses modes
------------------------------------------------------------

The missing modes are close but resolvable unit-circle neighbors, not the
effectively coincident small-radius absorption cluster. In the static
configuration, the missing positive-frequency pole is at ``71.53428 Hz`` and
its nearest neighbor is at ``70.94349 Hz``: a gap of ``0.59079 Hz`` at 48 kHz.
The four successive search FFT lengths are 8192, 16384, 32768, and 65536;
even the last has a frequency spacing of ``0.73242 Hz``.

Tracing that case shows 676, 686, 688, and 692 candidates as the grid is
refined. The nearby candidate consistently polishes to the neighboring pole,
leaving the other pole and its conjugate unresolved. Across all ten failing
configurations, missing-root nearest-neighbor gaps range from ``0.095`` to
``1.319 Hz``, at most about two samples of the finest angular grid. A 3-point
angular minimum test does not necessarily seed both roots in such a pair.

This is a coverage limitation of the prototype's grid initialization and
four-restart cap, not a need for arbitrary precision in these examples.
The prototype also omits the original LF method's quotient-deflation
completion stage. Local grid refinement or completion of the few missing
roots are distinct from accepting an incomplete result; omission is rejected
by the measured response errors above. The production algorithm was not
modified during this diagnosis.

Higher-resolution Kronecker follow-up
------------------------------------------------------------

The previously reported roughly 5% all-port relative L2 errors describe
waveform norms, not energy fractions. A 5% norm error corresponds to 0.25%
error-signal energy relative to reference energy. For equally energetic,
orthogonal modes, dropping two of 694 predicts ``sqrt(2/694) = 5.37%``
norm error. Actual errors depend on residues and the finite observation
window. These unit-circle modes persist, unlike the fast absorption modes.

``tests/benchmark_kronecker_resolution.py`` copies the LF function and changes
only its grid-attempt cap from four to ten (originally guarded by an AST
assertion; the kernel now exposes that cap directly).
All ten formerly failing configurations then return all 694 poles in
complex128. No reference roots seed either search or refinement.

Nine cases need FFT lengths 131072--524288 and take 0.12--0.26 seconds for
root search in single repetitions. The closest case, snapshot 05, needs
2097152 samples (0.02289 Hz spacing) and 2.15 seconds. These timings exclude
coefficient construction, matrix refinement, residues, and validation.
Before matrix refinement, one-second SISO relative IR errors range from
``3.4e-11`` to ``2.6e-7``. Applying the existing SVD null-vector Newton
refinement against the original FDN, then using the static residue helper,
reduces all ten errors to ``4.9e-12``--``5.9e-12`` in double precision.
This refinement is independent per pole; it is not an EAI iteration.
Production solver defaults remain unchanged.

The rotation tensor product has feedback eigenvalues
``exp(i*(+/-theta1 +/-theta2 +/-theta3))``. At three equal pi/4 angles,
the phases +pi/4 and -pi/4 each occur three times. Equal delays would turn
these repetitions into exact modal multiplicities; unequal delays change
the spectrum, so this observation alone does not establish the cause of
each close pair in the example.

A fixed-delay control in ``tests/benchmark_kronecker_spacing.py`` provides
additional evidence: the default Kronecker matrix has a minimum positive-
frequency gap of 0.59079 Hz; 20 seeded random SO(8) matrices have minimum
gaps of 4.00--14.69 Hz. Twenty small independent angle perturbations
(+/-0.05 radians) give 0.367--5.002 Hz. Thus this structured parameter
family is more prone to close pairs in these controls, and small angle
changes do not reliably eliminate them. This is not a general guarantee
about either matrix family.

Raw results are in
``docs/benchmarks/lindsey_fox_kronecker_higher_resolution.jsonl`` and
``docs/benchmarks/lindsey_fox_kronecker_spacing_controls.jsonl``.

Why the FLAMO decompositions take more than one second
------------------------------------------------------------

System sizes differ: the initial small-matrix benchmark had three delay
lines and 1116 poles, the paraunitary FLAMO case has four lines and 1003 poles,
and the absorption approximations retain 1578 poles on eight lines or 1958
poles on four lines. More importantly, the FLAMO adapter includes general
per-pole model derivatives and SVD refinement before residue extraction.
The static NumPy route has a cheaper residue path and no such pole-refinement
pass.

One warmed profiling repetition gave this breakdown in seconds (these are
individual profiling repetitions, not the earlier median timing table):

.. list-table:: Timed decomposition stages; response validation is excluded
   :header-rows: 1

   * - Stage
     - Two-tap FIR
     - First-order IIR
   * - FLAMO determinant sampling and coefficient recovery
     - 0.074
     - 0.067
   * - Lindsey–Fox root search and polishing
     - 0.326
     - 0.331
   * - Per-pole SVD refinement against the original model
     - 0.699
     - 0.558
   * - FLAMO residue extraction
     - 0.271
     - 0.311
   * - Pairing and other overhead
     - 0.024
     - 0.035
   * - Total
     - 1.394
     - 1.303

The root search remains sub-second. Refinement and residue extraction
dominate: they evaluate derivatives and perform SVDs in separate passes,
and refinement currently visits both members of each conjugate pair.
This identifies a postprocessing optimization opportunity without changing
the LF root finder. No such optimization is claimed by this profile.

Diagnostic data are saved as
``docs/benchmarks/lindsey_fox_kronecker_grid_diagnostic.jsonl``,
``docs/benchmarks/lindsey_fox_kronecker_neighbor_gaps.jsonl``, and
``docs/benchmarks/lindsey_fox_flamo_stage_profile.jsonl``.
