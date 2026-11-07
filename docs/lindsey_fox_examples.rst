Lindsey–Fox on the modal examples
========================================

This experiment uses the unchanged Lindsey–Fox root finder from PR 266 on
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

The existing public ``dss_to_pr(..., mode="lindsey_fox")`` accepts static
feedback matrices. It does not accept an arbitrary FLAMO model. To test
the three filtered examples, the benchmark has a separate experimental
adapter. No public FLAMO solver API or production root-finding algorithm
was changed.

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
