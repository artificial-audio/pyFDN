Lindsey–Fox modal decomposition
================================

Usage and supported models
--------------------------

Lindsey–Fox (LF) is an opt-in alternative for modal decomposition. The existing
DSS ``eig`` and FLAMO ``eai`` defaults are unchanged::

   pr, info = pyFDN.dss_to_pr(delays, A, B, C, D, mode="lindsey_fox")

   pr, info = pyFDN.flamo_to_pr(model, mode="lindsey_fox", num_poles=known_order)
   ir = pyFDN.pr_to_impz(*pr, ir_len)

The static DSS path requires finite real matrices, nonsingular ``A``, and
positive integer delays. It remains NumPy/SciPy-only, including on failure.
FLAMO requires a real float64 canonical FDN. Static, FIR matrix, and parallel
SOS feedback loops are supported; give ``num_poles`` explicitly for FIR loops.
Filter dynamics are included in the detected order of SOS loops.

``lf_max_seconds=3.0`` limits root search between numerical operations. One
operation may overrun the budget. Coefficient construction, matrix refinement,
residue extraction, and response validation are outside it.
This is an experimental solver for simple, sufficiently separated roots;
repeated roots and tightly clustered filter-zero modes can prevent completion.
A speedup is not guaranteed.

Acceptance and failure
----------------------

The scalar search must return the full unique root count, normalized backward
residuals at most ``1e-10``, and normalized coefficient reconstruction error
at most ``1e-7``. Roots are then refined against the original matrix. Real
systems refine one representative per conjugate pair before residue extraction.

Every internal input/output port is checked, including modes hidden by the
user's ``B`` or ``C``. FLAMO also checks the external response to reject dynamics
outside the modal representation. Validation uses ``max(256, 2*order+1)``
off-grid points outside the unit circle and every returned pole, at radius
``max(1, max(abs(poles))) * exp(1/max(4096, 8*order))``. The contour corresponds
to exponential windowing; this sampled acceptance test is not a uniform error
guarantee on the unit circle.

Both relative L2 error and peak error normalized by the reference peak must be
below ``response_tolerance=1e-8``. Search, pairing, completeness, residue, or
response failures raise directly. LF never calls EAI or another root solver,
and a budget or completeness failure does not return a partial decomposition.
Shared SVD refinement, residue extraction, and response validation do not run
Ehrlich-Aberth iteration.

Inspect ``meta['solver']``, ``requestedSolver``,
``expectedPoles``, ``returnedPoles``, ``complete``, ``approximate``, and
``responseValidation``. The latter reports ``relativeL2``, ``normalizedPeak``,
``points``, ``radius``, and ``referenceProbes``. For static DSS,
``errorBound=True`` identifies conservative upper bounds rather than measured
errors. Other cases use direct response evaluation. FLAMO additionally reports
``externalResponseValidation``.

LF requires a complete decomposition. Two-tap and IIR absorption can create
unresolved groups near filter zeros, which LF may fail to resolve within its
budget. Callers can run EAI, ``flamo_to_pr``'s default, as a separate solve.
For EAI only, ``allow_approximate=True`` permits a nonempty incomplete result
if both internal and external response checks pass. A warning and metadata
``complete=False``, ``approximate=True``, and ``omittedPoleCount`` report that
approximation. This does not establish where the omitted poles lie or make
their count a measure of response error. The flag does not change LF's contract.

Algorithm and numerical limits
-------------------------------

This independent implementation follows Sitton, Burrus, Fox and Treitel,
`Factoring very-high-degree polynomials (2003)
<https://doi.org/10.1109/MSP.2003.1253552>`_, with Laguerre polishing as in the
authors' `2006 description
<https://www.ece.rice.edu/dsp/software/polyhelp/LindseyFox3Stages.htm>`_. It is
not a port of the reference MATLAB implementation and does not implement its
quotient deflation, repeated-root recovery, or reciprocal search stage.

For static DSS, :func:`pyFDN.general_char_poly` samples
``det(I - diag(w**delays) @ A)`` on one circle and recovers ``M+1`` coefficients,
where ``M=sum(delays)``. Determinants are batched; cost is
``O(L*N**3 + L*log(L))`` for the next power of two ``L>M``. Coefficients are
constructed once and reused on every search circle, without principal-minor
enumeration.
Small coefficients can be obscured by determinant roundoff; terms are not
trimmed. FLAMO clears known SOS denominators before coefficient recovery.

The search uses 3-by-3 modulus minima on concentric FFT circles, independent
Laguerre polishing, up to ten finer grids, and at most ``2**21`` FFT samples.
Large batches evaluate Euler derivatives by FFT and an order-18 local Taylor
expansion; small batches use chunked Horner evaluation. Converged candidates
stop iterating. Spatial trees remove duplicates and a balanced FFT product
checks reconstructed coefficients. FFT residual checks include Taylor and
floating-point allowances. These use ordinary IEEE arithmetic, not formal
interval certificates.

For degree at least 2048, near-complete sets may try up to three curvature
seeding passes for close missing partners. At a simple root, ``P''/(2P')``
is the sum of inverse gaps to other roots. Removing the uniform-circle
background ``(degree-1)/(2*z)`` gives a heuristic partner estimate. Seeds must
lie within ``abs(z)/degree`` and away from existing roots, and use the same
Laguerre polishing and acceptance checks. This implementation-specific
heuristic leaves the 3-by-3 stencil unchanged and is not an EAI hybrid.

Conjugate pairing uses sorted matching and bounded local assignments, with
linear storage. Ambiguous tolerance-connected components above 512 poles
raise instead of allocating a global quadratic cost matrix. Internal response
validation stores rank-one residue factors; public residue arrays retain their
usual shape. Direct all-port validation can still require quadratic work in
pole count, so the solver is not intended as a general large-system guarantee.

Static DSS validation bound
----------------------------

For supported static systems, a residual bound can avoid dense modal response
sums on every validation point. Inconclusive or unsupported cases run the full
direct check with the same tolerance and grid. For ``H=P^-1``,

.. math::

   P(z)\widehat H(z)-I
   = Q(z)+\sum_k\frac{w_k P(p_k)u_k v_k}{z-p_k},
   \qquad P(z)=\operatorname{diag}(z^{m_i})-A.

``Q`` has degree at most ``max(delays)-1``. Its coefficients are modal moments,
accumulated in blocks of 32 with compensation, subtracting identity at power
zero. Absolute coefficient sums bound ``Q`` on the contour. Floating-point
allowances cover powers, short dot products, and compensated accumulation;
see Higham's `The Accuracy of Floating Point Summation (1993)
<https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf>`_. The following
residual bound is derived for this implementation.

Let ``alpha`` upper-bound ``||A||_2`` through Gershgorin bounds on ``A.T@A``,
and ``rho_lo``/``rho_hi`` enclose the validation radii. If
``d=min_i rho_lo**m_i-alpha>0``, the Frobenius error is bounded by

.. math::

   \|\widehat H(z)-H(z)\|_F
   \le \frac{\delta_Q+\sum_k\delta_k/|z-p_k|}{d},
   \qquad
   \delta_k\ge |w_k|\,\|P(p_k)u_k\|_2\,\|v_k\|_2.

The relative L2 denominator uses the lower bound
``||H||_F >= sqrt(N)/(rho_hi**max(m)+alpha)``. Up to 64 original reference
probes supply a lower bound on the reference peak, reduced by solve-error
allowances. Both error bounds must pass before direct evaluation is skipped.
The screen assumes ordinary IEEE round-to-nearest arithmetic and stable BLAS;
it is not directed-rounding interval arithmetic. It requires at least 512
poles and 32 ports, maximum delay 256, workspace at most 128 MiB, pole magnitudes
in ``[0.5,2]``, and a usable inverse bound. FLAMO uses direct validation.

Reproduce the example checks
----------------------------

From an editable checkout with the example dependencies installed, run::

   PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_lindsey_fox.py

This exercises all 30 repository modal configurations through the public
APIs and compares independent 48000-sample impulse responses for successful
solves. The runner asserts accuracy and completeness, with no timing assertions.
It runs notebook setup cells only; plotting and audio cells are excluded.
Static, Kronecker, paraunitary FIR, and SOS cases are included. The two absorption
cases may report an expected search failure; they never run EAI.

The recorded local run in ``benchmarks/lindsey_fox_examples.jsonl`` has 28
complete LF decompositions and two explicitly reported absorption search
failures. These results are sampled numerical evidence for these configurations,
not a guarantee for arbitrary models. To compare polynomial construction with principal minors::

   PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_general_char_poly.py --lines 4 8 12 16 --repeat 3
