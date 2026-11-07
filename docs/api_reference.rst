=============
API Reference
=============

All functions and classes are accessible from the top-level ``pyFDN`` namespace:

.. code-block:: python

   import pyFDN
   feedback = pyFDN.random_orthogonal(4)

The reference is organised by functional area, mirroring the package's module
structure. It covers the headline public API; a small number of low-level
helpers are exported for advanced/composability use but intentionally omitted
here (see ``tests/test_api_reference.py``).

----

Matrix Generators
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.random_orthogonal
   pyFDN.random_matrix_shift
   pyFDN.shift_matrix
   pyFDN.shift_matrix_distribute
   pyFDN.householder_matrix
   pyFDN.anderson_matrix
   pyFDN.complete_orthogonal
   pyFDN.nearest_orthogonal
   pyFDN.nearest_sign_agnostic_orthogonal
   pyFDN.degree_one_lossless
   pyFDN.schroeder_reverberator
   pyFDN.allpass_in_fdn
   pyFDN.construct_cascaded_paraunitary_matrix
   pyFDN.construct_paraunitary_from_elementals
   pyFDN.construct_velvet_feedback_matrix
   pyFDN.tiny_rotation_matrix
   pyFDN.rotation_matrix_from_angles
   pyFDN.rotation_kernel
   pyFDN.reflection_kernel
   pyFDN.kronecker_angles
   pyFDN.kronecker_matrix
   pyFDN.kronecker_transform
   pyFDN.fdn_matrix_gallery
   pyFDN.fdn_system_gallery
   pyFDN.filter_matrix_gallery
   pyFDN.fdn_build_gallery
   pyFDN.sample_delay_lengths
   pyFDN.FDNSystem
   pyFDN.FDNBuild

Allpass FDN
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.homogeneous_allpass_fdn
   pyFDN.rand_admissible_homogeneous_allpass
   pyFDN.complete_fdn
   pyFDN.complete_full_mimo_halmos
   pyFDN.complete_general_mimo_svd
   pyFDN.nested_allpass
   pyFDN.poletti_allpass
   pyFDN.series_allpass
   pyFDN.is_allpass
   pyFDN.is_uniallpass
   pyFDN.is_paraunitary

Scattering Delay Network
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.SDN

Acoustics & Absorption
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.sos_gain_per_sample_curves
   pyFDN.echo_density
   pyFDN.edc
   pyFDN.estimate_initial_level_bands
   pyFDN.estimate_rt_bands
   pyFDN.octave_bands
   pyFDN.octave_band_filterbank
   pyFDN.rt_to_gain_per_sample
   pyFDN.rt_to_slope
   pyFDN.slope_amplitude_to_level
   pyFDN.slope_to_rt

EQ Design (``pyFDN.eq``)
-------------------------

Explicit functions map either decay targets or gain targets onto a named
filter design. The same functions run in NumPy or Torch; the trainable
:class:`pyFDN.AttenuationFilter` and :class:`pyFDN.OutputEQ` modules use these
mappings inside a training loop. ``EQDesign`` is the literal choice of
``"graphic_eq"``, ``"first_order_shelf"``, ``"one_pole"``, or ``"two_stage"``
used by those modules. The target-to-EQ functions accept ``return_design=True`` when their
JSON-compatible design record is also needed for an ``FDNPreset``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.EQDesign
   pyFDN.decay_to_geq
   pyFDN.decay_to_first_order_shelf
   pyFDN.decay_to_one_pole
   pyFDN.decay_to_two_stage
   pyFDN.gain_to_geq
   pyFDN.gain_to_bounded_geq
   pyFDN.gain_to_first_order_shelf
   pyFDN.gain_to_one_pole
   pyFDN.gain_to_two_stage
   pyFDN.geq_design_matrix
   pyFDN.lowshelf_biquad
   pyFDN.highshelf_biquad
   pyFDN.peaking_biquad
   pyFDN.first_order_shelf_biquad
   pyFDN.one_pole_biquad
   pyFDN.probe_sos

Time-Domain Graph (``pyFDN.td``)
--------------------------------

Stateful block-processing operators, wired into a graph by the connectors and
rendered with ``.process_signal(signal)``. See :mod:`pyFDN.td`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.td.TimeOperator
   pyFDN.td.Identity
   pyFDN.td.Gain
   pyFDN.td.Delay
   pyFDN.td.AbsoluteValue
   pyFDN.td.DCBlocker
   pyFDN.td.ControllableFullWaveRect
   pyFDN.td.SDFD
   pyFDN.td.RingModulator
   pyFDN.td.PitchShift
   pyFDN.td.GranularPitchShift
   pyFDN.td.SOSBank
   pyFDN.td.MatrixFIR
   pyFDN.td.MatrixConvolver
   pyFDN.td.KroneckerMatrix
   pyFDN.td.TimeVaryingMatrix
   pyFDN.td.TimeVaryingKroneckerMatrix
   pyFDN.td.RecursionState
   pyFDN.td.Series
   pyFDN.td.Parallel
   pyFDN.td.Recursion

Delay Utilities
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.matrix_delay_approximation
   pyFDN.mgrpdelay
   pyFDN.ms_to_smp
   pyFDN.flamo_time_response
   pyFDN.flamo_freq_response

Building a FLAMO Graph
----------------------

An FDN as FLAMO modules, assembled from numpy values. The three filter hooks --
``post_delay`` inside the loop, ``post_matrix`` on the feedback path,
``post_output`` on the wet signal -- are the same three
:func:`pyFDN.process_dss` takes, in the same positions and under the same
names.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.assemble_fdn_core
   pyFDN.wrap_fdn_shell
   pyFDN.gain_module
   pyFDN.delay_module
   pyFDN.matrix_module
   pyFDN.fir_matrix_module
   pyFDN.sos_filter_module
   pyFDN.hook_module
   pyFDN.AttenuationFilter
   pyFDN.OutputEQ

Polynomial & Matrix Maths
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.adj_poly
   pyFDN.adjugate
   pyFDN.det_polynomial
   pyFDN.general_char_poly
   pyFDN.general_char_poly_fft
   pyFDN.interpolate_orthogonal
   pyFDN.is_orthogonal
   pyFDN.is_unilossless
   pyFDN.loop_tf
   pyFDN.matrix_convolution
   pyFDN.matrix_polyder
   pyFDN.matrix_polyval
   pyFDN.matrix_sqrt
   pyFDN.negpolyder
   pyFDN.outer_sum_approximation
   pyFDN.poly_degree
   pyFDN.polyder_rational
   pyFDN.polydiag

General Utilities
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.db_to_lin
   pyFDN.db_to_sq
   pyFDN.lin_to_db
   pyFDN.sq_to_db
   pyFDN.ensure_3d
   pyFDN.fade_out
   pyFDN.hertz_to_unit
   pyFDN.hertz_to_rad
   pyFDN.rad_to_hertz
   pyFDN.is_bounding_curve
   pyFDN.last_nonzero_indices
   pyFDN.max_corr
   pyFDN.mulaw_decode
   pyFDN.mulaw_encode
   pyFDN.peak_normalize
   pyFDN.pole_boundaries
   pyFDN.skew

Build Files, Packaged Examples & References
--------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.available_audio
   pyFDN.audio_metadata
   pyFDN.load_audio
   pyFDN.available_fdn_presets
   pyFDN.get_fdn_preset
   pyFDN.FDNPreset
   pyFDN.fdn_preset_to_dict
   pyFDN.fdn_preset_from_dict
   pyFDN.save_fdn_preset
   pyFDN.load_fdn_preset
   pyFDN.fdn_build_to_dict
   pyFDN.fdn_build_from_dict
   pyFDN.save_fdn_build
   pyFDN.load_fdn_build
   pyFDN.paper_reference
   pyFDN.paper_link

State-Space Translators
-----------------------

Convert a delay state-space (DSS) system -- ``A``, ``B``, ``C``, ``D``,
``delays`` -- to another representation. Each ``build_to_*`` function is a
thin wrapper over its ``dss_to_*`` counterpart that unpacks an
:class:`~pyFDN.FDNBuild` instead.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.dss_to_ss
   pyFDN.dss_to_impz
   pyFDN.build_to_impz
   pyFDN.dss_to_td
   pyFDN.build_to_td
   pyFDN.dss_to_tf
   pyFDN.dss_to_pr
   pyFDN.dss_to_flamo
   pyFDN.build_to_flamo
   pyFDN.flamo_to_pr
   pyFDN.flamo_decompose_for_pr
   pyFDN.flamo_extract_pr_decomposition
   pyFDN.FlamoDecompositionForPR
   pyFDN.impz_to_res
   pyFDN.mtf_to_impz
   pyFDN.pr_to_impz

Experimental Lindsey–Fox modal decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dss_to_pr(delays, A, B, C, D, mode="lindsey_fox")`` tests an FFT
grid-search alternative to the existing ``eig``, ``roots``, and ``eai``
backends. It searches the generalized characteristic polynomial on concentric
circles, polishes 3-by-3 modulus minima with Laguerre iterations, and checks
both the root count and reconstructed coefficients before extracting residues.
The coefficients are constructed once by :func:`pyFDN.general_char_poly_fft`:
sample ``det(I - diag(w**delays) @ A)`` on one unit circle, then recover the
``M+1`` coefficients with an inverse FFT, where ``M = sum(delays)``. Each
search circle reuses these coefficients with radius weighting and an FFT;
no further matrix determinants are needed during grid scanning or polishing.
The original :func:`pyFDN.general_char_poly` remains the principal-minor
reference used by the existing ``roots`` backend and other callers.
The annulus comes from the singular values of the static feedback matrix and
the smallest/largest delays. The default remains ``mode="eig"``.

This independent, restricted implementation follows Sitton, Burrus, Fox and
Treitel, `Factoring very-high-degree polynomials (2003)
<https://doi.org/10.1109/MSP.2003.1253552>`_, with Laguerre polishing as in
the authors' `2006 description
<https://www.ece.rice.edu/dsp/software/polyhelp/LindseyFox3Stages.htm>`_.
It requires real, nonsingular static feedback, positive integer delays, and
simple, sufficiently separated poles. It retries with finer grids but does
not implement quotient deflation, repeated-root recovery, or the reference
program's reciprocal search outside the unit circle. Failure raises
``RuntimeError`` in the scalar root kernel. The public modal route instead
refines against the original FDN matrix, checks the all-port response, and
falls back to EAI with a warning and metadata if LF fails. The fallback must
pass the same count and response checks. ``lf_max_seconds`` bounds only root
search; default eig/EAI behavior is unchanged.
Polynomial construction costs ``O(L N^3 + L log L)`` for ``N`` delay lines
and an FFT length ``L > M``. Determinants are evaluated in batches to avoid
storing all sampled matrices. Small coefficients can still be obscured by
determinant roundoff; terms are not thresholded or trimmed, and the known
constant/terminal coefficients are set directly to ``1`` and ``det(-A)``.
Conjugate pairing and polishing remain quadratic in the number of poles.
This is an experiment for long-delay FDNs, not a general replacement for
the existing methods.

``flamo_to_pr(model, mode="lindsey_fox", num_poles=known_order)`` provides
the same policy for real float64 canonical FLAMO FDNs, clearing known SOS
denominators before coefficient recovery. FIR loops require an explicit
order. Incomplete EAI fallback results are rejected unless
``allow_approximate=True`` and both all-port and external response checks
pass. Metadata distinguishes complete decompositions from validated
approximations. These sampled checks are not uniform response error bounds;
see :doc:`lindsey_fox_examples` for acceptance criteria and the 30-case run.

To compare one-time coefficient construction against principal minors::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_general_char_poly.py --lines 4 8 12 16 --repeat 3

One local run on macOS arm64/Python 3.11.15 with seeded ``A = 0.9 Q`` and
delays ``31 + 2 * arange(N)`` gave these median construction times in seconds:

.. list-table::
   :header-rows: 1

   * - Delay lines
     - Degree
     - Principal minors
     - FFT recovery
   * - 4
     - 136
     - 0.000078
     - 0.000102
   * - 8
     - 304
     - 0.001144
     - 0.000290
   * - 12
     - 504
     - 0.019713
     - 0.000468
   * - 16
     - 736
     - 0.343874
     - 0.001538

The maximum coefficient difference was below ``1e-14`` in all four cases.

Run the reproducible acceptance benchmark from the checkout::

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_lindsey_fox.py --scales 1 4 10 --repeat 3 --modes eig roots lindsey_fox eai

The benchmark reports complete decomposition time (including residues), pole
agreement with the first requested backend, and reconstructed impulse-response
error against time-domain simulation. Imports are warmed up, but each timed
EAI call includes its model construction. Timing assertions are intentionally
excluded from CI. One local run on macOS arm64, Python 3.11.15, yielded these
median times in seconds for a seeded three-line FDN with ``A = 0.95 Q``:

.. list-table::
   :header-rows: 1

   * - Poles
     - eig
     - roots
     - lindsey_fox
     - eai
   * - 117
     - 0.019031
     - 0.006029
     - 0.003532
     - 0.099798
   * - 450
     - 0.414931
     - 0.110904
     - 0.016855
     - 0.400557
   * - 1116
     - 2.857218
     - 0.723895
     - 0.057545
     - 1.021086

Across this run, all non-reference pole errors were below ``4e-14`` and all
impulse-response errors below ``1.3e-13``. These timings establish a promising
case for this implementation on one FDN family, not a universal speedup.

Coverage and timings on the repository's static, FIR, and SOS modal examples:

.. toctree::
   :maxdepth: 1

   lindsey_fox_examples

FDN Processing
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.process_dss
   pyFDN.process_fdn
   pyFDN.flamo_process

Training
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.build_fdn
   pyFDN.trainable_from_build
   pyFDN.trainable_from_preset
   pyFDN.build_set_decay
   pyFDN.Trainable
   pyFDN.train_fdn
   pyFDN.TrainLog
   pyFDN.LOSSLESS_ALIAS_DECAY_DB

Training Objectives
-------------------

An objective is a weighted sum of losses, composed with ``+`` and ``*``. Losses
on the impulse response read a :class:`pyFDN.Response`; losses on a model
parameter take a :class:`pyFDN.ParamRef` from :func:`pyFDN.param`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.Response
   pyFDN.model_response
   pyFDN.param
   pyFDN.params
   pyFDN.ParamRef
   pyFDN.Loss
   pyFDN.Match
   pyFDN.FlatMagnitude
   pyFDN.AsymmetricFlatMagnitude
   pyFDN.FlatSpectrogram
   pyFDN.SpectralFlatness
   pyFDN.MatchMagnitude
   pyFDN.MatchPhase
   pyFDN.MatchPhaseSpectrogram
   pyFDN.MatchMelMagnitude
   pyFDN.MatchSpectrogram
   pyFDN.MatchMelSpectrogram
   pyFDN.MatchImpulseResponse
   pyFDN.MatchEnergyDecay
   pyFDN.MatchCumulativeEnergy
   pyFDN.MatchMelEnergyDecayRelief
   pyFDN.MatchEchoDensity
   pyFDN.MatchEnvelope
   pyFDN.Energy
   pyFDN.Sparsity
   pyFDN.L1
   pyFDN.L2
   pyFDN.mimo_rir_eigenvalues_per_frequency
   pyFDN.energy_decay_curve

Plotting
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.plot_matrix
   pyFDN.plot_matrix_grid
   pyFDN.plot_system_matrix
   pyFDN.plot_fdn_parameter
   pyFDN.plot_fdn_build
   pyFDN.plot_db_per_sample
   pyFDN.plot_impulse_response
   pyFDN.plot_impulse_response_matrix
   pyFDN.plot_edc
   pyFDN.plot_spectrogram
   pyFDN.animate
   pyFDN.downsample_minmax

Notebook Display
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.labeled_audio

FLAMO Graph
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.plot_flamo_graph
   pyFDN.flamo_model_to_nodes
   pyFDN.flamo_nodes_flat
   pyFDN.extract_build
