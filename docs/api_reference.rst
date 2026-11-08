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
   pyFDN.householder_matrix
   pyFDN.anderson_matrix
   pyFDN.complete_orthogonal
   pyFDN.nearest_orthogonal
   pyFDN.nearest_sign_agnostic_orthogonal
   pyFDN.schroeder_reverberator
   pyFDN.allpass_in_fdn
   pyFDN.cascaded_paraunitary_matrix
   pyFDN.paraunitary_from_elementals
   pyFDN.tiny_rotation_matrix
   pyFDN.rotation_matrix_from_angles
   pyFDN.rotation_kernel
   pyFDN.reflection_kernel
   pyFDN.kronecker_angles
   pyFDN.kronecker_matrix
   pyFDN.kronecker_transform
   pyFDN.fdn_matrix_gallery
   pyFDN.fdn_system_gallery
   pyFDN.FEEDBACK_MATRIX_TYPES
   pyFDN.FILTER_MATRIX_TYPES
   pyFDN.SYSTEM_TYPES
   pyFDN.filter_matrix_gallery
   pyFDN.fdn_dss_gallery
   pyFDN.sample_delay_lengths
   pyFDN.DSS

Allpass FDN
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.homogeneous_allpass_fdn
   pyFDN.random_admissible_homogeneous_allpass
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
   pyFDN.sdn_to_flamo
   pyFDN.plot_sdn

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
used by those modules.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.EQDesign
   pyFDN.decay_to_graphic_eq
   pyFDN.decay_to_first_order_shelf
   pyFDN.decay_to_one_pole
   pyFDN.decay_to_two_stage
   pyFDN.gain_to_graphic_eq
   pyFDN.gain_to_bounded_graphic_eq
   pyFDN.gain_to_first_order_shelf
   pyFDN.gain_to_one_pole
   pyFDN.gain_to_two_stage
   pyFDN.graphic_eq_design_matrix
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
   pyFDN.td.ControllableFullWaveRectifier
   pyFDN.td.SignalDependentFractionalDelay
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
   pyFDN.matrix_group_delay
   pyFDN.ms_to_samples
   pyFDN.flamo_to_impz

Building a FLAMO Graph
----------------------

An FDN as FLAMO modules, assembled from numpy values. The three filter hooks --
``post_delay`` inside the loop, ``post_matrix`` on the feedback path,
``post_output`` on the wet signal -- are the same three
:func:`pyFDN.dss_to_td` takes, in the same positions and under the same
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
   pyFDN.AttenuationFilter
   pyFDN.OutputEQ

Polynomial & Matrix Maths
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.adjugate_polynomial
   pyFDN.adjugate
   pyFDN.det_polynomial
   pyFDN.general_char_poly
   pyFDN.interpolate_orthogonal
   pyFDN.is_orthogonal
   pyFDN.is_unilossless
   pyFDN.loop_transfer_function
   pyFDN.matrix_convolution
   pyFDN.outer_sum_approximation

General Utilities
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.lin_to_db
   pyFDN.sq_to_db
   pyFDN.fade_out
   pyFDN.hertz_to_unit
   pyFDN.hertz_to_rad
   pyFDN.rad_to_hertz
   pyFDN.is_bounding_curve
   pyFDN.max_corr
   pyFDN.mulaw_encode
   pyFDN.peak_normalize
   pyFDN.pole_boundaries

Saving, Packaged Examples & References
--------------------------------------

:func:`pyFDN.save_fdn` writes a :class:`~pyFDN.DSS` as readable JSON and
pickles anything else -- a ``td`` graph, a FLAMO model -- with the pyFDN and
Python versions that wrote it. Loading a pickle runs code from the file, so
only load pickles you trust. The packaged presets are DSS JSON files.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.available_audio
   pyFDN.audio_metadata
   pyFDN.load_audio
   pyFDN.available_fdn_presets
   pyFDN.load_fdn_preset
   pyFDN.save_fdn
   pyFDN.load_fdn
   pyFDN.resave_fdn
   pyFDN.paper_reference
   pyFDN.paper_link

State-Space Translators
-----------------------

Convert a delay state-space (DSS) system -- ``A``, ``B``, ``C``, ``D``,
``delays`` -- to another representation. A :class:`~pyFDN.DSS` unpacks
straight into each of them, with its filters alongside:
``pyFDN.dss_to_td(*dss, **filters)``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.dss_to_ss
   pyFDN.dss_to_impz
   pyFDN.dss_to_td
   pyFDN.dss_to_tf
   pyFDN.dss_to_pr
   pyFDN.dss_to_flamo
   pyFDN.flamo_to_dss
   pyFDN.flamo_to_pr
   pyFDN.flamo_decompose_for_pr
   pyFDN.FlamoDecompositionForPR
   pyFDN.impz_to_res
   pyFDN.tf_to_impz
   pyFDN.pr_to_impz
   pyFDN.PoleResidue

Experimental Lindsey–Fox modal decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``mode="lindsey_fox"`` to :func:`pyFDN.dss_to_pr` or
:func:`pyFDN.flamo_to_pr` for FFT grid search with Laguerre refinement.
The existing defaults remain ``"eig"`` for static DSS and ``"eai"`` for FLAMO.
The opt-in route checks completeness and the all-port response. Failed search,
completion, or response validation raises without switching to EAI. Direct EAI
calls may accept incomplete results with explicit ``allow_approximate=True``
and successful response checks; this option does not relax LF acceptance.

:func:`pyFDN.general_char_poly` recovers the static characteristic
coefficients once from determinant samples and reuses them on every search
circle. The static ``"roots"`` backend uses these FFT coefficients too.

All modal backends report solver and pole-count metadata. Complete eig, roots,
and direct EAI results have ``responseValidation=None``; LF always checks the
response. Unsupported multiple poles raise instead of producing zero residues.
Residues follow ``H(z) = D + sum_k R_k / (z - p_k)``; a simple pole at zero
contributes a unit delay.

Usage, supported models, numerical acceptance criteria, and a reproducible
benchmark on the modal examples:

.. toctree::
   :maxdepth: 1

   lindsey_fox_examples

FDN Processing
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.process_dss
   pyFDN.flamo_process

Training
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.trainable_fdn
   pyFDN.trainable_from_dss
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

Plotting
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.plot_matrix
   pyFDN.plot_matrix_grid
   pyFDN.plot_system_matrix
   pyFDN.plot_fdn_parameter
   pyFDN.plot_db_per_sample
   pyFDN.plot_impulse_response
   pyFDN.plot_impulse_response_matrix
   pyFDN.plot_edc
   pyFDN.plot_spectrogram
   pyFDN.animate

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
