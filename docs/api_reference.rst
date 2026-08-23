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
:class:`pyFDN.DecayFilter` and :class:`pyFDN.OutputEQ` modules use these
mappings inside a training loop. ``EQDesign`` is the literal choice of
``"graphic_eq"``, ``"first_order_shelf"``, or ``"one_pole"`` used by those
modules.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.EQDesign
   pyFDN.decay_to_geq
   pyFDN.decay_to_first_order_shelf
   pyFDN.decay_to_one_pole
   pyFDN.gain_to_geq
   pyFDN.gain_to_bounded_geq
   pyFDN.gain_to_first_order_shelf
   pyFDN.gain_to_one_pole
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
rendered with ``.process(signal)``. See :mod:`pyFDN.td`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.td.TimeOperator
   pyFDN.td.Identity
   pyFDN.td.Gain
   pyFDN.td.Delay
   pyFDN.td.AbsoluteValue
   pyFDN.td.SOSBank
   pyFDN.td.MatrixFIR
   pyFDN.td.MatrixConvolver
   pyFDN.td.TimeVaryingMatrix
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
:func:`pyFDN.process_fdn` takes, in the same positions and under the same names.

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
   pyFDN.DecayFilter
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
   pyFDN.load_fdn_preset
   pyFDN.fdn_build_to_dict
   pyFDN.fdn_build_from_dict
   pyFDN.save_fdn_build
   pyFDN.load_fdn_build
   pyFDN.paper_reference
   pyFDN.paper_link

State-Space Translators
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.dss_to_ss
   pyFDN.dss_to_impz
   pyFDN.build_to_impz
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

FDN Processing
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.process_fdn
   pyFDN.flamo_process

Training
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.build_fdn
   pyFDN.trainable_from_build
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
   pyFDN.FlatMagnitude
   pyFDN.AsymmetricFlatMagnitude
   pyFDN.FlatSpectrogram
   pyFDN.MatchMagnitude
   pyFDN.MatchSpectrogram
   pyFDN.MatchMelSpectrogram
   pyFDN.MatchImpulseResponse
   pyFDN.MatchEnergyDecay
   pyFDN.MatchCumulativeEnergy
   pyFDN.Energy
   pyFDN.Sparsity
   pyFDN.L1
   pyFDN.L2

Plotting
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.plot_matrix
   pyFDN.plot_matrix_grid
   pyFDN.plot_system_matrix
   pyFDN.plot_fdn_parameter
   pyFDN.plot_FDN_build
   pyFDN.plot_db_per_sample
   pyFDN.plot_impulse_response
   pyFDN.plot_impulse_response_matrix
   pyFDN.plot_edc
   pyFDN.plot_spectrogram
   pyFDN.animate
   pyFDN.downsampled_scatter
   pyFDN.downsample_minmax
   pyFDN.downsample_plotly_trace

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
