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
   pyFDN.FDNSystem

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

   pyFDN.absorption_filters
   pyFDN.first_order_absorption
   pyFDN.first_order_shelving_eq
   pyFDN.one_pole_absorption
   pyFDN.sos_gain_per_sample_curves
   pyFDN.echo_density
   pyFDN.edc
   pyFDN.absorption_to_rt
   pyFDN.estimate_initial_level_bands
   pyFDN.estimate_rt_bands
   pyFDN.rt_to_gain_per_sample
   pyFDN.rt_to_slope
   pyFDN.slope_to_rt

Graphic EQ
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.design_geq
   pyFDN.graphic_eq
   pyFDN.absorption_geq
   pyFDN.probe_sos
   pyFDN.shelving_filter
   pyFDN.bandpass_filter

DSP Components
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.FeedbackDelay
   pyFDN.FIRMatrixFilter
   pyFDN.SOSFilterBank

Delay Utilities
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.matrix_delay_approximation
   pyFDN.mgrpdelay
   pyFDN.ms_to_smp
   pyFDN.flamo_time_response

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

State-Space Translators
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.dss_to_ss
   pyFDN.dss_to_impz
   pyFDN.dss_to_tf
   pyFDN.dss_to_pr_direct
   pyFDN.dss_to_pr_flamo
   pyFDN.dss_to_flamo
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
   pyFDN.downsampled_scatter
   pyFDN.downsample_minmax
   pyFDN.downsample_plotly_trace

FLAMO Graph
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.plot_flamo_graph
   pyFDN.flamo_model_to_nodes
   pyFDN.flamo_nodes_flat
