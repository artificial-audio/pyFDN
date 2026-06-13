=============
API Reference
=============

All functions and classes are accessible from the top-level ``pyFDN`` namespace:

.. code-block:: python

   import pyFDN
   feedback = pyFDN.random_orthogonal(4)

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
   pyFDN.construct_cascaded_paraunitary_matrix
   pyFDN.construct_velvet_feedback_matrix
   pyFDN.tiny_rotation_matrix

Acoustics & Absorption
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.absorption_filters
   pyFDN.first_order_absorption
   pyFDN.first_order_shelving_eq
   pyFDN.one_pole_absorption
   pyFDN.echo_density
   pyFDN.edc
   pyFDN.absorption_to_rt
   pyFDN.rt_to_gain_per_sample
   pyFDN.rt_to_slope
   pyFDN.slope_to_rt

DSP Components
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.FeedbackDelay

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

   pyFDN.det_polynomial
   pyFDN.interpolate_orthogonal
   pyFDN.is_orthogonal
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
   pyFDN.is_bounding_curve
   pyFDN.last_nonzero_indices
   pyFDN.mulaw_decode
   pyFDN.mulaw_encode
   pyFDN.peak_normalize
   pyFDN.pole_boundaries
   pyFDN.skew
   pyFDN.is_almost_zero

State-Space Translators
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pyFDN.dss_to_ss
   pyFDN.dss_to_impz

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

   pyFDN.plot_db_per_sample
   pyFDN.plot_FDN_build
   pyFDN.plot_fdn_parameter
   pyFDN.plot_impulse_response
   pyFDN.plot_impulse_response_matrix
   pyFDN.plot_system_matrix
