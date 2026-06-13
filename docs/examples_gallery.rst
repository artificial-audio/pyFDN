================
Examples Gallery
================

Interactive Marimo notebooks demonstrating the core features of ``pyFDN``.
Click any example to open the fully rendered notebook on its own page. Each
example can also be downloaded as a ``.py`` file or viewed directly on GitHub.

----

Getting Started
---------------

.. list-table::
   :widths: 35 65
   :header-rows: 0

   * - `Vanilla FDN <_static/marimo/notebooks/example_vanilla_FDN.html>`_
     - Build a basic FDN from scratch with orthogonal feedback matrices and delay lines.
   * - `Process FDN <_static/marimo/notebooks/example_process_fdn.html>`_
     - Run audio through an FDN and listen to the output.

Absorption & Filters
---------------------

.. list-table::
   :widths: 35 65
   :header-rows: 0

   * - `Absorption Filters <_static/marimo/notebooks/example_absorption_filters.html>`_
     - Design frequency-dependent absorption filters for realistic room acoustics.
   * - `First-Order Absorption <_static/marimo/notebooks/example_first_order_absorption.html>`_
     - First-order shelving filter design targeting specific RT60 values at DC and Nyquist.

Advanced Topics
----------------

.. list-table::
   :widths: 35 65
   :header-rows: 0

   * - `Colorless FDN <_static/marimo/notebooks/example_colorless_FDN.html>`_
     - Design FDNs with flat frequency response to avoid metallic colouration.
   * - `Coupled Rooms <_static/marimo/notebooks/example_coupled_rooms.html>`_
     - Simulate coupled room acoustics by connecting multiple FDN structures.
   * - `Delay Matrix Density <_static/marimo/notebooks/example_delay_matrix_density.html>`_
     - Analyse echo density and mixing behaviour of different delay-line topologies.
   * - `Interpolate Matrix <_static/marimo/notebooks/example_interpolate_matrix.html>`_
     - Smoothly interpolate between orthogonal feedback matrices.
   * - `DSS to State-Space <_static/marimo/notebooks/example_dss_to_ss.html>`_
     - Convert delay state-space representations to standard state-space form.

Translation Examples
---------------------

.. list-table::
   :widths: 35 65
   :header-rows: 0

   * - `Direct DSS→PR <_static/marimo/notebooks/example_dss_to_pr_direct.html>`_
     - Numeric DSS-only pole-residue extraction with ``eig``, ``roots``, and ``polyeig`` modes, compared against the time-domain impulse response.
   * - `FLAMO DSS→PR <_static/marimo/notebooks/example_dss_to_pr_flamo.html>`_
     - In-depth DSS→pole-residue conversion with an SOS filter in the loop, including the refinement-fix derivation.
   * - `DSS to Transfer Function <_static/marimo/notebooks/example_dss_to_tf.html>`_
     - Convert the delay state-space form of an FDN into matrix transfer-function form and verify against direct simulation.
   * - `SDN Coefficients <_static/marimo/notebooks/example_sdn_coefficients.html>`_
     - Compute room-acoustics coefficients with a Scattering Delay Network and render the impulse response via FLAMO.

Allpass FDN Examples
--------------------

.. list-table::
   :widths: 35 65
   :header-rows: 0

   * - `Allpass FDN Completion <_static/marimo/notebooks/allpass_FDN_example_allpass_FDN_completion.html>`_
     - Construct ``b``, ``c``, and ``d`` for a given feedback matrix so the FDN becomes uniallpass.
   * - `Homogeneous Allpass (MIMO) <_static/marimo/notebooks/allpass_FDN_example_allpass_FDN_homogeneous_mimo.html>`_
     - MIMO allpass FDN with homogeneous decay, giving every pole the same decay rate with extra degrees of freedom.
   * - `Homogeneous Allpass (SISO) <_static/marimo/notebooks/allpass_FDN_example_allpass_FDN_homogeneous_siso.html>`_
     - SISO allpass FDN with homogeneous decay so all poles share the same decay rate.
   * - `Allpass FDN in FDN <_static/marimo/notebooks/allpass_FDN_example_allpass_FDN_in_FDN.html>`_
     - Embed an allpass MIMO FDN inside a larger FDN loop, with single input and stereo output.
   * - `Nested Allpass (Gardner) <_static/marimo/notebooks/allpass_FDN_example_allpass_FDN_nested.html>`_
     - Gardner's nested allpass structure, built by iteratively nesting feedforward/back allpass sections.
   * - `Allpass but not Uniallpass <_static/marimo/notebooks/allpass_FDN_example_allpass_FDN_not_uniallpass.html>`_
     - An FDN that is allpass only for specific delay lengths, built via a non-diagonal similarity transform.
   * - `Poletti's Allpass (MIMO) <_static/marimo/notebooks/allpass_FDN_example_allpass_FDN_poletti.html>`_
     - Poletti's unitary MIMO reverberator for reduced colouration in assisted reverberation systems.
   * - `Schroeder Series Allpass <_static/marimo/notebooks/allpass_FDN_example_allpass_FDN_schroeder.html>`_
     - Schroeder's cascade of first-order allpass sections realised as an FDN with diagonal feedback.
   * - `Schroeder Allpass in Loop <_static/marimo/notebooks/allpass_FDN_example_allpass_FDN_schroeder_in_loop.html>`_
     - Place Schroeder allpass filters behind the delays in the FDN loop to increase echo density, rendered with FLAMO.

----

.. note::

   All notebooks are located in the ``examples/`` directory of the repository.
   You can also browse them directly on
   `GitHub <https://github.com/artificial-audio/pyFDN/tree/main/examples>`_.
