================
Examples Gallery
================

Interactive Jupyter notebooks demonstrating the core features of ``pyFDN``.
Each example can be downloaded as a ``.ipynb`` file or viewed directly on GitHub.

----

Getting Started
---------------

.. list-table::
   :widths: 35 65
   :header-rows: 0

   * - :doc:`Vanilla FDN <examples/example_vanilla_FDN>`
     - Build a basic FDN from scratch with orthogonal feedback matrices and delay lines.
   * - :doc:`Process FDN <examples/example_process_fdn>`
     - Run audio through an FDN and listen to the output.

Absorption & Filters
---------------------

.. list-table::
   :widths: 35 65
   :header-rows: 0

   * - :doc:`Absorption Filters <examples/example_absorption_filters>`
     - Design frequency-dependent absorption filters for realistic room acoustics.
   * - :doc:`One-Pole Absorption <examples/example_one_pole_absorption>`
     - One-pole filter design targeting specific RT60 values at DC and Nyquist.
   * - :doc:`z-Domain Filters <examples/example_zFilter>`
     - Working with z-domain filter representations (ZFilter, ZFIR, ZSOS, ZTF).

Advanced Topics
----------------

.. list-table::
   :widths: 35 65
   :header-rows: 0

   * - :doc:`Colorless FDN <examples/example_colorless_FDN>`
     - Design FDNs with flat frequency response to avoid metallic colouration.
   * - :doc:`Coupled Rooms <examples/example_coupled_rooms>`
     - Simulate coupled room acoustics by connecting multiple FDN structures.
   * - :doc:`Delay Matrix Density <examples/example_delay_matrix_density>`
     - Analyse echo density and mixing behaviour of different delay-line topologies.
   * - :doc:`Interpolate Matrix <examples/example_interpolate_matrix>`
     - Smoothly interpolate between orthogonal feedback matrices.
   * - :doc:`DSS to State-Space <examples/example_dss_to_ss>`
     - Convert delay state-space representations to standard state-space form.

----

.. note::

   All notebooks are located in the ``examples/`` directory of the repository.
   You can also browse them directly on
   `GitHub <https://github.com/artificial-audio/pyFDN/tree/main/examples>`_.

.. toctree::
   :hidden:

   examples/example_vanilla_FDN
   examples/example_process_fdn
   examples/example_absorption_filters
   examples/example_one_pole_absorption
   examples/example_zFilter
   examples/example_colorless_FDN
   examples/example_coupled_rooms
   examples/example_delay_matrix_density
   examples/example_interpolate_matrix
   examples/example_dss_to_ss
   examples/allpass_FDN/example_allpass_FDN_completion
   examples/allpass_FDN/example_allpass_FDN_homogeneous_mimo
   examples/allpass_FDN/example_allpass_FDN_homogeneous_siso
   examples/allpass_FDN/example_allpass_FDN_in_FDN
   examples/allpass_FDN/example_allpass_FDN_nested
   examples/allpass_FDN/example_allpass_FDN_not_uniallpass
   examples/allpass_FDN/example_allpass_FDN_poletti
   examples/allpass_FDN/example_allpass_FDN_schroeder
   examples/allpass_FDN/example_allpass_FDN_schroeder_in_loop
   examples/example_dss_to_pr_direct
   examples/example_dss_to_pr_flamo
   examples/example_dss_to_tf
   examples/example_sdn_coefficients

