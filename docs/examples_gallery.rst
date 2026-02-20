================
Examples Gallery
================

Interactive Jupyter notebooks demonstrating the core features of ``pyFDN``.
Each example can be downloaded as a ``.ipynb`` file or viewed directly on GitHub.

----

Getting Started
---------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Vanilla FDN
      :link: examples/example_vanilla_FDN
      :link-type: doc

      Build a basic Feedback Delay Network from scratch with
      orthogonal feedback matrices and delay lines.

   .. grid-item-card:: Process FDN
      :link: examples/example_process_fdn
      :link-type: doc

      Run audio through an FDN and listen to the output.
      Covers the full signal-processing pipeline.

Absorption & Filters
---------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Absorption Filters
      :link: examples/example_absorption_filters
      :link-type: doc

      Design frequency-dependent absorption filters for
      realistic room acoustics simulation.

   .. grid-item-card:: One-Pole Absorption
      :link: examples/example_one_pole_absorption
      :link-type: doc

      One-pole filter design targeting specific RT60 values
      at DC and Nyquist frequencies.

   .. grid-item-card:: z-Domain Filters
      :link: examples/example_zFilter
      :link-type: doc

      Working with z-domain filter representations including
      ZFilter, ZFIR, ZSOS, and ZTF classes.

Advanced Topics
----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Colorless FDN
      :link: examples/example_colorless_FDN
      :link-type: doc

      Design FDNs with flat frequency response to avoid
      metallic colouration artefacts.

   .. grid-item-card:: Coupled Rooms
      :link: examples/example_coupled_rooms
      :link-type: doc

      Simulate coupled room acoustics by connecting
      multiple FDN structures.

   .. grid-item-card:: Delay Matrix Density
      :link: examples/example_delay_matrix_density
      :link-type: doc

      Analyse echo density and mixing behaviour of
      different delay-line topologies.

   .. grid-item-card:: Interpolate Matrix
      :link: examples/example_interpolate_matrix
      :link-type: doc

      Smoothly interpolate between orthogonal feedback
      matrices for time-varying reverberators.

   .. grid-item-card:: DSS to State-Space
      :link: examples/example_dss2ss
      :link-type: doc

      Convert delay state-space (DSS) representations
      to standard state-space form.

----

.. note::

   All notebooks are located in the ``examples/`` directory of the repository.
   You can also browse them directly on
   `GitHub <https://github.com/artificial-audio/pyFDN/tree/main/examples>`_.
