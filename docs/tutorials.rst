=========
Tutorials
=========

Longer-form teaching material: slide decks and guided walkthroughs that use the
example notebooks rather than replacing them.

DAFx 2026 — Feedback Delay Networks in Python
---------------------------------------------

A 90-minute tutorial given at the 28th International Conference on Digital Audio
Effects (DAFx 2026), Cambridge, MA. It builds an FDN from first principles,
analyses what makes one sound good or bad, and then optimises it with gradients.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ▶ Slides
      :link: _static/tutorials/dafx2026/slides.html

      The reveal.js deck. Press ``S`` for the speaker notes, ``O`` for the
      slide overview, ``?`` for all shortcuts.

   .. grid-item-card:: Participant guide
      :link: _static/tutorials/dafx2026/index.html

      Setup instructions, the session schedule, and the list of notebooks
      used in each segment.

**Presenters:** Sebastian J. Schlecht (FAU Erlangen-Nürnberg) and
Facundo Franchino (MIT).

**Prerequisites:** Python and NumPy, plus basic DSP — filtering, convolution,
and what a pole is. No prior experience with FDNs.

The notebooks used in the hands-on segments are all in the
:doc:`examples gallery <examples_gallery>`:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Segment
     - Notebooks
   * - Tour
     - ``example_process_fdn``, ``example_vanilla_FDN``
   * - Hands-on 1 — build an FDN
     - ``example_vanilla_FDN``, ``example_absorption_geq``,
       ``example_delay_matrix_density``
   * - Hands-on 2 — optimise an FDN
     - ``example_train_colorless_FDN``, ``example_colorless_FDN``,
       ``example_rir_to_fdn``
   * - Beyond the vanilla FDN
     - ``example_paraunitary_fdn``, ``example_scattering_fdn``,
       ``example_coupled_rooms``, ``example_multislope_rir_to_fdn``
   * - Real time
     - ``example_fdn_to_faust`` — compiling a design to FAUST with
       `adac <https://github.com/cucuwritescode/adac>`_

.. note::

   The deck is a snapshot of the toolbox as it was at the conference. Its
   sources live in ``docs/tutorials/dafx2026/`` and render without executing
   any Python, so it keeps working as the API moves on — see that directory's
   ``README.md`` for the reasoning and the freezing procedure.
