=======
History
=======

Unreleased
----------

* Add nonlinear and pitch-shifting time-domain operators for shimmer
  reverberation -- ``DCBlocker``, ``ControllableFullWaveRect``, ``SDFD``,
  ``RingModulator``, ``PitchShift`` and ``GranularPitchShift`` -- all usable as
  ``process_fdn`` hooks, plus the ``example_shimmer_fdn`` notebook that walks
  through each one inside the feedback loop of the same 8-line FDN.
* ``is_uniallpass`` no longer solves a singular Lyapunov equation when ``A`` is
  itself lossless (as in the allpass-in-FDN structure). The spectral radius is
  checked first and such a system is reported as not uniallpass, instead of the
  result depending on whether LAPACK chose to warn or raise on the ill-
  conditioned solve.

0.4.1 (2026-08-24)
------------------

* Let ``fdn_build_gallery`` and the target-to-EQ functions optionally return
  the design choices used to produce their coefficients, ready to store in an
  ``FDNPreset``. Build generation now also exposes the delay distribution and
  coprimality options and lives separately from the matrix galleries.
* **Breaking:** rename the ``Fs`` parameter to ``fs`` everywhere it is still
  spelled with a capital -- ``dss_to_flamo``, ``dss_to_pr``, ``delay_module``
  and ``SDN``. Every other sampling-rate argument in the package was already
  ``fs``, and the odd one out forced callers to remember which spelling each
  function wanted. ``SDN`` also renames its ``fs`` attribute and the ``"fs"``
  key of the dictionary ``SDN.compute()`` returns.

0.4.0 (2026-08-23)
------------------

* Add ``FDNPreset`` JSON documents: a baked ``FDNBuild`` plus catalog
  metadata and a controlled vocabulary for delays, matrices, and the three
  filter hooks. ``trainable_from_preset`` restores filter targets as meaningful
  FLAMO parameters only when they reproduce the baked coefficients.
* **Breaking:** replace ``train_fdn``'s ``mode`` string, and the ``target``,
  ``criteria``, ``sparsity_alpha`` and ``mss_nfft`` arguments that went with
  it, with a composed loss object. An objective is now written out --
  ``FlatMagnitude() + 0.2 * Sparsity(param(model, "feedback"))`` -- and every
  loss is a function of the impulse response, carried as a ``Response``, so
  losses own their reference data and one objective can fit two different
  references. ``pyFDN.train.objectives`` and ``build_objective`` are gone.
* **Breaking:** collect every EQ design in ``pyFDN.eq`` (formerly
  ``pyFDN.graphicEQ``) behind one ``EQDesign`` interface, with ``GraphicEQ``,
  ``FirstOrderShelf`` and ``OnePole`` sharing a single implementation across
  the numpy and torch backends.
* **Breaking:** remove ``absorption_filters`` and ``absorption_to_rt``; the FIR
  absorption path they served has no callers left.
* **Breaking:** name the three filter hooks ``post_delay``, ``post_matrix`` and
  ``post_output`` everywhere -- on ``FDNBuild``, in the FLAMO graph, in the
  plots and in the JSON schema -- and add the slot for the third. The build
  schema is version 2; version 1 files no longer load.
* Fix colorless training on a lossless FDN, whose poles sit on the unit circle
  and leave ``|H|`` unbounded. ``build_fdn`` now defaults ``alias_decay_db``
  from ``rt`` (``LOSSLESS_ALIAS_DECAY_DB`` when ``rt`` is None) and
  ``trainable_from_build`` threads it into every module, so a magnitude
  objective sees a bounded response.
* Train the decay: ``AttenuationFilter`` parametrizes the in-loop absorption filter
  by reverberation time per band, so the loop stays contractive for every value
  the parameter can take, and takes either one RT curve for the network or one
  per delay line. ``OutputEQ`` trains the output filter outside the recursion,
  the only part of an FDN that shapes the spectral envelope without touching
  the decay. Both are passed as the ``post_delay`` and ``post_output`` hooks of
  ``trainable_from_build``.
* Add the losses that go with it: ``AsymmetricFlatMagnitude``,
  ``FlatSpectrogram``, ``MatchEnergyDecay``, ``MatchCumulativeEnergy``,
  ``MatchMagnitude``, ``MatchSpectrogram``, ``MatchMelSpectrogram``,
  ``MatchImpulseResponse``, ``Energy``, ``Sparsity``, ``L1`` and ``L2``.
* Resolve trainable parameters by name: ``param(model, "feedback")`` fails
  where you write it with the list of available names, and ``params(model)``
  enumerates them.
* Export the FLAMO graph builders (``assemble_fdn_core``, ``wrap_fdn_shell``,
  ``gain_module``, ``delay_module``, ``matrix_module``, ``fir_matrix_module``,
  ``sos_filter_module``, ``hook_module``, ``AttenuationFilter``, ``OutputEQ``) from
  the top-level namespace.
* Make ``build_to_impz`` apply all three hooks, so it no longer rejects builds
  that carry an output EQ, and make ``extract_build`` refuse a hook it cannot
  bake rather than silently dropping it.
* Add an example notebook that fits every parameter of an FDN, decay included,
  to a measured concert-hall RIR, and document the loss rationale and
  measurements in ``docs/training_losses.rst``.
* Fix stereo orientation in ``labeled_audio``, which decoded a stereo render as
  thousands of channels.
* Check packaging metadata in CI and fail the documentation build on notebook
  cell errors.

0.3.0 (2026-08-17)
------------------

* **Breaking:** replace the ``pyFDN.dsp`` subpackage with ``pyFDN.td``. The
  top-level ``FeedbackDelay``, ``FIRMatrixFilter`` and ``SOSFilterBank``
  exports are gone; use ``pyFDN.td`` operators instead.
* **Breaking:** ship the packaged colorless-FDN presets as JSON rather than
  MATLAB ``.mat`` files.
* Add ``pyFDN.td``, a time-domain block-based processing graph built on NumPy
  alone (no torch, no FFT), with the ``Series``, ``Parallel`` and
  ``Recursion`` connectors and stateful ``Gain``, ``Delay``, ``SOSBank``,
  ``MatrixFIR``, ``MatrixConvolver`` and ``TimeVaryingMatrix`` operators.
* Add ``load_fdn_build``, ``save_fdn_build``, ``fdn_build_from_dict`` and
  ``fdn_build_to_dict`` for reading and writing FDN builds as JSON.
* Add an FDN-to-FAUST example that compiles a pyFDN design with adac, and a
  reverberation-enhancement example.
* Improve the example gallery and stop tracking generated documentation in the
  repository.
* Test on Python 3.14 in CI and the documentation build.

0.2.0 (2026-08-10)
------------------

* Package compact audio examples and predefined colorless-FDN coefficients so
  browser-hosted tutorials work without repository-relative files.
* Add public APIs for packaged audio, FDN presets, and paper references.
* Add generated documentation galleries and wheel-content checks.
* Keep decay-estimation dependencies optional for a smaller core installation.

0.1.0 (2025-07-02)
------------------

* First release on PyPI.
