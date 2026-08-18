=======
History
=======

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
