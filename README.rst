=====
pyFDN
=====


.. image:: https://img.shields.io/pypi/v/pyFDN.svg
        :target: https://pypi.python.org/pypi/pyFDN

.. image:: https://github.com/artificial-audio/pyFDN/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/artificial-audio/pyFDN/actions/workflows/ci.yml
        :alt: CI Status

.. image:: https://readthedocs.org/projects/pyFDN/badge/?version=latest
        :target: https://pyFDN.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/python-3.10%20%7C%203.11-blue
        :target: https://www.python.org/downloads/
        :alt: Python versions

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
        :target: https://opensource.org/licenses/MIT
        :alt: License

.. image:: https://img.shields.io/badge/coverage-68%25-brightgreen
    :target: https://github.com/artificial-audio/pyFDN/actions/workflows/ci.yml
    :alt: Test coverage


Python library for Feedback Delay Networks


* Free software: MIT license
* Documentation: https://pyFDN.readthedocs.io.


Overview
--------

``pyFDN`` provides building blocks for designing, simulating, and analysing Feedback Delay Networks (FDNs). The package focuses on reusable, tested helper functions that simplify typical FDN workflows such as creating orthogonal feedback matrices, designing loop filters, and inspecting pole locations. Using ``flamo`` as a dependency, ``pyFDN`` allows modular design of advanced FDN structure with DSP operations in time and frequency domain.


Highlights
----------

* Matrix polynomial helpers for evaluating, differentiating, and convolving FIR/IIR blocks.
* Loop analysis utilities including pole boundary estimation and curve bounding checks.
* Acoustic absorption design tools that translate RT60 targets into one-pole loop filters.
* Random orthogonal matrix generation to prototype energy-preserving feedback networks.


Installation
------------

Install the current release from PyPI::

    pip install pyFDN

For local development, create a virtual environment and install the package in
editable mode together with the optional tooling::

    python -m venv .venv
    source .venv/bin/activate
    pip install -e .


Quick start
-----------

.. The snippet below sketches the main steps involved in assembling a simple
.. four-delay FDN and inspecting its stability bounds::

..     from types import SimpleNamespace
..     import numpy as np
..     from pyFDN import one_pole_absorption, pole_boundaries, random_orthogonal

..     fs = 48_000
..     delays = np.array([331, 347, 359, 373], dtype=int)

..     # Energy-preserving feedback matrix (shape: N x N x 1)
..     feedback = random_orthogonal(len(delays))[..., np.newaxis]

..     # Match-loop decay targets (seconds) at DC and Nyquist
..     rt60 = np.full(len(delays), 0.6)
..     sos = one_pole_absorption(rt60, 1.4 * rt60, delays, fs)  # SOS format: (6, N)
..     # Convert SOS to b, a format for pole_boundaries: b shape (N, 1, 1), a shape (N, 1, 2)
..     b = sos[0:1, :].T[:, np.newaxis, :]  # b0, shape (N, 1, 1)
..     a = np.stack([sos[3, :], sos[4, :]], axis=1)[:, np.newaxis, :]  # [a0, a1], shape (N, 1, 2)
..     absorption = SimpleNamespace(b=b, a=a)

..     lower, upper, freqs = pole_boundaries(delays, absorption, feedback, fs)
..     print(f"Stability window @ {freqs[0]:.1f} Hz: {lower[0]:.3f} – {upper[0]:.3f}")

.. For matrix-polynomial manipulation without SciPy, the :mod:`pyFDN.auxiliary`
.. package exposes convenience functions such as ``matrix_convolution``,
.. ``matrix_polyval``, and ``TFMatrix``.


.. Repository index
.. ----------------

``src/pyFDN/auxiliary``
    Matrix polynomial routines (``matrix_convolution``, ``matrix_polyval``,
    ``matrix_polyder``), loop-analysis helpers (``pole_boundaries``,
    ``is_bounding_curve``), and absorption filter design utilities.
``src/pyFDN/generate``
    Random structure generators including ``random_orthogonal``.
``src/pyFDN/examples``
    Jupyter notebooks that demonstrate absorption design workflows.
``tests``
    Pytest-based regression suite covering the numerical helpers.
``docs``
    Sphinx project used to publish https://pyFDN.readthedocs.io/.


Development
-----------

Run the test suite (the configuration mirrors CI and emits coverage details)::

    tox -e py311

Or, inside an activated virtual environment::

    pytest --cov=src/pyFDN --cov-report=term-missing

For linting and packaging helpers see ``Makefile`` (``make lint``/``make docs``)
and ``tox.ini`` for multi-environment testing.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
