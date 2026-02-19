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
* Echo density (Abel & Huang 2006) for analysing reverberation and mixing time.
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

All main functions are accessible directly from the top-level ``pyFDN`` namespace::

    import numpy as np
    import pyFDN

    fs = 48_000
    delays = np.array([331, 347, 359, 373], dtype=int)

    # energy-preserving feedback matrix
    feedback = pyFDN.random_orthogonal(len(delays))

    # one-pole absorption filters targeting rt60 of 1.2s at dc and 0.9s at nyquist
    absorption = pyFDN.one_pole_absorption(1.2, 0.9, delays, fs)

    # convert delay state-space to standard state-space
    ss_matrix, b, c, d = pyFDN.dss2ss(delays, feedback)

Alternatively, import specific functions directly::

    from pyFDN import random_orthogonal, one_pole_absorption, mag2db

    feedback = random_orthogonal(4)
    absorption = one_pole_absorption(1.2, 0.9, [100, 150, 200, 250], 48_000)


.. Repository index
.. ----------------

``src/pyFDN/auxiliary``
    Matrix polynomial routines (``matrix_convolution``, ``matrix_polyval``,
    ``matrix_polyder``), loop-analysis helpers (``pole_boundaries``,
    ``is_bounding_curve``), and absorption filter design utilities.
``src/pyFDN/generate``
    Random structure generators including ``random_orthogonal``.
``examples``
    Jupyter notebooks: absorption design, vanilla FDN, delay feedback matrix density
    (compare topologies and echo density), colorless FDN, etc.
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
