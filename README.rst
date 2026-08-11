=====
pyFDN
=====

.. image:: docs/logo/logo_pyFDN_4.png
   :alt: pyFDN
   :align: center
   :width: 300px

.. image:: https://img.shields.io/badge/python-3.10%20%7C%203.11-blue
        :target: https://www.python.org/downloads/
        :alt: Python versions

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
        :target: https://opensource.org/licenses/MIT
        :alt: License

.. image:: https://img.shields.io/badge/coverage-68%25-brightgreen
    :target: https://github.com/artificial-audio/pyFDN/actions/workflows/ci.yml
    :alt: Test coverage

`Documentation <https://artificial-audio.github.io/pyFDN/>`_ |
`Examples <https://artificial-audio.github.io/pyFDN/examples_gallery.html>`_ |
`Report a bug <https://github.com/artificial-audio/pyFDN/issues>`_


Overview
--------

``pyFDN`` provides building blocks for designing, simulating, and analysing Feedback Delay Networks (FDNs). The package focuses on reusable, tested helper functions that simplify typical FDN workflows such as creating orthogonal feedback matrices, designing loop filters, and inspecting pole locations. Using ``flamo`` as a dependency, ``pyFDN`` allows modular design of advanced FDN structure with DSP operations in time and frequency domain.

The `pyFDN tutorial
<https://artificial-audio.github.io/pyfdn-tutorial-2026/>`_ provides an introduction to the package.

The source code is MIT-licensed. Small packaged tutorial resources retain the
separate terms recorded in ``src/pyFDN/resources/ATTRIBUTION.md``; in
particular, the Pori concert-hall impulse response is limited to
noncommercial use.


Highlights
----------

* Matrix polynomial helpers for evaluating, differentiating, and convolving FIR/IIR blocks.
* Loop analysis utilities including pole boundary estimation and curve bounding checks.
* Acoustic absorption design tools that translate RT targets into one-pole loop filters.
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

    # one-pole absorption filters targeting RT of 1.2 s at DC and 0.9 s at Nyquist
    absorption = pyFDN.one_pole_absorption(1.2, 0.9, delays, fs)

    # convert delay state-space to standard state-space (A_ss, b, c, d)
    A_ss, b, c, d = pyFDN.dss_to_ss(delays, feedback)

Alternatively, import specific functions directly::

    from pyFDN import random_orthogonal, one_pole_absorption, lin_to_db

    feedback = random_orthogonal(4)
    absorption = one_pole_absorption(1.2, 0.9, [100, 150, 200, 250], 48_000)


Development
-----------

Run the test suite (the configuration mirrors CI and emits coverage details)::

    tox -e py311

Or, inside an activated virtual environment::

    pytest --cov=src/pyFDN --cov-report=term-missing

For linting and packaging helpers see ``Makefile`` (``make lint``/``make docs``)
and ``tox.ini`` for multi-environment testing.


Helpful links
-----------

* `List of papers on Feedback Delay Networks <https://github.com/gdalsanto/delay-network-reverbs>`_ .
* `Multislope Estimation library <https://github.com/artificial-audio/multislope>`_ .
* `FLAMO (An Open-Source Library for Frequency-Domain Differentiable Audio Processing) <https://github.com/gdalsanto/flamo>`_ .
* `ADAC (Automatic Differentiable Audio Compilation) library <https://github.com/cucuwritescode/adac>`_ .