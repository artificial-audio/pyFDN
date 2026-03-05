============
Contributing
============

Contributions are welcome! This guide covers how to report issues, request features, set up your development environment, and run tests.


Issues and Requests
-------------------

Report bugs or request features on the `GitHub issue tracker <https://github.com/artificial-audio/pyFDN/issues>`_.

Reporting Bugs
^^^^^^^^^^^^^^

Please include:

* Operating system name and version
* Python version
* Detailed steps to reproduce
* Expected vs actual behavior

Requesting Features
^^^^^^^^^^^^^^^^^^^

Please include:

* Clear description of the feature
* Use case and motivation
* Proposed implementation approach (if applicable)

Pull Requests
^^^^^^^^^^^^^

* Include tests for new functionality
* Update documentation as needed
* Ensure all tests pass before submitting


Development Installation
------------------------

1. **Fork and clone the repository**:

   .. code-block:: console

      $ git clone git@github.com:your_name_here/pyFDN.git
      $ cd pyFDN

2. **Create a virtual environment**:

   .. code-block:: console

      $ python -m venv .venv
      $ source .venv/bin/activate

3. **Install pyFDN in editable mode with development dependencies**:

   .. code-block:: console

      $ pip install -e ".[dev]"

4. **Install FLAMO locally** (required for development):

   FLAMO is under active development alongside pyFDN. You are recommended to install it locally in editable mode:

   .. code-block:: console

      $ git clone <flamo-repository-url>
      $ cd flamo
      $ pip install -e .

   Then return to the pyFDN directory. The local FLAMO installation will be used
   by pyFDN during development and testing.

5. **Verify installation**:

   .. code-block:: console

      $ pytest tests/ -v


Building the Documentation
--------------------------

1. **Install docs dependencies** (if not already):

   .. code-block:: console

      $ pip install -e ".[docs]"

2. **Install pandoc** (system dependency required by ``nbsphinx``):

   .. code-block:: console

      $ brew install pandoc          # macOS
      $ sudo apt install pandoc      # Ubuntu / Debian

3. **Build the HTML docs**:

   .. code-block:: console

      $ make docs

   This runs ``sphinx-apidoc``, builds all pages (including executing the
   example notebooks), and opens the result in your browser.

   Alternatively, build only the HTML without opening a browser:

   .. code-block:: console

      $ cd docs
      $ make html

4. **Preview**: Open ``docs/_build/html/index.html`` in your browser.

.. note::

   The example notebooks are executed at build time by ``nbsphinx``. The
   first build may take a few minutes. Subsequent builds are faster because
   Sphinx caches the executed notebooks.


Repository Index
----------------

.. note::
    This section will be filled out later with the goal structure of the repository.


Testing Pipeline
----------------

**Configuration**

Tests are configured via ``pytest`` and ``tox.ini``. The test suite includes:

* Unit tests for individual functions
* Regression tests against MATLAB reference implementations
* Integration tests with FLAMO

**Running Tests**

Run all tests:

.. code-block:: console

   $ pytest

Run specific test file:

.. code-block:: console

   $ pytest tests/test_one_pole_absorption_regression.py

Run with coverage:

.. code-block:: console

   $ pytest --cov=pyFDN --cov-report=html

Run tests across multiple Python versions (requires tox):

.. code-block:: console

   $ tox

**Cross-Validation Example**

The regression test suite validates pyFDN outputs against MATLAB reference data.
Example: ``test_one_pole_absorption_regression.py`` compares Python-generated absorption filter coefficients and impulse responses with MATLAB FDN Toolbox results.

To run a specific cross-validation test:

.. code-block:: console

   $ pytest tests/test_one_pole_absorption_regression.py::test_one_pole_absorption_coefficients -v

The test loads MATLAB reference data from ``tests/reference/``, generates equivalent Python outputs, and validates numerical agreement within specified tolerances.

**Test Structure**

* ``tests/conftest.py``: Shared fixtures (e.g., ``loadmat`` for loading MATLAB files)
* ``tests/reference/``: MATLAB reference data files (``.mat`` format)
* ``tests/test_*.py``: Test modules organized by functionality


Code of Conduct
---------------

Please note that this project is released with a `Contributor Code of Conduct`_.
By participating in this project you agree to abide by its terms.

.. _`Contributor Code of Conduct`: CODE_OF_CONDUCT.rst
