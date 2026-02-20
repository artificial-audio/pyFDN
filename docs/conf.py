#!/usr/bin/env python
"""pyFDN documentation build configuration."""

import os
import sys

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pyFDN  # noqa: E402

# -- Symlink examples into docs so nbsphinx can find them --------------------
_examples_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
_examples_dst = os.path.join(os.path.dirname(__file__), 'examples')
if not os.path.exists(_examples_dst):
    os.symlink(_examples_src, _examples_dst)

# -- Project information ------------------------------------------------------
project = "pyFDN"
copyright = "2025, Sebastian J. Schlecht, Facundo Franchino, Jeremy Bai"
author = "Sebastian J. Schlecht, Facundo Franchino, Jeremy Bai"
version = pyFDN.__version__
release = pyFDN.__version__

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

# Autodoc settings
autodoc_default_options = {"members": True, "undoc-members": True}
autosummary_generate = True

# Napoleon settings (Google / NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# nbsphinx settings
nbsphinx_execute = "never"  # Don't re-execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note::

        | This page was generated from a Jupyter notebook.
        | :raw-html:`<a href="https://github.com/artificial-audio/pyFDN/blob/main/{{ docname }}"><img src="https://img.shields.io/badge/GitHub-view%20source-blue?logo=github" alt="View on GitHub"></a>`
        | :raw-html:`<a href="{{ env.docname.split('/')[-1] }}.ipynb" download><img src="https://img.shields.io/badge/Download-notebook-orange?logo=jupyter" alt="Download notebook"></a>`
"""

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Source settings
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Pygments
pygments_style = "default"
highlight_language = "python3"

# -- Options for HTML output --------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "pyFDN"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/artificial-audio/pyFDN",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    # Navigation
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "navigation_with_keys": True,
    "navigation_depth": 3,
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_context = {
    "default_mode": "light",
}

html_sidebars = {
    "index": [],
}

# -- Options for LaTeX output -------------------------------------------------
latex_documents = [
    (master_doc, "pyFDN.tex", "pyFDN Documentation", author, "manual"),
]
