#!/usr/bin/env python
"""pyFDN documentation build configuration."""

import os
import sys

# -- Path setup ---------------------------------------------------------------
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import pyFDN  # noqa: E402

# -- Project information ------------------------------------------------------
project = "pyFDN"
copyright = "2026, Artificial Audio Lab"
author = "Artificial Audio Lab"
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
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_marimo", 
]

# Autodoc settings
autodoc_default_options = {"members": True, "undoc-members": True}
autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings (Google / NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True


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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Pygments
pygments_style = "friendly"
pygments_dark_style = "monokai"
highlight_language = "python3"

# -- Options for HTML output --------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "pyFDN"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "logo/logo_pyFDN_3.png"
html_favicon = "logo/logo_pyFDN_3.png"

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
    "**": [],  # Remove left sidebar on all pages
}

# -- Options for LaTeX output -------------------------------------------------
latex_documents = [
    (master_doc, "pyFDN.tex", "pyFDN Documentation", author, "manual"),
]
# -- Marimo configuration -----------------------------------------------------
marimo_notebook_dir = '../examples'
marimo_default_height = '800px'
marimo_default_width = '100%'
marimo_click_to_load = 'overlay'  # Use overlay mode for better performance
marimo_load_button_text = "Load Interactive Notebook"

# Build notebooks serially in-process. The export-mode override below patches the
# builder in *this* process; joblib's parallel workers spawn fresh interpreters
# that never run conf.py, so they'd silently fall back to the hardcoded html-wasm
# export. Caching is disabled so the static export is deterministic across builds.
marimo_parallel_build = False
marimo_cache_notebooks = False

# -- Marimo export mode override ----------------------------------------------
# sphinx-marimo hardcodes ``marimo export html-wasm``, which ships each notebook
# as a Pyodide/WebAssembly app that executes the Python *in the browser*. Our
# examples import ``torch`` (via ``flamo``), and PyTorch has no Pyodide build, so
# every torch-dependent cell fails at runtime with ``ModuleNotFoundError: No
# module named 'torch'`` and the outputs never render.
#
# We override the builder's per-notebook export to use the static
# ``marimo export html`` instead. That runs each notebook once with this build's
# real interpreter (where torch/matplotlib are installed) and bakes the rendered
# outputs (code, plots, audio) into a self-contained HTML file. The notebooks are
# no longer live-interactive, but they render correctly everywhere — which is the
# only option that works for the torch/flamo examples.
#
# This patch lives in conf.py (our repo), so the installed sphinx-marimo package
# stays pristine and deployment (RTD / GitHub Actions) needs no vendored changes.
def _patch_marimo_static_export():
    import subprocess

    try:
        from sphinx_marimo import builder as _marimo_builder
    except Exception as exc:  # pragma: no cover - extension not installed
        print(f"[conf.py] sphinx-marimo not importable, skipping export patch: {exc}")
        return

    def _build_notebook_impl(self, notebook_path, output_dir):
        """Static-HTML replacement for MarimoBuilder._build_notebook_impl."""
        from pathlib import Path

        notebook_root = Path(marimo_notebook_dir).resolve()
        try:
            relative_path = notebook_path.relative_to(self.source_dir)
        except ValueError:
            relative_path = notebook_path.relative_to(notebook_root)

        output_name = str(relative_path).replace("/", "_").replace(".py", "")
        output_path = output_dir / f"{output_name}.html"

        # `html` (not `html-wasm`) executes the notebook server-side with the real
        # venv and embeds the outputs. marimo returns a non-zero exit code when any
        # cell raises during execution, but it still writes the HTML (with the error
        # baked in), so we log a warning and keep going rather than aborting the
        # whole docs build on one bad notebook.
        proc = subprocess.run(
            ["marimo", "export", "html", str(notebook_path), "-o", str(output_path), "--force"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(
                f"[conf.py] WARNING: '{relative_path}' had cell errors during static "
                f"export (output still written):\n{proc.stderr.strip()}"
            )

        return {
            "name": output_name,
            "path": str(relative_path),
            "output": f"notebooks/{output_name}.html",
        }

    _marimo_builder.MarimoBuilder._build_notebook_impl = _build_notebook_impl
    print("[conf.py] Patched sphinx-marimo to export static HTML (marimo export html)")


_patch_marimo_static_export()