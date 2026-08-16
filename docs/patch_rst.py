#!/usr/bin/env python3
"""Patch generated RST files to avoid duplicate object descriptions with api_reference."""

import re
from pathlib import Path

DOCS = Path(__file__).parent

# Replace pyFDN Module contents automodule with a link (avoids duplicates with api_reference)
pyfdn_rst = DOCS / "pyFDN.rst"
if pyfdn_rst.exists():
    text = pyfdn_rst.read_text()
    new = """Module contents
---------------

See :doc:`API Reference <api_reference>` for full documentation of all functions and classes."""
    text = re.sub(
        r"Module contents\n-+\n\n\.\. automodule:: pyFDN\n(   :[^\n]+\n)+",
        new + "\n",
        text,
    )
    pyfdn_rst.write_text(text)

# Sub-package "Module contents" sections otherwise re-document every member
# their __init__ re-exports. Keep the package docstring, but omit those members;
# api_reference and the submodule sections remain the canonical targets.
for package_rst in sorted(DOCS.glob("pyFDN.*.rst")):
    package = package_rst.stem  # e.g. "pyFDN.graphicEQ"
    text = package_rst.read_text()
    pattern = rf"(\.\. automodule:: {re.escape(package)}\n(   :[^\n]+\n)+)"
    match = re.search(pattern, text)
    if match:
        block = f".. automodule:: {package}\n   :no-index:\n   :no-members:\n"
        text = text[: match.start()] + block + text[match.end() :]
        package_rst.write_text(text)
