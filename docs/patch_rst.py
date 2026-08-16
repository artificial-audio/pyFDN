#!/usr/bin/env python3
"""Patch generated RST files to avoid duplicate object descriptions with api_reference."""

from pathlib import Path

DOCS = Path(__file__).parent

# Replace pyFDN Module contents automodule with a link (avoids duplicates with api_reference)
pyfdn_rst = DOCS / "pyFDN.rst"
if pyfdn_rst.exists():
    import re

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
