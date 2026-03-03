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

# Exclude DSP classes from pyFDN.dsp (documented in api_reference)
dsp_rst = DOCS / "pyFDN.dsp.rst"
if dsp_rst.exists():
    import re
    text = dsp_rst.read_text()
    # Add exclude-members (order of :members:/:undoc-members:/:show-inheritance: varies by sphinx-apidoc)
    for module, exclude in [
        ("pyFDN.dsp.dfiltmatrix", "DFiltMatrix"),
        ("pyFDN.dsp.feedback_delay", "FeedbackDelay"),
        ("pyFDN.dsp.filter_matrix", "FilterMatrix"),
        ("pyFDN.dsp", "DFiltMatrix, FeedbackDelay, FilterMatrix"),
    ]:
        # Match automodule block; add exclude-members if not present
        pattern = rf"(\.\. automodule:: {re.escape(module)}\n(   :[^\n]+\n)+)(?=\n(?:\n|\S)|\Z)"
        match = re.search(pattern, text)
        if match and ":exclude-members:" not in match.group(1):
            block = match.group(1).rstrip() + f"\n   :exclude-members: {exclude}\n"
            text = text[: match.start()] + block + text[match.end() :]
    # Remove duplicate exclude-members
    text = re.sub(r"(   :exclude-members:[^\n]+)\n\s*\1\n?", r"\1\n", text)
    dsp_rst.write_text(text)

# Exclude filter classes from pyFDN.auxiliary.filters (documented in api_reference;
# avoids "more than one target found for cross-reference 'shape'")
auxiliary_rst = DOCS / "pyFDN.auxiliary.rst"
if auxiliary_rst.exists():
    import re
    text = auxiliary_rst.read_text()
    pattern = r"(\.\. automodule:: pyFDN\.auxiliary\.filters\n(   :[^\n]+\n)+)(?=\n(?:\n|\S)|\Z)"
    match = re.search(pattern, text)
    if match and ":exclude-members:" not in match.group(1):
        block = match.group(1).rstrip() + "\n   :exclude-members: TFMatrix, ZFilter, ZFIR, ZScalar, ZTF, ZSOS\n"
        text = text[: match.start()] + block + text[match.end() :]
        text = re.sub(r"(   :exclude-members:[^\n]+)\n\s*\1\n?", r"\1\n", text)
        auxiliary_rst.write_text(text)
