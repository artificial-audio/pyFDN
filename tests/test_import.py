"""``import pyFDN`` must not load torch or flamo; the training API loads lazily."""

import subprocess
import sys


def test_import_does_not_load_torch_or_flamo():
    code = (
        "import sys, pyFDN; "
        "assert 'torch' not in sys.modules, 'torch loaded'; "
        "assert 'flamo' not in sys.modules, 'flamo loaded'; "
        "pyFDN.Match; "
        "assert 'torch' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_export_resolves():
    import pyFDN

    missing = [name for name in pyFDN.__all__ if not hasattr(pyFDN, name)]
    assert missing == []
    assert set(pyFDN.__all__) <= set(dir(pyFDN))
