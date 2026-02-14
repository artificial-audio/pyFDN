"""Regression tests for the flattened pyFDN public API."""

from __future__ import annotations

import json
import subprocess
import sys

import pyFDN
from pyFDN._public_api import EXPORT_MAP, EXPORTS


def _run_python_probe(script: str) -> dict[str, bool]:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_all_manifest_exports_resolve_from_pyFDN():
    """Every symbol in the manifest should resolve from top-level pyFDN."""
    for name in EXPORTS:
        assert hasattr(pyFDN, name)
        assert getattr(pyFDN, name) is not None


def test_dir_contains_manifest_exports():
    """dir(pyFDN) should expose all manifest entries for introspection/indexers."""
    assert set(EXPORTS).issubset(set(dir(pyFDN)))


def test_manifest_has_no_duplicates():
    """Manifest should be one-to-one by exported symbol name."""
    assert len(EXPORTS) == len(set(EXPORTS))
    assert len(EXPORTS) == len(EXPORT_MAP)


def test_import_pyFDN_does_not_eager_load_torch():
    """Base import should remain lightweight and avoid importing torch eagerly."""
    probe = """
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
import pyFDN

print(json.dumps({"torch_loaded": "torch" in sys.modules}))
"""
    result = _run_python_probe(probe)
    assert result["torch_loaded"] is False


def test_accessing_recursive_symbol_triggers_required_module_load():
    """Accessing recursive API should import torch via recursive modules."""
    probe = """
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd() / "src"))
import pyFDN

before = "torch" in sys.modules
_ = pyFDN.RecursionCore
after = "torch" in sys.modules

print(json.dumps({"before": before, "after": after}))
"""
    result = _run_python_probe(probe)
    assert result["before"] is False
    assert result["after"] is True
