"""Save and load any pyFDN object, and the presets that ship with pyFDN.

A :class:`~pyFDN.DSS` is saved as readable JSON: its five fields as nested
lists, one matrix row per line, plus the pyFDN version that wrote it.

Anything else -- a ``td`` graph, a FLAMO model -- is two pickles: a header
naming the pyFDN and Python versions that wrote it, then the object itself,
written with :mod:`cloudpickle` so FLAMO models (which hold lambdas) save too.
There is no format to maintain for these: when a refactor breaks an old file,
map the moved class in :data:`RENAMED` or rebuild the object, then
:func:`resave_fdn` it. Loading a pickle runs code from the file, so only load
pickles you or pyFDN wrote.
"""

from __future__ import annotations

import io
import json
import pickle
import platform
import re
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np

from pyFDN.dss import DSS

DSS_FORMAT = "pyfdn-dss"

RENAMED: dict[tuple[str, str], tuple[str, str]] = {}
"""``(old module, old name) -> (new module, new name)`` for classes that moved
since a pickle was saved, e.g. ``{("pyFDN.dss", "DSS"): ("pyFDN.system", "DSS")}``."""

_PRESETS = files("pyFDN.resources").joinpath("presets", "colorless_FDN")


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(*RENAMED.get((module, name), (module, name)))


def _versions() -> dict[str, str]:
    from pyFDN import __version__

    return {"pyfdn_version": __version__, "python": platform.python_version()}


def _dss_json(dss: DSS) -> str:
    fields = {name: np.asarray(value).tolist() for name, value in dss._asdict().items()}
    header = {"format": DSS_FORMAT, "pyfdn_version": _versions()["pyfdn_version"]}
    text = json.dumps({**header, **fields}, indent=2, allow_nan=False)
    # one row per line: collapse every innermost list onto a single line
    return re.sub(r"\[[^\[\]]*\]", lambda m: json.dumps(json.loads(m[0])), text) + "\n"


def _dss_from_json(data: bytes, where: str) -> DSS:
    fields = json.loads(data)
    missing = [name for name in DSS._fields if name not in fields]
    if fields.get("format") != DSS_FORMAT or missing:
        raise ValueError(f"{where} is not a {DSS_FORMAT} file (missing {missing})")
    return DSS(
        np.asarray(fields["delays"], dtype=int),
        *(np.asarray(fields[name], dtype=float) for name in "ABCD"),
    )


def save_fdn(obj: Any, path: str | PathLike[str]) -> None:
    """Save ``obj`` together with the version of pyFDN that wrote it.

    A :class:`~pyFDN.DSS` becomes readable JSON. Anything else that pickles --
    a ``td`` graph, a FLAMO model, a ``(dss, filters)`` pair -- is pickled. A
    FLAMO model's lambdas are stored as Python bytecode, so its file loads only
    on the same Python minor version; nothing else depends on it.
    """
    if isinstance(obj, DSS):
        Path(path).write_text(_dss_json(obj), encoding="utf-8")
    else:
        Path(path).write_bytes(pickle.dumps(_versions()) + cloudpickle.dumps(obj))


def _load(data: bytes, where: str) -> Any:
    if data.lstrip()[:1] == b"{":
        return _dss_from_json(data, where)
    file = io.BytesIO(data)
    header = pickle.load(file)
    try:
        return _Unpickler(file).load()
    except Exception as err:
        now = _versions()
        raise RuntimeError(
            f"{where} was saved with pyFDN {header['pyfdn_version']} on Python "
            f"{header['python']}; this is pyFDN {now['pyfdn_version']} on Python "
            f"{now['python']}. Map a moved class in pyFDN.save.RENAMED, or "
            "rebuild the object, and resave it."
        ) from err


def load_fdn(path: str | PathLike[str]) -> Any:
    """Load an object written by :func:`save_fdn`.

    A JSON file is read as data. A pickle runs code from the file, so only load
    pickles you trust.
    """
    return _load(Path(path).read_bytes(), str(path))


def resave_fdn(path: str | PathLike[str]) -> None:
    """Rewrite a saved file with the installed pyFDN, applying :data:`RENAMED`."""
    save_fdn(load_fdn(path), path)


def available_fdn_presets() -> tuple[str, ...]:
    """Return the names accepted by :func:`get_fdn_preset`."""
    names = (r.name for r in _PRESETS.iterdir() if r.name.endswith(".json"))
    return tuple(sorted(name.removesuffix(".json") for name in names))


def get_fdn_preset(name: str) -> DSS:
    """Return a packaged FDN as a :class:`~pyFDN.DSS`.

    The colorless presets come from Dal Santo et al., *Differentiable Feedback
    Delay Network For Colorless Reverberation* (MIT licence, see
    ``pyFDN/resources/ATTRIBUTION.md``). They were designed at 48 kHz.
    """
    name = name.removesuffix(".json")
    if name not in available_fdn_presets():
        choices = ", ".join(available_fdn_presets())
        raise ValueError(f"Unknown FDN preset '{name}'. Available presets: {choices}")
    return _load(_PRESETS.joinpath(f"{name}.json").read_bytes(), f"preset {name}")


__all__ = [
    "DSS_FORMAT",
    "RENAMED",
    "available_fdn_presets",
    "get_fdn_preset",
    "load_fdn",
    "resave_fdn",
    "save_fdn",
]
