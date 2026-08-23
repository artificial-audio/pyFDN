"""Load the small collection of predefined FDN builds."""

from __future__ import annotations

import json
import re
from importlib.resources import files

from .build import fdn_build_from_dict
from .generate.fdn_matrix_gallery import FDNBuild

_PRESET_ROOT = files("pyFDN.resources").joinpath("presets", "colorless_FDN")
_PRESET_RE = re.compile(r"colorless_(?P<initial>init_)?N(?P<size>\d+)_d(?P<delay>\d+)")


def available_fdn_presets() -> tuple[str, ...]:
    """Return the predefined FDN names accepted by :func:`load_fdn_preset`."""

    names = []
    for resource in _PRESET_ROOT.iterdir():
        match = re.fullmatch(r"colorless_(init_)?N(\d+)_d(\d+)\.json", resource.name)
        if match:
            initial = "init_" if match.group(1) else ""
            names.append(f"colorless_{initial}N{match.group(2)}_d{match.group(3)}")
    return tuple(sorted(names))


def load_fdn_preset(name: str, *, fs: float = 48_000.0) -> FDNBuild:
    """Load a predefined FDN as an :class:`pyFDN.FDNBuild`.

    The currently packaged presets are optimized and initial builds from the
    differentiable colorless-FDN examples. They are stored directly in pyFDN's
    readable, versioned JSON build format. Use :func:`pyFDN.build_set_decay`
    to add a desired reverberation time.
    """

    normalized_name = name.removesuffix(".json")
    match = _PRESET_RE.fullmatch(normalized_name)
    if not match:
        choices = ", ".join(available_fdn_presets())
        raise ValueError(f"Unknown FDN preset '{name}'. Available presets: {choices}")

    filename = f"{normalized_name}.json"
    resource = _PRESET_ROOT.joinpath(filename)
    if not resource.is_file():
        choices = ", ".join(available_fdn_presets())
        raise ValueError(f"Unknown FDN preset '{name}'. Available presets: {choices}")

    data = json.loads(resource.read_text(encoding="utf-8"))
    return fdn_build_from_dict(data, fs=fs)
