"""Load the small collection of predefined FDN coefficient sets."""

from __future__ import annotations

import re
from importlib.resources import as_file, files

import numpy as np
from scipy.io import loadmat
from scipy.linalg import expm

from .auxiliary.utils import skew
from .generate.fdn_matrix_gallery import FDNBuild

_PRESET_ROOT = files("pyFDN.resources").joinpath("presets", "colorless_FDN")
_PRESET_RE = re.compile(r"colorless_(?P<initial>init_)?N(?P<size>\d+)_d(?P<delay>\d+)")


def available_fdn_presets() -> tuple[str, ...]:
    """Return the predefined FDN names accepted by :func:`load_fdn_preset`."""

    names = []
    for resource in _PRESET_ROOT.iterdir():
        match = re.fullmatch(r"param_(init_)?N(\d+)_d(\d+)\.mat", resource.name)
        if match:
            initial = "init_" if match.group(1) else ""
            names.append(f"colorless_{initial}N{match.group(2)}_d{match.group(3)}")
    return tuple(sorted(names))


def load_fdn_preset(name: str, *, fs: float = 48_000.0) -> FDNBuild:
    """Load a predefined FDN as an :class:`pyFDN.FDNBuild`.

    The currently packaged presets are the optimized and initial coefficient
    sets from the differentiable colorless-FDN examples. Their stored matrix
    parameters are converted to an orthogonal feedback matrix before return.
    Use :func:`pyFDN.build_set_decay` to add a desired reverberation time.
    """

    match = _PRESET_RE.fullmatch(name.removesuffix(".mat"))
    if not match:
        choices = ", ".join(available_fdn_presets())
        raise ValueError(f"Unknown FDN preset '{name}'. Available presets: {choices}")

    initial = "init_" if match.group("initial") else ""
    filename = f"param_{initial}N{match.group('size')}_d{match.group('delay')}.mat"
    resource = _PRESET_ROOT.joinpath(filename)
    if not resource.is_file():
        choices = ", ".join(available_fdn_presets())
        raise ValueError(f"Unknown FDN preset '{name}'. Available presets: {choices}")

    with as_file(resource) as path:
        data = loadmat(path)

    delays = np.rint(np.asarray(data["m"], dtype=float).ravel()).astype(np.int64)
    feedback = expm(skew(np.asarray(data["A"], dtype=float)))
    input_gain = np.asarray(data["B"], dtype=float).reshape(-1, 1)
    output_gain = np.asarray(data["C"], dtype=float).reshape(1, -1)
    direct = np.zeros((output_gain.shape[0], input_gain.shape[1]))
    return FDNBuild(feedback, input_gain, output_gain, direct, delays, float(fs))
