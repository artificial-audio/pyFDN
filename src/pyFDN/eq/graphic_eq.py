"""Ten-band graphic equalizer design.

This module owns the graphic-EQ band layout, least-squares control problem,
and biquad assembly. Public function names use the established ``geq``
abbreviation; ``graphic_eq`` is used when naming the design itself.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from ..auxiliary.utils import hertz_to_rad
from ._backend import array_namespace
from .biquads import highshelf_biquad, lowshelf_biquad, peaking_biquad
from .probe_sos import probe_sos

CENTER_FREQUENCIES = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)
SHELVING_CROSSOVER = np.array([46.0, 11360.0])
BANDWIDTH_R = 2.7
N_GRAPHIC_EQ_BANDS = 10
N_GRAPHIC_EQ_SECTIONS = 11

_NUM_CONTROL = 100
_FFT_LEN = 2**16
_PROTOTYPE_GAIN_DB = 10.0


def _band_omega(fs: float) -> tuple[np.ndarray, np.ndarray]:
    return hertz_to_rad(CENTER_FREQUENCIES, fs), hertz_to_rad(SHELVING_CROSSOVER, fs)


def _geq_sections(
    center_omega: np.ndarray,
    shelving_omega: np.ndarray,
    bandwidth_r: float,
    gain_db: Any,
) -> Any:
    """Turn GEQ command gains into an unnormalized SOS bank."""
    xp = array_namespace(gain_db)
    if xp is np:
        gain_db = np.asarray(gain_db, dtype=float)
    expected = len(center_omega) + 3
    if gain_db.ndim == 0 or gain_db.shape[0] != expected:
        got = 0 if gain_db.ndim == 0 else gain_db.shape[0]
        raise ValueError(f"expected {expected} GEQ gains, got {got}")

    q = math.sqrt(bandwidth_r) / (bandwidth_r - 1.0)
    gains = 10.0 ** (gain_db / 20.0)
    sections = []
    for band in range(expected):
        gain = gains[band]
        if band == 0:
            zero, one = xp.zeros_like(gain), xp.ones_like(gain)
            b = xp.stack([gain, zero, zero], 0)
            a = xp.stack([one, zero, zero], 0)
        elif band == 1:
            b, a = lowshelf_biquad(shelving_omega[0], gain)
        elif band == expected - 1:
            b, a = highshelf_biquad(shelving_omega[1], gain)
        else:
            b, a = peaking_biquad(center_omega[band - 2], gain, q)
        sections.append(xp.concatenate([b, a], 0))
    return xp.stack(sections, 0)


@lru_cache(maxsize=8)
def _geq_control_problem(fs: float) -> tuple[np.ndarray, np.ndarray]:
    control_frequencies = np.round(np.logspace(0, np.log10(fs / 2.1), _NUM_CONTROL + 1))
    target_frequencies = np.concatenate([[1.0], CENTER_FREQUENCIES, [float(fs)]])
    interpolation = np.empty((len(control_frequencies), N_GRAPHIC_EQ_BANDS))
    for band in range(N_GRAPHIC_EQ_BANDS):
        unit = np.zeros(N_GRAPHIC_EQ_BANDS)
        unit[band] = 1.0
        interpolation[:, band] = np.interp(
            control_frequencies, target_frequencies, unit
        )

    center_omega, shelving_omega = _band_omega(fs)
    prototype = _geq_sections(
        center_omega,
        shelving_omega,
        BANDWIDTH_R,
        _PROTOTYPE_GAIN_DB * np.ones(N_GRAPHIC_EQ_SECTIONS),
    )
    system, _, _ = probe_sos(prototype, control_frequencies, _FFT_LEN, fs)
    return system / _PROTOTYPE_GAIN_DB, interpolation


def geq_design_matrix(fs: float) -> np.ndarray:
    """Return the constant map from ten band targets to eleven command gains."""
    system, interpolation = _geq_control_problem(float(fs))
    return np.linalg.pinv(system) @ interpolation


def gain_to_geq(
    gain_db: Any,
    fs: float,
    *,
    design_matrix: Any = None,
) -> Any:
    """Design a ten-band graphic EQ from amplitudes in dB.

    ``gain_db`` has shape ``(10,)`` or ``(10, n_channels)`` and is ordered as
    DC, 63 Hz through 8 kHz, and Nyquist.
    """
    xp = array_namespace(gain_db)
    if xp is np:
        gain_db = np.asarray(gain_db, dtype=float)
    if gain_db.ndim == 0 or gain_db.shape[0] != N_GRAPHIC_EQ_BANDS:
        got = 0 if gain_db.ndim == 0 else gain_db.shape[0]
        raise ValueError(f"graphic_eq takes {N_GRAPHIC_EQ_BANDS} gains, got {got}")

    matrix = geq_design_matrix(fs) if design_matrix is None else design_matrix
    if xp is not np and array_namespace(matrix) is np:
        matrix = xp.as_tensor(matrix, dtype=gain_db.dtype, device=gain_db.device)
    center_omega, shelving_omega = _band_omega(float(fs))
    sos = _geq_sections(center_omega, shelving_omega, BANDWIDTH_R, matrix @ gain_db)
    return sos / sos[:, 3:4, ...]


__all__ = [
    "BANDWIDTH_R",
    "CENTER_FREQUENCIES",
    "N_GRAPHIC_EQ_BANDS",
    "N_GRAPHIC_EQ_SECTIONS",
    "SHELVING_CROSSOVER",
    "gain_to_geq",
    "geq_design_matrix",
]
