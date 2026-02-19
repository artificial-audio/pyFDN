"""Delay related functions."""
from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import group_delay

from pyFDN.auxiliary.utils import ensure_3d
from pyFDN.auxiliary.math import outer_sum_approximation


def ms_to_smp(ms, fs):
    """Convert milliseconds to samples."""
    return np.round(np.array(ms) * fs / 1000).astype(int)


def mgrpdelay(matrix: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Group delay for each entry of an FIR matrix."""

    mat = ensure_3d(matrix)
    n, m, _ = mat.shape
    delays = []
    freq_ref = None
    for row in range(n):
        row_entries = []
        for col in range(m):
            coeffs = mat[row, col, :]
            if np.allclose(coeffs, 0):
                row_entries.append(np.full(512, np.nan, dtype=float))
                continue
            w, gd = group_delay((coeffs, [1.0]))
            if freq_ref is None:
                freq_ref = w
            if gd.size < w.size:
                padded = np.full(w.size, np.nan, dtype=float)
                padded[: gd.size] = gd
                gd = padded
            row_entries.append(gd)
        delays.append(row_entries)
    if freq_ref is None:
        freq_ref = np.linspace(0.0, np.pi, 512)
    return np.asarray(delays), freq_ref


def matrix_delay_approximation(matrix: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Rank-1 approximation of matrix group delay."""

    GD, _ = mgrpdelay(matrix)
    GD[np.isinf(GD)] = np.nan
    matrix_delay = np.nanmean(GD, axis=2)

    gdl, gdr = outer_sum_approximation(matrix_delay)
    approximation = gdl + gdr
    approximation_error = gdl[:, np.newaxis] + gdr[np.newaxis, :] - matrix_delay
    return approximation, approximation_error
