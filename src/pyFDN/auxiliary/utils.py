"""General utility functions."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

def ensure_3d(matrix: ArrayLike) -> np.ndarray:
    """Ensure the matrix has a trailing polynomial dimension."""

    arr = np.asarray(matrix)
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    if arr.ndim == 3:
        return arr
    raise ValueError("Expected a 2-D or 3-D array for polynomial matrices")


def last_nonzero_indices(mat: np.ndarray) -> np.ndarray:
    """Return 1-based indices of the last non-zero element along axis 2."""

    arr = ensure_3d(mat)
    nonzero = np.abs(arr) > 0
    if not np.any(nonzero):
        return np.zeros(arr.shape[:2], dtype=int)
    reversed_nonzero = nonzero[:, :, ::-1]
    first_true = np.argmax(reversed_nonzero, axis=2)
    has_nonzero = np.any(nonzero, axis=2)
    last = np.zeros_like(first_true)
    last[has_nonzero] = arr.shape[2] - first_true[has_nonzero]
    return last


def mag2db(magnitude: ArrayLike) -> np.ndarray:
    """Convert magnitudes to decibels with numerical guard."""

    mag = np.asarray(magnitude, dtype=float)
    tiny = np.finfo(float).tiny
    return 20.0 * np.log10(np.maximum(np.abs(mag), tiny))


def db2mag(db: ArrayLike) -> np.ndarray:
    """Convert decibel values to linear magnitude."""

    db_arr = np.asarray(db, dtype=float)
    return np.power(10.0, db_arr / 20.0)


def hertz2unit(hz: ArrayLike, fs: float) -> np.ndarray:
    """Convert frequency (Hz) to normalized frequency (0–1)."""
    return np.asarray(hz) / fs * 2
