from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def mag2db(magnitude: ArrayLike) -> np.ndarray:
    """Convert magnitudes to decibels with numerical guard."""

    mag = np.asarray(magnitude, dtype=float)
    tiny = np.finfo(float).tiny
    return 20.0 * np.log10(np.maximum(np.abs(mag), tiny))
