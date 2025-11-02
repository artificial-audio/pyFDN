from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

def rt60_to_slope(rt60: ArrayLike, fs: float) -> np.ndarray:
    """Convert a 60 dB decay time to an energy decay slope (dB per sample)."""

    rt_arr = np.asarray(rt60, dtype=float)
    return -60.0 / (rt_arr * fs)
