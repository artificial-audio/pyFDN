from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

def slope_to_rt60(slope: ArrayLike, fs: float) -> np.ndarray:
    """Convert slope (dB/sample) to T60 in seconds."""
    return -60.0 / (slope * fs)
