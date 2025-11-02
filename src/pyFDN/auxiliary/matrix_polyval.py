"""Matrix polynomial operations."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

def matrix_polyval(P: ArrayLike, z: complex) -> np.ndarray:
    """Evaluate a matrix polynomial ``P`` at the complex point ``z``."""

    P_arr = np.asarray(P)
    if P_arr.ndim != 3:
        raise ValueError("matrix_polyval expects a 3-D array")
    order = P_arr.shape[2]
    exponents = np.arange(order - 1, -1, -1, dtype=int)
    z_powers = (z ** exponents).reshape((1, 1, order))
    return np.sum(P_arr * z_powers, axis=2)