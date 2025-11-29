from __future__ import annotations
from numpy.typing import ArrayLike
import numpy as np

from pyFDN.helpers.utils import ensure_3d, last_nonzero_indices

def shift_matrix_distribute(mat: ArrayLike, sparsity: float, *, pulse_size: int | None = None) -> np.ndarray:
    """Randomly distribute time shifts for a polynomial matrix."""

    mat_arr = ensure_3d(mat)
    if pulse_size is None:
        indices = last_nonzero_indices(mat_arr)
        pulse_size = int(np.max(indices)) if indices.size else 1
        pulse_size = max(pulse_size, 1)

    n = mat_arr.shape[0]
    rng = np.random.default_rng()
    base = np.arange(n)
    rand_left_shift = np.floor(sparsity * (base + rng.random(n) * 0.99)).astype(int)
    return rand_left_shift * pulse_size
