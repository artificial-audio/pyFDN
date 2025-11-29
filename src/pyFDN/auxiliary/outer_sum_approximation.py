from __future__ import annotations
from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np

import math

def outer_sum_approximation(matrix: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Rank-1 approximation minimizing ``||u + v^T - matrix||_F``."""

    mat = np.asarray(matrix, dtype=float)
    max_val = np.max(mat)
    if max_val == 0:
        return np.zeros(mat.shape[0]), np.zeros(mat.shape[1])

    exp_mat = np.exp(mat / max_val)
    U, S, Vh = np.linalg.svd(exp_mat, full_matrices=False)
    eu = U[:, 0] * math.sqrt(S[0])
    ev = Vh[0, :] * math.sqrt(S[0])

    u = np.log(np.abs(eu)) * max_val
    v = np.log(np.abs(ev)) * max_val
    return u, v
