from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

def polydiag(p: ArrayLike) -> np.ndarray:
    """Construct a diagonal polynomial matrix from an array of polynomials."""

    arr = np.asarray(p)
    if arr.ndim != 2:
        raise ValueError("polydiag expects a 2-D array of shape (N, order)")
    n, order = arr.shape
    diag_mat = np.zeros((n, n, order), dtype=arr.dtype)
    for idx in range(n):
        diag_mat[idx, idx, :] = arr[idx, :]
    return diag_mat
