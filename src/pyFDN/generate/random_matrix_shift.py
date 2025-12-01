from __future__ import annotations
from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np

from pyFDN.generate.shift_matrix import shift_matrix
from pyFDN.auxiliary.utils import ensure_3d

def random_matrix_shift(
    max_shift: int, matrix: ArrayLike, matrix_rev: ArrayLike | None = None
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Randomly shift polynomial matrices in time."""

    mat = ensure_3d(matrix)
    rev = ensure_3d(matrix_rev) if matrix_rev is not None else None
    n = mat.shape[0]

    rng = np.random.default_rng()
    if max_shift >= n:
        rand_left = rng.permutation(max_shift)[:n]
        rand_right = rng.permutation(max_shift)[:n]
    elif max_shift <= 0:
        rand_left = np.zeros(n, dtype=int)
        rand_right = np.zeros(n, dtype=int)
    else:
        rand_left = rng.integers(0, max_shift, size=n)
        rand_right = rng.integers(0, max_shift, size=n)

    rand_left -= rand_left.min()
    rand_right -= rand_right.min()

    shifted = shift_matrix(mat, rand_left, "left")
    shifted = shift_matrix(shifted, rand_right, "right")

    if rev is None:
        shifted_rev = None
    else:
        shifted_rev = shift_matrix(rev, rand_right, "left")
        shifted_rev = shift_matrix(shifted_rev, rand_left, "right")

    return shifted, shifted_rev, rand_left, rand_right
