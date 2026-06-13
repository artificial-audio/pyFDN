"""Random paraunitary matrix from elemental degree-one factors.

Translation of constructParaunitaryFromElementals.m from fdnToolbox.

Reference:
    Vaidyanathan, "Multirate Systems and Filter Banks," Prentice Hall, 1993,
    p. 732.
"""

from __future__ import annotations

import numpy as np

from pyFDN.auxiliary.math import matrix_convolution
from pyFDN.auxiliary.utils import ensure_3d
from pyFDN.generate.degree_one_lossless import degree_one_lossless
from pyFDN.generate.random_orthogonal import random_orthogonal


def construct_paraunitary_from_elementals(
    n: int, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a random paraunitary matrix as a cascade of elemental factors.

    The matrix is a random orthogonal matrix multiplied by ``degree - 1``
    random degree-one lossless factors ``V(z) = (I - vv^T) + z^{-1} vv^T``.

    Parameters
    ----------
    n : int
        Size of the paraunitary matrix.
    degree : int
        Polynomial degree of the matrix (number of taps).

    Returns
    -------
    matrix : (n, n, degree) ndarray
        Random paraunitary FIR matrix in z^{-1} convention.
    v : (n, degree - 1) ndarray
        The unit-norm direction vectors of the elemental factors.
    """
    if degree < 1:
        raise ValueError("degree must be at least 1")

    matrix = ensure_3d(random_orthogonal(n))

    v = np.random.randn(n, degree - 1)
    v = v / np.sqrt(np.sum(v**2, axis=0, keepdims=True))
    for it in range(degree - 1):
        matrix = matrix_convolution(matrix, degree_one_lossless(v[:, it]))

    return matrix, v
