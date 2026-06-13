"""Degree-one lossless matrix polynomial.

Translation of degreeOneLossless.m from fdnToolbox.

Reference:
    Vaidyanathan, "Multirate Systems and Filter Banks," Prentice Hall, 1993, p. 732.
"""

from __future__ import annotations

import numpy as np


def degree_one_lossless(v: np.ndarray) -> np.ndarray:
    """Build the degree-one lossless polynomial matrix ``V(z) = (I - vv^T) + z^{-1} vv^T``.

    Args:
        v: Vector of shape ``(N,)`` or ``(N, 1)``.

    Returns:
        Polynomial matrix of shape ``(N, N, 2)`` where index ``[..., 0]``
        is the ``z^0`` coefficient and ``[..., 1]`` is the ``z^{-1}`` coefficient.
    """
    v = np.asarray(v, dtype=float).ravel()
    v = v / np.linalg.norm(v)
    vv = np.outer(v, v)
    N = len(v)
    V = np.zeros((N, N, 2))
    V[:, :, 0] = np.eye(N) - vv
    V[:, :, 1] = vv
    return V
