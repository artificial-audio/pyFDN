"""Householder reflection matrix.

Translation of householderMatrix.m from fdnToolbox.
"""

from __future__ import annotations

import numpy as np


def householder_matrix(u: np.ndarray) -> np.ndarray:
    """Create a Householder reflection matrix from a vector.

    ``H = I - 2 * (u u^T) / (u^T u)``

    Args:
        u: Vector orthogonal to the reflection hyperplane, shape ``(N,)``.

    Returns:
        Householder matrix of shape ``(N, N)``.  Orthogonal and symmetric.
    """
    u = np.asarray(u, dtype=float).ravel()
    u = u / np.linalg.norm(u)
    return np.eye(len(u)) - 2.0 * np.outer(u, u)
