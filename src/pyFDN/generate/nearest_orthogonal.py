"""Nearest orthogonal matrix via polar decomposition.

Translation of nearestOrthogonal.m from fdnToolbox.
"""

from __future__ import annotations

import numpy as np


def nearest_orthogonal(A: np.ndarray) -> np.ndarray:
    """Return the nearest orthogonal matrix to A in the Frobenius norm.

    Computed via the polar decomposition: ``B = U V^T`` where
    ``A = U S V^T`` is the SVD of A.

    Args:
        A: Square real matrix, shape ``(N, N)``.

    Returns:
        Orthogonal matrix of shape ``(N, N)``.
    """
    U, _, Vt = np.linalg.svd(np.asarray(A, dtype=float))
    return U @ Vt
