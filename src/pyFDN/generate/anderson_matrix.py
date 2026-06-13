"""Anderson block-circulant orthogonal matrix.

Translation of AndersonMatrix.m from fdnToolbox.

Reference:
    Anderson et al., "Flatter Frequency Response from Feedback Delay Network
    Reverbs," Proc. ICMC, 2015.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag

from .fdn_matrix_gallery import fdn_matrix_gallery


def anderson_matrix(
    N: int,
    K: int | None = None,
    matrix_type: str = "Hadamard",
) -> np.ndarray:
    """Build an N×N block-circulant orthogonal matrix.

    The matrix is block-diagonal with ``N/K`` blocks of size ``K×K``, then
    row-shifted by ``K`` to produce the block-circulant structure.

    Args:
        N: Total matrix size.
        K: Block size.  Defaults to the smallest prime factor of N.
        matrix_type: Type string passed to :func:`fdn_matrix_gallery` for each
                     block (default ``"Hadamard"``).

    Returns:
        Orthogonal matrix of shape ``(N, N)``.
    """
    if K is None:
        K = next(p for p in range(2, N + 1) if N % p == 0)

    if N % K != 0:
        raise ValueError(f"N ({N}) must be divisible by K ({K})")

    num_blocks = N // K
    blocks = [fdn_matrix_gallery(K, matrix_type) for _ in range(num_blocks)]
    A = block_diag(*blocks)
    return np.roll(A, K, axis=0)
