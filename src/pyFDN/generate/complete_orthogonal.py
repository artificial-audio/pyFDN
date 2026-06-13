"""Orthogonal completion of a feedback matrix.

Translation of completeOrthogonal.m from fdnToolbox.

Reference:
    Schlecht, "Allpass Feedback Delay Networks."
"""

from __future__ import annotations

import numpy as np


def complete_orthogonal(
    A: np.ndarray,
    num_io: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the orthogonal completion problem for feedback matrix A.

    Finds ``b``, ``c``, ``d`` such that ``V = [[A, b], [c, d]]`` is orthogonal,
    where ``d`` is ``(num_io, num_io)``.  The ``num_io`` smallest singular values
    of ``A`` must be strictly less than 1.

    The construction uses the SVD of A: for the ``num_io`` smallest singular
    values ``σ``, with left/right singular vectors ``U_s`` and ``V_s``

    .. code-block:: text

        b = U_s * diag(sqrt(1 - σ²))
        c = diag(sqrt(1 - σ²)) * V_s^T
        d = -diag(σ)

    Args:
        A: Feedback matrix of shape ``(N, N)``.
        num_io: Number of input/output channels.

    Returns:
        ``(b, c, d, V)`` with shapes ``(N, num_io)``, ``(num_io, N)``,
        ``(num_io, num_io)``, and ``(N + num_io, N + num_io)`` respectively.
    """
    A = np.asarray(A, dtype=float)

    U_A, sigma_A, Vt_A = np.linalg.svd(A)

    # Indices of the num_io smallest singular values
    idx = np.argsort(sigma_A)[:num_io]
    U_s = U_A[:, idx]  # (N, num_io)
    V_s = Vt_A[idx, :].T  # (N, num_io) right singular vectors

    scale = np.sqrt(np.maximum(1.0 - sigma_A[idx] ** 2, 0.0))
    b = U_s * scale  # (N, num_io)
    c = (V_s * scale).T  # (num_io, N)
    d = -np.diag(sigma_A[idx])  # (num_io, num_io)

    V = np.block([[A, b], [c, d]])
    return b, c, d, V
