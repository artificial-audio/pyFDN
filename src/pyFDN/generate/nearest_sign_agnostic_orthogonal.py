"""Nearest orthogonal matrix ignoring element signs.

Translation of nearestSignAgnosticOrthogonal.m from fdnToolbox.

Reference:
    Schlecht and Habets, "Sign-Agnostic Matrix Design for Spatial Artificial
    Reverberation with Feedback Delay Networks," AES Conf. on Spatial
    Reproduction, 2018.
"""

from __future__ import annotations

import numpy as np

from .nearest_orthogonal import nearest_orthogonal


def _sinkhorn_knopp(
    A: np.ndarray, max_iter: int = 1000, tol: float = 1e-9
) -> np.ndarray:
    """Normalise a non-negative matrix to doubly stochastic via Sinkhorn-Knopp."""
    B = A.copy()
    for _ in range(max_iter):
        B /= B.sum(axis=1, keepdims=True) + 1e-300
        B /= B.sum(axis=0, keepdims=True) + 1e-300
        if np.abs(B.sum(axis=0) - 1).max() < tol:
            break
    return B


def _sign_variable_exchange(
    sign_mat: np.ndarray, absolute: np.ndarray, max_iter: int = 100
) -> np.ndarray:
    """Alternate sign matrix and Procrustes step until sign pattern stabilises."""
    curr = sign_mat.copy()
    for _ in range(max_iter):
        prev = curr.copy()
        U, _, Vt = np.linalg.svd(np.sign(curr) * absolute)
        curr = U @ Vt
        if np.all(np.sign(curr) == np.sign(prev)):
            break
    return curr


def nearest_sign_agnostic_orthogonal(
    A: np.ndarray,
    max_trials: int = 100_000,
    tolerance: float = float(np.finfo(float).eps) * 1e5,
) -> np.ndarray:
    """Find the orthogonal matrix U minimising ``‖A − |U|‖_F``.

    Solves the non-convex problem by repeated random restarts followed by
    a sign-variable-exchange local search.

    Args:
        A: Input square matrix, shape ``(N, N)``.  Signs are ignored.
        max_trials: Number of random sign-pattern restarts.
        tolerance: Stop early when the Frobenius error is below this value.

    Returns:
        Orthogonal matrix of shape ``(N, N)``.
    """
    A = np.asarray(A, dtype=float)
    A = _sinkhorn_knopp(A**2) ** 0.5

    best_matrix = nearest_orthogonal(A)
    best_error = np.inf

    for _ in range(max_trials):
        new_orth = np.sign(np.random.randn(*A.shape))
        new_orth *= new_orth[0, :]
        new_orth *= new_orth[:, 0:1]

        B = _sign_variable_exchange(new_orth, A)
        distance = float(np.linalg.norm(A - np.abs(B), "fro"))
        if distance < best_error:
            best_matrix = B
            best_error = distance
            if best_error < tolerance:
                break

    return best_matrix
