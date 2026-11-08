"""Orthogonal feedback matrices.

Translations of fdnToolbox's randomOrthogonal, nearestOrthogonal,
nearestSignAgnosticOrthogonal, completeOrthogonal,
householderMatrix, AndersonMatrix and tinyRotationMatrix.

References:
    Schlecht and Habets, "Sign-Agnostic Matrix Design for Spatial Artificial
    Reverberation with Feedback Delay Networks," AES Conf. on Spatial
    Reproduction, 2018.

    Anderson et al., "Flatter Frequency Response from Feedback Delay Network
    Reverbs," Proc. ICMC, 2015.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag

from pyFDN.auxiliary.utils import as_generator


def random_orthogonal(
    N: int, *, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Generate a random orthogonal matrix distributed according to the Haar measure.

    ``rng`` is a NumPy ``Generator`` or an integer seed, as for every random
    function in pyFDN; ``None`` draws fresh entropy.
    """
    q, r = np.linalg.qr(as_generator(rng).standard_normal((N, N)))
    d = np.sign(np.diag(r))
    d[d == 0] = 1
    return q * d


def nearest_orthogonal(A: np.ndarray) -> np.ndarray:
    """Return the nearest orthogonal matrix to A in the Frobenius norm.

    Computed via the polar decomposition: ``B = U V^T`` where
    ``A = U S V^T`` is the SVD of A.

    Parameters
    ----------
    A
        Square real matrix, shape ``(N, N)``.

    Returns
    -------
    ndarray
        Orthogonal matrix of shape ``(N, N)``.
    """
    U, _, Vt = np.linalg.svd(np.asarray(A, dtype=float))
    return U @ Vt


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
    tol: float = float(np.finfo(float).eps) * 1e5,
    *,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Find the orthogonal matrix U minimising ``‖A − |U|‖_F``.

    Solves the non-convex problem by repeated random restarts followed by
    a sign-variable-exchange local search.

    Parameters
    ----------
    A
        Input square matrix, shape ``(N, N)``.  Signs are ignored.
    max_trials
        Number of random sign-pattern restarts.
    tol
        Stop early when the Frobenius error is below this value.
    rng
        Generator or integer seed for the random restarts.

    Returns
    -------
    ndarray
        Orthogonal matrix of shape ``(N, N)``.
    """
    A = np.asarray(A, dtype=float)
    A = _sinkhorn_knopp(A**2) ** 0.5

    rng = as_generator(rng)
    best_matrix = nearest_orthogonal(A)
    best_error = np.inf

    for _ in range(max_trials):
        new_orth = np.sign(rng.standard_normal(A.shape))
        new_orth *= new_orth[0, :]
        new_orth *= new_orth[:, 0:1]

        B = _sign_variable_exchange(new_orth, A)
        distance = float(np.linalg.norm(A - np.abs(B), "fro"))
        if distance < best_error:
            best_matrix = B
            best_error = distance
            if best_error < tol:
                break

    return best_matrix


def complete_orthogonal(
    A: np.ndarray,
    num_io: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the orthogonal completion problem for feedback matrix A.

    Finds ``B``, ``C``, ``D`` such that ``V = [[A, B], [C, D]]`` is orthogonal,
    where ``D`` is ``(num_io, num_io)``.  The ``num_io`` smallest singular values
    of ``A`` must be strictly less than 1.

    The construction uses the SVD of A: for the ``num_io`` smallest singular
    values ``σ``, with left/right singular vectors ``U_s`` and ``V_s``

    .. code-block:: text

        B = U_s * diag(sqrt(1 - σ²))
        C = diag(sqrt(1 - σ²)) * V_s^T
        D = -diag(σ)

    Parameters
    ----------
    A
        Feedback matrix of shape ``(N, N)``.
    num_io
        Number of input/output channels.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        ``(B, C, D, V)`` with shapes ``(N, num_io)``, ``(num_io, N)``,
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


def householder_matrix(u: np.ndarray) -> np.ndarray:
    """Create a Householder reflection matrix from a vector.

    ``H = I - 2 * (u u^T) / (u^T u)``

    Parameters
    ----------
    u
        Vector orthogonal to the reflection hyperplane, shape ``(N,)``.

    Returns
    -------
    ndarray
        Householder matrix of shape ``(N, N)``.  Orthogonal and symmetric.
    """
    u = np.asarray(u, dtype=float).ravel()
    u = u / np.linalg.norm(u)
    return np.eye(len(u)) - 2.0 * np.outer(u, u)


def anderson_matrix(
    N: int,
    K: int | None = None,
    matrix_type: str = "hadamard",
    *,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Build an N×N block-circulant orthogonal matrix.

    The matrix is block-diagonal with ``N/K`` blocks of size ``K×K``, then
    row-shifted by ``K`` to produce the block-circulant structure.

    Parameters
    ----------
    N
        Total matrix size.
    K
        Block size.  Defaults to the smallest prime factor of N.
    matrix_type
        Type string passed to :func:`fdn_matrix_gallery` for each
        block (default ``"hadamard"``).
    rng
        Generator or integer seed for random block types.

    Returns
    -------
    ndarray
        Orthogonal matrix of shape ``(N, N)``.
    """
    if K is None:
        K = next(p for p in range(2, N + 1) if N % p == 0)

    if N % K != 0:
        raise ValueError(f"N ({N}) must be divisible by K ({K})")

    from .fdn_matrix_gallery import fdn_matrix_gallery

    num_blocks = N // K
    rng = as_generator(rng)
    blocks = [fdn_matrix_gallery(K, matrix_type, rng=rng) for _ in range(num_blocks)]
    A = block_diag(*blocks)
    return np.roll(A, K, axis=0)


def rotation_matrix_from_angles(angles: ArrayLike, N: int | None = None) -> np.ndarray:
    """Orthogonal matrix with prescribed eigenvalue angles.

    Builds a block-diagonal matrix of 2x2 Givens rotations, one block per
    angle, so the eigenvalues are exp(+-1j * angles). For odd matrix sizes,
    a single eigenvalue at 1 is appended.

    Parameters
    ----------
    angles
        Eigenvalue angles in radians, one per conjugate pair.
    N
        Matrix size; either 2 * len(angles) or 2 * len(angles) + 1
        (default 2 * len(angles)).

    Returns
    -------
    ndarray
        Orthogonal matrix of shape (N, N).
    """
    angles = np.asarray(angles, dtype=float).ravel()
    num_pairs = angles.size
    if N is None:
        N = 2 * num_pairs
    if N not in (2 * num_pairs, 2 * num_pairs + 1):
        raise ValueError(f"Matrix size N={N} requires {N // 2} angles, got {num_pairs}")

    cos, sin = np.cos(angles), np.sin(angles)
    rotation = np.zeros((N, N))
    idx = 2 * np.arange(num_pairs)
    rotation[idx, idx] = cos
    rotation[idx, idx + 1] = -sin
    rotation[idx + 1, idx] = sin
    rotation[idx + 1, idx + 1] = cos
    if N % 2 == 1:
        rotation[-1, -1] = 1.0
    return rotation


def tiny_rotation_matrix(
    N: int,
    delta: float,
    spread: float = 0.1,
    *,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Dense orthogonal matrix with small eigenvalue angles.

    The eigenvalue angles are delta * pi, randomly spread by the factor
    spread. Small angles make the feedback matrix close to the identity in
    effect, which controls how quickly the network mixes (e.g. to model coupled
    rooms). The Givens-rotation core from :func:`rotation_matrix_from_angles`
    is conjugated by a random orthogonal matrix, so the result is dense but
    keeps the prescribed eigenvalues. For odd N one eigenvalue is at 1.

    Translation of tinyRotationMatrix.m from fdnToolbox.

    Parameters
    ----------
    N
        Matrix size.
    delta
        Mean normalized eigenvalue angle.
    spread
        Spreading of the eigenvalue angles (default 0.1).
    rng
        Generator or integer seed.

    Returns
    -------
    ndarray
        Orthogonal matrix of shape (N, N).
    """
    num_pairs = N // 2
    rng = as_generator(rng)
    frequency_spread = 2.0 * (rng.random(num_pairs) - 0.5) * spread + 1.0
    givens = rotation_matrix_from_angles(delta * np.pi * frequency_spread, N)
    q = random_orthogonal(N, rng=rng)
    return q @ givens @ q.T
