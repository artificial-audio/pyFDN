"""Paraunitary (lossless FIR) feedback matrices.

Translations of fdnToolbox's degreeOneLossless, shiftMatrix,
shiftMatrixDistribute, randomMatrixShift,
constructCascadedParaunitaryMatrix and
constructParaunitaryFromElementals.

Reference:
    Vaidyanathan, "Multirate Systems and Filter Banks," Prentice Hall, 1993,
    p. 732.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import hadamard

from pyFDN.auxiliary.math import matrix_convolution
from pyFDN.auxiliary.utils import ensure_3d, last_nonzero_indices

from .orthogonal import random_orthogonal


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


def shift_matrix(mat: ArrayLike, shift: ArrayLike, direction: str) -> np.ndarray:
    """Shift a polynomial matrix in time-domain by ``shift`` samples."""

    mat_arr = ensure_3d(mat).copy()
    shift_vec = np.asarray(shift, dtype=int)

    if direction == "left":
        if shift_vec.ndim != 1 or shift_vec.size != mat_arr.shape[0]:
            raise ValueError("Shift vector must match number of rows")
        required_space = last_nonzero_indices(mat_arr) + shift_vec[:, np.newaxis]
        additional_space = int(np.max(required_space) - mat_arr.shape[2])
        if additional_space > 0:
            pad = np.zeros(
                (mat_arr.shape[0], mat_arr.shape[1], additional_space),
                dtype=mat_arr.dtype,
            )
            mat_arr = np.concatenate([mat_arr, pad], axis=2)
        for idx in range(mat_arr.shape[0]):
            mat_arr[idx, :, :] = np.roll(mat_arr[idx, :, :], shift_vec[idx], axis=1)
        return mat_arr

    if direction == "right":
        if shift_vec.ndim != 1 or shift_vec.size != mat_arr.shape[1]:
            raise ValueError("Shift vector must match number of columns")
        required_space = last_nonzero_indices(mat_arr) + shift_vec[np.newaxis, :]
        additional_space = int(np.max(required_space) - mat_arr.shape[2])
        if additional_space > 0:
            pad = np.zeros(
                (mat_arr.shape[0], mat_arr.shape[1], additional_space),
                dtype=mat_arr.dtype,
            )
            mat_arr = np.concatenate([mat_arr, pad], axis=2)
        for idx in range(mat_arr.shape[1]):
            mat_arr[:, idx, :] = np.roll(mat_arr[:, idx, :], shift_vec[idx], axis=1)
        return mat_arr

    raise ValueError("direction must be 'left' or 'right'")


def shift_matrix_distribute(
    mat: ArrayLike, sparsity: float, *, pulse_size: int | None = None
) -> np.ndarray:
    """Randomly distribute time shifts for a polynomial matrix."""

    mat_arr = ensure_3d(mat)
    if pulse_size is None:
        indices = last_nonzero_indices(mat_arr)
        pulse_size = int(np.max(indices)) if indices.size else 1
        pulse_size = max(pulse_size, 1)

    n = mat_arr.shape[0]
    base = np.arange(n)
    rand_left_shift = np.floor(sparsity * (base + np.random.rand(n) * 0.99)).astype(int)
    return rand_left_shift * pulse_size


def random_matrix_shift(
    max_shift: int, matrix: ArrayLike, matrix_rev: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Randomly shift polynomial matrices in time."""

    mat = ensure_3d(matrix)
    rev = ensure_3d(matrix_rev) if matrix_rev is not None else None
    n = mat.shape[0]

    rand_left: np.ndarray
    rand_right: np.ndarray
    if max_shift >= n:
        rand_left = np.random.permutation(max_shift)[:n]
        rand_right = np.random.permutation(max_shift)[:n]
    elif max_shift <= 0:
        rand_left = np.zeros(n, dtype=int)
        rand_right = np.zeros(n, dtype=int)
    else:
        rand_left = np.asarray(np.random.randint(0, max_shift, size=n))
        rand_right = np.asarray(np.random.randint(0, max_shift, size=n))

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


def construct_cascaded_paraunitary_matrix(
    n: int,
    k: int,
    *,
    sparsity: float = 1.0,
    matrix_type: str = "Hadamard",
    gain_per_sample: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a paraunitary matrix and its reverse response."""

    matrix_type_lower = matrix_type.lower()
    if matrix_type_lower == "hadamard":
        if n & (n - 1) != 0:
            raise ValueError("Hadamard construction requires n to be a power of two")

        def generate_matrix(size: int) -> np.ndarray:
            return hadamard(size) / math.sqrt(size)
    elif matrix_type_lower == "random":
        generate_matrix = random_orthogonal  # type: ignore[assignment]
    else:
        raise ValueError("matrix_type must be 'Hadamard' or 'random'")

    sparsity_vector = np.concatenate(([sparsity], np.ones(max(k - 1, 0))))
    matrix = ensure_3d(generate_matrix(n))
    rev_matrix = ensure_3d(np.linalg.inv(matrix[:, :, 0]))

    pulse_size = 1
    for stage in range(k):
        shift_left = shift_matrix_distribute(
            matrix, sparsity_vector[stage], pulse_size=pulse_size
        )
        gain_diag = np.diag(np.power(gain_per_sample, shift_left))
        R1 = ensure_3d(generate_matrix(n) @ gain_diag)

        matrix = shift_matrix(matrix, shift_left, "left")
        matrix = matrix_convolution(R1, matrix)

        rev_matrix = shift_matrix(rev_matrix, shift_left, "right")
        R1_inv = ensure_3d(np.linalg.inv(R1[:, :, 0]))
        rev_matrix = matrix_convolution(rev_matrix, R1_inv)

        pulse_size = max(int(pulse_size * n * sparsity_vector[stage]), 1)

    return matrix, rev_matrix


def construct_velvet_feedback_matrix(
    n: int, stages: int, sparsity: float
) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper for ``construct_cascaded_paraunitary_matrix`` using Hadamard stages."""

    return construct_cascaded_paraunitary_matrix(
        n, stages, sparsity=sparsity, matrix_type="Hadamard"
    )


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
