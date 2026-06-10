"""Matrix polynomial and math operations."""

from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm, logm

from pyFDN.auxiliary.utils import ensure_3d, lin_to_db
from pyFDN.generate.is_almost_zero import is_almost_zero

if TYPE_CHECKING:
    import torch


def interpolate_orthogonal(A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
    """Geodesic interpolation between two orthogonal matrices.

    C(t) = A @ expm(t * logm(A.T @ B)). C(0)=A, C(1)=B; each C(t) is orthogonal.
    """
    M = A.T @ B
    return A @ expm(t * logm(M))


def is_orthogonal(Q: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if Q is orthogonal (Q.T @ Q ≈ I)."""
    return np.allclose(Q.T @ Q, np.eye(Q.shape[0]), atol=tol)


def _is_diagonally_equivalent_to_orthogonal(
    A: np.ndarray, tol: float = 1e-10
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Find Q = E @ A @ D with diagonal E, D such that Q is orthogonal.

    Implements the Berman–Parlett–Plemmons (1981) diagonal scaling approach via
    a rank-1 SVD of the Hadamard quotient C = inv(A) / A^T.

    Returns:
        (is_doe, Q, D, E)
    """
    inv_A = np.linalg.inv(A)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = inv_A / A.T
    # 0/0 entries are structurally zero in both numerator and denominator;
    # for orthogonal (or diagonally similar to orthogonal) matrices the true
    # limit is 1, so substituting 1 preserves the expected rank-1 structure.
    C = np.where(np.isnan(C), 1.0, C)
    U, s, Vh = np.linalg.svd(C)
    d2 = U[:, 0] * np.sqrt(s[0])
    e2 = Vh[0, :] * np.sqrt(s[0])
    # normalise so d2 is non-negative (absorb signs into e2)
    signs = np.sign(d2)
    d2 = d2 * signs
    e2 = e2 * signs
    D = np.diag(np.sqrt(d2))
    E = np.diag(np.sqrt(e2))
    Q = E @ A @ D
    return is_orthogonal(Q, tol=tol), Q, D, E


def is_unilossless(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Test whether A is diagonally similar to an orthogonal matrix.

    A is unilossless if there exists a diagonal D such that D^{-1} @ A @ D is
    orthogonal, i.e. the diagonal scaling is a similarity transform (inv(D) == E).

    Translates ``isDiagonallySimilarToOrthogonal.m`` from fdnToolbox.
    """
    is_doe, _Q, D, E = _is_diagonally_equivalent_to_orthogonal(A, tol=tol)
    return is_doe and is_almost_zero(np.linalg.inv(D) - E, tol=tol)


def poly_degree(polynomial: ArrayLike, tol: float | None = None) -> int:
    """Return the polynomial degree in the z^{-1} convention.

    Coefficients are ordered as [z^0, z^{-1}, z^{-2}, ...]; the degree is
    the index of the last coefficient whose magnitude is above the noise floor.
    """

    coeffs = np.asarray(polynomial)
    if coeffs.ndim != 1:
        coeffs = np.ravel(coeffs)
    if coeffs.size == 0:
        return 0

    if tol is None:
        tol = float(lin_to_db(np.finfo(float).eps))

    coeff_db = lin_to_db(coeffs)
    max_coeff = np.max(coeff_db)
    mask = coeff_db - max_coeff > tol
    active = np.nonzero(mask)[0]
    if active.size == 0:
        return 0

    return int(active[-1])


def polyder_rational(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of rational polynomial using quotient rule."""
    # Remove leading zeros
    b = np.trim_zeros(b, "f")
    a = np.trim_zeros(a, "f")

    if len(b) == 0:
        b = np.array([0.0])
    if len(a) == 0:
        a = np.array([1.0])

    # Compute derivatives of numerator and denominator
    db = np.polyder(b) if len(b) > 1 else np.array([0.0])
    da = np.polyder(a) if len(a) > 1 else np.array([0.0])

    # Apply quotient rule: (b/a)' = (b'*a - b*a') / a^2
    if len(db) == 0:
        db = np.array([0.0])
    if len(da) == 0:
        da = np.array([0.0])

    num1 = np.convolve(db, a)
    num2 = np.convolve(b, da)

    # Pad to same length
    max_len = max(len(num1), len(num2))
    if len(num1) < max_len:
        num1 = np.pad(num1, (max_len - len(num1), 0))
    if len(num2) < max_len:
        num2 = np.pad(num2, (max_len - len(num2), 0))

    q = num1 - num2
    p = np.convolve(a, a)

    # Remove leading zeros from result
    q = np.trim_zeros(q, "f")
    p = np.trim_zeros(p, "f")

    if len(q) == 0:
        q = np.array([0.0])
    if len(p) == 0:
        p = np.array([1.0])

    return q, p


def negpolyder(
    b: np.ndarray, a: np.ndarray, dont_truncate: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derivative of rational polynomial with negative exponents.

    Args:
        b: Numerator coefficients
        a: Denominator coefficients
        dont_truncate: Leading zeros are not truncated

    Returns:
        q: Numerator coefficients of derivative
        p: Denominator coefficients of derivative
    """
    # Flip for substitution x = z^-1
    b_flip = np.flip(b)
    a_flip = np.flip(a)

    # Compute derivative
    q, p = polyder_rational(b_flip, a_flip)

    # Flip for back substitution x^-1 = z
    q = np.flip(q)
    p = np.flip(p)

    # Multiply with -1/z^2
    q = np.convolve(q, np.array([0, 0, -1]))

    # Restore full length if truncation is not desired
    if dont_truncate:
        qq = np.zeros(len(a) + len(b) - 1)
        pp = np.zeros(len(a) + len(a) - 1)
        qq[: len(q)] = q
        pp[: len(p)] = p
        q = qq
        p = pp

    return q, p


def matrix_polyder(B: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of rational filter matrices in the z^{-1} convention.

    Coefficients are ordered as [z^0, z^{-1}, z^{-2}, ...] along axis 0.

    Args:
        B: Numerator coefficients, shape (order, N, M).
        A: Denominator coefficients, shape (order, N, M).

    Returns:
        Q: Numerator of the derivative, shape (order, N, M).
        P: Denominator of the derivative, shape (order, N, M).
    """
    order_B = B.shape[0]
    order_A = A.shape[0]
    Q = np.zeros((order_B, B.shape[1], B.shape[2]), dtype=complex)
    P = np.zeros((order_A, A.shape[1], A.shape[2]), dtype=complex)

    for it1 in range(B.shape[1]):
        for it2 in range(B.shape[2]):
            q, p = negpolyder(B[:, it1, it2], A[:, it1, it2])
            Q_len = min(len(q), order_B)
            P_len = min(len(p), order_A)
            Q[:Q_len, it1, it2] = q[:Q_len]
            P[:P_len, it1, it2] = p[:P_len]

    if np.allclose(Q.imag, 0.0):
        Q = Q.real.astype(float)
    if np.allclose(P.imag, 0.0):
        P = P.real.astype(float)

    return Q, P


def det_polynomial(polynomial_matrix: np.ndarray) -> np.ndarray:
    """Determinant of a polynomial matrix in the z^{-1} convention.

    Coefficients are ordered as [z^0, z^{-1}, z^{-2}, ...] along axis 2.
    Uses an FFT-based approach: evaluate at DFT points, compute scalar det
    at each frequency, then IFFT back.

    Args:
        polynomial_matrix: shape (N, N, L), polynomial matrix entries.

    Returns:
        determinant: 1-D array of determinant polynomial coefficients,
            ordered [z^0, z^{-1}, ...], trimmed to the actual degree.
    """
    N = polynomial_matrix.shape[1]
    length = polynomial_matrix.shape[2]
    fft_size = length * N
    is_real = np.isrealobj(polynomial_matrix)

    if is_real:
        freq_mat = np.fft.rfft(polynomial_matrix, fft_size, axis=2)
        n_freq = fft_size // 2 + 1
        freq_det = np.array([np.linalg.det(freq_mat[:, :, k]) for k in range(n_freq)])
        determinant = np.fft.irfft(freq_det, fft_size)
    else:
        freq_mat = np.fft.fft(polynomial_matrix, fft_size, axis=2)
        freq_det = np.array([np.linalg.det(freq_mat[:, :, k]) for k in range(fft_size)])
        determinant = np.fft.ifft(freq_det).real

    # Hard trim to theoretical degree bound: deg(det) <= N*(L-1)
    determinant = determinant[: fft_size - (N - 1)]

    # Data-driven trim of trailing numerical noise
    abs_max = np.max(np.abs(determinant))
    if abs_max > 0:
        tol = np.finfo(float).eps * N * abs_max
        nz = np.flatnonzero(np.abs(determinant) > tol)
        determinant = determinant[: nz[-1] + 1] if nz.size > 0 else determinant[:1]

    return determinant


def general_char_poly(delays: ArrayLike, A: np.ndarray) -> np.ndarray:
    """
    Generalized characteristic polynomial (GCP) for delay state-space.

    Implements the formula from Schlecht & Habets (2015), Time-varying feedback
    matrices in feedback delay networks. J. Acoust. Soc. Amer., 138(3), 1389-1398.
    Matches the reference generalCharPoly.m.

    Parameters
    ----------
    delays : array-like
        Vector of delays in samples (length ND); used with z^{-1} convention.
    A : ndarray
        Feedback matrix. If 2D (scalar matrix), GCP is built via submatrix
        determinants. If 3D of shape (N, N, L), polynomial matrix in z^{-1}.

    Returns
    -------
    p : ndarray
        Generalized characteristic polynomial coefficients in z^{-1} ordering
        (index 0 = z^0, index k = z^{-k}).
    """
    delays = np.asarray(delays, dtype=int).ravel()
    A = np.asarray(A, dtype=float)
    ND = len(delays)
    N = A.shape[1]

    if A.ndim == 2:
        # Scalar matrix: p_len = sum(d) + 1, iterate over submatrix determinants
        p_len = int(delays.sum()) + 1
        p = np.zeros(p_len)
        p[0] = 1.0
        for nn in range(1, N + 1):
            for ind_tup in combinations(range(ND), nn):
                ind = np.array(ind_tup)
                p_ind = int(delays[ind].sum())
                p[p_ind] += ((-1) ** nn) * np.linalg.det(A[np.ix_(ind, ind)])
        return p

    # Polyphase: A is (N, N, L); m = degree of det(A)
    det_A = det_polynomial(A)
    m = len(det_A) - 1
    p_len = int(delays.sum()) + 1 + m
    p = np.zeros(p_len)
    p[0] = 1.0
    for nn in range(1, N + 1):
        for ind_tup in combinations(range(ND), nn):
            ind = np.array(ind_tup)
            p_ind = int(delays[ind].sum())
            sub = A[np.ix_(ind, ind, np.arange(A.shape[2]))]
            dd = det_polynomial(sub)
            # det(A_sub(z)) * z^{-sum(m_sub)}: coefficient dd[j] of z^{-j}
            # lands at z^{-(p_ind + j)} (matches generalCharPoly.m polyphase branch)
            for j, c in enumerate(dd):
                p[p_ind + j] += ((-1) ** nn) * c
    return p


def matrix_polyval(P: ArrayLike, z: complex) -> np.ndarray:
    """Evaluate a matrix polynomial ``P`` at the complex point ``z``."""

    P_arr = np.asarray(P)
    if P_arr.ndim != 3:
        raise ValueError("matrix_polyval expects a 3-D array")
    order = P_arr.shape[2]
    exponents = np.arange(order - 1, -1, -1, dtype=int)
    z_powers = (z**exponents).reshape((1, 1, order))
    return np.sum(P_arr * z_powers, axis=2)


def polydiag(p: ArrayLike) -> np.ndarray:
    """Construct a diagonal polynomial matrix from an array of polynomials."""

    arr = np.asarray(p)
    if arr.ndim != 2:
        raise ValueError("polydiag expects a 2-D array of shape (N, order)")
    n, order = arr.shape
    diag_mat = np.zeros((n, n, order), dtype=arr.dtype)
    for idx in range(n):
        diag_mat[idx, idx, :] = arr[idx, :]
    return diag_mat


def matrix_convolution(A: ArrayLike, B: ArrayLike) -> np.ndarray:
    """Matrix polynomial multiplication by convolution."""

    A_arr = ensure_3d(A)
    B_arr = ensure_3d(B)
    if A_arr.shape[1] != B_arr.shape[0]:
        raise ValueError("Inner dimensions must agree")

    m, n, order_a = A_arr.shape
    _, k, order_b = B_arr.shape
    result = np.zeros((m, k, order_a + order_b - 1), dtype=np.result_type(A_arr, B_arr))

    for row in range(m):
        for col in range(k):
            acc = np.zeros(order_a + order_b - 1, dtype=result.dtype)
            for inter in range(n):
                acc += np.convolve(A_arr[row, inter, :], B_arr[inter, col, :])
            result[row, col, :] = acc
    return result


def outer_sum_approximation(matrix: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Rank-1 approximation minimizing ``||u + v^T - matrix||_F``."""

    mat = np.asarray(matrix, dtype=float)
    max_val = np.max(mat)
    if max_val == 0:
        return np.zeros(mat.shape[0]), np.zeros(mat.shape[1])

    exp_mat = np.exp(mat / max_val)
    U, S, Vh = np.linalg.svd(exp_mat, full_matrices=False)
    eu = U[:, 0] * math.sqrt(S[0])
    ev = Vh[0, :] * math.sqrt(S[0])

    u = np.log(np.abs(eu)) * max_val
    v = np.log(np.abs(ev)) * max_val
    return u, v


def matrix_sqrt(A: torch.Tensor) -> torch.Tensor:
    """Matrix square root via eigenvalue decomposition.

    sqrtm(A) = V @ sqrt(D) @ V^(-1) where A = V @ D @ V^(-1).

    Parameters
    ----------
    A : torch.Tensor
        Square matrix (real, will be cast to complex for eig).

    Returns
    -------
    torch.Tensor
        Real matrix square root of A.
    """
    import torch

    eigenvals, eigenvecs = torch.linalg.eig(A.to(torch.complex64))
    sqrt_eigenvals = torch.sqrt(eigenvals)
    return torch.real(
        eigenvecs @ torch.diag(sqrt_eigenvals) @ torch.linalg.inv(eigenvecs)
    ).float()
