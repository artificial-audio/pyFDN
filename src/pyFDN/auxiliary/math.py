"""Matrix polynomial and math operations."""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike

from pyFDN.auxiliary.utils import mag2db, ensure_3d

def poly_degree(polynomial: ArrayLike, var: str, tol: float | None = None) -> int:
    """Return the polynomial degree, matching ``polyDegree.m`` semantics."""

    coeffs = np.asarray(polynomial)
    if coeffs.ndim != 1:
        coeffs = np.ravel(coeffs)
    if coeffs.size == 0:
        return 0

    if tol is None:
        tol = mag2db(np.finfo(float).eps)

    coeff_db = mag2db(coeffs)
    max_coeff = np.max(coeff_db)
    mask = coeff_db - max_coeff > tol
    active = np.nonzero(mask)[0]
    if active.size == 0:
        return 0

    if var == "z^-1":
        return int(active[-1])
    if var == "z^1":
        return int(len(coeffs) - 1 - active[0])
    raise ValueError("var must be 'z^-1' or 'z^1'")


def polyder_rational(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of rational polynomial using quotient rule."""
    # Remove leading zeros
    b = np.trim_zeros(b, 'f')
    a = np.trim_zeros(a, 'f')
    
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
    q = np.trim_zeros(q, 'f')
    p = np.trim_zeros(p, 'f')
    
    if len(q) == 0:
        q = np.array([0.0])
    if len(p) == 0:
        p = np.array([1.0])
    
    return q, p


def negpolyder(b: np.ndarray, a: np.ndarray, dont_truncate: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
        qq[:len(q)] = q
        pp[:len(p)] = p
        q = qq
        p = pp
    
    return q, p


def matrix_polyder(B: np.ndarray, A: np.ndarray, var: str = 'z^1') -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function for polynomial derivative of filter matrices.
    
    Args:
        B: Numerator [FIR, N, M]
        A: Denominator [FIR, N, M]
        var: Variable type {'z^1', 'z^-1'}
        
    Returns:
        Q: Numerator of derivative
        P: Denominator of derivative
    """
    order_B = B.shape[0]
    order_A = A.shape[0]
    Q = np.zeros((order_B, B.shape[1], B.shape[2]), dtype=complex)
    P = np.zeros((order_A, A.shape[1], A.shape[2]), dtype=complex)
    
    for it1 in range(B.shape[1]):
        for it2 in range(B.shape[2]):
            if var == 'z^1':
                q, p = polyder_rational(B[:, it1, it2], A[:, it1, it2])
                Q_len = min(len(q), order_B)
                P_len = min(len(p), order_A)
                Q[:Q_len, it1, it2] = q[:Q_len]
                P[:P_len, it1, it2] = p[:P_len]
            elif var == 'z^-1':
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


def det_polynomial(polynomial_matrix: np.ndarray, var: str) -> np.ndarray:
    """
    Determinant of a polynomial matrix.
    
    Args:
        polynomial_matrix: numpy array of shape (N, N, degree) containing the polynomial coefficients
        var: 'z^1' or 'z^-1'    Returns:
        determinant: Determinant polynomial
    """
    tol_db = -200
    N = polynomial_matrix.shape[1]
    length = polynomial_matrix.shape[2]
    fft_size = length * N
    
    # Computation
    if var == 'z^-1':
        freq_mat = np.fft.fft(polynomial_matrix, fft_size, axis=2)
    elif var == 'z^1':
        freq_mat = np.fft.fft(np.flip(polynomial_matrix, axis=2), fft_size, axis=2)
    else:
        raise ValueError('Variable type not defined')
    
    freq_det = np.zeros(fft_size, dtype=complex)
    for it in range(fft_size):
        freq_det[it] = np.linalg.det(freq_mat[:, :, it])
    
    determinant = np.fft.ifft(freq_det, fft_size).real
    determinant = determinant[:-(N-1)]
    
    # Shorten the determinant numerically
    if var == 'z^-1':
        degree = poly_degree(determinant, var, tol_db)
        determinant = determinant[:degree+1]
    elif var == 'z^1':
        determinant = np.flip(determinant)
        degree = poly_degree(determinant, var, tol_db)
        determinant = determinant[-degree-1:] if degree >= 0 else determinant
    
    return determinant


def matrix_polyval(P: ArrayLike, z: complex) -> np.ndarray:
    """Evaluate a matrix polynomial ``P`` at the complex point ``z``."""

    P_arr = np.asarray(P)
    if P_arr.ndim != 3:
        raise ValueError("matrix_polyval expects a 3-D array")
    order = P_arr.shape[2]
    exponents = np.arange(order - 1, -1, -1, dtype=int)
    z_powers = (z ** exponents).reshape((1, 1, order))
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


def outer_sum_approximation(matrix: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
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
